from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from humanizer import __version__
from humanizer.analyzers.model_loader import download_model, is_model_available, model_info
from humanizer.config import AppConfig
from humanizer.pipeline import HumanizerPipeline

console = Console()

_KNOWN_COMMANDS = frozenset({"score", "humanize", "models", "help"})


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_humanized{input_path.suffix or '.txt'}")


def _preprocess_argv(argv: list[str] | None) -> list[str] | None:
    """mh draft.md  ->  mh humanize -f draft.md"""
    if not argv:
        return argv
    out = list(argv)
    if out[0] not in _KNOWN_COMMANDS and not out[0].startswith("-"):
        out = ["humanize", "-f", out[0], *out[1:]]
    elif (
        len(out) >= 2
        and out[0] == "humanize"
        and out[1] not in ("-f", "--file", "-t", "--text", "-o", "--output", "-c", "--config")
        and not out[1].startswith("-")
    ):
        out = ["humanize", "-f", out[1], *out[2:]]
    elif (
        len(out) >= 2
        and out[0] == "score"
        and out[1] not in ("-f", "--file", "-t", "--text", "-c", "--config")
        and not out[1].startswith("-")
    ):
        out = ["score", "-f", out[1], *out[2:]]
    return out


def _resolve_paths(args: argparse.Namespace) -> None:
    if getattr(args, "input_file", None) and not args.file:
        args.file = args.input_file
    if args.file and not args.output and args.command in (None, "humanize"):
        args.output = str(_default_output_path(Path(args.file)))


def _read_input(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    console.print("[red]No input. Use --file, --text, or pipe via stdin.[/red]")
    sys.exit(1)


def _print_span_report(pipeline: HumanizerPipeline, text: str, *, show_spans: bool) -> None:
    result = pipeline.run_sync(text)
    proxy = result.zerogpt_report
    seg = result.segment_report

    if proxy is not None:
        table = Table(title="ZeroGPT-Proxy Detection")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Document score (0=human, 100=AI)", f"{proxy.document_score:.1f}")
        table.add_row("ML-aligned score (ZeroGPT est.)", f"{proxy.ml_document_score:.1f}")
        table.add_row("Hot sentences", str(len(proxy.hot_sentences)))
        if seg and seg.legacy_report:
            table.add_row("Legacy proxy score", f"{seg.legacy_report.composite_score:.1f}")
        console.print(table)

        if show_spans and proxy.sentences:
            threshold = pipeline.config.detector.span_threshold
            console.print("\n[bold]Highlighted sentences (ZeroGPT-proxy):[/bold]")
            for s in sorted(proxy.sentences, key=lambda x: x.score, reverse=True):
                if s.score < threshold and s.index not in proxy.hot_sentences[:3]:
                    continue
                console.print(f"  [{s.index}] {s.score:.1f}  {s.text[:90]!r}")
            console.print(f"\n[dim]Flags: {len(proxy.hot_sentences)} sentences above threshold ({threshold:.1f})[/dim]")
        return

    det = result.iterations[0].detection if result.iterations else None
    if det is None:
        return
    table = Table(title="AI Likelihood Analysis (legacy)")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_row("Composite (0=human, 100=AI)", f"{det.composite_score:.1f}")
    console.print(table)


def _apply_config_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.target is not None:
        config.pipeline.target_ai_score = args.target
    if getattr(args, "iterations", None) is not None:
        config.pipeline.max_iterations = args.iterations
    if getattr(args, "max_passes", None) is not None:
        config.pipeline.max_passes = args.max_passes
    if getattr(args, "span_threshold", None) is not None:
        config.detector.span_threshold = args.span_threshold


def _ensure_ml_deps(config: AppConfig) -> bool:
    if config.detector.engine != "onnx":
        return True
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        console.print(
            "[red]ONNX detector requires ML dependencies. Install with:[/red]\n"
            "  pip install -e \".[full]\"\n"
            "[dim]Or set detector.engine = \"legacy\" in config.[/dim]"
        )
        return False
    if not is_model_available():
        console.print("[yellow]Downloading ONNX detector model (~120 MB, one-time)...[/yellow]")
        try:
            download_model()
        except Exception as exc:
            console.print(f"[red]Model download failed: {exc}[/red]")
            return False
    return True


async def _run_humanize(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    _apply_config_overrides(config, args)

    if not _ensure_ml_deps(config):
        return 1

    text = _read_input(args)
    pipeline = HumanizerPipeline(config)
    engine = config.pipeline.engine

    if engine == "llm" and not config.llm.api_key:
        console.print("[red]LLM engine selected but no API key.[/red]")
        return 1

    max_label = (
        f"Max passes: {config.pipeline.max_passes}"
        if engine == "segment"
        else f"Max iterations: {config.pipeline.max_iterations}"
    )
    console.print(Panel(f"Processing {len(text):,} characters...", title="Manuscript Humanizer"))
    console.print(
        f"[dim]Engine: {engine} | Target score: <={config.pipeline.target_ai_score} | "
        f"{max_label}[/dim]\n"
    )

    initial = pipeline.run_sync(text)
    console.print(f"Initial ZeroGPT-proxy score: [yellow]{initial.initial_score:.1f}[/yellow]")

    if getattr(args, "show_spans", False) and initial.zerogpt_report:
        _print_span_report(pipeline, text, show_spans=True)

    result = await pipeline.run(text)

    console.print(f"\nFinal ML-aligned score: [{'green' if result.success else 'yellow'}]{result.final_score:.1f}[/]")
    if initial.zerogpt_report is not None:
        expected = pipeline.proxy_scorer.expected_zerogpt_score(result.final)
        console.print(f"Expected ZeroGPT (~):  [{'green' if expected <= 10 else 'yellow'}]{expected:.1f}[/]")
        console.print(f"[dim]Composite proxy: {result.zerogpt_report.document_score:.1f}[/dim]")
    console.print(f"Passes used:      {len(result.iterations)}")
    console.print(f"Target met:       {'[green]yes[/green]' if result.success else '[yellow]no[/yellow]'}")
    if config.pipeline.one_shot:
        console.print(
            "[dim]One-shot mode: verify on https://www.zerogpt.com/ "
            "(built-in proxy may not match live detectors).[/dim]"
        )

    if result.iterations:
        last = result.iterations[-1]
        if last.changed_sentences:
            console.print(f"Sentences edited: {last.changed_sentences}")
        if last.applied:
            console.print("[dim]Applied transforms:[/dim]")
            for tag in last.applied[:8]:
                console.print(f"  {tag}")

    if args.output:
        Path(args.output).write_text(result.final, encoding="utf-8")
        console.print(f"\n[green]Saved to {args.output}[/green]")
        console.print("[dim]Verify manually at https://www.zerogpt.com/[/dim]")
    else:
        console.print("\n" + "─" * 60 + "\n")
        console.print(result.final)

    return 0 if result.success else 2


def _run_models_download(_args: argparse.Namespace) -> int:
    console.print("[bold]Downloading AIGC detector ONNX model...[/bold]")
    try:
        path = download_model(force=getattr(_args, "force", False))
        info = model_info(path)
        console.print(f"[green]Model ready at {info['cache_dir']}[/green]")
        return 0
    except Exception as exc:
        console.print(f"[red]Download failed: {exc}[/red]")
        return 1


def _add_common_args(parser: argparse.ArgumentParser, *, positional: bool = False) -> None:
    parser.add_argument("-c", "--config", default="config.toml", help="Config file path")
    parser.add_argument("-f", "--file", help="Input file path")
    if positional:
        parser.add_argument(
            "input_file",
            nargs="?",
            help="Input file (shortcut for --file)",
        )
    parser.add_argument("-t", "--text", help="Input text string")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: <input>_humanized.<ext>)",
    )
    parser.add_argument("--target", type=float, help="Target AI score (default from config)")
    parser.add_argument("--iterations", type=int, help="Max rewrite iterations (legacy engine)")
    parser.add_argument("--max-passes", type=int, dest="max_passes", help="Max targeted passes")
    parser.add_argument("--span-threshold", type=float, dest="span_threshold", help="Sentence flag threshold")
    parser.add_argument("--show-spans", action="store_true", help="Print sentence heatmap")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = _preprocess_argv(argv) or []
    parser = argparse.ArgumentParser(
        prog="mh",
        description=(
            "Humanize academic manuscripts for lower AI-detector scores.\n\n"
            "Examples:\n"
            "  mh manuscript.md\n"
            "  mh humanize draft.txt -o clean.txt\n"
            "  mh score manuscript.md\n"
            "  echo \"text\" | mh score"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_common_args(parser, positional=True)

    sub = parser.add_subparsers(dest="command")
    score_parser = sub.add_parser("score", help="Analyze AI likelihood without rewriting")
    humanize_parser = sub.add_parser(
        "humanize",
        help="Rewrite text (default when you pass a file path)",
    )
    models_parser = sub.add_parser("models", help="Manage ML models")
    models_dl = models_parser.add_subparsers(dest="models_command")
    dl_parser = models_dl.add_parser("download", help="Download ONNX detector model")
    dl_parser.add_argument("--force", action="store_true", help="Re-download model")

    for sub_parser in (score_parser, humanize_parser):
        _add_common_args(sub_parser, positional=True)

    args = parser.parse_args(argv)
    command = args.command or "humanize"
    args.command = command
    _resolve_paths(args)

    if command == "models":
        if args.models_command == "download":
            return _run_models_download(args)
        models_parser.print_help()
        return 1

    if command == "score":
        config = AppConfig.load(args.config)
        _apply_config_overrides(config, args)
        if not _ensure_ml_deps(config):
            return 1
        text = _read_input(args)
        _print_span_report(HumanizerPipeline(config), text, show_spans=getattr(args, "show_spans", True))
        return 0

    return asyncio.run(_run_humanize(args))


if __name__ == "__main__":
    raise SystemExit(main())
