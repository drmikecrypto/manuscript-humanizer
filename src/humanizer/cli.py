from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from humanizer import __version__
from humanizer.analyzers.model_loader import download_model, is_model_available, model_info
from humanizer.config import AppConfig
from humanizer.io import backup_file, load_document, resolve_output_path, save_document
from humanizer.io.documents import DocumentPayload
from humanizer.pipeline import HumanizerPipeline

console = Console()
logger = logging.getLogger("humanizer.cli")

_KNOWN_COMMANDS = frozenset({"score", "humanize", "models", "help"})
_SUPPORTED_EXTENSIONS = frozenset({".md", ".txt", ".tex", ".docx", ".pdf"})


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
    if not args.file:
        return
    input_path = Path(args.file)
    if getattr(args, "in_place", False):
        args.output = str(input_path)
    elif not args.output and args.command in (None, "humanize"):
        args.output = str(resolve_output_path(input_path))


def _load_input(args: argparse.Namespace) -> DocumentPayload:
    if args.text:
        return DocumentPayload(text=args.text)
    if args.file:
        return load_document(Path(args.file))
    if not sys.stdin.isatty():
        return DocumentPayload(text=sys.stdin.read(), format="text")
    console.print("[red]No input. Use a file path, --file, --text, or pipe via stdin.[/red]")
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
    if getattr(args, "aggression", None) is not None:
        config.pipeline.aggression = args.aggression
    if getattr(args, "allow_tone_down", False):
        config.pipeline.allow_tone_down = True
    if getattr(args, "engine", None) is not None:
        config.pipeline.engine = args.engine


def _preflight_input(payload: DocumentPayload, *, verbose: bool) -> bool:
    """Validate input format and optional dependencies before pipeline run."""
    if payload.source_path:
        ext = payload.source_path.suffix.lower()
        if ext and ext not in _SUPPORTED_EXTENSIONS:
            console.print(f"[red]Unsupported input extension: {ext}[/red]")
            return False
        if ext == ".pdf":
            try:
                import fitz  # noqa: F401
            except ImportError:
                console.print(
                    "[red]PDF input requires PyMuPDF. Install with:[/red]\n"
                    "  pip install -e \".[full]\""
                )
                return False
        if ext == ".docx":
            try:
                import docx  # noqa: F401
            except ImportError:
                console.print(
                    "[red]DOCX input requires python-docx. Install with:[/red]\n"
                    "  pip install -e \".[full]\""
                )
                return False
        if verbose:
            console.print(f"[dim]Pre-flight OK: {payload.source_path.name} ({payload.format})[/dim]")
    return True


def _result_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str)


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
    verbose = getattr(args, "verbose", False)
    as_json = getattr(args, "json_output", False)

    if verbose:
        logging.basicConfig(level=logging.INFO)

    if not _ensure_ml_deps(config):
        return 1

    payload = _load_input(args)
    if not _preflight_input(payload, verbose=verbose):
        return 1
    text = payload.text
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
    if not as_json:
        console.print(Panel(f"Processing {len(text):,} characters...", title="Manuscript Humanizer"))
        if payload.source_path:
            console.print(f"[dim]Input: {payload.source_path} ({payload.format})[/dim]")
        console.print(
            f"[dim]Engine: {engine} | Target score: <={config.pipeline.target_ai_score} | "
            f"{max_label}[/dim]\n"
        )

    initial = pipeline.run_sync(text)
    if not as_json:
        console.print(f"Initial ZeroGPT-proxy score: [yellow]{initial.initial_score:.1f}[/yellow]")

    if getattr(args, "show_spans", False) and initial.zerogpt_report and not as_json:
        _print_span_report(pipeline, text, show_spans=True)

    result = await pipeline.run(text)

    if not as_json:
        console.print(f"\nFinal ML-aligned score: [{'green' if result.success else 'yellow'}]{result.final_score:.1f}[/]")
        if initial.zerogpt_report is not None:
            expected = pipeline.proxy_scorer.expected_zerogpt_score(result.final)
            console.print(f"Expected ZeroGPT (~):  [{'green' if expected <= 10 else 'yellow'}]{expected:.1f}[/]")
            console.print(f"[dim]Composite proxy: {result.zerogpt_report.document_score:.1f}[/dim]")
        console.print(f"Passes used:      {len(result.iterations)}")
        console.print(f"Target met:       {'[green]yes[/green]' if result.success else '[yellow]no[/yellow]'}")
        if result.quality_report is not None:
            qr = result.quality_report
            console.print(
                f"Quality gate:     {'[green]pass[/green]' if qr.passed else '[red]fail[/red]'} "
                f"(overlap={qr.similarity:.2f}, length={qr.length_ratio:.2f}x)"
            )
            if qr.issues:
                for issue in qr.issues[:4]:
                    console.print(f"  [dim]- {issue}[/dim]")
        if not result.success:
            console.print(
                "[dim]Quality or proxy target not reached; verify meaning before detector score. "
                "https://www.zerogpt.com/[/dim]"
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
        out_path = Path(args.output)
        qr_failed = result.quality_report is not None and not result.quality_report.passed
        force_save = getattr(args, "force_save", False)
        saved_path: str | None = None
        if qr_failed and not force_save:
            if not as_json:
                console.print(
                    "[yellow]Output not saved — quality gate failed. "
                    "Use --force-save to override.[/yellow]"
                )
        else:
            if qr_failed and force_save and not as_json:
                console.print("[yellow]Saving despite quality gate failure (--force-save).[/yellow]")
            if getattr(args, "in_place", False) and payload.source_path:
                backup = backup_file(payload.source_path)
                if backup:
                    if verbose and not as_json:
                        console.print(f"[dim]Backup: {backup}[/dim]")
                    if payload.format == "pdf":
                        payload.meta["pdf_backup"] = str(backup)
            elif payload.format == "pdf" and payload.source_path:
                payload.meta["pdf_backup"] = str(payload.source_path)
            payload.meta["force_save"] = force_save
            saved_path = str(save_document(out_path, result.final, payload))
            if not as_json:
                console.print(f"\n[green]Saved to {saved_path}[/green]")
                console.print("[dim]Verify manually at https://www.zerogpt.com/[/dim]")
    else:
        if not as_json:
            console.print("\n" + "─" * 60 + "\n")
            console.print(result.final)

    if as_json:
        summary = {
            "command": "humanize",
            "success": result.success,
            "initial_score": result.initial_score,
            "final_score": result.final_score,
            "expected_zerogpt": pipeline.proxy_scorer.expected_zerogpt_score(result.final),
            "document_score": (
                result.zerogpt_report.document_score if result.zerogpt_report else None
            ),
            "passes": len(result.iterations),
            "engine": engine,
            "saved_path": saved_path,
            "quality": (
                {
                    "passed": result.quality_report.passed,
                    "similarity": result.quality_report.similarity,
                    "length_ratio": result.quality_report.length_ratio,
                    "issues": result.quality_report.issues,
                }
                if result.quality_report
                else None
            ),
            "warnings": getattr(result, "warnings", []),
        }
        print(_result_json(summary))

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
        help="Output file path (default: <input>_humanized.<ext>; use -i for same file)",
    )
    parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="Overwrite the input file (creates a timestamped .bak backup first)",
    )
    parser.add_argument("--target", type=float, help="Target AI score (default from config)")
    parser.add_argument("--iterations", type=int, help="Max rewrite iterations (legacy engine)")
    parser.add_argument("--max-passes", type=int, dest="max_passes", help="Max targeted passes")
    parser.add_argument("--span-threshold", type=float, dest="span_threshold", help="Sentence flag threshold")
    parser.add_argument("--show-spans", action="store_true", help="Print sentence heatmap")
    parser.add_argument(
        "--force-save",
        action="store_true",
        help="Save output even when the quality gate fails",
    )
    parser.add_argument(
        "--aggression",
        choices=["conservative", "high"],
        help="Template aggression (default from config)",
    )
    parser.add_argument(
        "--allow-tone-down",
        action="store_true",
        dest="allow_tone_down",
        help="Allow endorsement tone-down in fidelity checks",
    )
    parser.add_argument(
        "--engine",
        choices=["segment", "local", "llm"],
        help="Rewrite engine override (default from config)",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Machine-readable JSON summary")
    parser.add_argument("--verbose", action="store_true", help="Show pipeline and I/O diagnostics")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = _preprocess_argv(argv) or []
    parser = argparse.ArgumentParser(
        prog="mh",
        description=(
            "Humanize academic manuscripts for lower AI-detector scores.\n\n"
            "Supports .txt, .md, .docx, .pdf (install formats: pip install -e \".[full]\").\n\n"
            "Examples:\n"
            "  mh manuscript.docx\n"
            "  mh thesis.pdf -i\n"
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
        payload = _load_input(args)
        if not _preflight_input(payload, verbose=getattr(args, "verbose", False)):
            return 1
        pipeline = HumanizerPipeline(config)
        if getattr(args, "json_output", False):
            result = pipeline.run_sync(payload.text)
            proxy = result.zerogpt_report
            print(
                _result_json(
                    {
                        "command": "score",
                        "initial_score": result.initial_score,
                        "document_score": proxy.document_score if proxy else None,
                        "ml_document_score": proxy.ml_document_score if proxy else None,
                        "hot_sentences": len(proxy.hot_sentences) if proxy else 0,
                    }
                )
            )
            return 0
        _print_span_report(
            pipeline, payload.text, show_spans=getattr(args, "show_spans", True)
        )
        return 0

    return asyncio.run(_run_humanize(args))


if __name__ == "__main__":
    raise SystemExit(main())
