from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from humanizer import __version__
from humanizer.config import AppConfig
from humanizer.pipeline import HumanizerPipeline

console = Console()


def _read_input(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    console.print("[red]No input. Use --file, --text, or pipe via stdin.[/red]")
    sys.exit(1)


def _print_score_report(pipeline: HumanizerPipeline, text: str) -> None:
    result = pipeline.run_sync(text)
    det = result.iterations[0].detection
    table = Table(title="AI Likelihood Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_row("Composite (0=human, 100=AI)", f"{det.composite_score:.1f}")
    table.add_row("Pattern score", f"{det.details.get('pattern_score', 0):.1f}")
    table.add_row("Statistical score", f"{det.details.get('statistical_score', 0):.1f}")
    table.add_row("Burstiness (higher=human)", f"{det.statistics.burstiness:.1f}")
    table.add_row("Repetition (higher=AI)", f"{det.statistics.repetition_score:.1f}")
    table.add_row("Pattern hits", str(det.pattern.hit_count))
    console.print(table)
    if det.pattern.hits:
        console.print("\n[bold]Top pattern hits:[/bold]")
        for hit in det.pattern.hits[:8]:
            console.print(f"  [{hit.pattern_id}] {hit.matched!r}")


async def _run_humanize(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    if args.target is not None:
        config.pipeline.target_ai_score = args.target
    if args.iterations is not None:
        config.pipeline.max_iterations = args.iterations

    text = _read_input(args)
    pipeline = HumanizerPipeline(config)

    if not config.llm.api_key:
        console.print(
            "[red]No API key. Set llm.api_key in config.toml or OPENAI_API_KEY env var.[/red]"
        )
        return 1

    console.print(Panel(f"Processing {len(text):,} characters...", title="Manuscript Humanizer"))
    console.print(f"[dim]Target score: ≤{config.pipeline.target_ai_score} | Max iterations: {config.pipeline.max_iterations}[/dim]\n")

    initial = pipeline.run_sync(text)
    console.print(f"Initial AI score: [yellow]{initial.initial_score:.1f}[/yellow]")

    result = await pipeline.run(text)

    console.print(f"\nFinal AI score:   [{'green' if result.success else 'yellow'}]{result.final_score:.1f}[/]")
    console.print(f"Iterations used:  {len(result.iterations)}")
    console.print(f"Target met:       {'[green]yes[/green]' if result.success else '[yellow]no[/yellow]'}")

    if args.output:
        Path(args.output).write_text(result.final, encoding="utf-8")
        console.print(f"\n[green]Saved to {args.output}[/green]")
    else:
        console.print("\n" + "─" * 60 + "\n")
        console.print(result.final)

    return 0 if result.success else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="manuscript-humanizer",
        description="Detector-aware academic manuscript humanizer",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-c", "--config", default="config.toml", help="Config file path")
    parser.add_argument("-f", "--file", help="Input file path")
    parser.add_argument("-t", "--text", help="Input text string")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--target", type=float, help="Target AI score (default from config)")
    parser.add_argument("--iterations", type=int, help="Max rewrite iterations")

    sub = parser.add_subparsers(dest="command")
    score_parser = sub.add_parser("score", help="Analyze AI likelihood without rewriting")
    humanize_parser = sub.add_parser("humanize", help="Rewrite text (default)")

    # Mirror top-level args on subcommands so `mh score -f x.txt` works.
    for sub_parser in (score_parser, humanize_parser):
        sub_parser.add_argument("-c", "--config", default="config.toml")
        sub_parser.add_argument("-f", "--file")
        sub_parser.add_argument("-t", "--text")
        sub_parser.add_argument("-o", "--output")
        sub_parser.add_argument("--target", type=float)
        sub_parser.add_argument("--iterations", type=int)

    args = parser.parse_args(argv)
    command = args.command or "humanize"

    if command == "score":
        config = AppConfig.load(args.config)
        text = _read_input(args)
        _print_score_report(HumanizerPipeline(config), text)
        return 0

    return asyncio.run(_run_humanize(args))


if __name__ == "__main__":
    raise SystemExit(main())
