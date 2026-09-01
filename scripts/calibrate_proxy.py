#!/usr/bin/env python3
"""Score a manuscript before/after humanization and print proxy component breakdown."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

from humanizer.analyzers.zerogpt_proxy import ZeroGPTProxyScorer
from humanizer.config import AppConfig
from humanizer.io import load_document
from humanizer.pipeline import HumanizerPipeline


def _score_row(label: str, text: str, scorer: ZeroGPTProxyScorer) -> dict[str, str | float]:
    report = scorer.analyze(text)
    top = sorted(report.sentences, key=lambda s: s.score, reverse=True)[:3]
    return {
        "label": label,
        "chars": len(text),
        "document_score": round(report.document_score, 2),
        "ml_document_score": round(report.ml_document_score, 2),
        "hot_sentences": len(report.hot_sentences),
        "top_hot": " | ".join(f"{s.score:.0f}:{s.text[:50]!r}" for s in top),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate offline ZeroGPT proxy scores")
    parser.add_argument("input", type=Path, help="Input manuscript (.md, .txt, .docx, .pdf)")
    parser.add_argument("-o", "--output", type=Path, help="Optional CSV output path")
    parser.add_argument("-c", "--config", default="config.toml", help="Config file path")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    config = AppConfig.load(args.config)
    payload = load_document(args.input)
    pipeline = HumanizerPipeline(config)
    scorer = pipeline.proxy_scorer

    before = _score_row("before", payload.text, scorer)
    result = asyncio.run(pipeline.run(payload.text))
    after: dict[str, str | float] = _score_row("after", result.final, scorer)
    if result.quality_report is not None:
        after["quality_pass"] = result.quality_report.passed
        after["overlap"] = round(result.quality_report.similarity, 3)

    rows = [before, after]
    fieldnames = list(rows[0].keys())
    for key in rows[1]:
        if key not in fieldnames:
            fieldnames.append(key)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")

    for row in rows:
        print(
            f"{row['label']:8} doc={row['document_score']:.1f} "
            f"ml={row['ml_document_score']:.1f} hot={row['hot_sentences']}"
        )
    delta = float(before["document_score"]) - float(after["document_score"])
    print(f"Improvement: {delta:.1f} proxy points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
