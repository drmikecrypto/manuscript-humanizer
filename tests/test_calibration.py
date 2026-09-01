"""Proxy calibration regression on demo manuscript."""

from pathlib import Path

import pytest

from humanizer.config import AppConfig
from humanizer.pipeline import HumanizerPipeline
from humanizer.validators.fidelity import build_manuscript_quality_report


@pytest.fixture
def demo_text() -> str:
    path = Path("examples/demo_manuscript.md")
    if not path.exists():
        pytest.skip("demo_manuscript.md not available")
    return path.read_text(encoding="utf-8")


def test_demo_manuscript_proxy_improves(demo_text: str):
    config = AppConfig()
    config.pipeline.max_passes = 10
    config.pipeline.rewrite_all_sentences = True
    pipeline = HumanizerPipeline(config)
    before_score = pipeline.proxy_scorer.analyze(demo_text).document_score

    import asyncio

    result = asyncio.run(pipeline.run(demo_text))
    after_score = pipeline.proxy_scorer.analyze(result.final).document_score
    assert result.quality_report is not None
    assert result.quality_report.passed
    assert after_score < before_score
    assert before_score - after_score >= 5.0


def test_manuscript_quality_report_allows_structural_edit(demo_text: str):
    edited = demo_text.replace(
        "were randomly divided into 8 groups",
        "were allocated to 8 groups",
        1,
    )
    report = build_manuscript_quality_report(demo_text, edited)
    assert report.passed
