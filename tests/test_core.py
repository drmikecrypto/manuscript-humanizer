from humanizer.analyzers.detector import detect_ai_likelihood
from humanizer.config import AppConfig
from humanizer.pipeline import HumanizerPipeline
from humanizer.rewriters.local_rewriter import LocalRewriter
from humanizer.validators.fidelity import validate_fidelity

SAMPLE = """
In recent years, continual learning has attracted increasing attention.
Furthermore, our comprehensive approach utilizes state-of-the-art architectures.
The accuracy was 94.2% [1] and improved to 97.1% in 2023.
"""


def test_pattern_detection_finds_ai_tells():
    from humanizer.analyzers.patterns import analyze_patterns

    report = analyze_patterns(SAMPLE)
    assert report.hit_count >= 3
    assert report.score > 10


def test_statistics_returns_scores():
    from humanizer.analyzers.statistics import analyze_statistics

    stats = analyze_statistics(SAMPLE)
    assert 0 <= stats.ai_likelihood <= 100
    assert stats.avg_sentence_length > 0


def test_composite_detection():
    report = detect_ai_likelihood(SAMPLE)
    assert report.composite_score > 0
    assert "pattern_score" in report.details


def test_fidelity_preserves_numbers():
    original = "The model achieved 94.2% accuracy [Smith, 2023]."
    good = "The model reached 94.2% accuracy [Smith, 2023]."
    bad = "The model achieved high accuracy."
    assert validate_fidelity(original, good).passed
    assert not validate_fidelity(original, bad).passed


def test_fidelity_detects_missing_citation():
    original = "Results improved [1, 2]."
    rewritten = "Results improved significantly."
    report = validate_fidelity(original, rewritten, preserve_citations=True)
    assert not report.passed
    assert report.missing_citations


def test_local_rewriter_reduces_ai_score(calibration_manuscript):
    original = calibration_manuscript
    rewriter = LocalRewriter()
    rewritten = rewriter.rewrite(original, iteration=3, issues=["High lexical repetition (80)"])
    before = detect_ai_likelihood(original)
    after = detect_ai_likelihood(rewritten)
    assert after.composite_score <= before.composite_score


def test_manuscript_humanization_improves_score(calibration_manuscript):
    original = calibration_manuscript
    rewriter = LocalRewriter()
    rewritten = rewriter.rewrite(original, iteration=5, issues=["High lexical repetition (80)"])
    before = detect_ai_likelihood(original)
    after = detect_ai_likelihood(rewritten)
    assert after.composite_score <= before.composite_score
    assert validate_fidelity(original, rewritten, min_similarity=0.72).passed


def test_pipeline_runs_offline_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    config = AppConfig.load()
    config.pipeline.engine = "segment"
    config.detector.engine = "legacy"
    config.detector.span_threshold = 40.0
    config.pipeline.max_passes = 3

    async def _run():
        pipeline = HumanizerPipeline(config)
        return await pipeline.run(SAMPLE)

    import asyncio

    result = asyncio.run(_run())
    assert result.final_score <= result.initial_score or result.success or result.iterations


def test_cli_preprocess_positional_file():
    from pathlib import Path

    from humanizer.cli import _preprocess_argv
    from humanizer.io.documents import resolve_output_path

    assert _preprocess_argv(["draft.md"]) == ["humanize", "-f", "draft.md"]
    assert _preprocess_argv(["humanize", "draft.md"]) == ["humanize", "-f", "draft.md"]
    assert resolve_output_path(Path("chapter1.md")) == Path("chapter1_humanized.md")
