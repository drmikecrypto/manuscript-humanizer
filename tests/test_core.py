from humanizer.analyzers.detector import detect_ai_likelihood
from humanizer.analyzers.patterns import analyze_patterns
from humanizer.analyzers.statistics import analyze_statistics
from humanizer.validators.fidelity import validate_fidelity

SAMPLE = """
In recent years, continual learning has attracted increasing attention.
Furthermore, our comprehensive approach utilizes state-of-the-art architectures.
The accuracy was 94.2% [1] and improved to 97.1% in 2023.
"""


def test_pattern_detection_finds_ai_tells():
    report = analyze_patterns(SAMPLE)
    assert report.hit_count >= 3
    assert report.score > 10


def test_statistics_returns_scores():
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
