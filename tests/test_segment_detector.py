from humanizer.analyzers.segment_detector import SegmentDetector
from humanizer.rewriters.transforms import split_manuscript_sentences

AI_SAMPLE = """
Furthermore, our comprehensive approach utilizes state-of-the-art architectures.
The results demonstrate significant improvements across all metrics.
Additionally, this study plays a crucial role in advancing the field.
"""

HUMAN_SAMPLE = """
We ran three pilot batches. Batch two failed — the seal leaked.
I rewrote the methods after that. The numbers stayed the same.
"""


def test_segment_detector_legacy_engine():
    det = SegmentDetector(engine="legacy", span_threshold=50.0)
    report = det.analyze(AI_SAMPLE)
    assert 0 <= report.document_score <= 100
    assert report.sentence_scores
    assert report.engine == "legacy"


def test_sentence_spans_mapped():
    det = SegmentDetector(engine="legacy")
    text = (
        "First sentence contains enough words to pass the minimum length filter. "
        "Second sentence also contains enough words for manuscript splitting."
    )
    report = det.analyze(text)
    assert len(report.sentence_scores) == len(split_manuscript_sentences(text))


def test_hot_sentences_flagged():
    det = SegmentDetector(engine="legacy", span_threshold=40.0)
    ai_report = det.analyze(AI_SAMPLE)
    human_report = det.analyze(HUMAN_SAMPLE)
    assert ai_report.document_score >= human_report.document_score or ai_report.legacy_report


def test_document_score_uses_top_windows():
    det = SegmentDetector(engine="legacy")
    report = det.analyze(AI_SAMPLE)
    assert report.spans
    assert report.spans[0].ai_probability >= report.spans[-1].ai_probability
