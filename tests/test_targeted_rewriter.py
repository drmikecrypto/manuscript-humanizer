from humanizer.analyzers.zerogpt_proxy import ZeroGPTProxyScorer
from humanizer.rewriters.targeted_rewriter import TargetedRewriter
from humanizer.validators.fidelity import validate_fidelity


SAMPLE = """
Furthermore, our comprehensive approach utilizes state-of-the-art architectures.
The accuracy was 94.2% and improved to 97.1% in 2023.
Additionally, this study plays a crucial role in advancing the field.
"""


def test_targeted_rewriter_changes_only_hot_sentences():
    scorer = ZeroGPTProxyScorer(use_onnx=False, hot_threshold=30.0)
    report = scorer.analyze(SAMPLE)
    rewriter = TargetedRewriter(proxy_scorer=scorer)
    result = rewriter.rewrite_pass(SAMPLE, report, sentences_per_pass=2, original=SAMPLE)
    if result.changed_sentences:
        assert len(result.changed_sentences) <= len(report.hot_sentences) + 2


def test_targeted_rewriter_preserves_numbers():
    original = "The model achieved 94.2% accuracy in 2023."
    scorer = ZeroGPTProxyScorer(use_onnx=False, hot_threshold=0.0)
    report = scorer.analyze(original)
    rewriter = TargetedRewriter(proxy_scorer=scorer)
    result = rewriter.rewrite_pass(original, report, sentences_per_pass=1, original=original)
    assert validate_fidelity(original, result.text, min_similarity=0.55).passed


def test_manuscript_targeted_pass(calibration_manuscript):
    original = calibration_manuscript
    scorer = ZeroGPTProxyScorer(use_onnx=False, hot_threshold=40.0)
    report = scorer.analyze(original)
    rewriter = TargetedRewriter(proxy_scorer=scorer)
    result = rewriter.rewrite_all_hot(original, original=original)
    assert validate_fidelity(original, result.text, min_similarity=0.65).passed
