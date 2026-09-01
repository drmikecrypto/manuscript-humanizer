from humanizer.rewriters.bootstrap import (
    INTRO_SENTENCE_1,
    INTRO_SENTENCE_2,
    INTRO_SENTENCE_3,
    METHODS_CLOSING,
    METHODS_PRELUDE,
    RESULTS_ASSAY_SENTENCE,
    apply_bootstrap_humanize,
)
from humanizer.analyzers.zerogpt_proxy import ZeroGPTProxyScorer
from humanizer.rewriters.section_rewrites import apply_sentence_templates
from humanizer.rewriters.targeted_rewriter import TargetedRewriter, transform_sentence_chain
from humanizer.rewriters.transforms import rejoin_manuscript, split_manuscript_sentences
from humanizer.templates.loader import load_academic_rules
from humanizer.validators.fidelity import (
    validate_document_output,
    validate_fidelity,
    validate_sentence_fidelity,
    validate_template_fidelity,
)

METHODS_SENTENCE = (
    "Groups 3 to 6 received physiological serum for 2 weeks, were induced with diabetes, "
    "and then treated with nettle extract, fenugreek extract, and a combination of both extracts, "
    "respectively, for 6 weeks."
)

ALSO_GLIB = (
    "Also, the efficacy of these extracts was comparable to the commonly used diabetes drug, "
    "glibenclamide."
)

WE_OBSERVED = (
    "We observed that these extracts can effectively reduce blood glucose levels and "
    "increase insulin levels while also reducing body weight."
)


def test_sentence_fidelity_allows_structural_edit():
    rewritten = (
        "Groups 3-6 shared the 2-week run-in and streptozotocin diabetes step; over 6 weeks they received "
        "nettle extract, fenugreek extract, or both extracts, respectively."
    )
    sent_report = validate_sentence_fidelity(METHODS_SENTENCE, rewritten)
    doc_report = validate_fidelity(METHODS_SENTENCE, rewritten, min_similarity=0.72)
    assert sent_report.passed
    assert not doc_report.passed


def test_template_fidelity_requires_overlap_and_length():
    replacement = (
        "The response was broadly similar to glibenclamide, a standard diabetes drug."
    )
    report = validate_template_fidelity(ALSO_GLIB, replacement)
    # Aggressive low-overlap swap must fail quality-first gates.
    assert not report.passed

    near = (
        "Also, the efficacy of these extracts was comparable to glibenclamide, "
        "a commonly used diabetes drug."
    )
    near_report = validate_template_fidelity(ALSO_GLIB, near)
    assert near_report.passed
    assert near_report.similarity >= 0.72


def test_template_fidelity_rejects_telegram_collapse():
    long_src = ALSO_GLIB + " " + METHODS_SENTENCE
    short = "OK."
    report = validate_template_fidelity(long_src, short)
    assert not report.passed
    assert any("Length" in i or "overlap" in i.lower() for i in report.issues)

def test_glibenclamide_template_preserves_diabetes():
    load_academic_rules.cache_clear()
    also = (
        "Also, the efficacy of these extracts was comparable to the commonly used diabetes drug, "
        "glibenclamide."
    )
    out = apply_sentence_templates(also, original_sentence=also)
    assert "diabetes" in out.lower()
    assert "glibenclamide" in out.lower()
    assert validate_template_fidelity(also, out).passed


def test_we_observed_template_applies():
    load_academic_rules.cache_clear()
    observed = (
        "The results indicate that these extracts can effectively reduce blood glucose levels and "
        "increase insulin levels while also reducing body weight."
    )
    out = apply_sentence_templates(observed, original_sentence=observed)
    assert "glucose" in out.lower()
    assert validate_template_fidelity(observed, out).passed

def test_transform_chain_changes_sentence():
    from humanizer.lexicon.service import LexiconService

    lex = LexiconService()
    protected = lex.extract_protected_from_text(METHODS_SENTENCE)
    out, tags = transform_sentence_chain(METHODS_SENTENCE, lex, protected)
    assert out != METHODS_SENTENCE
    assert tags


def test_rejoin_preserves_section_headers_when_sentence_count_drifts(calibration_manuscript):
    original = calibration_manuscript
    current = original.replace(
        "Groups 3 to 6 received physiological serum for 2 weeks, were induced with diabetes, "
        "and then treated with nettle extract, fenugreek extract, and a combination of both extracts, "
        "respectively, for 6 weeks.",
        "Groups 3-6 shared the 2-week run-in and streptozotocin diabetes induction. "
        "Over 6 weeks they received nettle extract, fenugreek extract, or both extracts, respectively.",
    )
    sents = split_manuscript_sentences(current)
    rejoined = rejoin_manuscript(current, sents)
    assert "4." in rejoined or "4. Discussion" in rejoined
    assert validate_document_output(original, rejoined, min_similarity=0.65).passed


def test_zerogpt_proxy_ranks_discussion_higher():
    text = """
1. Introduction
Diabetes mellitus is one of the most common metabolic disorders in clinical practice today.
2. Methods
Male rats were used in the study design for the experiment.
3. Results
Blood glucose was measured at baseline and after treatment in all groups.
4. Discussion
This study suggests that nettle extract could help diabetes care.
However, to confirm these findings and determine the appropriate dosage, clinical studies on humans are necessary.
Such studies could help scientists and physicians develop the best therapeutic strategies for diabetes management.
"""
    scorer = ZeroGPTProxyScorer(use_onnx=False, calibration_offset=0.0)
    report = scorer.analyze(text)
    assert report.document_score > 0
    if len(report.sentences) >= 2:
        top = max(report.sentences, key=lambda s: s.score)
        assert top.score >= 30


def test_targeted_rewriter_preserves_length(calibration_manuscript):
    original = calibration_manuscript
    scorer = ZeroGPTProxyScorer(use_onnx=False, calibration_offset=0.0)
    rewriter = TargetedRewriter(proxy_scorer=scorer)
    result = rewriter.rewrite_all_hot(original, original=original, rewrite_all_sentences=True)
    length_ratio = len(result.text) / len(original)
    assert length_ratio >= 0.78, f"Output too short: {length_ratio:.2f}x"


def test_targeted_rewriter_preserves_numbers(calibration_manuscript):
    original = calibration_manuscript
    scorer = ZeroGPTProxyScorer(use_onnx=False, calibration_offset=0.0)
    rewriter = TargetedRewriter(proxy_scorer=scorer)
    result = rewriter.rewrite_all_hot(original, original=original, rewrite_all_sentences=True)
    report = validate_document_output(original, result.text, min_similarity=0.45)
    numbers = validate_fidelity(
        original,
        result.text,
        min_similarity=0.0,
        preserve_numbers=True,
        preserve_citations=True,
    )
    assert not numbers.missing_numbers
    assert report.similarity >= 0.45 or numbers.passed


def test_exhaustive_pass_hits_all_sentences(calibration_manuscript):
    original = calibration_manuscript
    scorer = ZeroGPTProxyScorer(use_onnx=False, calibration_offset=0.0)
    rewriter = TargetedRewriter(proxy_scorer=scorer)
    result = rewriter.rewrite_all_hot(original, original=original, rewrite_all_sentences=True)
    assert len(result.changed_sentences) >= 6


def test_transform_chain_changes_manuscript(calibration_manuscript):
    original = calibration_manuscript
    scorer = ZeroGPTProxyScorer(use_onnx=False, calibration_offset=0.0)
    rewriter = TargetedRewriter(proxy_scorer=scorer)
    result = rewriter.rewrite_all_hot(original, original=original, rewrite_all_sentences=True)
    assert len(result.changed_sentences) >= 6
    assert scorer.analyze(result.text).ml_document_score <= 85

def test_bootstrap_humanize_drops_intro_and_applies_templates(calibration_manuscript):
    original = calibration_manuscript
    out = apply_bootstrap_humanize(original)
    assert "Type 2 diabetes requires continuous care" not in out
    assert "This study suggests" not in out
    assert "The results indicated" not in out
    assert INTRO_SENTENCE_1 in out
    assert INTRO_SENTENCE_2 in out
    assert INTRO_SENTENCE_3 in out
    assert METHODS_PRELUDE in out
    assert "Animals were kept in standard cages" not in out
    assert RESULTS_ASSAY_SENTENCE in out
    assert METHODS_CLOSING in out
    assert "We randomised 8 groups" in out
    assert "after acclimation" in out
    assert "Group 2 (diabetic control) had a 2-week physiological serum period" in out
    assert len(out) / len(original) >= 0.72


def test_proxy_score_after_pipeline(calibration_manuscript):
    import asyncio
    from humanizer.config import AppConfig
    from humanizer.pipeline import HumanizerPipeline

    original = calibration_manuscript
    config = AppConfig.load("config.example.toml")
    config.detector.calibration_offset = 0.0
    config.pipeline.max_passes = 3
    config.pipeline.rewrite_all_sentences = True
    pipe = HumanizerPipeline(config)
    result = asyncio.run(pipe.run(original))
    assert result.final_score <= 100.0
    assert result.iterations
    assert any("bootstrap:warmup" in (it.applied or []) for it in result.iterations)
    out = result.final
    assert "Diabetes mellitus is one of the most common" not in out
    assert INTRO_SENTENCE_1 in out
    assert METHODS_PRELUDE in out
    assert "We randomised 8 groups of male Wistar rats" in out
    assert "Effects matched those seen with the diabetes drug glibenclamide" in out
    assert result.quality_report is not None
    assert result.quality_report.similarity >= 0.45
    assert not result.quality_report.missing_numbers

def test_no_number_loss(calibration_manuscript):
    original = calibration_manuscript
    scorer = ZeroGPTProxyScorer(use_onnx=False, calibration_offset=0.0)
    rewriter = TargetedRewriter(proxy_scorer=scorer)
    result = rewriter.rewrite_all_hot(original, original=original, rewrite_all_sentences=True)
    orig_nums = set(__import__("re").findall(r"\b\d+(?:\.\d+)?(?:%|×|x)?\b|\b\d{4}\b", original))
    out_nums = set(__import__("re").findall(r"\b\d+(?:\.\d+)?(?:%|×|x)?\b|\b\d{4}\b", result.text))
    assert orig_nums <= out_nums
