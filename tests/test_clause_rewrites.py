"""Clause rewrite fidelity tests."""

from humanizer.rewriters.clause_rewrites import apply_clause_rewrites
from humanizer.validators.fidelity import validate_template_fidelity


def test_clause_rewrite_methods_allocation():
    original = (
        "In this study, male Wistar rats weighing 190-210 grams were randomly divided into 8 groups."
    )
    out = apply_clause_rewrites(original, original_sentence=original, section="methods")
    assert "randomly divided" not in out.lower() or "allocated" in out.lower()
    assert "190-210" in out
    assert validate_template_fidelity(original, out).passed


def test_clause_rewrite_strips_furthermore_in_discussion():
    original = "Furthermore, nettle extract lowered glucose in diabetic rats."
    out = apply_clause_rewrites(original, original_sentence=original, section="discussion")
    assert "Furthermore" not in out
    assert "nettle" in out.lower()


def test_results_observed_phrase():
    original = "The results indicated that glucose fell in treated rats."
    out = apply_clause_rewrites(original, original_sentence=original, section="results")
    assert "indicated" not in out.lower() or "observed" in out.lower()
