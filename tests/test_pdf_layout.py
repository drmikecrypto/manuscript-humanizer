"""Unit tests for pdf_layout helpers."""

from humanizer.analyzers.zerogpt_proxy import blend_ml_pattern
from humanizer.io.pdf_layout import (
    JUSTIFY_EDGE_TOLERANCE_PT,
    approx_measure,
    pdf_font_name,
    reflow_to_width_budgets,
)


def test_pdf_font_name_times_bold():
    assert pdf_font_name("Times-Bold", flags=0) == "tibo"


def test_reflow_preserves_tokens():
    measure = approx_measure(11.0)
    original = ["short line here", "another line here", "final line"]
    out = reflow_to_width_budgets(
        "one two three four five six seven eight",
        original,
        [200.0, 200.0, 200.0],
        measure,
    )
    assert out is not None
    assert " ".join(out).split() == "one two three four five six seven eight".split()


def test_blend_ml_pattern():
    assert blend_ml_pattern(40.0, 80.0) == 40.0 * 0.35 + 80.0 * 0.65


def test_justify_tolerance_constant():
    assert JUSTIFY_EDGE_TOLERANCE_PT == 2.0
