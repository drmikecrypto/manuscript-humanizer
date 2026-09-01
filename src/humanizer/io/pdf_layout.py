"""Shared PDF line reflow and text-width measurement helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Average character width as fraction of font size (base-14 approximation).
_CHAR_WIDTH_RATIO = 0.52

JUSTIFY_EDGE_TOLERANCE_PT = 2.0
JUSTIFY_MIN_FONT_RATIO = 0.80
_FONT_SHRINK_STEP = 0.97


def pdf_font_name(raw: str, flags: int = 0) -> str:
    """Map PDF embedded font names to PyMuPDF base-14 names when possible."""
    lower = (raw or "").lower()
    bold = "bold" in lower or bool(flags & 2**4)
    italic = "italic" in lower or "oblique" in lower or bool(flags & 2**1)
    if "times" in lower or "serif" in lower or "roman" in lower:
        if bold and italic:
            return "tibi"
        if bold:
            return "tibo"
        if italic:
            return "titi"
        return "tiro"
    if "cour" in lower:
        return "cobo" if bold else "cour"
    if bold and italic:
        return "hebi"
    if bold:
        return "hebo"
    if italic:
        return "heit"
    return "helv"


def approx_measure(fontsize: float, fontname: str = "helv") -> Callable[[str], float]:
    """Return a text-width measurer without a PDF page (offline reflow)."""
    del fontname  # ratio is close enough across base-14 fonts for reflow budgets
    unit = fontsize * _CHAR_WIDTH_RATIO

    def measure(text: str) -> float:
        return len(text) * unit

    return measure


def page_text_measurer(fitz: Any, fontsize: float, fontname: str) -> Callable[[str], float]:
    """Return a measurer backed by PyMuPDF fitz.get_text_length."""

    def measure(text: str) -> float:
        if not text:
            return 0.0
        return float(fitz.get_text_length(text, fontname=fontname, fontsize=fontsize))

    return measure


def char_budgets_from_lines(original_lines: list[str], fontsize: float = 11.0) -> list[float]:
    """Convert character-length line budgets to approximate point widths."""
    measure = approx_measure(fontsize)
    return [max(measure(line.rstrip()), measure(" " * 10)) for line in original_lines]


def width_budgets_from_meta(
    line_meta: list[dict[str, Any]],
    original_lines: list[str],
    *,
    fontsize: float = 11.0,
) -> list[float]:
    """Extract per-line width budgets from PDF line metadata."""
    budgets: list[float] = []
    for meta, line in zip(line_meta, original_lines):
        tw = meta.get("target_width")
        if tw is not None and float(tw) > 0:
            budgets.append(float(tw))
        else:
            bbox = meta.get("bbox")
            if bbox and len(bbox) >= 4:
                budgets.append(max(float(bbox[2]) - float(bbox[0]), 10.0))
            else:
                budgets.append(char_budgets_from_lines([line], fontsize=fontsize)[0])
    return budgets


def reflow_to_width_budgets(
    new_text: str,
    original_lines: list[str],
    budgets_pt: list[float],
    measure: Callable[[str], float],
) -> list[str] | None:
    """
    Keep the same number of lines; pack whole words within each line's width budget.

    Returns None if the reflow would drop tokens vs new_text (caller should keep originals).
    """
    if not original_lines:
        return [new_text]
    if len(original_lines) == 1:
        return [new_text]

    words = new_text.split()
    if not words:
        return list(original_lines)

    if len(budgets_pt) != len(original_lines):
        budgets_pt = char_budgets_from_lines(original_lines)

    space_w = measure(" ")
    out: list[str] = []
    idx = 0
    for budget in budgets_pt:
        chunk: list[str] = []
        while idx < len(words):
            trial_words = chunk + [words[idx]]
            trial = " ".join(trial_words)
            trial_w = measure(trial)
            if trial_w > budget and chunk:
                break
            chunk.append(words[idx])
            idx += 1
        out.append(" ".join(chunk))

    if idx < len(words):
        remaining = " ".join(words[idx:])
        out[-1] = f"{out[-1]} {remaining}".strip() if out[-1] else remaining

    joined = " ".join(line for line in out if line).split()
    if joined != words:
        return None
    return out


def reflow_to_line_lengths(
    new_text: str,
    original_lines: list[str],
    *,
    fontsize: float = 11.0,
    line_meta: list[dict[str, Any]] | None = None,
    measure: Callable[[str], float] | None = None,
) -> list[str] | None:
    """Reflow using width budgets from metadata or character approximation."""
    if measure is None:
        measure = approx_measure(fontsize)
    if line_meta and len(line_meta) == len(original_lines):
        budgets = width_budgets_from_meta(line_meta, original_lines, fontsize=fontsize)
    else:
        budgets = char_budgets_from_lines(original_lines, fontsize=fontsize)
    return reflow_to_width_budgets(new_text, original_lines, budgets, measure)


def justify_line_words(
    page: Any,
    fitz: Any,
    words: list[str],
    *,
    x_start: float,
    target_width: float,
    y: float,
    fontsize: float,
    fontname: str,
    insert_kwargs: dict[str, Any],
    min_font_ratio: float = JUSTIFY_MIN_FONT_RATIO,
) -> tuple[float, bool]:
    """
    Insert words spanning exactly target_width (left and right margins aligned).

    Compresses or expands inter-word gaps; scales font down when glyphs alone exceed width.
    Returns (final_fontsize, font_was_scaled).
    """
    if not words:
        return fontsize, False

    orig_fontsize = fontsize
    scaled = False
    kwargs = dict(insert_kwargs)

    def measure_widths(fs: float) -> list[float]:
        measure = page_text_measurer(fitz, fs, fontname)
        return [measure(word) for word in words]

    if len(words) == 1:
        widths = measure_widths(fontsize)
        min_fs = orig_fontsize * min_font_ratio
        while widths[0] > target_width and fontsize > min_fs:
            fontsize *= _FONT_SHRINK_STEP
            widths = measure_widths(fontsize)
            scaled = True
        kwargs["fontsize"] = fontsize
        kwargs["fontname"] = fontname
        page.insert_text((x_start, y), words[0], **kwargs)
        return fontsize, scaled

    widths = measure_widths(fontsize)
    min_fs = orig_fontsize * min_font_ratio
    while sum(widths) > target_width and fontsize > min_fs:
        fontsize *= _FONT_SHRINK_STEP
        widths = measure_widths(fontsize)
        scaled = True

    gap = (target_width - sum(widths)) / (len(words) - 1)
    kwargs["fontsize"] = fontsize
    kwargs["fontname"] = fontname
    x = x_start
    for i, word in enumerate(words):
        page.insert_text((x, y), word, **kwargs)
        x += widths[i] + gap

    if scaled:
        logger.debug(
            "Justify scaled font %.2f -> %.2f for %d words in %.1fpt width",
            orig_fontsize,
            fontsize,
            len(words),
            target_width,
        )

    return fontsize, scaled
