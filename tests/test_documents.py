"""Tests for multi-format document I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from humanizer.io.documents import (
    _align_pdf_lines,
    _propagate_paragraph_justify,
    backup_file,
    load_document,
    resolve_output_path,
    save_document,
)
from humanizer.io.pdf_layout import JUSTIFY_EDGE_TOLERANCE_PT


def test_resolve_output_path_defaults():
    src = Path("chapter1.docx")
    assert resolve_output_path(src) == Path("chapter1_humanized.docx")
    assert resolve_output_path(src, in_place=True) == src
    assert resolve_output_path(src, output=Path("out.pdf")) == Path("out.pdf")


def test_text_roundtrip(tmp_path: Path):
    src = tmp_path / "draft.md"
    src.write_text("# Title\n\nFurthermore, our approach utilizes state-of-the-art methods.", encoding="utf-8")
    payload = load_document(src)
    assert "Furthermore" in payload.text
    out = tmp_path / "draft_humanized.md"
    save_document(out, "Our approach uses strong methods.", payload)
    assert "strong methods" in out.read_text(encoding="utf-8")


def test_docx_roundtrip(tmp_path: Path):
    docx = pytest.importorskip("docx")
    src = tmp_path / "paper.docx"
    doc = docx.Document()
    doc.add_paragraph("Introduction paragraph with AI phrasing.")
    doc.add_paragraph("Methods paragraph about the study design.")
    doc.save(str(src))

    payload = load_document(src)
    assert payload.format == "docx"
    assert "Introduction" in payload.text

    humanized = "Intro paragraph rewritten.\n\nMethods section rewritten."
    out = tmp_path / "paper_humanized.docx"
    save_document(out, humanized, payload)

    reread = load_document(out)
    assert "rewritten" in reread.text.lower()


def test_in_place_docx_creates_backup(tmp_path: Path):
    docx = pytest.importorskip("docx")
    src = tmp_path / "thesis.docx"
    doc = docx.Document()
    doc.add_paragraph("Original thesis text.")
    doc.save(str(src))

    payload = load_document(src)
    backup = backup_file(src)
    assert backup is not None
    assert backup.exists()
    save_document(src, "Humanized thesis text.", payload)
    assert "Humanized" in load_document(src).text


def test_pdf_extract_and_write(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    src = tmp_path / "article.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sample PDF body text for humanization.")
    doc.save(str(src))
    doc.close()

    payload = load_document(src)
    assert payload.format == "pdf"
    assert "Sample PDF" in payload.text
    payload.meta["pdf_backup"] = str(src)
    payload.meta["force_save"] = True

    out = tmp_path / "article_humanized.pdf"
    save_document(out, "Rewritten PDF body text.", payload)
    assert out.exists()
    assert out.stat().st_size > 0
    reread = load_document(out)
    assert "Rewritten" in reread.text


def test_pdf_preserves_letterhead_and_line_order(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    src = tmp_path / "letter.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in (
        "Jane Scholar",
        "jane@example.edu",
        "To: Admissions Committee",
        "Dear committee members,",
        "I am applying for the advertised PhD position in computational neuroscience.",
        "My prior work includes pathway analysis and a controlled animal study.",
        "Sincerely,",
        "Jane Scholar",
    ):
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    doc.save(str(src))
    doc.close()

    payload = load_document(src)
    payload.meta["pdf_backup"] = str(src)
    humanized = payload.text.replace(
        "I am applying for the advertised PhD position in computational neuroscience.",
        "I am applying for the PhD position in computational neuroscience at your lab.",
    )
    out = tmp_path / "letter_humanized.pdf"
    save_document(out, humanized, payload)

    reread = load_document(out)
    lines = reread.text.splitlines()
    assert lines[0] == "Jane Scholar"
    assert lines[1] == "jane@example.edu"
    assert lines[2].startswith("To:")
    assert lines[3].startswith("Dear")
    assert "your lab" in reread.text
    assert "Sincerely," in reread.text
    assert lines[-1] == "Jane Scholar"

    opened = fitz.open(str(out))
    sorted_text = opened[0].get_text("text", sort=True)
    opened.close()
    assert sorted_text.index("Dear") < sorted_text.index("your lab")
    assert sorted_text.index("your lab") < sorted_text.index("Sincerely,")


def test_pdf_preserves_font_size_on_edited_line(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    src = tmp_path / "letter.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Dear committee,", fontsize=12)
    page.insert_text((72, 120), "Original body line about the advertised PhD role.", fontsize=12)
    page.insert_text((72, 140), "Sincerely,", fontsize=12)
    doc.save(str(src))
    doc.close()

    payload = load_document(src)
    payload.meta["pdf_backup"] = str(src)
    humanized = payload.text.replace(
        "Original body line about the advertised PhD role.",
        "Rewritten body line about the PhD role at your laboratory group.",
    )
    out = tmp_path / "letter_humanized.pdf"
    save_document(out, humanized, payload)

    before = fitz.open(str(src))
    after = fitz.open(str(out))

    def body_size(pdf):
        for block in pdf[0].get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                text = "".join(span.get("text", "") for span in line["spans"])
                if "body line" in text.lower() or "phd role" in text.lower():
                    return line["spans"][0]["size"]
        return None

    orig_size = body_size(before)
    new_size = body_size(after)
    before.close()
    after.close()
    assert orig_size is not None
    assert new_size is not None
    assert abs(orig_size - new_size) < 0.2


def _make_justified_pdf(
    path: Path,
    lines: list[str],
    *,
    x0: float = 72,
    x1: float = 500,
    y_start: float = 100,
    line_height: float = 18,
    fontsize: float = 11,
) -> None:
    """Create a PDF with manually justified body lines (non-final lines only)."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()

    def measure(text: str) -> float:
        return float(fitz.get_text_length(text, fontname="helv", fontsize=fontsize))

    for i, line_text in enumerate(lines):
        y = y_start + i * line_height
        words = line_text.split()
        is_last = i == len(lines) - 1
        if len(words) >= 3 and not is_last:
            space_w = measure(" ")
            word_widths = [measure(word) for word in words]
            total = sum(word_widths) + space_w * (len(words) - 1)
            target = x1 - x0
            extra_per = (target - total) / (len(words) - 1)
            x = x0
            for j, word in enumerate(words):
                page.insert_text((x, y), word, fontsize=fontsize, fontname="helv")
                x += word_widths[j] + space_w + extra_per
        else:
            page.insert_text((x0, y), line_text, fontsize=fontsize, fontname="helv")
    doc.save(str(path))
    doc.close()


def _line_right_edge(pdf_path: Path, needle: str) -> float | None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open(str(pdf_path))
    try:
        for block in doc[0].get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                text = "".join(span.get("text", "") for span in line["spans"])
                if needle in text:
                    return float(line["bbox"][2])
    finally:
        doc.close()
    return None


def _line_and_block_right(pdf_path: Path, needle: str) -> tuple[float | None, float | None]:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open(str(pdf_path))
    try:
        for block in doc[0].get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            block_text = ""
            for line in block["lines"]:
                block_text += "".join(span.get("text", "") for span in line["spans"])
            if needle not in block_text:
                continue
            block_right = float(block["bbox"][2])
            line_right = max(float(line["bbox"][2]) for line in block["lines"])
            return line_right, block_right
    finally:
        doc.close()
    return None, None


def test_pdf_justified_line_redraw(tmp_path: Path):
    src = tmp_path / "justified.pdf"
    body_lines = [
        "I am applying for the advertised PhD position in computational neuroscience at your institution.",
        "My prior work includes pathway analysis and a controlled animal study with rigorous methods.",
        "Sincerely,",
    ]
    _make_justified_pdf(src, ["Dear committee,"] + body_lines)

    payload = load_document(src)
    # Index 1 is the first long body line (index 0 is "Dear committee,").
    assert payload.meta["pdf_pages_meta"][0][1]["justify"] is True
    payload.meta["pdf_backup"] = str(src)
    payload.meta["force_save"] = True

    original_right = _line_right_edge(src, "advertisedPhD") or _line_right_edge(src, "Iamapplying")
    lines = payload.text.splitlines()
    lines[1] = (
        "I am enthusiastically applying for the PhD position in computational neuroscience "
        "at your laboratory institution with extensive relevant research background."
    )
    humanized = "\n".join(lines)
    out = tmp_path / "justified_humanized.pdf"
    save_document(out, humanized, payload)

    line_right, block_right = _line_and_block_right(out, "enthusiastically")
    if line_right is None:
        line_right, block_right = _line_and_block_right(out, "laboratory")
    assert original_right is not None
    assert line_right is not None
    assert block_right is not None
    assert abs(line_right - block_right) < JUSTIFY_EDGE_TOLERANCE_PT


def test_pdf_justified_longer_rewrite(tmp_path: Path):
    src = tmp_path / "longer.pdf"
    lines = [
        "I am applying for the advertised PhD position in computational neuroscience at your institution.",
        "My prior work includes pathway analysis and a controlled animal study with rigorous methods.",
        "Sincerely,",
    ]
    _make_justified_pdf(src, ["Dear committee,"] + lines)

    payload = load_document(src)
    payload.meta["pdf_backup"] = str(src)
    payload.meta["force_save"] = True

    edited = payload.text.splitlines()
    edited[1] = (
        "I am enthusiastically applying for the PhD position in computational neuroscience "
        "at your laboratory institution with extensive relevant research background experience."
    )
    out = tmp_path / "longer_humanized.pdf"
    save_document(out, "\n".join(edited), payload)

    line_right, block_right = _line_and_block_right(out, "enthusiastically")
    assert line_right is not None
    assert block_right is not None
    assert abs(line_right - block_right) < JUSTIFY_EDGE_TOLERANCE_PT


def test_pdf_justify_propagates_to_paragraph():
    page_lines = [
        "This is the first line of a long justified paragraph in the document here.",
        "Thissecondlinehasnospacesbutislongenoughtobeabodylineinparagraph",
        "Short last.",
    ]
    page_meta = [
        {"justify": True, "is_last_in_block": False, "bbox": [72, 100, 500, 118]},
        {"justify": False, "is_last_in_block": False, "bbox": [72, 118, 500, 136]},
        {"justify": False, "is_last_in_block": True, "bbox": [72, 136, 200, 154]},
    ]
    _propagate_paragraph_justify(page_lines, page_meta)
    assert page_meta[0]["justify"] is True
    assert page_meta[1]["justify"] is True
    assert page_meta[2]["justify"] is False


def test_pdf_last_line_not_justified(tmp_path: Path):
    src = tmp_path / "last_line.pdf"
    lines = [
        "This is a longer justified line that should stretch across the full paragraph width here.",
        "This is a shorter last line.",
    ]
    _make_justified_pdf(src, lines)

    payload = load_document(src)
    meta = payload.meta["pdf_pages_meta"][0]
    assert meta[0]["justify"] is True
    assert meta[0]["is_last_in_block"] is False
    assert meta[1]["is_last_in_block"] is True
    assert meta[1]["justify"] is False


def test_pdf_reflow_matches_outbound(tmp_path: Path):
    from humanizer.io.pdf_layout import approx_measure, reflow_to_width_budgets

    original_lines = [
        "I am applying for the advertised PhD position in computational neuroscience.",
        "My prior work includes pathway analysis and controlled animal study methods.",
    ]
    joined = (
        "I am applying for the PhD position in computational neuroscience at your laboratory group "
        "with strong background in pathway analysis and controlled animal study methods."
    )
    budgets = [400.0, 400.0]
    measure = approx_measure(11.0)
    expected = reflow_to_width_budgets(joined, original_lines, budgets, measure)
    pages_meta = [[{"target_width": 400, "bbox": [72, 100, 472, 118]}] * 2]
    aligned = _align_pdf_lines(original_lines, joined, pages_meta)
    assert expected is not None
    assert aligned == expected


def test_pdf_quality_guard_reverts_bad_reflow(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    src = tmp_path / "guard.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Dear committee,", fontsize=12)
    page.insert_text(
        (72, 120),
        "Original body line about the advertised PhD role in computational neuroscience.",
        fontsize=12,
    )
    page.insert_text((72, 140), "Sincerely,", fontsize=12)
    doc.save(str(src))
    doc.close()

    payload = load_document(src)
    payload.meta["pdf_backup"] = str(src)
    bad = (
        "Dear committee,\n"
        "Quantum chromodynamics bears no relation to veterinary medicine or neuroscience whatsoever.\n"
        "Sincerely,"
    )
    out = tmp_path / "guard_out.pdf"
    save_document(out, bad, payload)

    reread = load_document(out)
    assert "Original body line" in reread.text
    assert "Quantum chromodynamics" not in reread.text


def test_unsupported_extension_raises(tmp_path: Path):
    bad = tmp_path / "file.xyz"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(bad)
