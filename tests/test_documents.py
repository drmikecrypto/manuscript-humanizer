"""Tests for multi-format document I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from humanizer.io.documents import (
    DocumentPayload,
    backup_file,
    load_document,
    resolve_output_path,
    save_document,
)


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

    out = tmp_path / "article_humanized.pdf"
    save_document(out, "Rewritten PDF body text.", payload)
    assert out.exists()
    assert out.stat().st_size > 0


def test_unsupported_extension_raises(tmp_path: Path):
    bad = tmp_path / "file.xyz"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(bad)
