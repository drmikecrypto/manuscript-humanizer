from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".tex", ".rst", ".html", ".htm"})
DOCX_EXTENSIONS = frozenset({".docx"})
PDF_EXTENSIONS = frozenset({".pdf"})
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | DOCX_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class DocumentPayload:
    """Text plus metadata needed to write back in the original format."""

    text: str
    source_path: Path | None = None
    format: str = "text"
    meta: dict[str, Any] = field(default_factory=dict)


def is_supported_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def _require_docx():
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            'Word support requires: pip install -e ".[formats]" or pip install python-docx'
        ) from exc
    return Document


def _require_pymupdf():
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            'PDF support requires: pip install -e ".[formats]" or pip install pymupdf'
        ) from exc
    return fitz


def _distribute_text(original_blocks: list[str], humanized: str) -> list[str]:
    """Map humanized prose back onto the original block count (paragraphs / pages)."""
    n = len(original_blocks)
    if n == 0:
        return []
    cleaned = humanized.strip()
    if not cleaned:
        return [""] * n
    if n == 1:
        return [cleaned]

    parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if len(parts) == n:
        return parts
    if len(parts) > n:
        merged = parts[: n - 1] + ["\n\n".join(parts[n - 1 :])]
        return merged

    words = cleaned.split()
    if not words:
        return [""] * n

    weights = [max(len(block.split()), 1) for block in original_blocks]
    total = sum(weights)
    out: list[str] = []
    idx = 0
    for i, weight in enumerate(weights):
        if i == n - 1:
            out.append(" ".join(words[idx:]))
            break
        take = max(1, round(len(words) * weight / total))
        out.append(" ".join(words[idx : idx + take]))
        idx += take
    return out


def _read_text(path: Path) -> DocumentPayload:
    return DocumentPayload(
        text=path.read_text(encoding="utf-8"),
        source_path=path,
        format="text",
    )


def _read_docx(path: Path) -> DocumentPayload:
    Document = _require_docx()
    doc = Document(str(path))
    paragraphs = [para.text for para in doc.paragraphs]
    text = "\n\n".join(paragraphs)
    return DocumentPayload(
        text=text,
        source_path=path,
        format="docx",
        meta={"paragraphs": paragraphs},
    )


def _read_pdf(path: Path) -> DocumentPayload:
    fitz = _require_pymupdf()
    doc = fitz.open(str(path))
    try:
        pages = [page.get_text("text").strip() for page in doc]
    finally:
        doc.close()
    text = "\n\n".join(p for p in pages if p)
    return DocumentPayload(
        text=text,
        source_path=path,
        format="pdf",
        meta={"pages": pages},
    )


def load_document(path: Path | str) -> DocumentPayload:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return _read_text(path)
    if ext in DOCX_EXTENSIONS:
        return _read_docx(path)
    if ext in PDF_EXTENSIONS:
        return _read_pdf(path)
    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _write_text(path: Path, text: str, payload: DocumentPayload) -> None:
    path.write_text(text, encoding="utf-8")


def _write_docx(path: Path, text: str, payload: DocumentPayload) -> None:
    Document = _require_docx()
    source = payload.source_path
    original_paragraphs: list[str] = payload.meta.get("paragraphs", [])
    if not original_paragraphs and source and source.exists():
        original_paragraphs = [p.text for p in Document(str(source)).paragraphs]

    if source and source.exists() and source.suffix.lower() == ".docx":
        doc = Document(str(source))
    else:
        doc = Document()

    new_paragraphs = _distribute_text(original_paragraphs or [""], text)
    if doc.paragraphs:
        for idx, para in enumerate(doc.paragraphs):
            para.text = new_paragraphs[idx] if idx < len(new_paragraphs) else ""
        for extra in new_paragraphs[len(doc.paragraphs) :]:
            doc.add_paragraph(extra)
    else:
        for block in new_paragraphs:
            doc.add_paragraph(block)
    doc.save(str(path))


def _write_pdf(path: Path, text: str, payload: DocumentPayload) -> None:
    fitz = _require_pymupdf()
    pages_src: list[str] = payload.meta.get("pages") or [text]
    blocks = _distribute_text(pages_src, text)

    doc = fitz.open()
    try:
        for block in blocks:
            page = doc.new_page(width=595, height=842)  # A4
            margin = 50
            rect = fitz.Rect(margin, margin, page.rect.width - margin, page.rect.height - margin)
            page.insert_textbox(
                rect,
                block,
                fontsize=11,
                fontname="helv",
                align=fitz.TEXT_ALIGN_LEFT,
            )
        if not blocks:
            doc.new_page()
        doc.save(str(path))
    finally:
        doc.close()


def save_document(path: Path | str, text: str, payload: DocumentPayload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS or payload.format == "text":
        _write_text(path, text, payload)
    elif ext in DOCX_EXTENSIONS or payload.format == "docx":
        _write_docx(path, text, payload)
    elif ext in PDF_EXTENSIONS or payload.format == "pdf":
        _write_pdf(path, text, payload)
    else:
        _write_text(path.with_suffix(".txt"), text, payload)
        path = path.with_suffix(".txt")
    return path


def resolve_output_path(
    input_path: Path,
    *,
    output: Path | None = None,
    in_place: bool = False,
) -> Path:
    if in_place:
        return input_path
    if output is not None:
        return output
    return input_path.with_name(f"{input_path.stem}_humanized{input_path.suffix or '.txt'}")


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.stem}.bak.{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup
