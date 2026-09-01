"""Map manuscript sentences to academic section template packs."""

from __future__ import annotations


from humanizer.rewriters.transforms import (
    SECTION_HEADER_ONLY_RE,
    is_section_header,
    split_manuscript_sentences,
    split_sentences,
)

_SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "introduction": ("introduction", "background"),
    "methods": ("method", "material"),
    "results": ("result", "finding"),
    "discussion": ("discussion", "conclusion"),
}


def normalize_section_name(header: str) -> str:
    """Map a section header line to a template pack name."""
    lower = header.lower()
    for name, keywords in _SECTION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return name
    return "fallback"


def build_sentence_section_map(full_text: str, sentences: list[str] | None = None) -> list[str]:
    """Return template section name for each editable sentence index."""
    sents = sentences if sentences is not None else split_manuscript_sentences(full_text)
    if not sents:
        return []

    mapping: list[str] = []
    current = "fallback"
    pending = list(sents)
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal current
        if not paragraph_lines:
            return
        para = " ".join(paragraph_lines).strip()
        paragraph_lines.clear()
        if not para:
            return
        for sent in split_sentences(para):
            sent = sent.strip()
            if len(sent.split()) < 5:
                continue
            if SECTION_HEADER_ONLY_RE.match(sent):
                current = normalize_section_name(sent)
                continue
            if pending and (sent == pending[0] or pending[0] in sent or sent in pending[0]):
                mapping.append(current)
                pending.pop(0)

    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if is_section_header(line) or SECTION_HEADER_ONLY_RE.match(line):
            flush_paragraph()
            current = normalize_section_name(line)
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    while len(mapping) < len(sents):
        mapping.append(current)
    return mapping[: len(sents)]


def section_for_sentence_index(text: str, sentence_index: int) -> str:
    """Section pack name for a sentence index in full document text."""
    sentences = split_manuscript_sentences(text)
    if sentence_index < 0 or sentence_index >= len(sentences):
        return "fallback"
    return build_sentence_section_map(text, sentences)[sentence_index]
