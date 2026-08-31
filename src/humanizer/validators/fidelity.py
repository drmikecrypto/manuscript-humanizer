from __future__ import annotations

import re
from dataclasses import dataclass, field

from humanizer.analyzers.detector import CITATION_RE, NUMBER_RE


@dataclass
class FidelityReport:
    passed: bool
    similarity: float
    missing_numbers: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    length_ratio: float = 1.0
    issues: list[str] = field(default_factory=list)


def _extract_numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def _extract_citations(text: str) -> set[str]:
    return set(CITATION_RE.findall(text))


def _token_overlap(a: str, b: str) -> float:
    """Lexical overlap proxy for meaning preservation (no embedding dependency)."""
    tokens_a = set(re.findall(r"\b\w{3,}\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w{3,}\b", b.lower()))
    if not tokens_a:
        return 1.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def validate_fidelity(
    original: str,
    rewritten: str,
    *,
    min_similarity: float = 0.82,
    min_length_ratio: float = 0.78,
    max_length_ratio: float = 1.5,
    preserve_numbers: bool = True,
    preserve_citations: bool = True,
) -> FidelityReport:
    issues: list[str] = []
    similarity = _token_overlap(original, rewritten)

    missing_numbers: list[str] = []
    missing_citations: list[str] = []

    if preserve_numbers:
        orig_nums = _extract_numbers(original)
        new_nums = _extract_numbers(rewritten)
        missing_numbers = sorted(orig_nums - new_nums)
        if missing_numbers:
            issues.append(f"Missing numbers: {', '.join(missing_numbers[:5])}")

    if preserve_citations:
        orig_cites = _extract_citations(original)
        new_cites = _extract_citations(rewritten)
        missing_citations = sorted(orig_cites - new_cites)
        if missing_citations:
            issues.append(f"Missing citations: {', '.join(missing_citations[:3])}")

    len_ratio = len(rewritten) / max(len(original), 1)
    if len_ratio < min_length_ratio or len_ratio > max_length_ratio:
        issues.append(f"Length drift: {len_ratio:.2f}x")

    if similarity < min_similarity:
        issues.append(f"Low token overlap: {similarity:.2f} < {min_similarity:.2f}")

    passed = (
        similarity >= min_similarity
        and not missing_numbers
        and not missing_citations
        and min_length_ratio <= len_ratio <= max_length_ratio
    )

    return FidelityReport(
        passed=passed,
        similarity=similarity,
        missing_numbers=missing_numbers,
        missing_citations=missing_citations,
        length_ratio=len_ratio,
        issues=issues,
    )


_PROTECTED_TERM_RE = re.compile(
    r"\b(streptozotocin|glibenclamide|wistar|nettle|fenugreek|insulin|diabetes)\b",
    re.IGNORECASE,
)


def _protected_terms_preserved(original: str, rewritten: str) -> tuple[bool, list[str]]:
    orig_terms = {m.group(0).lower() for m in _PROTECTED_TERM_RE.finditer(original)}
    new_lower = rewritten.lower()
    missing = [t for t in sorted(orig_terms) if t not in new_lower]
    return not missing, missing


def validate_sentence_fidelity(
    original_sentence: str,
    rewritten_sentence: str,
    *,
    min_similarity: float = 0.40,
    preserve_numbers: bool = True,
    preserve_citations: bool = True,
) -> FidelityReport:
    """Gate a single-sentence rewrite (hard: numbers/citations/terms; soft: overlap)."""
    issues: list[str] = []
    similarity = _token_overlap(original_sentence, rewritten_sentence)

    missing_numbers: list[str] = []
    missing_citations: list[str] = []

    if preserve_numbers:
        orig_nums = _extract_numbers(original_sentence)
        new_nums = _extract_numbers(rewritten_sentence)
        missing_numbers = sorted(orig_nums - new_nums)
        if missing_numbers:
            issues.append(f"Missing numbers: {', '.join(missing_numbers[:5])}")

    if preserve_citations:
        orig_cites = _extract_citations(original_sentence)
        new_cites = _extract_citations(rewritten_sentence)
        missing_citations = sorted(orig_cites - new_cites)
        if missing_citations:
            issues.append(f"Missing citations: {', '.join(missing_citations[:3])}")

    terms_ok, missing_terms = _protected_terms_preserved(original_sentence, rewritten_sentence)
    if not terms_ok:
        issues.append(f"Missing protected terms: {', '.join(missing_terms[:5])}")

    len_ratio = len(rewritten_sentence) / max(len(original_sentence), 1)
    if len_ratio < 0.72 or len_ratio > 2.0:
        issues.append(f"Length drift: {len_ratio:.2f}x")

    passed = (
        similarity >= min_similarity
        and not missing_numbers
        and not missing_citations
        and terms_ok
        and 0.72 <= len_ratio <= 2.0
    )

    return FidelityReport(
        passed=passed,
        similarity=similarity,
        missing_numbers=missing_numbers,
        missing_citations=missing_citations,
        length_ratio=len_ratio,
        issues=issues,
    )


def validate_template_fidelity(
    original_sentence: str,
    rewritten_sentence: str,
    *,
    preserve_numbers: bool = True,
    preserve_citations: bool = True,
) -> FidelityReport:
    """Hard gates only for curated JSON template rules (no overlap gate)."""
    issues: list[str] = []
    similarity = _token_overlap(original_sentence, rewritten_sentence)

    missing_numbers: list[str] = []
    missing_citations: list[str] = []

    if preserve_numbers:
        orig_nums = _extract_numbers(original_sentence)
        new_nums = _extract_numbers(rewritten_sentence)
        missing_numbers = sorted(orig_nums - new_nums)
        if missing_numbers:
            issues.append(f"Missing numbers: {', '.join(missing_numbers[:5])}")

    if preserve_citations:
        orig_cites = _extract_citations(original_sentence)
        new_cites = _extract_citations(rewritten_sentence)
        missing_citations = sorted(orig_cites - new_cites)
        if missing_citations:
            issues.append(f"Missing citations: {', '.join(missing_citations[:3])}")

    terms_ok, missing_terms = _protected_terms_preserved(original_sentence, rewritten_sentence)
    if not terms_ok:
        issues.append(f"Missing protected terms: {', '.join(missing_terms[:5])}")

    len_ratio = len(rewritten_sentence) / max(len(original_sentence), 1)
    if len_ratio < 0.72 or len_ratio > 2.5:
        issues.append(f"Length drift: {len_ratio:.2f}x")

    passed = (
        not missing_numbers
        and not missing_citations
        and terms_ok
        and 0.72 <= len_ratio <= 2.5
    )

    return FidelityReport(
        passed=passed,
        similarity=similarity,
        missing_numbers=missing_numbers,
        missing_citations=missing_citations,
        length_ratio=len_ratio,
        issues=issues,
    )


def validate_document_output(
    original: str,
    rewritten: str,
    *,
    min_similarity: float = 0.65,
    min_length_ratio: float = 0.78,
    max_length_ratio: float = 1.5,
    preserve_numbers: bool = True,
    preserve_citations: bool = True,
) -> FidelityReport:
    """Final output gate before writing file."""
    return validate_fidelity(
        original,
        rewritten,
        min_similarity=min_similarity,
        min_length_ratio=min_length_ratio,
        max_length_ratio=max_length_ratio,
        preserve_numbers=preserve_numbers,
        preserve_citations=preserve_citations,
    )
