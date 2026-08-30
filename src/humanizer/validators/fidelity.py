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
    if len_ratio < 0.6 or len_ratio > 1.6:
        issues.append(f"Length drift: {len_ratio:.2f}x")

    passed = (
        similarity >= min_similarity
        and not missing_numbers
        and not missing_citations
        and 0.65 <= len_ratio <= 1.5
    )

    return FidelityReport(
        passed=passed,
        similarity=similarity,
        missing_numbers=missing_numbers,
        missing_citations=missing_citations,
        length_ratio=len_ratio,
        issues=issues,
    )
