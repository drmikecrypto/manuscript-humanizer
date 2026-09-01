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
    claim_strength_delta: float = 0.0
    missing_content_units: list[str] = field(default_factory=list)
    invented_numbers: list[str] = field(default_factory=list)


# Stronger endorsement / praise lexicon → weaker. Reject if rewrite loses strength.
_STRENGTH_RANK: dict[str, int] = {
    "strongest": 5,
    "exceptional": 5,
    "outstanding": 5,
    "remarkable": 4,
    "excellent": 4,
    "ideally suited": 4,
    "significant": 3,
    "rigorous": 3,
    "comprehensive": 3,
    "strong": 3,
    "solid": 2,
    "sound": 2,
    "steady": 2,
    "routine": 1,
    "usable": 1,
}

_STRENGTH_PHRASE_RE = re.compile(
    r"\b("
    + "|".join(re.escape(k) for k in sorted(_STRENGTH_RANK, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

_QUOTED_TITLE_RE = re.compile(r'"([^"]{8,})"')
_SUPERVISION_RE = re.compile(
    r"\b(direct supervision|thesis supervisor|supervised|under my)\b",
    re.IGNORECASE,
)
_PROGRAM_RE = re.compile(
    r"\b(KI SciLifeLab|computational neuroscience|doctoral|PhD|ESHG\s*2025)\b",
    re.IGNORECASE,
)


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


def _claim_strength_score(text: str) -> float:
    matches = _STRENGTH_PHRASE_RE.findall(text)
    if not matches:
        return 0.0
    return float(sum(_STRENGTH_RANK.get(m.lower(), 0) for m in matches))


def _claim_strength_delta(original: str, rewritten: str) -> float:
    """Positive = rewrite is weaker than source."""
    return _claim_strength_score(original) - _claim_strength_score(rewritten)


def _extract_content_units(text: str) -> list[str]:
    """Named facts / titles / supervision claims that must survive a rewrite."""
    units: list[str] = []
    for m in _QUOTED_TITLE_RE.finditer(text):
        units.append(f'"{m.group(1)}"')
    for m in NUMBER_RE.finditer(text):
        units.append(m.group(0))
    for m in _SUPERVISION_RE.finditer(text):
        units.append(m.group(0).lower())
    for m in _PROGRAM_RE.finditer(text):
        units.append(m.group(0).lower())
    # HLA / markers common in abstracts
    for m in re.finditer(r"\b(CD44|CD133|EPCAM|HLA-A0[24]:0[12])\b", text, re.I):
        units.append(m.group(0).upper() if m.group(0).upper().startswith("HLA") else m.group(0))
    return units


def _missing_content_units(original: str, rewritten: str) -> list[str]:
    missing: list[str] = []
    new_lower = rewritten.lower()
    for unit in _extract_content_units(original):
        if unit.startswith('"') and unit.endswith('"'):
            inner = unit[1:-1]
            if inner.lower() not in new_lower and unit not in rewritten:
                missing.append(unit)
        elif unit.lower() not in new_lower and unit not in rewritten:
            missing.append(unit)
    return missing


def _invented_numbers(original: str, rewritten: str) -> list[str]:
    """Digits present in rewrite but absent from source (e.g. invented 'eleven'/years)."""
    orig = _extract_numbers(original)
    new = _extract_numbers(rewritten)
    return sorted(new - orig)


def validate_fidelity(
    original: str,
    rewritten: str,
    *,
    min_similarity: float = 0.82,
    min_length_ratio: float = 0.78,
    max_length_ratio: float = 1.5,
    preserve_numbers: bool = True,
    preserve_citations: bool = True,
    allow_tone_down: bool = False,
    require_content_units: bool = False,
    reject_invented_numbers: bool = False,
    require_protected_terms: bool = False,
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

    strength_delta = _claim_strength_delta(original, rewritten)
    if not allow_tone_down and strength_delta > 2.0:
        issues.append(f"Claim strength downgraded (delta={strength_delta:.1f})")

    missing_units: list[str] = []
    if require_content_units:
        missing_units = _missing_content_units(original, rewritten)
        if missing_units:
            issues.append(f"Missing content units: {', '.join(missing_units[:5])}")

    invented: list[str] = []
    if reject_invented_numbers:
        invented = _invented_numbers(original, rewritten)
        if invented:
            issues.append(f"Invented numbers: {', '.join(invented[:5])}")

    missing_protected: list[str] = []
    if require_protected_terms:
        terms_ok, missing_protected = _protected_terms_preserved(original, rewritten)
        if not terms_ok:
            issues.append(f"Missing protected terms: {', '.join(missing_protected[:5])}")

    passed = (
        similarity >= min_similarity
        and not missing_numbers
        and not missing_citations
        and min_length_ratio <= len_ratio <= max_length_ratio
        and (allow_tone_down or strength_delta <= 2.0)
        and (not require_content_units or not missing_units)
        and (not reject_invented_numbers or not invented)
        and (not require_protected_terms or not missing_protected)
    )

    return FidelityReport(
        passed=passed,
        similarity=similarity,
        missing_numbers=missing_numbers,
        missing_citations=missing_citations,
        length_ratio=len_ratio,
        issues=issues,
        claim_strength_delta=strength_delta,
        missing_content_units=missing_units,
        invented_numbers=invented,
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
    min_similarity: float = 0.65,
    min_length_ratio: float = 0.72,
    max_length_ratio: float = 1.25,
    allow_tone_down: bool = False,
    require_content_units: bool = True,
    reject_invented_numbers: bool = True,
) -> FidelityReport:
    """Quality-first gates for curated JSON template rules.

    Defaults suit academic near-paraphrases. Outbound short-form callers pass
    stricter bands (0.72 / 0.85–1.15).
    """
    return validate_fidelity(
        original_sentence,
        rewritten_sentence,
        min_similarity=min_similarity,
        min_length_ratio=min_length_ratio,
        max_length_ratio=max_length_ratio,
        preserve_numbers=preserve_numbers,
        preserve_citations=preserve_citations,
        allow_tone_down=allow_tone_down,
        require_content_units=require_content_units,
        reject_invented_numbers=reject_invented_numbers,
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
    allow_tone_down: bool = False,
    require_content_units: bool = False,
    reject_invented_numbers: bool = False,
    require_protected_terms: bool = False,
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
        allow_tone_down=allow_tone_down,
        require_content_units=require_content_units,
        reject_invented_numbers=reject_invented_numbers,
        require_protected_terms=require_protected_terms,
    )


def build_manuscript_quality_report(
    original: str,
    rewritten: str,
    *,
    min_similarity: float = 0.55,
    min_length_ratio: float = 0.78,
) -> FidelityReport:
    """Quality gate for long-form academic manuscripts."""
    return validate_document_output(
        original,
        rewritten,
        min_similarity=min_similarity,
        min_length_ratio=min_length_ratio,
        max_length_ratio=1.5,
        require_protected_terms=True,
    )


def build_quality_report(original: str, rewritten: str) -> FidelityReport:
    """Informational QC report (strict short-form defaults)."""
    return validate_fidelity(
        original,
        rewritten,
        min_similarity=0.72,
        min_length_ratio=0.85,
        max_length_ratio=1.15,
        preserve_numbers=True,
        preserve_citations=True,
        allow_tone_down=False,
        require_content_units=True,
        reject_invented_numbers=True,
    )
