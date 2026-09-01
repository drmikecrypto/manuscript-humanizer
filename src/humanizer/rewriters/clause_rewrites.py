"""Domain-safe clause-level rewrites for academic manuscripts."""

from __future__ import annotations

import re

from humanizer.validators.fidelity import validate_template_fidelity

# Generalized pattern families (not fixture-specific sentences).
CLAUSE_RULES: list[tuple[str, str, str]] = [
    # (section, pattern, replacement) — section "all" applies everywhere
    (
        "methods",
        r"\bIn this study,\s+(.+?)\s+were randomly divided into\b",
        r"\1 were allocated to",
    ),
    (
        "methods",
        r"\bwere randomly divided into\b",
        "were allocated to",
    ),
    (
        "methods",
        r"\bwere induced with diabetes using streptozotocin\b",
        "received streptozotocin diabetes induction",
    ),
    (
        "results",
        r"\bThe results indicated that\b",
        "We observed that",
    ),
    (
        "results",
        r"\bThe results indicate that\b",
        "The data show that",
    ),
    (
        "results",
        r"\bindicated that\b",
        "showed that",
    ),
    (
        "results",
        r"\bMeasurements of (.+?) were performed\b",
        r"We measured \1",
    ),
    (
        "discussion",
        r"\bThis study suggests that\b",
        "Our findings suggest that",
    ),
    (
        "discussion",
        r"\bFurthermore,\s*",
        "",
    ),
    (
        "discussion",
        r"\bMoreover,\s*",
        "",
    ),
    (
        "all",
        r"\bIn order to\b",
        "to",
    ),
    (
        "all",
        r"\bplays a (?:crucial|key|vital|pivotal) role\b",
        "matters",
    ),
    (
        "introduction",
        r"\bone of the most common\b",
        "a common",
    ),
    (
        "introduction",
        r"\bcharacterized by\b",
        "marked by",
    ),
]


def apply_clause_rewrites(
    sentence: str,
    *,
    original_sentence: str | None = None,
    section: str = "fallback",
) -> str:
    """Apply section-scoped clause templates with fidelity gating."""
    baseline = original_sentence or sentence
    current = sentence
    for rule_section, pattern, repl in CLAUSE_RULES:
        if rule_section not in (section, "all", "fallback"):
            continue
        if not re.search(pattern, current, flags=re.IGNORECASE):
            continue
        candidate = re.sub(pattern, repl, current, flags=re.IGNORECASE).strip()
        if not candidate or candidate == current:
            continue
        if validate_template_fidelity(baseline, candidate).passed:
            current = candidate
    return current
