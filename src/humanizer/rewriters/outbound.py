from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from humanizer.rewriters.targeted_rewriter import RewriteResult
from humanizer.validators.fidelity import validate_template_fidelity

if TYPE_CHECKING:
    from humanizer.rewriters.targeted_rewriter import TargetedRewriter

# Letterhead, contact, section labels — never rewrite.
FROZEN_LINE_PREFIXES = (
    "To:",
    "Re:",
    "Dear ",
    "To Whom It May Concern",
    "Sincerely",
    "Email:",
    "Private email:",
    "Phone:",
    "Date:",
    "Letter of",
    "On behalf",
    "Dr. ",
    "Professor",
    "Faculty",
    "Institutional",
    "Author:",
    "Technical writing",
    "First-author",
    "TITL",
    "B ACK",
    "MATER",
    "R ESULTS",
    "CO NCL",
    "Integrative pathway",
    "Integrating pathway",
    "Pathway integration",
    "Institutional Signature",
)

LETTERHEAD_RE = re.compile(
    r"^(?:Professor of|Faculty of|Email:|Private email:|Phone:|Date:|Institutional)",
    re.I,
)

FROZEN_EXACT = frozenset(
    {
        "GitHub",
        "Shiraz, Iran (relocating)",
        "Mohammad Javad Darabi",
        "Islamic Azad University of Shushtar",
        "Shushtar, Iran",
    }
)

CONTACT_RE = re.compile(r"^[\w.+-]+@[\w.-]+\.\w+|^\+?\d[\d\s-]{8,}$", re.I)

# Module-level aggression for outbound (set from pipeline / callers).
_aggression: str = "conservative"
_allow_tone_down: bool = False


def set_outbound_aggression(aggression: str = "conservative", *, allow_tone_down: bool = False) -> None:
    global _aggression, _allow_tone_down
    _aggression = aggression if aggression in {"conservative", "high"} else "conservative"
    _allow_tone_down = allow_tone_down


@dataclass
class OutboundRegion:
    """Contiguous lines that are either frozen letterhead or editable body prose."""

    frozen: bool
    lines: list[str]
    start: int  # line index in original splitlines()


def _is_frozen_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped in FROZEN_EXACT:
        return True
    if any(stripped.startswith(prefix) for prefix in FROZEN_LINE_PREFIXES):
        return True
    if LETTERHEAD_RE.match(stripped):
        return True
    if CONTACT_RE.match(stripped):
        return True
    if re.match(r"^https?://", stripped):
        return True
    # Title lines in abstracts (short, no sentence end)
    if len(stripped.split()) <= 14 and stripped.endswith(":") and stripped.isupper() is False:
        if not stripped.endswith((".", "!", "?")):
            return True
    return False


def split_outbound_regions(text: str) -> list[OutboundRegion]:
    lines = text.splitlines()
    regions: list[OutboundRegion] = []
    body_buf: list[str] = []
    body_start = 0

    def flush_body(end_idx: int) -> None:
        nonlocal body_buf, body_start
        if body_buf:
            regions.append(OutboundRegion(frozen=False, lines=body_buf, start=body_start))
            body_buf = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            flush_body(idx)
            continue
        if _is_frozen_line(stripped):
            flush_body(idx)
            regions.append(OutboundRegion(frozen=True, lines=[line], start=idx))
            continue
        if not body_buf:
            body_start = idx
        body_buf.append(line)

    flush_body(len(lines))
    return regions


def split_outbound_blocks(text: str) -> list[str]:
    """Legacy flat block list (headers as single lines, body as joined paragraphs)."""
    blocks: list[str] = []
    for region in split_outbound_regions(text):
        if region.frozen:
            blocks.extend(line.strip() for line in region.lines if line.strip())
        else:
            blocks.append(" ".join(line.strip() for line in region.lines))
    return blocks


def _reflow_to_line_lengths(new_text: str, original_lines: list[str]) -> list[str] | None:
    """Width-based reflow shared with PDF write path (character-width approximation)."""
    from humanizer.io.pdf_layout import reflow_to_line_lengths

    return reflow_to_line_lengths(new_text, original_lines)


def _apply_regions_to_lines(original: str, regions: list[OutboundRegion]) -> str:
    lines = original.splitlines()
    for region in regions:
        if region.frozen:
            continue
        if len(" ".join(region.lines).strip()) < 40:
            continue
        joined = " ".join(line.strip() for line in region.lines)
        new_lines = _reflow_to_line_lengths(joined, region.lines)
        if new_lines is None:
            # Unsafe reflow — keep original region lines.
            continue
        for offset, new_line in enumerate(new_lines):
            line_idx = region.start + offset
            if line_idx < len(lines):
                lines[line_idx] = new_line
    return "\n".join(lines) + ("\n" if original.endswith("\n") else "")


@lru_cache(maxsize=1)
def _outbound_rules_path():
    from humanizer.templates.loader import _outbound_templates_root

    return _outbound_templates_root() / "letters.json"


@lru_cache(maxsize=1)
def _abstract_rules_path():
    from humanizer.templates.loader import _outbound_templates_root

    return _outbound_templates_root() / "abstracts.json"


@lru_cache(maxsize=1)
def load_abstract_rules() -> list[tuple[str, str]]:
    import json

    path = _abstract_rules_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = [(item["pattern"], item["replacement"]) for item in data.get("rules", []) if item.get("pattern")]
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


def _is_abstract_document(text: str) -> bool:
    lower = text.lower()
    return "conference abstract" in lower or "technical writing sample" in lower or "b ack gr o und" in lower


def _is_recommendation_document(text: str) -> bool:
    lower = text.lower()
    return "letter of recommendation" in lower or "to whom it may concern" in lower


def _is_motivation_document(text: str) -> bool:
    lower = text.lower()
    return "phd computational neuroscience" in lower or "ki scilifelab" in lower


@lru_cache(maxsize=1)
def load_outbound_rules() -> list[tuple[str, str]]:
    import json

    path = _outbound_rules_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = [(item["pattern"], item["replacement"]) for item in data.get("rules", []) if item.get("pattern")]
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


def _fidelity_ok(original: str, candidate: str) -> bool:
    report = validate_template_fidelity(
        original,
        candidate,
        min_similarity=0.72,
        min_length_ratio=0.85,
        max_length_ratio=1.15,
        allow_tone_down=_allow_tone_down,
    )
    return report.passed


def _apply_rules_to_block(block: str, rules: list[tuple[str, str]]) -> str:
    current = block
    for pattern, repl in rules:
        if not re.search(pattern, current, flags=re.IGNORECASE):
            continue
        candidate = re.sub(pattern, repl, current, flags=re.IGNORECASE).strip()
        if not candidate:
            continue
        # Gate against the pre-rule span so each swap must preserve meaning vs current text.
        if _fidelity_ok(current, candidate):
            current = candidate
    return current


def _apply_stylometric_pass(block: str) -> str:
    """Light local AI-tell transforms — fidelity-gated, no full-sentence swaps."""
    from humanizer.rewriters.transforms import (
        apply_phrase_replacements,
        diversify_sentence_openers,
        reduce_lexical_repetition,
        rejoin_manuscript,
        soften_em_dashes,
        split_manuscript_sentences,
    )

    current = apply_phrase_replacements(block, extended=False)
    current = reduce_lexical_repetition(current)
    current = soften_em_dashes(current)
    sentences = split_manuscript_sentences(current)
    sentences = diversify_sentence_openers(sentences, iteration=1)
    current = rejoin_manuscript(current, sentences)
    if _fidelity_ok(block, current):
        return current
    return block


def _humanize_body_region(
    lines: list[str],
    rules: list[tuple[str, str]],
    *,
    baseline_lines: list[str] | None = None,
    skip_stylometric: bool = False,
) -> list[str]:
    del baseline_lines  # polish disabled in quality-first path
    joined = " ".join(line.strip() for line in lines)
    if len(joined) < 40:
        return lines

    updated = _apply_rules_to_block(joined, rules)
    if not skip_stylometric:
        updated = _apply_stylometric_pass(updated)
    if updated == joined:
        return lines

    # Document-level fidelity vs original joined body.
    if not _fidelity_ok(joined, updated):
        return lines

    new_lines = _reflow_to_line_lengths(updated, lines)
    if new_lines is None:
        return lines
    return new_lines


def _outbound_polish_regions(text: str, baseline: str) -> str:
    """Conservative path: skip stacked ZeroGPT polish (meaning-damaging)."""
    del baseline
    if _aggression != "high":
        return text
    from humanizer.rewriters.section_rewrites import apply_zerogpt_polish

    regions = split_outbound_regions(text)
    lines = text.splitlines()
    changed = False
    for region in regions:
        if region.frozen:
            continue
        joined = " ".join(line.strip() for line in region.lines)
        polished = apply_zerogpt_polish(joined, joined)
        if polished.strip() == joined.strip() or not _fidelity_ok(joined, polished):
            continue
        new_lines = _reflow_to_line_lengths(polished, region.lines)
        if new_lines is None:
            continue
        for offset, new_line in enumerate(new_lines):
            idx = region.start + offset
            if idx < len(lines):
                lines[idx] = new_line
        changed = True
    if not changed:
        return text
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _select_rules_for_document(text: str) -> list[tuple[str, str]]:
    """Conservative: letters/abstracts only. High: also stack ZeroGPT packs."""
    if _is_abstract_document(text):
        rules = list(load_abstract_rules())
        if _aggression == "high":
            from humanizer.templates.loader import load_abstract_zerogpt_rules

            rules = rules + load_abstract_zerogpt_rules()
        return rules

    rules = list(load_outbound_rules())
    if _aggression == "high":
        from humanizer.templates.loader import (
            load_motivation_zerogpt_rules,
            load_recommendation_zerogpt_rules,
            load_zerogpt_pass_rules,
        )

        if _is_recommendation_document(text):
            rules = rules + load_recommendation_zerogpt_rules()
        elif _is_motivation_document(text):
            rules = rules + load_motivation_zerogpt_rules()
        rules = rules + load_zerogpt_pass_rules()
    return rules


def apply_abstract_humanize(text: str, *, baseline: str | None = None) -> str:
    rules = _select_rules_for_document(text)
    if not rules and _aggression == "conservative":
        # Still allow local stylometric AI-tell cleanup.
        rules = []
    baseline_text = baseline or text
    regions = split_outbound_regions(text)
    baseline_regions = {r.start: r for r in split_outbound_regions(baseline_text)}
    for region in regions:
        if region.frozen:
            continue
        base_region = baseline_regions.get(region.start)
        base_lines = base_region.lines if base_region and not base_region.frozen else region.lines
        region.lines = _humanize_body_region(
            region.lines,
            rules,
            baseline_lines=base_lines,
            skip_stylometric=False,
        )
    return _apply_regions_to_lines(text, regions)


def apply_outbound_humanize(text: str, *, baseline: str | None = None) -> str:
    """Rewrite body prose only; letterhead, titles, and section labels stay untouched."""
    if _is_abstract_document(text):
        return apply_abstract_humanize(text, baseline=baseline)

    rules = _select_rules_for_document(text)
    baseline_text = baseline or text
    regions = split_outbound_regions(text)
    baseline_regions = {r.start: r for r in split_outbound_regions(baseline_text)}
    for region in regions:
        if region.frozen:
            continue
        base_region = baseline_regions.get(region.start)
        base_lines = base_region.lines if base_region and not base_region.frozen else region.lines
        region.lines = _humanize_body_region(
            region.lines,
            rules,
            baseline_lines=base_lines,
            skip_stylometric=False,
        )
    result = _apply_regions_to_lines(text, regions)
    return result.strip() + "\n"


def apply_outbound_targeted_pass(
    text: str,
    baseline: str,
    rewriter: TargetedRewriter,
    *,
    hot_threshold: float | None = None,
    rewrite_all_sentences: bool = True,
) -> RewriteResult:
    del rewriter, hot_threshold, rewrite_all_sentences, baseline
    # Targeted ML paraphrase on short letters breaks layout; templates only.
    return RewriteResult(text=text, changed_sentences=[], applied=[])


def apply_outbound_iterative(
    text: str,
    baseline: str,
    *,
    max_rounds: int = 12,
) -> RewriteResult:
    """Repeat template passes on body prose only, preserving line layout."""
    current = text
    applied: list[str] = []
    # Conservative: one pass is enough; stacking erodes meaning.
    rounds = 1 if _aggression == "conservative" else max_rounds
    for round_num in range(rounds):
        templated = apply_outbound_humanize(current, baseline=baseline)
        polished = _outbound_polish_regions(templated, baseline)
        if polished.strip() == current.strip():
            break

        def _body_joined(doc: str) -> str:
            return " ".join(
                " ".join(line.strip() for line in region.lines)
                for region in split_outbound_regions(doc)
                if not region.frozen
            )

        base_body = _body_joined(text)
        new_body = _body_joined(polished)
        if base_body and new_body and not _fidelity_ok(base_body, new_body):
            break
        current = polished
        applied.append(f"outbound:round{round_num + 1}")
    changed = current.strip() != text.strip()
    return RewriteResult(text=current, changed_sentences=[0] if changed else [], applied=applied)
