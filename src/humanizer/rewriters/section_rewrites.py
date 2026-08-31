from __future__ import annotations

import re

from humanizer.templates.loader import apply_template_rules, load_academic_rules, load_zerogpt_pass_rules
from humanizer.validators.fidelity import validate_template_fidelity


def apply_sentence_templates(sentence: str, *, original_sentence: str | None = None) -> str:
    """Apply template rules to a single sentence with hard-gate fidelity only."""
    baseline = original_sentence or sentence
    current = sentence
    for pattern, repl in load_academic_rules():
        if not re.search(pattern, current, flags=re.IGNORECASE):
            continue
        candidate = re.sub(pattern, repl, current, flags=re.IGNORECASE)
        if validate_template_fidelity(baseline, candidate).passed:
            current = candidate
    return current


def apply_zerogpt_polish(text: str, baseline: str) -> str:
    """Final pass using ZeroGPT-calibrated templates (from live detector feedback)."""
    from humanizer.rewriters.transforms import rejoin_manuscript, split_manuscript_sentences

    orig_sents = split_manuscript_sentences(baseline)
    sentences = split_manuscript_sentences(text)
    if not sentences:
        return text
    polished: list[str] = []
    for idx, sentence in enumerate(sentences):
        orig = orig_sents[idx] if idx < len(orig_sents) else sentence
        current = sentence
        for pattern, repl in load_zerogpt_pass_rules():
            if not re.search(pattern, current, flags=re.IGNORECASE):
                continue
            candidate = re.sub(pattern, repl, current, flags=re.IGNORECASE)
            if validate_template_fidelity(orig, candidate).passed:
                current = candidate
        polished.append(current)
    return rejoin_manuscript(text, polished)


def apply_safe_section_rewrites(text: str, *, min_similarity: float = 0.72) -> str:
    """Document-level template pass (legacy compatibility)."""
    del min_similarity
    return apply_template_rules(text)


# Backward compatibility alias
ALL_SECTION_RULES = load_academic_rules()


def rewrite_section(header: str, body: str) -> str:
    del header
    return apply_template_rules(body)


def rewrite_methods_section(body: str) -> str:
    return apply_template_rules(body)


def rewrite_results_section(body: str) -> str:
    return apply_template_rules(body)


def rewrite_discussion_section(body: str) -> str:
    return apply_template_rules(body)


def rewrite_introduction_section(body: str) -> str:
    return apply_template_rules(body)
