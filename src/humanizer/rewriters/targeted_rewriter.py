from __future__ import annotations

import re
from dataclasses import dataclass

from humanizer.analyzers.patterns import analyze_patterns
from humanizer.analyzers.zerogpt_proxy import ZeroGPTProxyReport, ZeroGPTProxyScorer, _get_onnx_score
from humanizer.lexicon.service import LexiconService
from humanizer.rewriters.lexicon_rewriter import apply_local_paraphrase
from humanizer.rewriters.rhythm import apply_rhythm_pass, break_parallel_structures, merge_intro_type2_sentences
from humanizer.rewriters.section_rewrites import apply_sentence_templates, apply_zerogpt_polish
from humanizer.rewriters.transforms import (
    OPENER_ALTERNATIVES,
    apply_phrase_replacements,
    rejoin_manuscript,
    split_manuscript_sentences,
)
from humanizer.validators.fidelity import (
    validate_sentence_fidelity,
    validate_template_fidelity,
)

AI_VOCAB_IN_SENTENCE = re.compile(
    r"\b(furthermore|moreover|additionally|utilize|leverage|comprehensive|robust|"
    r"significantly|substantially|demonstrate|facilitate|delve|underscores?)\b",
    re.IGNORECASE,
)


@dataclass
class RewriteResult:
    text: str
    changed_sentences: list[int]
    applied: list[str]


def _zerogpt_pick_score(sentence: str) -> float:
    """Blend ONNX with pattern tells — ZeroGPT tracks the latter more than our ML model."""
    ml = _get_onnx_score(sentence)
    pat = analyze_patterns(sentence).score
    return ml * 0.35 + pat * 0.65


def _candidate_pick_score(original: str, candidate: str) -> float:
    """Score rewrite candidates; penalise aggressive shortening."""
    base = _zerogpt_pick_score(candidate)
    len_ratio = len(candidate) / max(len(original), 1)
    if len_ratio < 0.72:
        return base + (0.72 - len_ratio) * 250.0
    if len_ratio < 0.85:
        return base + (0.85 - len_ratio) * 40.0
    return base


def _apply_opener_pass(sentence: str) -> str:
    match = re.match(r"^(\w+)(\s*,?\s+)", sentence)
    if not match:
        return sentence
    opener = match.group(1).lower()
    alts = OPENER_ALTERNATIVES.get(opener)
    if not alts:
        return sentence
    rest = sentence[match.end() :]
    return alts[0].capitalize() + match.group(2) + rest


def _apply_burstiness_pass(sentence: str) -> str:
    if len(sentence.split()) < 22:
        return sentence
    if re.search(
        r"\b(reducing|lowering|increasing|raising|decreasing|reduced|increased|decreased)\b.*,.*,.*and\b",
        sentence,
        re.IGNORECASE,
    ):
        return sentence
    parts = re.split(r";\s*", sentence, maxsplit=1)
    if len(parts) != 2 or len(parts[0].split()) <= 6:
        return sentence
    a, b = parts[0].strip(), parts[1].strip()
    if not a.endswith("."):
        a += "."
    if b and b[0].islower():
        b = b[0].upper() + b[1:]
    return f"{a} {b}"


def _apply_vocab_pass(sentence: str, lexicon: LexiconService, protected: set[str]) -> str:
    current = sentence
    for word in re.findall(r"\b[a-zA-Z]{4,}\b", sentence):
        if word in protected or lexicon.is_protected(word):
            continue
        if not AI_VOCAB_IN_SENTENCE.search(word):
            continue
        alt = lexicon.suggest_swap(word, rng_seed=hash(word) & 0xFFFF)
        if not alt:
            continue
        replaced = lexicon.safe_replace_token(current, word, alt, protected)
        if replaced:
            return replaced
    return current


def _try_step(original: str, current: str, step_fn, *, min_similarity: float = 0.35) -> str:
    candidate = step_fn(current)
    if candidate == current:
        return current
    if validate_sentence_fidelity(original, candidate, min_similarity=min_similarity).passed:
        return candidate
    return current


def _try_template_step(original: str, current: str, step_fn) -> str:
    candidate = step_fn(current)
    if candidate == current:
        return current
    if validate_template_fidelity(original, candidate).passed:
        return candidate
    return current


def _sentence_rewrite_allowed(original_sentence: str, candidate: str) -> bool:
    """Hard gates for a single-sentence edit (no document overlap gate)."""
    if candidate == original_sentence:
        return False
    template_ok = validate_template_fidelity(original_sentence, candidate).passed
    if template_ok:
        return True
    return validate_sentence_fidelity(original_sentence, candidate, min_similarity=0.35).passed


def _collect_sentence_candidates(
    sentence: str,
    original_sentence: str,
    lexicon: LexiconService,
    protected: set[str],
    *,
    doc_score: float,
) -> list[tuple[str, list[str]]]:
    """Build all local rewrite candidates for ML minimization."""
    seen: set[str] = {sentence}
    candidates: list[tuple[str, list[str]]] = []

    def add(text: str, tags: list[str]) -> None:
        if text in seen:
            return
        if not _sentence_rewrite_allowed(original_sentence, text):
            return
        seen.add(text)
        candidates.append((text, tags))

    def add_with_template_hops(text: str, tags: list[str]) -> None:
        add(text, tags)
        hop = apply_sentence_templates(text, original_sentence=original_sentence)
        if hop != text:
            add(hop, tags + ["template_hop"])

    chain_out, chain_tags = transform_sentence_chain(
        sentence, lexicon, protected, doc_score=doc_score
    )
    if chain_out != sentence:
        add_with_template_hops(chain_out, chain_tags)

    template_out = apply_sentence_templates(sentence, original_sentence=original_sentence)
    if template_out != sentence:
        add_with_template_hops(template_out, ["template"])

    for intensity in ("light", "medium", "strong"):
        para = apply_local_paraphrase(
            sentence,
            lexicon,
            protected,
            original=original_sentence,
            intensity=intensity,
            min_pattern_score=0.0,
        )
        if para != sentence:
            add_with_template_hops(para, [f"paraphrase:{intensity}"])

    rhythm_out = apply_rhythm_pass(sentence, original=original_sentence, min_similarity=0.35)
    if rhythm_out != sentence:
        add_with_template_hops(rhythm_out, ["rhythm"])

    parallel_out = break_parallel_structures(sentence)
    if parallel_out != sentence:
        add_with_template_hops(parallel_out, ["parallel"])

    phrase_out = apply_phrase_replacements(sentence)
    if phrase_out != sentence:
        add_with_template_hops(phrase_out, ["phrase"])

    return candidates


def transform_sentence_chain(
    sentence: str,
    lexicon: LexiconService,
    protected: set[str],
    *,
    max_rounds: int = 3,
    doc_score: float = 100.0,
) -> tuple[str, list[str]]:
    """Multi-step transform chain with sentence-level fidelity after each step."""
    original = sentence
    current = sentence
    applied: list[str] = []
    template_applied = False
    rounds = 5 if doc_score > 10 else max_rounds

    def _template_step(s: str) -> str:
        nonlocal template_applied
        out = apply_sentence_templates(s, original_sentence=original)
        if out != s:
            template_applied = True
        return out

    def _parallel_step(s: str) -> str:
        if template_applied:
            return s
        return break_parallel_structures(s)

    steps: list[tuple[str, object]] = [
        ("template", _template_step),
        ("rhythm", lambda s: apply_rhythm_pass(s, original=original, min_similarity=0.48, skip_parallel=True)),
        ("parallel", _parallel_step),
        ("phrase", apply_phrase_replacements),
        ("vocab", lambda s: _apply_vocab_pass(s, lexicon, protected)),
        ("opener", _apply_opener_pass),
        ("burstiness", _apply_burstiness_pass),
        (
            "paraphrase",
            lambda s: apply_local_paraphrase(
                s, lexicon, protected, original=original, intensity="light"
            )
            if not template_applied
            else s,
        ),
        ("template_final", _template_step),
    ]

    for _ in range(rounds):
        round_changed = False
        template_applied = False
        for name, fn in steps:
            if name in ("template", "template_final"):
                next_text = _try_template_step(original, current, fn)
            else:
                next_text = _try_step(original, current, fn)
            if next_text != current:
                current = next_text
                applied.append(name)
                round_changed = True
                if name == "template":
                    template_applied = True
        if not round_changed:
            break

    return current, applied


class TargetedRewriter:
    """Edit flagged sentences using a multi-step structural transform chain."""

    def __init__(
        self,
        lexicon: LexiconService | None = None,
        min_similarity: float = 0.72,
        proxy_scorer: ZeroGPTProxyScorer | None = None,
    ) -> None:
        self.lexicon = lexicon or LexiconService()
        self.min_similarity = min_similarity
        self.proxy = proxy_scorer or ZeroGPTProxyScorer()

    def rewrite_pass(
        self,
        text: str,
        report: ZeroGPTProxyReport | None = None,
        *,
        sentences_per_pass: int = 0,
        original: str | None = None,
        hot_threshold: float | None = None,
        top_n: int = 0,
        rewrite_all_sentences: bool = False,
    ) -> RewriteResult:
        baseline = original or text
        sentences = split_manuscript_sentences(text)
        if not sentences:
            return RewriteResult(text=text, changed_sentences=[], applied=[])

        proxy_report = report or self.proxy.analyze(text)
        threshold = hot_threshold if hot_threshold is not None else self.proxy.hot_threshold

        protected = self.lexicon.extract_protected_from_text(baseline)
        original_sentences = split_manuscript_sentences(baseline)
        orig_map = {
            i: original_sentences[i] if i < len(original_sentences) else s
            for i, s in enumerate(sentences)
        }

        ranked = sorted(proxy_report.sentences, key=lambda s: (s.ml_score, s.score), reverse=True)

        if rewrite_all_sentences:
            hot_indices = [s.index for s in ranked if s.index < len(sentences)]
        else:
            hot_indices = [i for i in proxy_report.hot_sentences if i < len(sentences)]
            if not hot_indices:
                hot_indices = [
                    s.index for s in ranked
                    if (s.ml_score >= 55.0 or s.score >= threshold) and s.index < len(sentences)
                ]
            if top_n > 0:
                for s in ranked[:top_n]:
                    if s.index < len(sentences) and s.index not in hot_indices:
                        hot_indices.append(s.index)

        if sentences_per_pass > 0:
            hot_indices = hot_indices[:sentences_per_pass]

        changed: list[int] = []
        applied_tags: list[str] = []
        new_sentences = list(sentences)
        current_ml_doc = proxy_report.ml_document_score

        for s in ranked:
            idx = s.index
            if idx not in hot_indices:
                continue
            sentence = sentences[idx]
            old_ml = s.ml_score
            orig_sentence = orig_map.get(idx, sentence)

            candidates = _collect_sentence_candidates(
                sentence,
                orig_sentence,
                self.lexicon,
                protected,
                doc_score=current_ml_doc,
            )
            if not candidates:
                continue

            best_text, best_tags = min(
                candidates,
                key=lambda c: (
                    _candidate_pick_score(orig_sentence, c[0])
                    + (
                        -4.0
                        if any(t in c[1] for t in ("template", "template_hop", "template_final"))
                        else 0.0
                    ),
                    -len(c[0]),
                ),
            )
            best_ml = _get_onnx_score(best_text)
            old_pick = _zerogpt_pick_score(sentence)
            new_pick = _zerogpt_pick_score(best_text)
            if best_text == sentence:
                continue

            trial = list(new_sentences)
            trial[idx] = best_text
            trial_text = rejoin_manuscript(text, trial)
            if not _sentence_rewrite_allowed(orig_sentence, best_text):
                continue

            trial_proxy = self.proxy.analyze(trial_text)
            trial_ml = next((x.ml_score for x in trial_proxy.sentences if x.index == idx), best_ml)
            ml_doc_improved = trial_proxy.ml_document_score <= current_ml_doc - 0.3
            ml_sent_improved = trial_ml < old_ml - 0.3
            pick_improved = new_pick < old_pick - 1.0
            doc_score_worse = trial_proxy.ml_document_score > current_ml_doc + 2.0
            structural = any(
                t in best_tags
                for t in ("template", "template_final", "template_hop", "rhythm", "parallel", "phrase")
            )
            template_ok = structural and (pick_improved or trial_ml <= old_ml + 3.0)

            if doc_score_worse and not pick_improved:
                continue

            if not ml_sent_improved and not ml_doc_improved and not template_ok and not pick_improved:
                continue
            if trial_ml > old_ml + 8.0 and not pick_improved:
                continue

            new_sentences[idx] = best_text
            changed.append(idx)
            applied_tags.append(f"sentence[{idx}]:{'+'.join(best_tags)}")
            current_ml_doc = min(current_ml_doc, trial_proxy.ml_document_score)

        result_text = rejoin_manuscript(text, new_sentences)
        merged = merge_intro_type2_sentences(result_text)
        if merged != result_text:
            merge_len = len(merged) / max(len(baseline), 1)
            merge_proxy = self.proxy.analyze(merged)
            if (
                merge_len >= 0.78
                and merge_proxy.ml_document_score
                <= self.proxy.analyze(result_text).ml_document_score
            ):
                result_text = merged
                applied_tags.append("merge:intro_type2")
        polished = apply_zerogpt_polish(result_text, baseline)
        polish_len = len(polished) / max(len(baseline), 1)
        if polished != result_text and polish_len >= 0.78:
            result_text = polished
            applied_tags.append("polish:zerogpt")
        return RewriteResult(text=result_text, changed_sentences=changed, applied=applied_tags)

    def rewrite_all_hot(
        self,
        text: str,
        *,
        original: str | None = None,
        hot_threshold: float | None = None,
        top_n: int = 0,
        rewrite_all_sentences: bool = True,
    ) -> RewriteResult:
        """Rewrite hot or all sentences in one pass."""
        return self.rewrite_pass(
            text,
            sentences_per_pass=0,
            original=original,
            hot_threshold=hot_threshold,
            top_n=top_n,
            rewrite_all_sentences=rewrite_all_sentences,
        )
