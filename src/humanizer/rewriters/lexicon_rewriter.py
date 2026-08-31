from __future__ import annotations

import re

from humanizer.analyzers.patterns import analyze_patterns
from humanizer.lexicon.service import LexiconService
from humanizer.validators.fidelity import validate_sentence_fidelity

AI_VOCAB = re.compile(
    r"\b(furthermore|moreover|additionally|utilize|leverage|comprehensive|robust|"
    r"significantly|substantially|demonstrate|facilitate|delve|underscores?|"
    r"characterized|various|indicate|suggests?|effective(?:ly)?|comparable|"
    r"complications|treatment|strategies|performed|recorded)\b",
    re.IGNORECASE,
)

_SWAP_COUNTS = {"light": 1, "medium": 2, "strong": 3}


def apply_local_paraphrase(
    sentence: str,
    lexicon: LexiconService,
    protected: set[str],
    *,
    original: str,
    intensity: str = "medium",
    min_pattern_score: float = 12.0,
) -> str:
    """Swap 1-3 non-protected tokens via WordNet synonyms to reduce pattern hits."""
    if analyze_patterns(sentence).score < min_pattern_score:
        return sentence
    max_swaps = _SWAP_COUNTS.get(intensity, 2)
    current = sentence
    baseline_score = analyze_patterns(sentence).score
    swaps_done = 0

    candidates = re.findall(r"\b[a-zA-Z]{4,}\b", sentence)
    # Prefer words that match AI vocab or pattern-heavy terms
    priority = [w for w in candidates if AI_VOCAB.search(w)]
    rest = [w for w in candidates if w not in priority]
    ordered = priority + rest

    for word in ordered:
        if swaps_done >= max_swaps:
            break
        if word in protected or lexicon.is_protected(word):
            continue
        alt = lexicon.suggest_swap(word, rng_seed=hash((sentence, word)) & 0xFFFF)
        if not alt or alt.lower() == word.lower():
            continue
        replaced = lexicon.safe_replace_token(current, word, alt, protected)
        if not replaced:
            continue
        if not validate_sentence_fidelity(original, replaced, min_similarity=0.35).passed:
            continue
        new_score = analyze_patterns(replaced).score
        if new_score <= baseline_score + 5:
            current = replaced
            swaps_done += 1

    return current
