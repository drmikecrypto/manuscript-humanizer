from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class StatisticalReport:
    burstiness: float  # 0-100, higher = more human-like variance
    avg_sentence_length: float
    sentence_length_std: float
    repetition_score: float  # 0-100, higher = more repetitive (AI-like)
    opener_diversity: float  # 0-100, higher = more diverse
    ai_likelihood: float  # 0-100 composite


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _sentence_lengths(sentences: list[str]) -> list[int]:
    return [len(s.split()) for s in sentences]


def _burstiness(lengths: list[int]) -> float:
    if len(lengths) < 2:
        return 50.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 50.0
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std = math.sqrt(variance)
    # Coefficient of variation mapped to 0-100; humans typically > 0.4 CV
    cv = std / mean
    return min(100.0, cv * 120)


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "been",
    "have", "has", "had", "not", "but", "into", "than", "then", "when", "after", "before",
    "during", "which", "their", "they", "them", "also", "both", "each", "such", "using",
    "used", "between", "among", "within", "without", "while", "where", "these", "those",
}


# Domain terms that legitimately repeat in academic manuscripts.
DOMAIN_TERMS = {
    "diabetes", "diabetic", "glucose", "insulin", "extract", "extracts", "serum",
    "rats", "treatment", "control", "group", "groups", "weeks", "levels", "study",
    "results", "induced", "physiological", "blood", "weight", "patient", "patients",
    "clinical", "therapy", "hyperglycemia", "streptozotocin", "glibenclamide",
}


def _repetition_score(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    content_words = [w for w in words if w not in STOPWORDS and w not in DOMAIN_TERMS]
    if len(content_words) < 10:
        return 0.0
    unique_ratio = len(set(content_words)) / len(content_words)
    return max(0.0, min(100.0, (1 - unique_ratio) * 200))


def _opener_diversity(sentences: list[str]) -> float:
    if not sentences:
        return 50.0
    openers = [s.split()[0].lower() if s.split() else "" for s in sentences]
    unique = len(set(openers))
    ratio = unique / len(openers)
    return ratio * 100


def analyze_statistics(text: str) -> StatisticalReport:
    sentences = _sentences(text)
    lengths = _sentence_lengths(sentences)
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    std_len = math.sqrt(sum((x - avg_len) ** 2 for x in lengths) / len(lengths)) if len(lengths) > 1 else 0.0

    burst = _burstiness(lengths)
    rep = _repetition_score(text)
    opener = _opener_diversity(sentences)

    # Structural template density — overlapping segments flag uniform AI Methods/Results prose.
    template_hits = sum(
        len(re.findall(pat, text, flags=re.IGNORECASE))
        for pat in (
            r"\bwere induced with diabetes\b",
            r"\bfor 2 weeks?\b",
            r"\bfor 6 weeks?\b",
            r"\bGroup \d+\b",
            r"\b(?:reducing|increasing|decreasing)\b",
        )
    )
    template_penalty = min(30.0, template_hits * 2.5)

    ai_likelihood = (
        template_penalty * 0.15
        + (100 - burst) * 0.40
        + rep * 0.30
        + (100 - opener) * 0.15
    )
    ai_likelihood = max(0.0, min(100.0, ai_likelihood))

    return StatisticalReport(
        burstiness=burst,
        avg_sentence_length=avg_len,
        sentence_length_std=std_len,
        repetition_score=rep,
        opener_diversity=opener,
        ai_likelihood=ai_likelihood,
    )
