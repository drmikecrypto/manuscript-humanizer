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


def _repetition_score(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    if len(words) < 10:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    # Low unique ratio = repetitive = AI-like
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

    # Low burstiness + high repetition + low opener diversity = AI-like
    ai_likelihood = (
        (100 - burst) * 0.45
        + rep * 0.35
        + (100 - opener) * 0.20
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
