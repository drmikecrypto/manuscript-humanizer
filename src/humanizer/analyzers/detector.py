from __future__ import annotations

import re
from dataclasses import dataclass, field

from humanizer.analyzers.patterns import PatternReport, analyze_patterns
from humanizer.analyzers.statistics import StatisticalReport, analyze_statistics


@dataclass
class DetectionReport:
    pattern: PatternReport
    statistics: StatisticalReport
    composite_score: float  # 0-100, higher = more AI-like
    external_score: float | None = None
    details: dict[str, float] = field(default_factory=dict)


def compute_composite(
    pattern_score: float,
    stat_score: float,
    external: float | None = None,
    external_weight: float = 0.4,
) -> float:
    internal = pattern_score * 0.55 + stat_score * 0.45
    if external is not None:
        return internal * (1 - external_weight) + external * external_weight
    return internal


def detect_ai_likelihood(
    text: str,
    external_score: float | None = None,
    external_weight: float = 0.4,
) -> DetectionReport:
    pattern = analyze_patterns(text)
    stats = analyze_statistics(text)
    composite = compute_composite(
        pattern.score,
        stats.ai_likelihood,
        external_score,
        external_weight,
    )
    return DetectionReport(
        pattern=pattern,
        statistics=stats,
        composite_score=composite,
        external_score=external_score,
        details={
            "pattern_score": pattern.score,
            "statistical_score": stats.ai_likelihood,
            "burstiness": stats.burstiness,
            "repetition": stats.repetition_score,
        },
    )


# Citation and number patterns for preservation checks
CITATION_RE = re.compile(
    r"\[[\d,\s\-–]+\]|\([\w\s]+,?\s*\d{4}[a-z]?\)|\bdoi:\s*\S+",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:%|×|x)?\b|\b\d{4}\b",
)
