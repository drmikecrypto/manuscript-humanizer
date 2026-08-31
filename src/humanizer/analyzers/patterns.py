from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PatternHit:
    pattern_id: str
    category: str
    severity: int  # 1=critical, 2=moderate, 3=minor
    matched: str
    span: tuple[int, int]


@dataclass
class PatternReport:
    hits: list[PatternHit] = field(default_factory=list)
    score: float = 0.0

    @property
    def hit_count(self) -> int:
        return len(self.hits)


# Academic + general AI tells. Severity 1 = strong detector signal.
AI_PATTERNS: list[tuple[str, str, int, str]] = [
    # (id, category, severity, regex)
    ("P01", "stock_phrase", 1, r"\b(in recent years|over the past (?:few )?years|in today's (?:world|landscape))\b"),
    ("P02", "stock_phrase", 1, r"\b(it is (?:important|worth|crucial) to note that)\b"),
    ("P03", "stock_phrase", 1, r"\b(plays a (?:crucial|pivotal|vital|key) role)\b"),
    ("P04", "stock_phrase", 1, r"\b(paves the way for|sheds light on|serves as a testament)\b"),
    ("P05", "stock_phrase", 1, r"\b(to the best of our knowledge)\b"),
    ("P06", "stock_phrase", 1, r"\b(extensive experiments (?:demonstrate|show|confirm))\b"),
    ("P07", "ai_vocab", 1, r"\b(delve|delving|underscores?|tapestry|landscape|intricac(?:y|ies)|multifaceted)\b"),
    ("P08", "ai_vocab", 1, r"\b(furthermore|moreover|additionally|in addition)\b"),
    ("P09", "ai_vocab", 2, r"\b(comprehensive|robust|novel framework|cutting-edge|state-of-the-art)\b"),
    ("P10", "ai_vocab", 2, r"\b(leverage|utilize|facilitate|enhance|foster)\b"),
    ("P11", "structure", 1, r"\b(firstly|secondly|thirdly|first, .{0,80} second, .{0,80} third,)\b"),
    ("P12", "structure", 2, r"—"),  # em-dash overuse checked by frequency
    ("P13", "hedging", 2, r"\b(it (?:can|could|may|might) be (?:argued|seen|noted) that)\b"),
    ("P14", "hedging", 2, r"\b(is expected to|has the potential to|offers promising)\b"),
    ("P15", "inflation", 1, r"\b(revolutioniz(?:e|ing)|transformative paradigm|groundbreaking)\b"),
    ("P16", "inflation", 2, r"\b(significant(?:ly)?|substantial(?:ly)?|remarkable(?:ly)?)\b"),
    ("P17", "passive_stack", 2, r"\b(was (?:conducted|performed|carried out) by)\b"),
    ("P18", "conclusion", 1, r"\b(in conclusion|to summarize|in summary|overall,)\b"),
    ("P19", "rule_of_three", 2, r"\b(\w+ly, \w+ly, (?:and )?\w+ly)\b"),
    ("P20", "vague_source", 2, r"\b(researchers (?:have )?(?:found|shown|demonstrated) that)\b"),
    ("P21", "academic", 1, r"\b(this (?:paper|study|work) (?:aims to|seeks to|attempts to))\b"),
    ("P22", "academic", 2, r"\b(our (?:proposed )?method|the proposed (?:method|approach|framework))\b"),
    ("P23", "filler", 3, r"\b(it is clear that|needless to say|as a matter of fact)\b"),
    ("P24", "filler", 3, r"\b(in order to)\b"),
    # Academic manuscript tells (ZeroGPT-aligned)
    ("P25", "academic", 1, r"\bthis study suggests\b"),
    ("P26", "academic", 1, r"\bthe results indicate\b"),
    ("P27", "academic", 1, r"\bclinical studies on humans are necessary\b"),
    ("P28", "academic", 1, r"\bsuch studies could help scientists and physicians\b"),
    ("P29", "academic", 1, r"\bwere induced with diabetes\b"),
    ("P30", "academic", 2, r"\b(reducing|lowering).{0,40}(increasing|raising).{0,40}(decreasing|lowering)\b"),
    ("P31", "academic", 2, r"\bcharacterized by impaired glucose metabolism\b"),
    ("P32", "academic", 2, r"\bdevelop the best therapeutic strategies\b"),
    ("P33", "academic", 1, r"\bwere performed at various stages\b"),
    ("P34", "academic", 1, r"\bcan effectively reduce\b"),
    ("P35", "academic", 1, r"\bcomparable to the (?:commonly|usually|often) used\b"),
    ("P36", "academic", 1, r"\bmay serve as complementary\b"),
    ("P37", "academic", 1, r"\b(besides|also|further), the efficacy of these\b"),
    ("P38", "academic", 1, r"\bwere allocated to \d+ groups\b"),
    ("P39", "academic", 1, r"\breceived physiological serum for 2 weeks\b"),
    ("P40", "academic", 1, r"\bwere (?:done|performed) at (?:multiple|different|several|various) stages\b"),
    ("P41", "academic", 2, r"\bmeasurements of blood glucose, insulin levels\b"),
    ("P42", "academic", 1, r"\b(chiefly|primarily) divided into two main types\b"),
    ("P43", "academic", 2, r"\bdescribed by impaired glucose metabolism\b"),
    ("P44", "academic", 1, r"\bthe results (?:indicated|suggested) that nettle\b"),
    ("P45", "academic", 1, r"\bcandidate adjunct options herbal\b"),
]

_COMPILED = [(pid, cat, sev, re.compile(pat, re.IGNORECASE)) for pid, cat, sev, pat in AI_PATTERNS]


def analyze_patterns(text: str) -> PatternReport:
    hits: list[PatternHit] = []
    for pid, cat, sev, regex in _COMPILED:
        for match in regex.finditer(text):
            hits.append(
                PatternHit(
                    pattern_id=pid,
                    category=cat,
                    severity=sev,
                    matched=match.group(0),
                    span=(match.start(), match.end()),
                )
            )

    # Em-dash frequency penalty
    em_count = text.count("—") + text.count(" - ")
    if em_count >= 3:
        hits.append(PatternHit("P12_freq", "structure", 2, f"{em_count} em-dashes", (0, 0)))

    # Score: weighted by severity and density per 1000 words
    words = max(len(text.split()), 1)
    density = len(hits) / (words / 1000)
    severity_sum = sum(h.severity for h in hits)
    score = min(100.0, severity_sum * 2.5 + density * 8)
    return PatternReport(hits=hits, score=score)
