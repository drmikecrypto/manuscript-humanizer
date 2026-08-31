from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

from humanizer.analyzers.detector import detect_ai_likelihood
from humanizer.analyzers.patterns import analyze_patterns
from humanizer.rewriters.transforms import split_manuscript_sentences

PARALLEL_TRIPLE_RE = re.compile(
    r"\b(reducing|lowering|increasing|raising|decreasing|reduced|increased|decreased|"
    r"fell|rose|dropped|lowered|raised)\b"
    r".*,\s*"
    r"\b(reducing|lowering|increasing|raising|decreasing|reduced|increased|decreased|"
    r"fell|rose|dropped|lowered|raised)\b"
    r".*,?\s*and\s*"
    r"\b(reducing|lowering|increasing|raising|decreasing|reduced|increased|decreased|"
    r"fell|rose|dropped|lowered|raised)\b",
    re.IGNORECASE,
)

_onnx_classifier: object | None = None


def _get_onnx_score(sentence: str) -> float:
    global _onnx_classifier
    try:
        if _onnx_classifier is None:
            from humanizer.analyzers.segment_detector import _OnnxClassifier

            _onnx_classifier = _OnnxClassifier()
        return float(_onnx_classifier.predict_proba(sentence))  # type: ignore[union-attr]
    except Exception:
        return detect_ai_likelihood(sentence).composite_score


@dataclass
class SentenceProxyScore:
    index: int
    text: str
    score: float
    template_score: float
    ml_score: float
    legacy_score: float
    parallel_score: float
    opener_penalty: float
    length_penalty: float


@dataclass
class ZeroGPTProxyReport:
    document_score: float
    ml_document_score: float = 0.0
    sentences: list[SentenceProxyScore] = field(default_factory=list)
    hot_sentences: list[int] = field(default_factory=list)

    @property
    def highlighted(self) -> list[SentenceProxyScore]:
        return [s for s in self.sentences if s.index in self.hot_sentences]


class ZeroGPTProxyScorer:
    """Offline proxy aligned with ZeroGPT-style sentence highlighting."""

    def __init__(
        self,
        *,
        hot_threshold: float = 45.0,
        use_onnx: bool = True,
        calibration_offset: float = 8.0,
        top_n_sentences: int = 5,
    ) -> None:
        self.hot_threshold = hot_threshold
        self.use_onnx = use_onnx
        self.calibration_offset = calibration_offset
        self.top_n_sentences = top_n_sentences

    def score_sentence(
        self,
        sentence: str,
        *,
        opener_counts: dict[str, int],
        doc_median_words: float,
    ) -> SentenceProxyScore:
        template = min(100.0, analyze_patterns(sentence).score * 1.2)
        legacy = detect_ai_likelihood(sentence).composite_score
        ml = _get_onnx_score(sentence) if self.use_onnx else legacy
        parallel = 80.0 if PARALLEL_TRIPLE_RE.search(sentence) else 0.0

        opener_match = re.match(r"^(\w+)", sentence.strip())
        opener_penalty = 0.0
        if opener_match:
            count = opener_counts.get(opener_match.group(1).lower(), 0)
            if count >= 2:
                opener_penalty = min(100.0, 30.0 + count * 15.0)

        word_count = len(sentence.split())
        length_penalty = 0.0
        if doc_median_words > 0 and abs(word_count - doc_median_words) <= 3:
            length_penalty = 25.0

        combined = (
            template * 0.15
            + ml * 0.50
            + legacy * 0.10
            + parallel * 0.15
            + opener_penalty * 0.05
            + length_penalty * 0.05
        )
        return SentenceProxyScore(
            index=-1,
            text=sentence,
            score=min(100.0, combined),
            template_score=template,
            ml_score=ml,
            legacy_score=legacy,
            parallel_score=parallel,
            opener_penalty=opener_penalty,
            length_penalty=length_penalty,
        )

    def analyze(self, text: str) -> ZeroGPTProxyReport:
        sentences = split_manuscript_sentences(text)
        if not sentences:
            return ZeroGPTProxyReport(document_score=0.0)

        opener_counts: dict[str, int] = {}
        for s in sentences:
            m = re.match(r"^(\w+)", s.strip())
            if m:
                key = m.group(1).lower()
                opener_counts[key] = opener_counts.get(key, 0) + 1

        word_counts = [len(s.split()) for s in sentences]
        doc_median = float(median(word_counts)) if word_counts else 0.0

        scored: list[SentenceProxyScore] = []
        for idx, sentence in enumerate(sentences):
            s = self.score_sentence(sentence, opener_counts=opener_counts, doc_median_words=doc_median)
            s.index = idx
            scored.append(s)

        top_n = min(self.top_n_sentences, len(scored))
        top = sorted((s.score for s in scored), reverse=True)[:top_n]
        doc_score = sum(top) / len(top) if top else 0.0
        top_ml = sorted((s.ml_score for s in scored), reverse=True)[:top_n]
        ml_doc_score = sum(top_ml) / len(top_ml) if top_ml else 0.0
        hot = [s.index for s in scored if s.ml_score >= 55.0 or s.score >= self.hot_threshold]

        return ZeroGPTProxyReport(
            document_score=doc_score,
            ml_document_score=ml_doc_score,
            sentences=scored,
            hot_sentences=hot,
        )

    def rescore_document(self, text: str) -> float:
        return self.analyze(text).document_score

    def expected_zerogpt_score(self, text: str) -> float:
        """Rough mapping from ML proxy to expected ZeroGPT % (fixture-calibrated)."""
        report = self.analyze(text)
        return min(100.0, report.ml_document_score + self.calibration_offset * 0.5)
