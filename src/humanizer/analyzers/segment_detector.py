from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from humanizer.analyzers.detector import DetectionReport, detect_ai_likelihood
from humanizer.analyzers.model_loader import default_cache_dir, download_model, is_model_available
from humanizer.rewriters.transforms import split_manuscript_sentences

_WORD_RE = re.compile(r"\S+")
_ML_BLEND_WEIGHT = 0.35  # ONNX saturates high; legacy carries discriminative signal


@dataclass
class SpanScore:
    start: int
    end: int
    text: str
    ai_probability: float
    sentence_index: int


@dataclass
class SentenceScore:
    index: int
    text: str
    ai_probability: float
    start: int
    end: int


@dataclass
class SegmentDetectionReport:
    document_score: float
    spans: list[SpanScore] = field(default_factory=list)
    hot_sentences: list[int] = field(default_factory=list)
    sentence_scores: list[SentenceScore] = field(default_factory=list)
    engine: str = "onnx"
    legacy_report: DetectionReport | None = None

    @property
    def hot_spans(self) -> list[SpanScore]:
        return [s for s in self.spans if s.ai_probability >= 60.0]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in split_manuscript_sentences(text):
        idx = text.find(sentence, cursor)
        if idx < 0:
            idx = cursor
        start = idx
        end = start + len(sentence)
        spans.append((start, end, sentence))
        cursor = end
    if not spans and text.strip():
        spans.append((0, len(text), text.strip()))
    return spans


def _blend_score(ml: float, legacy: float) -> float:
    return _ML_BLEND_WEIGHT * ml + (1.0 - _ML_BLEND_WEIGHT) * legacy


def _windows(text: str, window_words: int, overlap: float) -> list[tuple[int, int, str]]:
    words = list(_WORD_RE.finditer(text))
    if not words:
        return []
    if len(words) <= window_words:
        return [(0, len(text), text.strip())]

    step = max(1, int(window_words * (1 - overlap)))
    result: list[tuple[int, int, str]] = []
    for i in range(0, len(words), step):
        chunk = words[i : i + window_words]
        if not chunk:
            break
        start = chunk[0].start()
        end = chunk[-1].end()
        snippet = text[start:end].strip()
        if snippet:
            result.append((start, end, snippet))
        if i + window_words >= len(words):
            break
    return result


class _OnnxClassifier:
    def __init__(self, cache_dir: Path | None = None) -> None:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        import onnxruntime as ort  # type: ignore[import-untyped]

        root = cache_dir or default_cache_dir()
        if not is_model_available(root):
            download_model(root)

        tok_path = root / "tokenizer.json"
        model_path = root / "model_quantized.onnx"
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self.tokenizer.enable_truncation(max_length=512)
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict_proba(self, text: str) -> float:
        enc = self.tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        logits = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        probs = _softmax(logits)
        return float(probs[0][1] * 100.0)


class SegmentDetector:
    """Windowed ML detector with sentence heatmap."""

    def __init__(
        self,
        *,
        engine: str = "onnx",
        span_threshold: float = 60.0,
        window_words: int = 150,
        window_overlap: float = 0.5,
        calibration_offset: float = 0.0,
        cache_dir: Path | None = None,
    ) -> None:
        self.engine = engine
        self.span_threshold = span_threshold
        self.window_words = window_words
        self.window_overlap = window_overlap
        self.calibration_offset = calibration_offset
        self.cache_dir = cache_dir
        self._onnx: _OnnxClassifier | None = None

    def _get_onnx(self) -> _OnnxClassifier:
        if self._onnx is None:
            self._onnx = _OnnxClassifier(self.cache_dir)
        return self._onnx

    def _score_text_ml(self, text: str) -> float:
        if not text.strip():
            return 0.0
        try:
            return self._get_onnx().predict_proba(text)
        except Exception:
            return detect_ai_likelihood(text).composite_score

    def _score_text_legacy(self, text: str) -> float:
        return detect_ai_likelihood(text).composite_score

    def _combined_score(self, text: str) -> float:
        legacy = self._score_text_legacy(text)
        if self.engine == "onnx":
            try:
                ml = self._score_text_ml(text)
                score = _blend_score(ml, legacy)
            except Exception:
                score = legacy
        else:
            score = legacy
        return max(0.0, min(100.0, score + self.calibration_offset))

    def score_span(self, text: str) -> float:
        return self._combined_score(text)

    def analyze(self, text: str) -> SegmentDetectionReport:
        legacy = detect_ai_likelihood(text)
        sent_spans = _sentence_spans(text)
        windows = _windows(text, self.window_words, self.window_overlap)

        window_scores: list[tuple[int, int, str, float]] = []
        for start, end, snippet in windows:
            score = self._combined_score(snippet)
            window_scores.append((start, end, snippet, score))

        sentence_ai: list[float] = []
        sentence_score_objs: list[SentenceScore] = []
        for idx, (s_start, s_end, sentence) in enumerate(sent_spans):
            covering = [sc for ws, we, _, sc in window_scores if not (we <= s_start or ws >= s_end)]
            legacy_sent = self._score_text_legacy(sentence)
            if covering:
                ai = max(covering)
            else:
                ai = self._combined_score(sentence)
            # Prefer legacy ranking when sentence is short
            if len(sentence.split()) < 20:
                ai = max(ai, legacy_sent)
            sentence_ai.append(ai)
            sentence_score_objs.append(
                SentenceScore(index=idx, text=sentence, ai_probability=ai, start=s_start, end=s_end)
            )

        span_scores: list[SpanScore] = []
        for idx, (s_start, s_end, sentence) in enumerate(sent_spans):
            span_scores.append(
                SpanScore(
                    start=s_start,
                    end=s_end,
                    text=sentence,
                    ai_probability=sentence_ai[idx],
                    sentence_index=idx,
                )
            )
        span_scores.sort(key=lambda s: s.ai_probability, reverse=True)

        if window_scores:
            top = sorted((sc for _, _, _, sc in window_scores), reverse=True)[:3]
            doc_score = sum(top) / len(top)
        elif sentence_ai:
            top = sorted(sentence_ai, reverse=True)[:3]
            doc_score = sum(top) / len(top)
        else:
            doc_score = self._combined_score(text)

        if self.engine == "onnx":
            doc_score = _blend_score(
                sum(sorted((self._score_text_ml(w[2]) for w in window_scores), reverse=True)[:3])
                / max(1, min(3, len(window_scores))),
                legacy.composite_score,
            )

        hot = [s.index for s in sentence_score_objs if s.ai_probability >= self.span_threshold]

        engine_label = self.engine
        if self.engine == "onnx" and not is_model_available():
            engine_label = "legacy-fallback"

        return SegmentDetectionReport(
            document_score=max(0.0, min(100.0, doc_score + self.calibration_offset)),
            spans=span_scores,
            hot_sentences=hot,
            sentence_scores=sentence_score_objs,
            engine=engine_label,
            legacy_report=legacy,
        )

    def rescore_sentences(self, text: str, sentence_indices: list[int]) -> SegmentDetectionReport:
        report = self.analyze(text)
        if sentence_indices:
            hot = [i for i in report.hot_sentences if i in sentence_indices]
            if hot:
                report.hot_sentences = hot
        return report
