from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from humanizer.analyzers.detector import DetectionReport, detect_ai_likelihood
from humanizer.analyzers.external import fetch_external_score
from humanizer.config import AppConfig
from humanizer.rewriters.llm_rewriter import LLMRewriter
from humanizer.validators.fidelity import FidelityReport, validate_fidelity


@dataclass
class IterationResult:
    iteration: int
    text: str
    detection: DetectionReport
    fidelity: FidelityReport | None = None


@dataclass
class PipelineResult:
    original: str
    final: str
    iterations: list[IterationResult] = field(default_factory=list)
    success: bool = False
    initial_score: float = 0.0
    final_score: float = 0.0


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if end < len(text):
            # break at paragraph or sentence boundary
            break_at = max(chunk.rfind("\n\n"), chunk.rfind(". "))
            if break_at > size // 2:
                chunk = chunk[: break_at + 1]
                end = start + len(chunk)
        chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else end
    return [c for c in chunks if c]


def _issues_from_detection(report: DetectionReport) -> list[str]:
    issues: list[str] = []
    for hit in report.pattern.hits[:12]:
        issues.append(f"[{hit.pattern_id}] {hit.category}: \"{hit.matched}\"")
    stats = report.statistics
    if stats.burstiness < 35:
        issues.append(f"Low sentence-length variance (burstiness={stats.burstiness:.0f}) — vary sentence lengths")
    if stats.repetition_score > 40:
        issues.append(f"High lexical repetition ({stats.repetition_score:.0f}) — vary word choice")
    if stats.opener_diversity < 50:
        issues.append("Many sentences start the same way — diversify openers")
    return issues


class HumanizerPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._rewriter: LLMRewriter | None = None

    @property
    def rewriter(self) -> LLMRewriter:
        if self._rewriter is None:
            self._rewriter = LLMRewriter(self.config.llm)
        return self._rewriter

    async def _score(self, text: str) -> DetectionReport:
        external = await fetch_external_score(text, self.config.detector)
        return detect_ai_likelihood(
            text,
            external_score=external,
            external_weight=self.config.detector.external_weight,
        )

    def _score_sync(self, text: str) -> DetectionReport:
        return detect_ai_likelihood(text)

    async def run(self, text: str) -> PipelineResult:
        pipe = self.config.pipeline
        result = PipelineResult(original=text, final=text)

        initial = await self._score(text)
        result.initial_score = initial.composite_score

        chunks = _chunk_text(text, pipe.chunk_size, pipe.chunk_overlap)
        current_chunks = list(chunks)

        for iteration in range(1, pipe.max_iterations + 1):
            rewritten_chunks: list[str] = []
            chunk_reports: list[DetectionReport] = []

            for chunk in current_chunks:
                detection = await self._score(chunk)
                chunk_reports.append(detection)

                if detection.composite_score <= pipe.target_ai_score:
                    rewritten_chunks.append(chunk)
                    continue

                issues = _issues_from_detection(detection)
                rewritten = self.rewriter.rewrite(chunk, iteration=iteration, issues=issues)

                fidelity = validate_fidelity(
                    chunk,
                    rewritten,
                    min_similarity=pipe.min_meaning_similarity,
                    preserve_numbers=pipe.preserve_numbers,
                    preserve_citations=pipe.preserve_citations,
                )

                if not fidelity.passed:
                    refined = self.rewriter.refine(rewritten, fidelity.issues)
                    fidelity2 = validate_fidelity(
                        chunk,
                        refined,
                        min_similarity=pipe.min_meaning_similarity,
                        preserve_numbers=pipe.preserve_numbers,
                        preserve_citations=pipe.preserve_citations,
                    )
                    rewritten = refined if fidelity2.passed else chunk
                else:
                    rewritten = rewritten

                rewritten_chunks.append(rewritten)

            combined = "\n\n".join(rewritten_chunks)
            final_detection = await self._score(combined)

            result.iterations.append(
                IterationResult(
                    iteration=iteration,
                    text=combined,
                    detection=final_detection,
                )
            )

            if self.config.output.save_intermediate:
                self._save_run(iteration, combined, final_detection)

            current_chunks = _chunk_text(combined, pipe.chunk_size, pipe.chunk_overlap)
            result.final = combined
            result.final_score = final_detection.composite_score

            if final_detection.composite_score <= pipe.target_ai_score:
                result.success = True
                break

        return result

    def run_sync(self, text: str) -> PipelineResult:
        """Score-only path without LLM (for audit/dry-run)."""
        pipe = self.config.pipeline
        detection = self._score_sync(text)
        result = PipelineResult(
            original=text,
            final=text,
            initial_score=detection.composite_score,
            final_score=detection.composite_score,
            success=detection.composite_score <= pipe.target_ai_score,
        )
        result.iterations.append(IterationResult(iteration=0, text=text, detection=detection))
        return result

    def _save_run(self, iteration: int, text: str, detection: DetectionReport) -> None:
        out_dir = Path(self.config.output.intermediate_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"iter_{iteration}_{ts}.txt"
        meta = out_dir / f"iter_{iteration}_{ts}_score.txt"
        path.write_text(text, encoding="utf-8")
        meta.write_text(
            f"composite={detection.composite_score:.1f}\n"
            f"pattern={detection.details.get('pattern_score', 0):.1f}\n"
            f"statistical={detection.details.get('statistical_score', 0):.1f}\n",
            encoding="utf-8",
        )
