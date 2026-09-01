from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re

from humanizer.analyzers.detector import DetectionReport, detect_ai_likelihood
from humanizer.analyzers.external import fetch_external_score
from humanizer.analyzers.segment_detector import SegmentDetectionReport, SegmentDetector
from humanizer.analyzers.zerogpt_proxy import ZeroGPTProxyReport, ZeroGPTProxyScorer
from humanizer.config import AppConfig
from humanizer.lexicon.service import LexiconService
from humanizer.rewriters.bootstrap import apply_bootstrap_humanize, is_bootstrap_manuscript
from humanizer.rewriters.outbound import (
    apply_outbound_humanize,
    apply_outbound_iterative,
    set_outbound_aggression,
)
from humanizer.rewriters.local_rewriter import LocalRewriter
from humanizer.rewriters.llm_rewriter import LLMRewriter
from humanizer.rewriters.targeted_rewriter import TargetedRewriter
from humanizer.rewriters.section_rewrites import apply_zerogpt_polish
from humanizer.validators.fidelity import (
    FidelityReport,
    build_manuscript_quality_report,
    build_quality_report,
    validate_document_output,
    validate_fidelity,
)


SHORT_FORM_MARKERS = (
    "conference abstract",
    "letter of recommendation",
    "dear members of the selection",
    "to whom it may concern",
    "technical writing sample",
    "cover letter",
)


def _is_short_form_document(text: str) -> bool:
    """Letters and 1-page abstracts — template warm-up beats segment ML passes."""
    lower = text.lower()
    if any(m in lower for m in SHORT_FORM_MARKERS):
        return len(text) < 12000
    return len(text) < 2800


def _maybe_apply_burstiness_pass(text: str, baseline: str) -> str:
    """Apply burstiness adjustment when sentence-length CV is uniformly low."""
    from statistics import mean, stdev

    from humanizer.rewriters.transforms import (
        adjust_burstiness,
        is_section_header,
        rejoin_manuscript,
        split_manuscript_sentences,
    )

    sentences = split_manuscript_sentences(text)
    if len(sentences) < 4:
        return text
    lengths = [len(s.split()) for s in sentences if s.strip() and not is_section_header(s)]
    if len(lengths) < 4:
        return text
    avg = mean(lengths)
    if avg <= 0:
        return text
    cv = stdev(lengths) / avg
    if cv >= 0.35:
        return text
    adjusted = adjust_burstiness(sentences, target_low=True)
    candidate = rejoin_manuscript(text, adjusted)
    if build_manuscript_quality_report(baseline, candidate).passed:
        return candidate
    return text


@dataclass
class IterationResult:
    iteration: int
    text: str
    detection: DetectionReport | None = None
    segment_detection: SegmentDetectionReport | None = None
    zerogpt_report: ZeroGPTProxyReport | None = None
    fidelity: FidelityReport | None = None
    changed_sentences: list[int] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    original: str
    final: str
    iterations: list[IterationResult] = field(default_factory=list)
    success: bool = False
    initial_score: float = 0.0
    final_score: float = 0.0
    segment_report: SegmentDetectionReport | None = None
    zerogpt_report: ZeroGPTProxyReport | None = None
    quality_report: FidelityReport | None = None


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if end < len(text):
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
        issues.append(f"Low sentence-length variance (burstiness={stats.burstiness:.0f})")
    if stats.repetition_score > 40:
        issues.append(f"High lexical repetition ({stats.repetition_score:.0f})")
    if stats.opener_diversity < 50:
        issues.append("Many sentences start the same way")
    return issues


def _structural_issues(text: str) -> list[str]:
    issues: list[str] = []
    if len(re.findall(r"\bwere induced with diabetes\b", text, flags=re.IGNORECASE)) >= 2:
        issues.append("Repeated induction protocol template")
    if re.search(
        r"\b(reducing|lowering).{0,50}, (increasing|raising).{0,50}, and (decreasing|lowering)",
        text,
        flags=re.IGNORECASE,
    ):
        issues.append("Parallel reduce/increase/decrease clause")
    if re.search(r"\bAdditionally,\b", text, flags=re.IGNORECASE):
        issues.append("Transition stack")
    return issues


class HumanizerPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._rewriter: LocalRewriter | LLMRewriter | TargetedRewriter | None = None
        self._detector: SegmentDetector | None = None
        self._proxy: ZeroGPTProxyScorer | None = None
        self._lexicon: LexiconService | None = None

    @property
    def lexicon(self) -> LexiconService:
        if self._lexicon is None:
            self._lexicon = LexiconService(
                domains=self.config.lexicon.domains,
                protect_domain_terms=self.config.lexicon.protect_domain_terms,
            )
        return self._lexicon

    @property
    def segment_detector(self) -> SegmentDetector:
        if self._detector is None:
            det = self.config.detector
            self._detector = SegmentDetector(
                engine=det.engine,
                span_threshold=det.span_threshold,
                window_words=det.window_words,
                window_overlap=det.window_overlap,
                calibration_offset=det.calibration_offset,
            )
        return self._detector

    @property
    def proxy_scorer(self) -> ZeroGPTProxyScorer:
        if self._proxy is None:
            self._proxy = ZeroGPTProxyScorer(
                hot_threshold=self.config.detector.span_threshold,
                use_onnx=self.config.detector.engine == "onnx",
                calibration_offset=self.config.detector.calibration_offset,
            )
        return self._proxy

    @property
    def rewriter(self) -> LocalRewriter | LLMRewriter | TargetedRewriter:
        if self._rewriter is None:
            engine = self.config.pipeline.engine
            if engine == "llm":
                self._rewriter = LLMRewriter(self.config.llm)
            elif engine == "local":
                self._rewriter = LocalRewriter(self.config.pipeline.min_meaning_similarity)
            else:
                self._rewriter = TargetedRewriter(
                    lexicon=self.lexicon,
                    min_similarity=self.config.pipeline.min_meaning_similarity,
                    proxy_scorer=self.proxy_scorer,
                )
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

    def _proxy_score(self, text: str) -> ZeroGPTProxyReport:
        return self.proxy_scorer.analyze(text)

    def _segment_score(self, text: str) -> SegmentDetectionReport:
        return self.segment_detector.analyze(text)

    async def _run_segment(self, text: str) -> PipelineResult:
        pipe = self.config.pipeline
        set_outbound_aggression(pipe.aggression, allow_tone_down=pipe.allow_tone_down)
        result = PipelineResult(original=text, final=text)

        initial = self._proxy_score(text)
        result.initial_score = initial.document_score
        result.zerogpt_report = initial
        result.segment_report = self._segment_score(text)

        current = text
        baseline = text
        is_bootstrap = is_bootstrap_manuscript(text)
        is_short = _is_short_form_document(baseline) and not is_bootstrap

        # Quality-first short-form: stricter overlap/length than detector-tuned 0.40/0.55
        if is_short:
            warm_min_sim = max(pipe.min_meaning_similarity, 0.72)
            warm_min_len = 0.85
        elif is_bootstrap:
            # Bootstrap templates rewrite many AI-stock sentences; gate on numbers/length.
            warm_min_sim = 0.45
            warm_min_len = 0.70
        else:
            warm_min_sim = 0.72
            warm_min_len = 0.78

        ms_qc_fn = (
            (lambda o, r: build_manuscript_quality_report(o, r, min_similarity=0.45, min_length_ratio=0.70))
            if is_bootstrap
            else build_manuscript_quality_report
        )
        best_passing_text = baseline
        best_passing_doc = initial.document_score

        # Warm-up: outbound/bootstrap template pass
        if is_bootstrap:
            warmed = apply_bootstrap_humanize(text)
            warmup_tag = "bootstrap:warmup"
        else:
            outbound_result = apply_outbound_iterative(text, text, max_rounds=8)
            warmed = outbound_result.text
            warmup_tag = "outbound:warmup"

        if warmed.strip() != text.strip():
            fidelity = validate_document_output(
                text,
                warmed,
                min_similarity=warm_min_sim,
                min_length_ratio=warm_min_len,
                max_length_ratio=1.15 if is_short else 1.5,
                preserve_numbers=pipe.preserve_numbers,
                preserve_citations=pipe.preserve_citations,
                allow_tone_down=pipe.allow_tone_down,
                require_content_units=is_short,
                reject_invented_numbers=is_short,
            )
            if fidelity.passed:
                current = warmed
                warm_proxy = self._proxy_score(current)
                if ms_qc_fn(baseline, current).passed:
                    best_passing_text = current
                    best_passing_doc = warm_proxy.document_score
                result.iterations.append(
                    IterationResult(
                        iteration=0,
                        text=current,
                        zerogpt_report=warm_proxy,
                        detection=detect_ai_likelihood(current),
                        segment_detection=self._segment_score(current),
                        fidelity=fidelity,
                        applied=[warmup_tag],
                    )
                )
            # else: keep original — never ship a damaged letter/abstract

        best_text = current
        best_proxy = self._proxy_score(best_text)
        best_doc = best_proxy.document_score
        if is_short:
            result.quality_report = build_quality_report(baseline, best_text)
        else:
            result.quality_report = validate_document_output(
                baseline,
                best_text,
                min_similarity=0.45 if is_bootstrap else pipe.min_meaning_similarity,
                min_length_ratio=0.70 if is_bootstrap else 0.78,
                max_length_ratio=1.5,
                preserve_numbers=pipe.preserve_numbers,
                preserve_citations=pipe.preserve_citations,
            )

        if is_short:
            result.final = best_text
            result.final_score = best_proxy.document_score
            result.zerogpt_report = best_proxy
            quality_ok = result.quality_report.passed if result.quality_report else False
            result.success = best_text.strip() != baseline.strip() and quality_ok
            return result

        min_length_ratio = 0.78

        assert isinstance(self.rewriter, TargetedRewriter)

        for pass_num in range(1, pipe.max_passes + 1):
            proxy = self._proxy_score(current)

            if proxy.document_score <= pipe.target_ai_score:
                result.success = True
                result.final = current
                result.final_score = proxy.document_score
                result.zerogpt_report = proxy
                result.iterations.append(
                    IterationResult(
                        iteration=pass_num,
                        text=current,
                        zerogpt_report=proxy,
                        detection=detect_ai_likelihood(current),
                        segment_detection=self._segment_score(current),
                    )
                )
                break

            hot_top_n = 0 if pipe.rewrite_all_sentences else 8
            rewrite = self.rewriter.rewrite_all_hot(
                current,
                original=baseline,
                hot_threshold=self.config.detector.span_threshold,
                top_n=hot_top_n,
                rewrite_all_sentences=pipe.rewrite_all_sentences,
            )

            if rewrite.text.strip() == current.strip():
                result.iterations.append(
                    IterationResult(
                        iteration=pass_num,
                        text=current,
                        zerogpt_report=proxy,
                        detection=detect_ai_likelihood(current),
                        segment_detection=self._segment_score(current),
                    )
                )
                break

            fidelity = validate_fidelity(
                current,
                rewrite.text,
                min_similarity=0.88,
                preserve_numbers=pipe.preserve_numbers,
                preserve_citations=pipe.preserve_citations,
            )
            if not fidelity.passed:
                doc_fidelity = validate_document_output(
                    text,
                    rewrite.text,
                    min_similarity=0.45,
                    preserve_numbers=pipe.preserve_numbers,
                    preserve_citations=pipe.preserve_citations,
                )
                if not doc_fidelity.passed:
                    numbers_only = validate_fidelity(
                        text,
                        rewrite.text,
                        min_similarity=0.0,
                        preserve_numbers=pipe.preserve_numbers,
                        preserve_citations=pipe.preserve_citations,
                    )
                    if numbers_only.missing_numbers or numbers_only.missing_citations:
                        result.iterations.append(
                            IterationResult(
                                iteration=pass_num,
                                text=current,
                                zerogpt_report=proxy,
                                detection=detect_ai_likelihood(current),
                                fidelity=numbers_only,
                            )
                        )
                        break
                else:
                    fidelity = doc_fidelity

            length_ratio = len(rewrite.text) / max(len(baseline), 1)
            if length_ratio < min_length_ratio:
                result.iterations.append(
                    IterationResult(
                        iteration=pass_num,
                        text=current,
                        zerogpt_report=proxy,
                        detection=detect_ai_likelihood(current),
                        fidelity=fidelity,
                        changed_sentences=rewrite.changed_sentences,
                        applied=rewrite.applied,
                    )
                )
                break

            trial_proxy = self._proxy_score(rewrite.text)
            if trial_proxy.document_score > proxy.document_score + 1.5:
                result.iterations.append(
                    IterationResult(
                        iteration=pass_num,
                        text=current,
                        zerogpt_report=proxy,
                        detection=detect_ai_likelihood(current),
                        fidelity=fidelity,
                        changed_sentences=rewrite.changed_sentences,
                        applied=rewrite.applied + ["rejected:worse_score"],
                    )
                )
                break

            current = rewrite.text
            new_proxy = trial_proxy
            if new_proxy.document_score < best_doc:
                best_text = current
                best_doc = new_proxy.document_score
                best_proxy = new_proxy
            ms_qc = ms_qc_fn(baseline, current)
            if ms_qc.passed and new_proxy.document_score < best_passing_doc:
                best_passing_text = current
                best_passing_doc = new_proxy.document_score

            result.iterations.append(
                IterationResult(
                    iteration=pass_num,
                    text=current,
                    zerogpt_report=new_proxy,
                    detection=detect_ai_likelihood(current),
                    segment_detection=self._segment_score(current),
                    fidelity=fidelity,
                    changed_sentences=rewrite.changed_sentences,
                    applied=rewrite.applied,
                )
            )

            if self.config.output.save_intermediate:
                self._save_run(pass_num, current, new_proxy, rewrite.applied)

            result.final = current
            result.final_score = new_proxy.document_score
            result.zerogpt_report = new_proxy
            result.segment_report = self._segment_score(current)

            if new_proxy.document_score <= pipe.target_ai_score:
                result.success = True
                break

        if not result.iterations:
            result.iterations.append(
                IterationResult(
                    iteration=0,
                    text=text,
                    zerogpt_report=initial,
                    detection=detect_ai_likelihood(text),
                    segment_detection=result.segment_report,
                )
            )

        result.final = _maybe_apply_burstiness_pass(
            best_text if best_doc <= self._proxy_score(current).document_score else current,
            baseline,
        )
        if not ms_qc_fn(baseline, result.final).passed:
            if ms_qc_fn(baseline, best_passing_text).passed:
                result.final = best_passing_text
            elif ms_qc_fn(baseline, best_text).passed:
                result.final = best_text

        result.zerogpt_report = self._proxy_score(result.final)
        result.final_score = result.zerogpt_report.document_score
        result.quality_report = ms_qc_fn(baseline, result.final)
        result.success = (
            result.final_score <= pipe.target_ai_score and result.quality_report.passed
        )

        return result

    async def _run_legacy(self, text: str) -> PipelineResult:
        pipe = self.config.pipeline
        result = PipelineResult(original=text, final=text)

        initial = await self._score(text)
        result.initial_score = initial.composite_score

        chunks = _chunk_text(text, pipe.chunk_size, pipe.chunk_overlap)
        current_chunks = list(chunks)

        for iteration in range(1, pipe.max_iterations + 1):
            rewritten_chunks: list[str] = []

            for chunk in current_chunks:
                detection = await self._score(chunk)

                if detection.composite_score <= pipe.target_ai_score:
                    rewritten_chunks.append(chunk)
                    continue

                issues = _issues_from_detection(detection) + _structural_issues(chunk)
                assert isinstance(self.rewriter, (LocalRewriter, LLMRewriter))
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

                rewritten_chunks.append(rewritten)

            combined = "\n\n".join(rewritten_chunks)
            final_detection = await self._score(combined)

            result.iterations.append(
                IterationResult(iteration=iteration, text=combined, detection=final_detection)
            )

            if self.config.output.save_intermediate:
                proxy = self._proxy_score(combined)
                self._save_run(iteration, combined, proxy, [])

            current_chunks = _chunk_text(combined, pipe.chunk_size, pipe.chunk_overlap)
            result.final = combined
            result.final_score = final_detection.composite_score

            if final_detection.composite_score <= pipe.target_ai_score:
                result.success = True
                break

        return result

    async def run(self, text: str) -> PipelineResult:
        if self.config.pipeline.engine == "segment":
            return await self._run_segment(text)
        return await self._run_legacy(text)

    def run_sync(self, text: str) -> PipelineResult:
        pipe = self.config.pipeline
        if self.config.pipeline.engine == "segment":
            proxy = self._proxy_score(text)
            seg = self._segment_score(text)
            result = PipelineResult(
                original=text,
                final=text,
                initial_score=proxy.document_score,
                final_score=proxy.document_score,
                success=proxy.document_score <= pipe.target_ai_score,
                zerogpt_report=proxy,
                segment_report=seg,
            )
            result.iterations.append(
                IterationResult(
                    iteration=0,
                    text=text,
                    zerogpt_report=proxy,
                    detection=detect_ai_likelihood(text),
                    segment_detection=seg,
                )
            )
            return result

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

    def _save_run(
        self,
        iteration: int,
        text: str,
        proxy: ZeroGPTProxyReport,
        applied: list[str],
    ) -> None:
        out_dir = Path(self.config.output.intermediate_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"iter_{iteration}_{ts}.txt"
        meta = out_dir / f"iter_{iteration}_{ts}_score.txt"
        path.write_text(text, encoding="utf-8")
        hot_lines = [
            f"zerogpt_proxy_score={proxy.document_score:.1f}",
            f"hot_sentences={len(proxy.hot_sentences)}",
        ]
        for tag in applied:
            hot_lines.append(f"  applied: {tag}")
        for s in sorted(proxy.sentences, key=lambda x: x.score, reverse=True)[:5]:
            hot_lines.append(f"  [{s.index}] {s.score:.1f} {s.text[:80]!r}")
        meta.write_text("\n".join(hot_lines) + "\n", encoding="utf-8")
