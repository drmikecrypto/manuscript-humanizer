from __future__ import annotations

from humanizer.analyzers.detector import detect_ai_likelihood
from humanizer.rewriters.transforms import refine_text, transform_text
from humanizer.validators.fidelity import validate_fidelity


class LocalRewriter:
    """Fully offline rewriter — no API keys or network calls."""

    def __init__(self, min_similarity: float = 0.72) -> None:
        self._min_similarity = min_similarity

    def rewrite(
        self,
        text: str,
        *,
        iteration: int = 1,
        issues: list[str] | None = None,
    ) -> str:
        candidates = [
            transform_text(text, iteration=iteration, issues=issues, intensity="light"),
            transform_text(text, iteration=iteration, issues=issues, intensity="medium"),
            transform_text(text, iteration=iteration, issues=issues, intensity="strong"),
        ]

        best_text = text
        best_score = detect_ai_likelihood(text).composite_score

        for candidate in candidates:
            if not validate_fidelity(
                text, candidate, min_similarity=self._min_similarity
            ).passed:
                continue
            score = detect_ai_likelihood(candidate).composite_score
            if score < best_score:
                best_score = score
                best_text = candidate

        return best_text

    def refine(self, text: str, problems: list[str]) -> str:
        return refine_text(text, problems)
