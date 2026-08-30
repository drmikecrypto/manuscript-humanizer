from __future__ import annotations

from openai import OpenAI

from humanizer.config import LLMConfig
from humanizer.prompts.academic import (
    ACADEMIC_SYSTEM_PROMPT,
    REFINE_USER_TEMPLATE,
    REWRITE_USER_TEMPLATE,
)


class LLMRewriter:
    def __init__(self, config: LLMConfig) -> None:
        kwargs: dict = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = OpenAI(**kwargs)
        self._config = config

    def rewrite(
        self,
        text: str,
        *,
        iteration: int = 1,
        issues: list[str] | None = None,
    ) -> str:
        issue_block = "\n".join(f"- {i}" for i in (issues or [])) or "- General AI-pattern cleanup"
        user_msg = REWRITE_USER_TEMPLATE.format(
            iteration=iteration,
            issues=issue_block,
            text=text,
        )
        response = self._client.chat.completions.create(
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            messages=[
                {"role": "system", "content": ACADEMIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def refine(self, text: str, problems: list[str]) -> str:
        user_msg = REFINE_USER_TEMPLATE.format(
            problems="\n".join(f"- {p}" for p in problems),
            text=text,
        )
        response = self._client.chat.completions.create(
            model=self._config.model,
            temperature=max(0.5, self._config.temperature - 0.2),
            max_tokens=self._config.max_tokens,
            messages=[
                {"role": "system", "content": ACADEMIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()
