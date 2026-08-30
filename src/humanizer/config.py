from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


@dataclass
class LLMConfig:
    provider: str = "openai"
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.85
    max_tokens: int = 8192


@dataclass
class PipelineConfig:
    target_ai_score: float = 15.0
    max_iterations: int = 5
    min_meaning_similarity: float = 0.82
    preserve_citations: bool = True
    preserve_numbers: bool = True
    chunk_size: int = 3500
    chunk_overlap: int = 200


@dataclass
class DetectorConfig:
    gptzero_api_key: str = ""
    originality_api_key: str = ""
    use_external_detector: bool = False
    external_weight: float = 0.4


@dataclass
class OutputConfig:
    show_diff: bool = False
    save_intermediate: bool = True
    intermediate_dir: str = ".humanizer_runs"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        config_path = Path(path or "config.toml")
        if not config_path.exists():
            example = config_path.parent / "config.example.toml"
            if example.exists():
                data: dict[str, Any] = tomllib.loads(example.read_text(encoding="utf-8"))
            else:
                data = {}
        else:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))

        cfg = cls()
        if llm := data.get("llm"):
            cfg.llm = LLMConfig(**{k: llm[k] for k in LLMConfig.__dataclass_fields__ if k in llm})
        if pipe := data.get("pipeline"):
            cfg.pipeline = PipelineConfig(
                **{k: pipe[k] for k in PipelineConfig.__dataclass_fields__ if k in pipe}
            )
        if det := data.get("detector"):
            cfg.detector = DetectorConfig(
                **{k: det[k] for k in DetectorConfig.__dataclass_fields__ if k in det}
            )
        if out := data.get("output"):
            cfg.output = OutputConfig(**{k: out[k] for k in OutputConfig.__dataclass_fields__ if k in out})

        cfg._resolve_api_keys()
        return cfg

    def _resolve_api_keys(self) -> None:
        if not self.llm.api_key:
            env_map = {
                "openai": "OPENAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "custom": "LLM_API_KEY",
            }
            env_key = env_map.get(self.llm.provider, "OPENAI_API_KEY")
            self.llm.api_key = os.environ.get(env_key, "")

        if not self.llm.base_url:
            defaults = {
                "deepseek": "https://api.deepseek.com",
            }
            self.llm.base_url = defaults.get(self.llm.provider, "")

        if not self.detector.gptzero_api_key:
            self.detector.gptzero_api_key = os.environ.get("GPTZERO_API_KEY", "")

    def save_example(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "llm": self.llm.__dict__,
            "pipeline": self.pipeline.__dict__,
            "detector": self.detector.__dict__,
            "output": self.output.__dict__,
        }
        path.write_bytes(tomli_w.dumps(data))
