from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

MODEL_REPO = "Eslzzyl/aigc-detector-en-onnx"
MODEL_FILES = ("onnx/model_quantized.onnx", "tokenizer.json")
HF_BASE = "https://huggingface.co"


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "manuscript-humanizer" / "models" / "aigc-detector-en"


def is_model_available(cache_dir: Path | None = None) -> bool:
    root = cache_dir or default_cache_dir()
    return (root / "model_quantized.onnx").exists() and (root / "tokenizer.json").exists()


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "manuscript-humanizer/0.2"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def download_model(cache_dir: Path | None = None, *, force: bool = False) -> Path:
    """Download ONNX AIGC detector model and tokenizer to cache."""
    root = cache_dir or default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)

    if not force and is_model_available(root):
        return root

    for remote in MODEL_FILES:
        url = f"{HF_BASE}/{MODEL_REPO}/resolve/main/{remote}"
        name = Path(remote).name
        dest = root / name
        _download_file(url, dest)

    meta = {"repo": MODEL_REPO, "files": list(MODEL_FILES)}
    (root / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def model_info(cache_dir: Path | None = None) -> dict[str, str | bool]:
    root = cache_dir or default_cache_dir()
    return {
        "cache_dir": str(root),
        "available": is_model_available(root),
        "repo": MODEL_REPO,
    }
