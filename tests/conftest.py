"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

_PRIVATE_MS = Path("tests/test_manuscripts/orginal_manuscript.md")
_PUBLIC_MS = Path("examples/demo_manuscript.md")


def load_calibration_manuscript() -> str:
    for path in (_PRIVATE_MS, _PUBLIC_MS):
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        "No calibration manuscript found. "
        "Use examples/demo_manuscript.md or add a local copy under tests/test_manuscripts/."
    )


@pytest.fixture
def calibration_manuscript() -> str:
    return load_calibration_manuscript()
