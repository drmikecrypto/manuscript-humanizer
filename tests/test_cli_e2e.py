"""End-to-end CLI smoke tests."""

import subprocess
import sys
from pathlib import Path


def test_cli_humanize_demo_manuscript(tmp_path: Path):
    demo = Path("examples/demo_manuscript.md")
    if not demo.exists():
        return
    out = tmp_path / "demo_humanized.md"
    proc = subprocess.run(
        [sys.executable, "-m", "humanizer.cli", "humanize", "-f", str(demo), "-o", str(out)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=180,
    )
    assert proc.returncode in (0, 2)
    assert out.exists()
    assert len(out.read_text(encoding="utf-8")) > 100


def test_cli_score_demo_manuscript():
    demo = Path("examples/demo_manuscript.md")
    if not demo.exists():
        return
    proc = subprocess.run(
        [sys.executable, "-m", "humanizer.cli", "score", "-f", str(demo)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=120,
    )
    assert proc.returncode == 0
    assert "ZeroGPT" in proc.stdout or "score" in proc.stdout.lower()
