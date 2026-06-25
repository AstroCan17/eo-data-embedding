"""The Phase-0 green-light gate, wrapped for pytest.

Deselected by default (see `addopts` in pyproject.toml) — run with `pytest -m slow`.
Exercises the full synthetic pipeline: encode -> parquet store -> FAISS -> few-shot probe.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_phase0_smoke_runs_clean():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "phase0_smoke.py"), "--n", "120"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
