"""The numpy reference of the tile pipeline reproduces HF greedy on a local checkpoint (skipped without it)."""
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.path.expanduser(os.environ.get("ETX_REFERENCE_MODEL", "~/models/qwen2.5-0.5b"))
PY = os.path.expanduser(os.environ.get("ETX_REFERENCE_PYTHON", sys.executable))


@pytest.mark.skipif(not (os.path.isdir(MODEL) and os.path.exists(PY)), reason="needs the local Qwen2.5-0.5B checkpoint and a torch venv")
def test_ref_numpy_matches_hf():
    out = run_command([PY, "examples/llm/ref_numpy.py", "--model", MODEL, "--gen", "4", "--hf"], timeout=600, cwd=ROOT)
    assert "generated: [279, 1156, 97941, 572]" in out.stdout, out.stdout[-500:]   # HF greedy on the default prompt (verified 2026-09-25)
    assert out.returncode == 0 and "tokens equal up to 4/4" in out.stdout, out.stdout[-1000:]
