"""`etx compile` refusing an invalid D4 placement, moved from tests/test_dsv2_d4.py, through external APIs only."""
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("rows,needle", [("40", "row boundary 40 is not a multiple of 32"),
                                         ("1408", "rows [1408, 1408): outside [0, 1408) or empty")])
def test_etx_compile_refuses_an_invalid_placement(tmp_path, rows, needle):
    env = dict(os.environ, ETX_DSV2_D4="1", ETX_DSV2_D4_ROWS=rows)
    p = run_command([sys.executable, "-m", "etx", "compile", "examples/dsv2lite/graph.py", "--arch", "gfx942",
                     "--out", str(tmp_path / "plan"), "--inline-tiles", "--relay", "on"], cwd=ROOT, env=env)
    assert p.returncode != 0 and "PlacementError" in p.stdout and needle in p.stdout, p.stdout[-2000:]
    assert not (tmp_path / "plan" / "megakernel_d0.hip").exists()
