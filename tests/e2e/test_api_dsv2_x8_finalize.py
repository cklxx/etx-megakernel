"""The x8 correctness final lane writing a result when its dependencies are missing, moved from tests/test_deepseek_open_prep.py."""
import json
import os
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_final_lane_reports_missing_dependencies(tmp_path):
    env = dict(os.environ, ETX_SESS_LOGDIR=str(tmp_path), ETX_DSV2_FLEET=str(tmp_path), ETX_DSV2_DATA=str(tmp_path))
    run = run_command([sys.executable, '-m', 'examples.dsv2lite.x8_correctness', 'finalize'], cwd=ROOT, env=env)
    assert run.returncode == 0, run.stdout[-1500:]
    out = json.loads((tmp_path / 'x8_summary.json').read_text())
    assert out['status'] == 'INCOMPLETE' and out['references_ready']['status'] == 'MISSING'
