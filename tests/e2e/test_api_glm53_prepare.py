"""The examples.glm53.prepare command, moved from tests/test_glm53_preparation.py, through external APIs only."""
import json
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_executable_blocker_is_offline_and_exits_nonzero(tmp_path):
    out = tmp_path / 'blocker.json'
    run = run_command([sys.executable, '-m', 'examples.glm53.prepare', '--require-model', '--out', str(out)], cwd=ROOT)
    assert run.returncode == 2 and json.loads(out.read_text())['status'] == 'ASSETS-REQUIRED'
