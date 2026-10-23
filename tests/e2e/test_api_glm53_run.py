"""examples.glm53.run refusing options at qualification, moved from tests/test_glm53_options.py, through external APIs only."""
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_qualify_cannot_supply_options_to_override_existing_identity(tmp_path):
    run = run_command([sys.executable, '-m', 'examples.glm53.run', 'qualify', '--stage', str(tmp_path),
                       '--build', str(tmp_path), '--out', str(tmp_path), '--options', str(tmp_path/'options.json')],
                      cwd=ROOT)
    assert run.returncode == 2, run.stdout
