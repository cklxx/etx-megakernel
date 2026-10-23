"""The columns experiment command, moved from tests/test_experiment_columns.py, through external APIs only."""
import json
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def columns(*argv):
    return run_command([sys.executable, '-m', 'examples.experiment_program.columns', *argv], cwd=ROOT)


def test_cli_all_flags_off_no_native_inputs():
    for record in ('G02','G03','G05','G13'):
        run=columns('--record',record,'--dry-run')
        assert run.returncode==0, run.stdout
        report=json.loads(run.stdout)
        assert report['enabled'] is False
        assert report['status']=='BLOCKED_NATIVE_CAPTURE'
    run=columns('--record','G02','--execute')
    assert run.returncode==2
    assert json.loads(run.stdout)['enabled'] is False
