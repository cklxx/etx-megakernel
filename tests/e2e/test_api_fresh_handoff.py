"""The fresh-handoff plan/analyze commands on unrun and edited matrices, moved from tests/test_fresh_handoff.py."""
import json
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def handoff(*argv):
    return run_command([sys.executable, '-m', 'etx.tools.fresh_handoff', *map(str, argv)], cwd=ROOT)


def planned():
    run = handoff('plan')
    assert run.returncode == 0, run.stdout
    return json.loads(run.stdout)


def test_missing_runs_fail_closed(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'plan': planned()}))
    run = handoff('analyze', tmp_path)
    report = json.loads(run.stdout[run.stdout.index('{'):])
    assert report['mechanism_gate'] == 'not_passed'
    assert len(report['cases']) == 144
    assert all(c['state'] == 'fail' for c in report['cases'])
    assert report['paired'] == []


def test_edited_matrix_cannot_pass(tmp_path):
    plan = planned()
    plan['cases'] = []
    (tmp_path / 'manifest.json').write_text(json.dumps({'plan': plan}))
    run = handoff('analyze', tmp_path)
    assert run.returncode != 0 and 'frozen' in run.stdout
