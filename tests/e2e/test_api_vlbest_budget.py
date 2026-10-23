"""The vlbest --dry-run budget print, moved from tests/test_vlbest_session.py, through external APIs only."""
import os
from pathlib import Path
import sys

import pytest
from etx.external import load_job_session
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('override,cap', [(None, 8400), ('6000', 6000)])
def test_effective_budget_print(override, cap):
    env = {k: v for k, v in os.environ.items() if k != 'ETX_SESS_MAX_SECONDS'}
    if override is not None:
        env['ETX_SESS_MAX_SECONDS'] = override
    budget = f'BUDGET vm-life=150m runner-cap={cap / 60:g}m'
    if override is not None:
        budget += ' override=ETX_SESS_MAX_SECONDS'
    proc = run_command([sys.executable, '-m', 'examples.sessions.vlbest', '--dry-run'], cwd=ROOT, env=env)
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.splitlines()[:2] == [
        budget, f'Session hard stop: {cap}s from runner start (includes setup)']


def test_dry_run_lists_every_job_in_order():
    env = {k: v for k, v in os.environ.items() if k != 'ETX_SESS_MAX_SECONDS'}
    jobs = load_job_session(ROOT / 'examples/sessions/vlbest.py').jobs
    proc = run_command([sys.executable, '-m', 'examples.sessions.vlbest', '--dry-run'], cwd=ROOT, env=env)
    assert proc.returncode == 0, proc.stdout
    assert 'BUDGET vm-life=150m runner-cap=140m' in proc.stdout
    rows = [line for line in proc.stdout.splitlines() if '  reads: ' in line]
    assert len(rows) == len(jobs)
    assert all(j.name in rows[i] for i, j in enumerate(jobs))
