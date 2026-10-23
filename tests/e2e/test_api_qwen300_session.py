"""Qwen300 session commands, moved from tests/test_qwen300_session.py, through external APIs only."""
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / 'examples/sessions/qwen300.py'


@pytest.mark.parametrize('seconds,accepted', [
    (None, True), ('6600', True), ('1500', True), ('6601', False),
    ('8400', False), ('0', False), ('-1', False), ('nan', False),
    ('inf', False), ('invalid', False),
])
def test_qwen300_cap_remains_110_minutes(seconds, accepted):
    env = dict(os.environ)
    env.pop('ETX_SESS_MAX_SECONDS', None)
    if seconds is not None:
        env['ETX_SESS_MAX_SECONDS'] = seconds
    result = run_command([sys.executable, '-m', 'etx.tools.gpu_jobs', str(SESSION), '--dry-run'],
                         cwd=ROOT, env=env, timeout=15)
    assert (result.returncode == 0) is accepted, result.stdout
    if accepted:
        assert f'Session hard stop: {seconds or 6600}s' in result.stdout
    else:
        assert 'Qwen300 cap must be positive and <=6600 seconds' in result.stdout


@pytest.mark.parametrize('args', [
    ['--job', '3'], ['--inside', 'hf'], ['--inside', 'probe'],
    ['--inside', 'bench', '--precision', 'fp8'],
])
def test_removed_jobs_and_modes_refused(args):
    assert run_command([sys.executable, '-m', 'examples.sessions.qwen300', *args], cwd=ROOT).returncode == 2
