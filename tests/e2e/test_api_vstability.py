"""The vstability summary job, moved from tests/test_session_references.py, through external APIs only."""
import json
import os
from pathlib import Path

from etx.external import load_job_session
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_vstability_summary_uses_current_logdir(tmp_path):
    home = tmp_path / 'home'
    stale = home / 'sess'
    current = stale / 'session-current'
    current.mkdir(parents=True)
    for rnd in range(1, 7):
        (stale / f'vs_r{rnd}.json').write_text(json.dumps({'median_ms': 99}))
        (current / f'vs_r{rnd}.json').write_text(json.dumps({'median_ms': rnd}))
    jobs = load_job_session(ROOT / 'examples/sessions/vstability.py').jobs
    assert all('--json ${ETX_SESS_LOGDIR:-$HOME/sess}/' in job.cmd for job in jobs[:-1])
    result = run_command(['bash', '-c', jobs[-1].cmd], cwd=ROOT, timeout=10,
                         env=dict(os.environ, HOME=str(home), ETX_SESS_LOGDIR=str(current)))
    assert result.returncode == 0, result.stdout
    assert 'VSTAB 1.000000 6.000000 5.000000 3.500000' in result.stdout
