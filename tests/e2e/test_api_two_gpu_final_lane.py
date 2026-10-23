"""The two-GPU sessions' final lane, run exactly as the runner runs it, through external APIs only."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.external import load_job_session
from etx.testing import qualification_record, run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('session, flag', [('qwen_batch_correctness', '--qwen-only'),
                                           ('glm53_model', '--glm-only')])
def test_session_final_lane_command_runs_as_the_runner_runs_it(tmp_path, session, flag):
    # The exact `final` cmd, under bash -c from the repository with the runner's interpreter
    # first on PATH (etx/tools/gpu_jobs.py), against a complete and an empty record.
    jobs = load_job_session(str(ROOT / f'examples/sessions/{session}.py')).jobs
    final, = [j for j in jobs if j.lane == 'final']
    assert flag in final.cmd
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    for root, code in ((qualification_record(tmp_path / 'good'), 0), (tmp_path / 'empty', 2)):
        root.mkdir(parents=True, exist_ok=True)
        done = run_command(['bash', '-c', final.cmd], cwd=ROOT, env=dict(env, ETX_SESS_LOGDIR=str(root)))
        assert done.returncode == code, done.stdout
        assert 'unrecognized arguments' not in done.stdout
        out, = final.writes
        assert json.loads((root / out).read_text())['status'] == ('OK' if code == 0 else 'FAIL')
