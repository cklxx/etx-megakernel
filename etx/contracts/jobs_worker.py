"""Isolated scheduler host; private transport, not another scheduling engine."""
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys

# -I excludes caller-controlled cwd/PYTHONPATH. Load only this installed tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from etx.api.jobs import JobSession, JobSessionRequest, JobSpec
from etx.contracts.jobs import _run_jobs


def main(directory, parent):
    directory = Path(directory)

    stopping = False

    def interrupted(signum, frame):
        nonlocal stopping
        # A second signal must not interrupt the first signal's abort/sweep.
        if stopping:
            return
        stopping = True
        raise KeyboardInterrupt(f'scheduler interrupted by signal {signum}')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    if sys.platform == 'linux':
        import ctypes
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError('cannot establish scheduler parent-death signal')
        if os.getppid() != parent:
            raise KeyboardInterrupt('scheduler parent exited during setup')
    payload = json.loads((directory / 'request.json').read_text())
    observations, response = [], {}
    context = nullcontext(observations)
    if payload['observe']:
        from etx.testing.jobs import _observe_local_job_processes
        context = _observe_local_job_processes()
    try:
        fields = payload['request']
        if isinstance(fields['session'], dict):
            session = fields['session']
            session['jobs'] = tuple(JobSpec(**job) for job in session['jobs'])
            fields['session'] = JobSession(**session)
        request = JobSessionRequest(**fields)
        with context as observations:
            response['result'] = asdict(_run_jobs(request))
    except BaseException as error:
        response['error'] = {'type': type(error).__name__, 'message': str(error),
                             'code': error.code if isinstance(error, SystemExit) else None}
    response['observations'] = [row.to_dict() for row in observations]
    (directory / 'response.json').write_text(json.dumps(response))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
