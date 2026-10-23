"""Bounded scheduler conformance seams; observations never expose live processes.

Logical GPU lanes in these probes run CPU shell commands only. No probe reports
GPU qualification, provisions a machine, or accepts replacement implementations.
"""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

from etx.api.types import Configuration
from etx.experiments import jobs as engine


def job_transport_interrupt_probe(directory, *, kill_parent=False):
    """Interrupt an API caller while its isolated scheduler owns a CPU child."""
    import os
    import signal
    import subprocess
    import time
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    code = '''
import sys
from etx.external import JobSpec, JobSession, JobSessionRequest, run_jobs
root = sys.argv[1]
job = JobSpec('sleeper', 'echo ready > "$READY"; sleep 5; echo escaped > "$LATE"; sleep 60',
              lane='cpu', env={'READY': root+'/ready', 'LATE': root+'/late'}, timeout=90)
run_jobs(JobSessionRequest(JobSession((job,), max_seconds=95), logdir=root+'/records', gpus=(), poll_s=.01))
'''
    with (directory / 'caller.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-c', code, str(directory)],
                                 cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            while not (directory / 'ready').is_file():
                if child.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('scheduler interrupt probe did not become ready')
                time.sleep(.01)
            child.send_signal(signal.SIGKILL if kill_parent else signal.SIGINT)
            # Cleanup may take the scheduler's whole sweep budget (etx.session.proc.SWEEP_S, 30 s) plus its
            # shim wait on a loaded host; the late marker fires 5 s after ready if the child escaped.
            code = child.wait(timeout=45)
            deadline = time.monotonic() + 45
            records = list((directory / 'records').iterdir())
            while not all((path / 'SEAL').is_file() for path in records):
                if time.monotonic() >= deadline:
                    raise RuntimeError('interrupted scheduler did not seal after cleanup')
                time.sleep(.01)
            time.sleep(5.5)
            return Configuration({'caller_returncode': code, 'late_child_output': (directory / 'late').exists(),
                                  'seals': [json.loads((path / 'SEAL').read_text()) for path in records]})
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


@contextmanager
def observe_unrelated_child():
    """Own one bounded sleeper outside the scheduler and report whether it survived."""
    import subprocess
    process = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(30)'],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    observations = []
    try:
        yield observations
    finally:
        observations.append(Configuration({'survived_scheduler': process.poll() is None}))
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def observe_job_processes():
    """Request bounded immutable child observations from the isolated scheduler."""
    from etx.contracts.jobs import _job_observer
    observations = []
    token = _job_observer.set(observations)
    try:
        yield observations
    finally:
        _job_observer.reset(token)


@contextmanager
def _observe_local_job_processes():
    """Observe at most 32 job launches and poll them once on context exit.

    The returned list is populated with immutable snapshots on exit. Only the
    PATH and test JOB_MARKER environment fields are retained, not ambient secrets.
    No Popen object or private callable crosses the testing facade.
    """
    observations, launched = [], []
    popen = engine.subprocess.Popen

    def capture(args, **kwargs):
        if len(launched) >= 32:
            raise ValueError('job process probe is limited to 32 launches')
        proc = popen(args, **kwargs)
        command = args[-1]
        launched.append((proc, {
            'command': command,
            'uses_runner_shim': args == [sys.executable, '-I', engine.sproc.__file__,
                                         '--', 'bash', '-c', command],
            'path': kwargs.get('env', {}).get('PATH'),
            'marker': kwargs.get('env', {}).get('JOB_MARKER'),
        }))
        return proc

    try:
        with patch.object(engine.subprocess, 'Popen', capture):
            yield observations
    finally:
        observations.extend(Configuration({**fields, 'returncode': proc.poll()})
                            for proc, fields in launched)


def job_runner_probe(kind, directory, *, lane='cpu'):
    """Run one fixed, CPU-only private runner conformance scenario."""
    if lane not in ('cpu', 'gpu'):
        raise ValueError('probe lane must be cpu or gpu (CPU commands only)')
    directory = Path(directory)
    if kind == 'atomic-status':
        directory.mkdir(parents=True, exist_ok=True)
        jobs = [engine.Job(name='x', cmd='true')]
        runner = engine.Runner(jobs, [], 1, directory)
        runner._write_status()
        replace, calls = engine.os.replace, []

        def checked_replace(src, dst):
            calls.append((Path(src), json.loads(dst.read_text())[0]['status'],
                          json.loads(src.read_text())[0]['status']))
            replace(src, dst)

        with patch.object(engine.os, 'replace', checked_replace):
            jobs[0].status = 'fail'
            runner._write_status()
        return Configuration({
            'replacements': len(calls),
            'old_statuses': [old for _, old, _ in calls],
            'new_statuses': [new for _, _, new in calls],
            'temporary_files_removed': all(not path.exists() for path, _, _ in calls),
            'final_status': json.loads((directory / 'status.json').read_text())[0]['status'],
        })
    if kind == 'occupied-directory':
        job = engine.Job(name='probe', lane=lane, cmd='true', cwd=str(directory.parent), timeout=2)
        code = engine.Runner([job], [0], 1, directory, out=io.StringIO(),
                             poll_s=.01, max_seconds=3).run()
        before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        error = None
        try:
            engine.Runner([engine.Job(name='again', cmd='true', lane=lane)], [0], 1,
                          directory, out=io.StringIO(), max_seconds=3).run()
        except SystemExit as exc:
            error = str(exc.code)
        after = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        return Configuration({'returncode': code, 'error': error, 'unchanged': before == after})
    raise ValueError(f'unknown job runner probe: {kind}')
