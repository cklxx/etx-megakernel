"""Fixed, bounded scheduler fault probes, isolated from the caller's processes."""
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

from etx.api.types import Configuration
from etx.experiments import jobs as engine
from etx.session import proc as supervisor


_KINDS = ('launch-failure', 'status-failure', 'status-interrupt', 'status-exit',
          'final-status-failure', 'folded-forward', 'folded-valid')


def job_fault_probe(kind, directory):
    """Return detached observations from one fixed CPU-only fault scenario."""
    if kind not in _KINDS:
        raise ValueError(f'unknown job fault probe: {kind}')
    output = io.StringIO()
    code = supervisor.run(
        [sys.executable, '-c',
         'import json, sys; from etx.testing.jobs_faults import _probe; '
         'print(json.dumps(_probe(sys.argv[1], sys.argv[2])))',
         kind, str(Path(directory).resolve())],
        cwd=str(Path(__file__).resolve().parents[2]), sink=output.write,
        deadline_s=8, poll_s=.01)
    if code != 0:
        raise RuntimeError(f'job fault probe failed ({code}): {output.getvalue()}')
    return Configuration(json.loads(output.getvalue()))


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _probe(kind, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record_dir = directory / 'record'
    if kind.startswith('folded-'):
        jobs = [engine.Job('first', 'true', gpu=1,
                           reads=['second'] if kind == 'folded-forward' else []),
                engine.Job('second', 'true', gpu=0)]
        engine.validate_jobs(jobs)  # Valid before the missing pin is folded.
        error = None
        with patch.object(engine.subprocess, 'Popen') as launch:
            try:
                runner = engine.Runner(jobs, [0], 1, record_dir, out=io.StringIO())
            except SystemExit as exc:
                error = str(exc)
            return {'error': error, 'launches': launch.call_count,
                    'pins': [j.gpu for j in jobs], 'record_created': record_dir.exists()}

    ready = directory / 'started'
    command = f'printf "%s" "$$" > {shlex.quote(str(ready))}; sleep 30'
    jobs = [engine.Job(name, command, lane='cpu', cwd=str(directory), timeout=2)
            for name in ('first', 'second')]
    if kind == 'final-status-failure':
        jobs = [engine.Job('first', f'printf "%s" "$$" > {shlex.quote(str(ready))}',
                           lane='cpu', cwd=str(directory), timeout=2)]
    runner = engine.Runner(jobs, [], 2, record_dir, out=io.StringIO(), poll_s=.01, max_seconds=3)
    launched, child_pids, seal_observations = [], [], []
    popen, write_status, seal = engine.subprocess.Popen, runner._write_status, engine.record.seal
    status_calls = 0

    def fail():
        end = time.monotonic() + 2
        while not ready.exists() or not ready.read_text():
            if time.monotonic() >= end:
                raise RuntimeError('fault probe child did not become ready')
            time.sleep(.005)
        child_pids.append(int(ready.read_text()))
        error = {'status-interrupt': KeyboardInterrupt, 'status-exit': SystemExit}.get(kind, OSError)
        raise error('injected scheduler fault')

    def launch(*args, **kwargs):
        if kind == 'launch-failure' and launched:
            fail()
        proc = popen(*args, **kwargs)
        launched.append(proc)
        return proc

    def publish():
        nonlocal status_calls
        status_calls += 1
        if kind == 'final-status-failure':
            if jobs[0].status == 'ok':
                fail()
        elif kind != 'launch-failure' and status_calls >= 2:
            fail()  # Persistent publication failure, including the abort retry.
        write_status()

    def stopped():
        return all(p.poll() is not None for p in launched) and all(not _alive(p) for p in child_pids)

    def sealing(*args, **kwargs):
        seal_observations.append(stopped())
        return seal(*args, **kwargs)

    error = None
    try:
        with patch.object(engine.subprocess, 'Popen', launch), \
                patch.object(runner, '_write_status', publish), \
                patch.object(engine.record, 'seal', sealing):
            try:
                runner.run()
            except BaseException as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
        sealed = json.loads((record_dir / 'SEAL').read_text())
        verification, messages = engine.record.verify(record_dir)
        return {'error': error, 'launches': len(launched), 'stopped': stopped(),
                'stopped_before_seal': seal_observations,
                'statuses': [j.status for j in jobs],
                'sealed_jobs': sealed['jobs'], 'complete': sealed['complete'],
                'failure': sealed.get('failure'),
                'verification': verification, 'verification_messages': messages}
    finally:
        # A failing implementation must not make a conformance probe leak work.
        active = [j for j in jobs if j.proc is not None and j.proc.poll() is None]
        if active:
            runner._kill(*active)
