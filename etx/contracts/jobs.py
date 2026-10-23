"""Adapt immutable scheduler requests to the single legacy scheduling engine."""
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from etx.api.jobs import JobOutcome, JobSession, JobSessionRequest, JobSessionResult, JobSpec
from etx.api.types import Configuration
from etx.experiments import jobs as engine
from etx.session import record


def load_job_session(path):
    jobs, limit, local_only, lease_path = engine._load(path)
    declarations = []
    for job in jobs:
        values = {name: getattr(job, name) for name in JobSpec.__dataclass_fields__ if name != 'schema_version'}
        values['cwd'] = os.fspath(job.cwd)
        values['env'] = {key: str(value) for key, value in job.env.items()}
        declarations.append(JobSpec(**values))
    return JobSession(tuple(declarations), limit, local_only, str(lease_path) if lease_path is not None else None)


def verify_job_record(directory):
    code, lines = record.verify(Path(directory))
    return Configuration({'returncode': code, 'messages': lines, 'kind': 'session-record-integrity',
                          'device_qualification': False})


_job_observer = ContextVar('job_process_observer', default=None)


def run_jobs(request):
    """Own the scheduler in a child, never in the caller's process tree supervisor."""
    if not isinstance(request, JobSessionRequest):
        raise TypeError('run_jobs requires JobSessionRequest')
    values = {name: getattr(request, name) for name in request.__dataclass_fields__ if name != 'session'}
    if isinstance(request.session, str):
        values['session'] = request.session
    else:
        session = request.session
        values['session'] = dict(jobs=[job.to_dict() for job in session.jobs],
                                 max_seconds=session.max_seconds, local_only=session.local_only,
                                 lease_path=session.lease_path, schema_version=session.schema_version)
    observer = _job_observer.get()
    with tempfile.TemporaryDirectory(prefix='etx-job-worker-') as temporary:
        directory = Path(temporary)
        (directory / 'request.json').write_text(json.dumps({'request': values, 'observe': observer is not None}))
        worker = Path(__file__).with_name('jobs_worker.py').resolve()
        # Inherit environment without serializing it. Only declared overrides are
        # part of the immutable input. The worker imports packaged code, not cwd.
        from etx.contracts.worker import stream_worker
        code = stream_worker([sys.executable, '-I', str(worker), str(directory), str(os.getpid())])
        response = directory / 'response.json'
        if not response.is_file():
            raise RuntimeError(f'scheduler worker exited {code} without a completed response')
        payload = json.loads(response.read_text())
        if observer is not None:
            observer.extend(Configuration(row) for row in payload['observations'])
        if 'error' in payload:
            error = payload['error']
            if error['type'] == 'SystemExit':
                raise SystemExit(error['code'])
            exceptions = {cls.__name__: cls for cls in (ValueError, TypeError, FileExistsError,
                          FileNotFoundError, PermissionError, OSError, RuntimeError, KeyboardInterrupt)}
            raise exceptions.get(error['type'], RuntimeError)(error['message'])
        if code != 0:
            raise RuntimeError(f'scheduler worker exited {code}')
        fields = payload['result']
        fields['jobs'] = tuple(JobOutcome(**job) for job in fields['jobs'])
        return JobSessionResult(**fields)


def _run_jobs(request):
    if not isinstance(request, JobSessionRequest):
        raise TypeError('run_jobs requires JobSessionRequest')
    if isinstance(request.session, str):
        jobs, max_seconds, local_only, lease_path = engine._load(request.session)
    else:
        session = request.session
        jobs = [engine.Job(**{k: v for k, v in spec.to_dict().items() if k != 'schema_version'})
                for spec in session.jobs]
        engine.validate_jobs(jobs)
        max_seconds = engine.session_limit(session.max_seconds)
        local_only = record.LOCAL_ONLY if session.local_only is None else session.local_only
        lease_path = session.lease_path
    jobs = [j for j in jobs if j.machine == request.machine]
    if request.only:
        keep = {j.name for j in jobs if re.search(request.only, j.name)}
        dropped = {w for j in jobs if j.name not in keep for w in j.writes} | {j.name for j in jobs} - keep
        for j in jobs:
            j.reads = [r for r in j.reads if r.lstrip('?') not in dropped]
        jobs = [j for j in jobs if j.name in keep]
    gpus = list(request.gpus)

    def result(code, directory=None):
        return JobSessionResult(code, str(directory) if directory is not None else None,
                                tuple(JobOutcome(j.name, j.status, j.rc, j.start, j.end, j.on_gpu,
                                                 str(j.log) if j.log is not None else None, j.note)
                                      for j in jobs), max_seconds,
                                'not-executed' if request.dry_run else 'host-process')

    if request.dry_run:
        if max_seconds is not None:
            print(f'Session hard stop: {max_seconds:g}s from runner start (includes setup)')
        for i, j in enumerate(jobs):
            pin = f'GPU {j.gpu}' if j.gpu is not None else ('any GPU' if j.lane == 'gpu' else 'cpu')
            pin = 'final' if j.lane == 'final' else pin
            print(f"{i:3d} {j.name:32} {pin:8} timeout {j.timeout:5.0f}s  reads: {', '.join(j.reads) or '-'}"
                  + (f"  writes: {', '.join(j.writes)}" if j.writes else ''))
        return result(0)
    # A runnable GPU lane without an available lane would wait forever. Dry runs
    # remain useful for inspecting such declarations without executing them.
    if not gpus and any(j.lane == 'gpu' for j in jobs):
        raise ValueError('GPU jobs require at least one logical GPU lane')
    lease = None
    if lease_path is not None:
        if request.only or gpus != [0] or request.machine != 'A' or max_seconds is None:
            raise SystemExit('leased session requires complete graph, one GPU lane 0, machine A, finite cap')
        from etx.session.lease import Lease
        lease = Lease(lease_path, max_seconds)
        max_seconds = lease.remaining
    try:
        logdir = record.create(Path(os.path.expanduser(request.logdir)))
        code = engine.Runner(jobs, gpus, request.cpu_slots, logdir, request.progress_s,
                             poll_s=request.poll_s, max_seconds=max_seconds, local_only=local_only).run()
        return result(code, logdir)
    finally:
        if lease is not None:
            lease.close()
