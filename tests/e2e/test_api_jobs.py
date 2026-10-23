"""Host-process scheduler E2E: logical GPU lanes execute CPU commands only.

These tests establish scheduling and record behavior, never GPU correctness or
performance. All workloads enter through the external facade.
"""
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
import io
import json
import os
from pathlib import Path
import shlex
import sys
import textwrap
import time

import pytest

from etx.external import (JobSpec, JobSession, JobSessionRequest, load_job_session,
                          run_jobs, verify_job_record, ProcessRequest, execute)
from etx.testing import observe_job_processes, observe_unrelated_child, job_transport_interrupt_probe


ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path, specs, gpus=(0, 1), cpu_slots=2, max_seconds=10, poll_s=.05):
    session = JobSession(tuple(JobSpec(cwd=str(tmp_path), **s) for s in specs),
                         max_seconds=max_seconds)
    out = io.StringIO()
    start = time.monotonic()
    with redirect_stdout(out):
        result = run_jobs(JobSessionRequest(session, logdir=str(tmp_path / 'logs'),
                                           gpus=gpus, cpu_slots=cpu_slots,
                                           progress_s=.3, poll_s=poll_s))
    return result, {j.name: j for j in result.jobs}, out.getvalue(), time.monotonic() - start


def _cli(tmp_path, session, *args, environment=None):
    return execute(ProcessRequest((sys.executable, '-m', 'etx.tools.gpu_jobs', str(session), *args),
                                  str(ROOT), 15, environment=environment or {}),
                   output=tmp_path)


def test_parallel_pins_deps_gates_timeouts(tmp_path):
    specs = [
        dict(name='a', cmd='echo GPU=$GPU HIP=$HIP_VISIBLE_DEVICES; sleep 0.5'),
        dict(name='b', cmd='echo GPU=$GPU; sleep 0.5'),
        dict(name='pin1', cmd='echo GPU=$GPU', gpu=1, reads=['a', 'b']),
        dict(name='build', cmd='echo built > $ETX_SESS_LOGDIR/built.txt', lane='cpu', writes=['built.txt']),
        dict(name='bad', cmd='echo oops; exit 3', lane='cpu'),
        dict(name='needs_bad', cmd='echo never', reads=['bad']),
        dict(name='gated_yes', cmd='echo ran', reads=['built.txt']),
        dict(name='gated_no', cmd='echo ran', reads=['never-written.txt']),
        dict(name='slow', cmd='echo start; sleep 30', timeout=.6, summary=['start'], lane='cpu'),
        dict(name='any', cmd='echo done', reads=['?bad']),
    ]
    result, j, out, dt = _run(tmp_path, specs)
    assert result.returncode == 1
    assert result.backend == 'host-process'
    assert j['a'].status == j['b'].status == 'ok'
    assert {j['a'].on_gpu, j['b'].on_gpu} == {0, 1}  # two logical lanes, not GPUs
    assert abs(j['a'].start - j['b'].start) < .3
    assert j['pin1'].on_gpu == 1 and 'GPU=1' in Path(j['pin1'].log).read_text()
    assert 'HIP=' + str(j['a'].on_gpu) in Path(j['a'].log).read_text()
    assert j['needs_bad'].status == 'skipped' and j['any'].status == 'ok'
    assert j['gated_yes'].status == 'ok' and j['gated_no'].status == 'skipped'
    assert j['slow'].status == 'timeout' and dt < 15
    assert 'start' in out and 'session done' in out
    status = json.loads((Path(result.record_directory) / 'status.json').read_text())
    assert {s['name']: s['status'] for s in status}['bad'] == 'fail'


def test_reads_wait_for_their_writer_and_the_gpu_lane_keeps_list_order(tmp_path):
    """A logical lane cannot overtake pending work; failed/missing writes skip readers."""
    _, j, _, _ = _run(tmp_path, [
        dict(name='prep_pack', lane='cpu', cmd='sleep .4; echo m > $ETX_SESS_LOGDIR/data.manifest', writes=['data.manifest']),
        dict(name='build_bad', lane='cpu', cmd='exit 1', writes=['b_bad/run']),
        dict(name='build_slow', lane='cpu', cmd='sleep .6; mkdir -p $ETX_SESS_LOGDIR/b_s; touch $ETX_SESS_LOGDIR/b_s/run',
             writes=['b_s/run']),
        dict(name='liar', lane='cpu', cmd='true', writes=['promised.txt']),
        dict(name='g_bad', reads=['data.manifest', 'b_bad/run'], cmd='echo never'),
        dict(name='g_slow', reads=['data.manifest', 'b_s/run'], cmd='test -f $ETX_SESS_LOGDIR/data.manifest'),
        dict(name='g_ready', reads=['data.manifest'], cmd='test -f $ETX_SESS_LOGDIR/data.manifest'),
        dict(name='g_lie', reads=['promised.txt'], cmd='echo never'),
    ], gpus=(0,))
    assert j['g_bad'].status == 'skipped' and 'build_bad fail' in j['g_bad'].note
    assert j['g_slow'].status == j['g_ready'].status == 'ok'
    assert j['g_slow'].start >= j['build_slow'].end and j['g_ready'].start >= j['g_slow'].start
    assert j['liar'].status == 'fail' and 'declared output missing' in j['liar'].note
    assert j['g_lie'].status == 'skipped'


def test_load_and_dry_run(tmp_path, capsys):
    path = tmp_path / 's.py'
    path.write_text('JOBS = [dict(name="x", cmd="true"), dict(name="y", cmd="true", reads="x", machine="B")]\n')
    session = load_job_session(path)
    assert session.jobs[1].reads == ('x',)
    result = run_jobs(JobSessionRequest(session, dry_run=True, machine='B'))
    assert result.returncode == 0 and result.backend == 'not-executed'
    assert result.record_directory is None
    assert [job.name for job in result.jobs] == ['y']
    assert 'y' in capsys.readouterr().out
    assert _cli(tmp_path / 'dry-run', path, '--dry-run', '--machine', 'B').returncode == 0
    assert 'y' in (tmp_path / 'dry-run/output.log').read_text()
    path.write_text('JOBS = [dict(name="x", cmd="true", after=["y"])]\n')
    with pytest.raises(SystemExit, match='after'):
        load_job_session(path)
    path.write_text('JOBS = [dict(name="x", cmd="true", reads=["typo"])]\n')
    with pytest.raises(SystemExit, match='unknown dependency'):
        load_job_session(path)


def test_child_uses_runner_interpreter_environment(tmp_path):
    stale = tmp_path / 'system-bin'
    stale.mkdir()
    python = stale / 'python3'
    python.write_text('#!/bin/sh\necho WRONG-INTERPRETER\nexit 99\n')
    python.chmod(0o755)
    inherited_path = str(stale) + os.pathsep + os.environ['PATH']
    probe = 'import os, sys; print(sys.prefix); print(os.environ["JOB_MARKER"])'
    commands = [f'python3 -u -c {shlex.quote(probe)}',
                f'env python3 -u -c {shlex.quote(probe)}', shlex.quote(str(python))]
    with observe_job_processes() as spawned:
        result, jobs, _, _ = _run(tmp_path, [
            dict(name=f'probe{i}', lane='cpu', cmd=cmd,
                 env=dict(PATH=inherited_path, JOB_MARKER='preserved'))
            for i, cmd in enumerate(commands)
        ], gpus=())
    assert result.returncode == 1
    assert [j.status for j in jobs.values()] == ['ok', 'ok', 'fail']
    for i in (0, 1):
        assert f'{sys.prefix}\npreserved\n' in Path(jobs[f'probe{i}'].log).read_text()
    assert 'WRONG-INTERPRETER' in Path(jobs['probe2'].log).read_text()
    assert len(spawned) == len(commands)
    for observation, cmd in zip(spawned, commands):
        assert observation['command'] == cmd and observation['uses_runner_shim']
        assert observation['path'] == str(Path(sys.executable).parent) + os.pathsep + inherited_path
        assert observation['marker'] == 'preserved'


def test_pin_folds_onto_one_gpu(tmp_path):
    result, j, _, _ = _run(tmp_path, [dict(name='p1', cmd='echo GPU=$GPU', gpu=1),
                                     dict(name='p0', cmd='echo GPU=$GPU', gpu=0)], gpus=(0,))
    assert result.returncode == 0 and j['p1'].on_gpu == 0 and 'absent' in j['p1'].note


def test_live_status_before_report_job(tmp_path):
    reader = ('import json, os; from pathlib import Path; '
              's={x["name"]:x["status"] for x in json.loads('
              '(Path(os.environ["ETX_SESS_LOGDIR"])/"status.json").read_text())}; '
              'assert s["bad"]=="fail"; assert s["skip"]=="skipped"; '
              'assert s["slow"]=="timeout"; print("LIVE-STATUS OK")')
    result, jobs, _, _ = _run(tmp_path, [
        dict(name='bad', lane='cpu', cmd='exit 3'),
        dict(name='skip', lane='cpu', reads=['bad'], cmd='exit 99'),
        dict(name='slow', lane='cpu', cmd='sleep 10', timeout=.1),
        dict(name='report', lane='final', cmd=f'{shlex.quote(sys.executable)} -c {shlex.quote(reader)}'),
    ])
    assert result.returncode == 1 and jobs['report'].status == 'ok'
    assert 'LIVE-STATUS OK' in Path(jobs['report'].log).read_text()


def test_every_session_gets_a_new_sealed_record(tmp_path):
    session, root = tmp_path / 'session.py', tmp_path / 'sess'
    for marker in ('first', 'second'):
        session.write_text(f'JOBS = [dict(name="probe", lane="cpu", cmd="echo {marker}")]\n')
        result = _cli(tmp_path / marker, session, '--gpus', '', '--logdir', str(root))
        assert result.returncode == 0, (tmp_path / marker / 'output.log').read_text()
        if marker == 'first':
            first = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    records = sorted(root.iterdir())
    assert len(records) == 2 and all(verify_job_record(r)['returncode'] == 0 for r in records)
    assert {p: p.read_bytes() for p in records[0].rglob('*') if p.is_file()} == first
    assert 'second' in (records[1] / 'probe.log').read_text()
    assert json.loads((records[1] / 'SEAL').read_text())['jobs'] == {'probe': 'ok'}


@pytest.mark.parametrize('lane', ['cpu', 'gpu'])
def test_job_environment_names_the_record(tmp_path, monkeypatch, lane):
    monkeypatch.setenv('ETX_SESS_LOGDIR', '/stale/parent')
    result, jobs, _, _ = _run(tmp_path, [
        dict(name='probe', lane=lane, env={'ETX_SESS_LOGDIR': '/stale/job'},
             cmd='printf "%s\\n" "$ETX_SESS_LOGDIR"; test -f "$ETX_SESS_LOGDIR/status.json"'),
    ], gpus=(0,))
    assert result.returncode == 0
    assert Path(jobs['probe'].log).read_text().splitlines()[-1] == result.record_directory


def test_session_hard_stop_kills_all_lanes_and_skips_tail(tmp_path):
    with observe_job_processes() as processes:
        result, jobs, _, elapsed = _run(tmp_path, [
            dict(name=name, cmd='echo started; sleep 30', lane=lane)
            for name, lane in [('cpu', 'cpu'), ('gpu', 'gpu')]
        ] + [dict(name='tail', cmd='echo must-not-run', lane='final')],
            gpus=(0,), cpu_slots=1, max_seconds=.15, poll_s=.01)
    assert result.returncode == 1 and elapsed < 3
    assert [j.status for j in jobs.values()] == ['timeout', 'timeout', 'skipped']
    assert all(j.note == 'session hard stop' for j in jobs.values())
    assert len(processes) == 2 and all(p['returncode'] is not None for p in processes)
    record = Path(result.record_directory)
    assert not (record / 'tail.log').exists()
    assert [j['status'] for j in json.loads((record / 'status.json').read_text())] == [
        'timeout', 'timeout', 'skipped']


@pytest.mark.skipif(not os.path.isdir('/proc/self/task'), reason='descendant sweep reads /proc (Linux, the box)')
def test_hard_stop_kills_children_forked_during_the_stop(tmp_path):
    # R15: CPU child script deliberately forks; it is executed only by run_jobs.
    child = tmp_path / 'fork.py'
    child.write_text(textwrap.dedent(f'''
        import os, time
        for _ in range(250):
            if os.fork() == 0:
                time.sleep(1.5)
                open(os.path.join({str(tmp_path)!r}, 'late-%d' % os.getpid()), 'w').write('x')
                os._exit(0)
            time.sleep(.0001)
        time.sleep(2)
    '''))
    result, jobs, _, _ = _run(tmp_path, [
        dict(name='forker', cmd=f'timeout --kill-after=5s 5s {shlex.quote(sys.executable)} {shlex.quote(str(child))}; echo after',
             lane='cpu', timeout=5),
    ], gpus=(0,), cpu_slots=1, max_seconds=.8, poll_s=.005)
    assert result.returncode == 1
    time.sleep(2.5)
    assert jobs['forker'].status == 'timeout' and not list(tmp_path.glob('late-*'))


@pytest.mark.parametrize('limit', ['0', '-1', "float('inf')", "float('nan')", "'2400'"])
def test_invalid_session_budget_refused(tmp_path, monkeypatch, limit):
    monkeypatch.delenv('ETX_SESS_MAX_SECONDS', raising=False)
    path = tmp_path / 'session.py'
    path.write_text(f'JOBS = []\nMAX_SESSION_SECONDS = {limit}\n')
    with pytest.raises(SystemExit, match='finite and positive'):
        load_job_session(path)


@pytest.mark.parametrize('override,expected', [(None, 8400), ('6000', 6000)])
def test_session_budget_override(tmp_path, monkeypatch, override, expected):
    monkeypatch.delenv('ETX_SESS_MAX_SECONDS', raising=False)
    if override is not None:
        monkeypatch.setenv('ETX_SESS_MAX_SECONDS', override)
    path = tmp_path / 'session.py'
    path.write_text('JOBS = []\nMAX_SESSION_SECONDS = 8400\n')
    assert load_job_session(path).max_seconds == expected


@pytest.mark.parametrize('override', ['nope', '', '0', '-1', 'inf', 'nan'])
def test_invalid_session_budget_override_refused(tmp_path, monkeypatch, override):
    monkeypatch.setenv('ETX_SESS_MAX_SECONDS', override)
    path = tmp_path / 'session.py'
    path.write_text('JOBS = []\nMAX_SESSION_SECONDS = 8400\n')
    with pytest.raises(SystemExit) as exc:
        run_jobs(JobSessionRequest(str(path), dry_run=True))
    assert exc.value.code == 'MAX_SESSION_SECONDS must be finite and positive'
    result = _cli(tmp_path / 'invalid', path, '--dry-run', environment={'ETX_SESS_MAX_SECONDS': override})
    assert result.returncode == 1
    assert 'MAX_SESSION_SECONDS must be finite and positive' in (tmp_path / 'invalid/output.log').read_text()


def test_declarations_and_outcomes_are_immutable(tmp_path):
    reads, env = [], {'MARKER': 'original'}
    spec = JobSpec(name='immutable', cmd='true', lane='cpu', reads=reads, env=env)
    reads.append('mutated')
    env['MARKER'] = 'mutated'
    assert spec.reads == () and spec.env['MARKER'] == 'original'
    with pytest.raises(FrozenInstanceError):
        spec.cmd = 'false'
    session = JobSession([spec])
    with pytest.raises(FrozenInstanceError):
        session.max_seconds = 1
    assert session.jobs == (spec,)
    result, jobs, _, _ = _run(tmp_path, [dict(name='probe', lane='cpu', cmd='true')], gpus=())
    assert isinstance(result.jobs, tuple) and isinstance(jobs['probe'].log, str)
    with pytest.raises(FrozenInstanceError):
        result.returncode = 1
    with pytest.raises(FrozenInstanceError):
        jobs['probe'].status = 'fail'
    assert verify_job_record(result.record_directory)['returncode'] == 0


@pytest.mark.parametrize('kill_parent', [False, True])
def test_interrupted_api_caller_leaves_no_running_job(tmp_path, kill_parent):
    if kill_parent and sys.platform != 'linux':
        pytest.skip('parent-death signal uses Linux prctl')
    result = job_transport_interrupt_probe(tmp_path / 'interrupt', kill_parent=kill_parent)
    assert result['caller_returncode'] != 0
    assert not result['late_child_output']
    assert len(result['seals']) == 1
    assert result['seals'][0]['complete'] is False
    assert result['seals'][0]['jobs']['sleeper'] == 'fail'


def test_concurrent_schedulers_do_not_sweep_each_others_children(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    requests = []
    for name, command, timeout in (('timeout', 'sleep 30', .15), ('survivor', 'sleep .5; echo survived', 3)):
        session = JobSession((JobSpec(name, command, lane='cpu', timeout=timeout),), max_seconds=5)
        requests.append(JobSessionRequest(session, logdir=str(tmp_path / name), gpus=(), poll_s=.01))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_jobs, requests))
    assert [result.returncode for result in results] == [1, 0]
    assert results[0].jobs[0].status == 'timeout'
    assert results[1].jobs[0].status == 'ok'
    assert 'survived' in Path(results[1].jobs[0].log).read_text()


def test_timeout_does_not_kill_unrelated_caller_child(tmp_path):
    with observe_unrelated_child() as observations:
        result, jobs, _, _ = _run(tmp_path, [dict(name='slow', lane='cpu', cmd='sleep 30', timeout=.1)],
                                  gpus=(), poll_s=.01)
    assert result.returncode == 1 and jobs['slow'].status == 'timeout'
    assert len(observations) == 1 and observations[0]['survived_scheduler']


def test_legacy_loader_normalizes_environment_and_path_cwd(tmp_path):
    path = tmp_path / 'session.py'
    path.write_text('from pathlib import Path\nJOBS = [dict(name="probe", lane="cpu", '
                    f'cwd=Path({str(tmp_path)!r}), env={{"OMP_NUM_THREADS":4}}, '
                    'cmd=\'printf "%s\\\\n" "$OMP_NUM_THREADS"; pwd\')]\n')
    session = load_job_session(path)
    assert session.jobs[0].cwd == str(tmp_path) and session.jobs[0].env['OMP_NUM_THREADS'] == '4'
    request = JobSessionRequest(session, logdir=str(tmp_path / 'records'), gpus=(), poll_s=.01)
    result = run_jobs(request)
    assert result.returncode == 0
    log = Path(result.jobs[0].log).read_text()
    assert f'4\n{tmp_path}\n' in log


def test_only_filter_preserves_historical_dependency_selection(tmp_path):
    session = JobSession((
        JobSpec('build', 'exit 99', lane='cpu', writes=('built.txt',)),
        JobSpec('selected', 'echo selected', lane='cpu', reads=('build', 'built.txt')),
        JobSpec('remote', 'exit 99', lane='cpu', machine='B'),
    ), max_seconds=10)
    result = run_jobs(JobSessionRequest(session, only='selected|remote', gpus=(),
                                       logdir=str(tmp_path / 'records'), poll_s=.01))
    assert result.returncode == 0 and [job.name for job in result.jobs] == ['selected']
    assert result.jobs[0].status == 'ok'
    assert verify_job_record(result.record_directory)['returncode'] == 0


def test_lease_retries_reuse_budget_and_close_on_record_failure(tmp_path, monkeypatch):
    monkeypatch.delenv('ETX_SESS_MAX_SECONDS', raising=False)
    lease = tmp_path / 'lease.json'
    session = JobSession((JobSpec('cpu', 'true', lane='cpu'),), max_seconds=30,
                         lease_path=str(lease))
    occupied = tmp_path / 'not-a-directory'
    occupied.write_text('preserved')
    with pytest.raises(FileExistsError):
        run_jobs(JobSessionRequest(session, logdir=str(occupied)))
    first = lease.read_bytes()
    result = run_jobs(JobSessionRequest(session, logdir=str(tmp_path / 'records'), poll_s=.01))
    assert result.returncode == 0 and 0 < result.max_seconds < 30
    again = run_jobs(JobSessionRequest(session, logdir=str(tmp_path / 'records'), poll_s=.01))
    assert again.returncode == 0 and again.max_seconds < result.max_seconds
    assert lease.read_bytes() == first and occupied.read_text() == 'preserved'
    assert result.record_directory != again.record_directory


@pytest.mark.parametrize('controls', [
    {'only': 'cpu'}, {'gpus': (1,)}, {'gpus': ()}, {'machine': 'B'},
])
def test_leased_session_refuses_partial_or_other_machine_execution(tmp_path, controls):
    lease = tmp_path / 'lease.json'
    session = JobSession((JobSpec('cpu', 'true', lane='cpu'),), max_seconds=30,
                         lease_path=str(lease))
    with pytest.raises(SystemExit, match='complete graph'):
        run_jobs(JobSessionRequest(session, logdir=str(tmp_path / 'records'), **controls))
    assert not lease.exists() and not (tmp_path / 'records').exists()


def test_record_integrity_failure_is_not_a_successful_job_result(tmp_path):
    result, _, _, _ = _run(tmp_path, [dict(name='cpu', lane='cpu', cmd='echo original')], gpus=())
    assert result.returncode == 0
    Path(result.jobs[0].log).write_text('corrupted')
    verification = verify_job_record(result.record_directory)
    assert verification['returncode'] == 2
    assert verification['device_qualification'] is False
    assert any('MISMATCH cpu.log' in line for line in verification['messages'])


@pytest.mark.parametrize('path', sorted((ROOT / 'examples/sessions').rglob('*.py')),
                         ids=lambda path: path.name)
def test_existing_session_declarations_are_supported(path):
    session = load_job_session(path)
    assert all(isinstance(job, JobSpec) for job in session.jobs)


def test_dotted_legacy_names_are_safe_log_basenames(tmp_path):
    session = tmp_path / 'session.py'
    session.write_text('JOBS = [dict(name="dl_qwen2.5-1.5b", lane="cpu", cmd="true")]\n')
    declaration = load_job_session(session)
    assert declaration.jobs[0].name == 'dl_qwen2.5-1.5b'
    result = run_jobs(JobSessionRequest(declaration, logdir=str(tmp_path / 'records'), gpus=()))
    assert result.returncode == 0
    assert Path(result.jobs[0].log).name == 'dl_qwen2.5-1.5b.log'


@pytest.mark.parametrize('fields', [
    {'name': '../escape'}, {'name': '.'}, {'name': '..'}, {'lane': 'other'}, {'gpu': -1}, {'gpu': True},
    {'timeout': 0}, {'timeout': float('inf')}, {'quiet_s': float('nan')},
    {'reads': ['?']}, {'writes': ['']}, {'env': {'BAD=KEY': 'value'}},
    {'env': {'MARKER': 1}}, {'schema_version': 2},
])
def test_invalid_job_declarations_are_refused(fields):
    with pytest.raises(ValueError):
        JobSpec(**{'name': 'probe', 'cmd': 'true', **fields})
