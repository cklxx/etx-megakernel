"""etx.tools.gpu_jobs: two GPU lanes run in parallel, pins hold, failed dependencies skip, gates decide, timeouts
kill, every job's full output lands in its log."""
import io
import json
import os
from pathlib import Path
import shlex
import sys
import textwrap
import time

import pytest

from etx.tools import gpu_jobs as gj


def _run(tmp_path, specs, gpus=(0, 1), cpu_slots=2):
    jobs = [gj.Job(cwd=str(tmp_path), **s) for s in specs]
    out = io.StringIO()
    r = gj.Runner(jobs, list(gpus), cpu_slots, tmp_path / "logs", progress_s=0.3, out=out, poll_s=0.05)
    t0 = time.time()
    rc = r.run()
    return rc, {j.name: j for j in jobs}, out.getvalue(), time.time() - t0


def test_parallel_pins_deps_gates_timeouts(tmp_path):
    specs = [
        dict(name="a", cmd="echo GPU=$GPU HIP=$HIP_VISIBLE_DEVICES; sleep 0.5"),
        dict(name="b", cmd="echo GPU=$GPU; sleep 0.5"),
        dict(name="pin1", cmd="echo GPU=$GPU", gpu=1, reads=["a", "b"]),
        dict(name="build", cmd="echo built > $ETX_SESS_LOGDIR/built.txt", lane="cpu", writes=["built.txt"]),
        dict(name="bad", cmd="echo oops; exit 3", lane="cpu"),
        dict(name="needs_bad", cmd="echo never", reads=["bad"]),
        dict(name="gated_yes", cmd="echo ran", reads=["built.txt"]),
        dict(name="gated_no", cmd="echo ran", reads=["never-written.txt"]),
        dict(name="slow", cmd="echo start; sleep 30", timeout=0.6, summary=["start"], lane="cpu"),
        dict(name="any", cmd="echo done", reads=["?bad"]),
    ]
    rc, j, out, dt = _run(tmp_path, specs)
    assert rc == 1                                           # "bad" failed and "slow" timed out
    assert j["a"].status == j["b"].status == "ok"
    assert {j["a"].on_gpu, j["b"].on_gpu} == {0, 1}          # both GPUs used at once
    assert abs(j["a"].start - j["b"].start) < 0.3
    assert j["pin1"].on_gpu == 1 and "GPU=1" in (tmp_path / "logs/pin1.log").read_text()
    assert "HIP=" + str(j["a"].on_gpu) in (tmp_path / "logs/a.log").read_text()
    assert j["needs_bad"].status == "skipped" and j["any"].status == "ok"
    assert j["gated_yes"].status == "ok" and j["gated_no"].status == "skipped"
    assert j["slow"].status == "timeout" and dt < 15
    assert "start" in out and "session done" in out
    st = json.loads((tmp_path / "logs/status.json").read_text())
    assert {s["name"]: s["status"] for s in st}["bad"] == "fail"


def test_reads_wait_for_their_writer_and_the_gpu_lane_keeps_list_order(tmp_path):
    """The d1d5 loss (2bb3597) and R07: a reader starts only after its writer succeeded and the file exists; a GPU job
    never overtakes an earlier pending one; a skip advances the lane; a missing declared output fails its job."""
    rc, j, _, _ = _run(tmp_path, [
        dict(name="prep_pack", lane="cpu", cmd="sleep .4; echo m > $ETX_SESS_LOGDIR/data.manifest", writes=["data.manifest"]),
        dict(name="build_bad", lane="cpu", cmd="exit 1", writes=["b_bad/run"]),
        dict(name="build_slow", lane="cpu", cmd="sleep .6; mkdir -p $ETX_SESS_LOGDIR/b_s; touch $ETX_SESS_LOGDIR/b_s/run",
             writes=["b_s/run"]),
        dict(name="liar", lane="cpu", cmd="true", writes=["promised.txt"]),
        dict(name="g_bad", reads=["data.manifest", "b_bad/run"], cmd="echo never"),
        dict(name="g_slow", reads=["data.manifest", "b_s/run"], cmd="test -f $ETX_SESS_LOGDIR/data.manifest"),
        dict(name="g_ready", reads=["data.manifest"], cmd="test -f $ETX_SESS_LOGDIR/data.manifest"),
        dict(name="g_lie", reads=["promised.txt"], cmd="echo never"),
    ], gpus=(0,))
    assert j["g_bad"].status == "skipped" and "build_bad fail" in j["g_bad"].note
    assert j["g_slow"].status == j["g_ready"].status == "ok"
    assert j["g_slow"].start >= j["build_slow"].end and j["g_ready"].start >= j["g_slow"].start   # no overtaking
    assert j["liar"].status == "fail" and "declared output missing" in j["liar"].note
    assert j["g_lie"].status == "skipped"


def test_load_and_dry_run(tmp_path, capsys):
    f = tmp_path / "s.py"
    f.write_text('JOBS = [dict(name="x", cmd="true"), dict(name="y", cmd="true", reads="x", machine="B")]\n')
    jobs = gj.load_jobs(f)
    assert jobs[1].reads == ["x"]
    assert gj.main([str(f), "--dry-run", "--machine", "B"]) == 0
    assert "y" in capsys.readouterr().out
    f.write_text('JOBS = [dict(name="x", cmd="true", after=["y"])]\n')         # the old edge fields are gone
    with pytest.raises(SystemExit, match="after"):
        gj.load_jobs(f)
    f.write_text('JOBS = [dict(name="x", cmd="true", reads=["typo"])]\n')
    with pytest.raises(SystemExit, match="unknown dependency"):
        gj.load_jobs(f)


def test_child_uses_runner_interpreter_environment(tmp_path, monkeypatch):
    stale = tmp_path / 'system-bin'
    stale.mkdir()
    python = stale / 'python3'
    python.write_text('#!/bin/sh\necho WRONG-INTERPRETER\nexit 99\n')
    python.chmod(0o755)
    inherited_path = str(stale) + os.pathsep + os.environ['PATH']
    probe = 'import os, sys; print(sys.prefix); print(os.environ["JOB_MARKER"])'
    commands = [f'python3 -u -c {shlex.quote(probe)}',
                f'env python3 -u -c {shlex.quote(probe)}',
                f'{shlex.quote(str(python))}']
    spawned = []
    popen = gj.subprocess.Popen

    def capture(args, **kwargs):
        spawned.append((args, kwargs['env']))
        return popen(args, **kwargs)

    monkeypatch.setattr(gj.subprocess, 'Popen', capture)
    rc, jobs, _, _ = _run(tmp_path, [
        dict(name=f'probe{i}', lane='cpu', cmd=cmd,
             env=dict(PATH=inherited_path, JOB_MARKER='preserved'))
        for i, cmd in enumerate(commands)
    ], gpus=())
    assert rc == 1  # the explicitly selected interpreter still exits 99
    assert [j.status for j in jobs.values()] == ['ok', 'ok', 'fail']
    for i in (0, 1):
        assert f'{sys.prefix}\npreserved\n' in jobs[f'probe{i}'].log.read_text()
    assert 'WRONG-INTERPRETER' in jobs['probe2'].log.read_text()
    for (args, env), cmd in zip(spawned, commands):
        assert args == [sys.executable, '-I', gj.sproc.__file__, '--', 'bash', '-c', cmd]   # under the job shim
        assert env['PATH'] == str(Path(sys.executable).parent) + os.pathsep + inherited_path
        assert env['JOB_MARKER'] == 'preserved'


def test_pin_folds_onto_one_gpu(tmp_path):
    rc, j, out, _ = _run(tmp_path, [dict(name="p1", cmd="echo GPU=$GPU", gpu=1), dict(name="p0", cmd="echo GPU=$GPU", gpu=0)], gpus=(0,))
    assert rc == 0 and j["p1"].on_gpu == 0 and "absent" in j["p1"].note


def test_live_status_before_report_job(tmp_path):
    import shlex
    import sys
    reader = "import json; from pathlib import Path; s={x['name']:x['status'] for x in json.loads(Path('logs/status.json').read_text())}; assert s['bad']=='fail'; assert s['skip']=='skipped'; assert s['slow']=='timeout'; print('LIVE-STATUS OK')"
    rc, jobs, _, _ = _run(tmp_path, [
        dict(name="bad", lane="cpu", cmd="exit 3"),
        dict(name="skip", lane="cpu", reads=["bad"], cmd="exit 99"),
        dict(name="slow", lane="cpu", cmd="sleep 10", timeout=.1),
        dict(name="report", lane="final",
             cmd=f"{shlex.quote(sys.executable)} -c {shlex.quote(reader)}"),
    ])
    assert rc == 1 and jobs["report"].status == "ok"
    assert "LIVE-STATUS OK" in jobs["report"].log.read_text()


def test_status_atomic_replace(tmp_path, monkeypatch):
    jobs = [gj.Job(name="x", cmd="true")]
    runner = gj.Runner(jobs, [], 1, tmp_path)
    runner._write_status()
    replace = gj.os.replace
    calls = []
    def checked_replace(src, dst):
        assert json.loads(dst.read_text())[0]["status"] == "pending"
        assert json.loads(src.read_text())[0]["status"] == "fail"
        calls.append(src)
        replace(src, dst)
    monkeypatch.setattr(gj.os, "replace", checked_replace)
    jobs[0].status = "fail"
    runner._write_status()
    assert len(calls) == 1 and not calls[0].exists()
    assert json.loads((tmp_path / "status.json").read_text())[0]["status"] == "fail"


def test_every_session_gets_a_new_sealed_record(tmp_path):
    import subprocess
    from etx.session import record
    session, root = tmp_path / "session.py", tmp_path / "sess"
    env = dict(os.environ, PYTHONPATH=str(Path(gj.__file__).resolve().parents[2]))
    for marker in ("first", "second"):          # same job name, same --logdir: two records, the first untouched
        session.write_text(f'JOBS = [dict(name="probe", lane="cpu", cmd="echo {marker}")]\n')
        result = subprocess.run([sys.executable, "-m", "etx.tools.gpu_jobs", str(session), "--gpus", "",
                                 "--logdir", str(root)], env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        if marker == "first":
            first = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    records = sorted(p for p in root.iterdir())
    assert len(records) == 2 and all(record.verify(r)[0] == 0 for r in records)
    assert {p: p.read_bytes() for p in records[0].rglob("*") if p.is_file()} == first
    assert "second" in (records[1] / "probe.log").read_text()
    assert json.loads((records[1] / "SEAL").read_text())["jobs"] == {"probe": "ok"}


@pytest.mark.parametrize('lane', ['cpu', 'gpu'])
def test_job_environment_names_the_record_and_a_used_directory_is_refused(tmp_path, monkeypatch, lane):
    logdir = tmp_path / 'logs'
    monkeypatch.setenv('ETX_SESS_LOGDIR', '/stale/parent')
    job = gj.Job(name='probe', lane=lane, cwd=str(tmp_path), env={'ETX_SESS_LOGDIR': '/stale/job'},
                 cmd='printf "%s\\n" "$ETX_SESS_LOGDIR"; test -f "$ETX_SESS_LOGDIR/status.json"')
    assert gj.Runner([job], [0], 1, logdir, out=io.StringIO(), poll_s=.01).run() == 0
    assert job.log.read_text().splitlines()[-1] == str(logdir)
    with pytest.raises(SystemExit, match='not empty'):
        gj.Runner([gj.Job(name='again', cmd='true')], [0], 1, logdir, out=io.StringIO()).run()


def test_session_hard_stop_kills_all_lanes_and_skips_tail(tmp_path):
    jobs = [gj.Job(name=name, cmd='echo started; sleep 30', lane=lane, cwd=str(tmp_path))
            for name, lane in [('cpu', 'cpu'), ('gpu', 'gpu')]]
    jobs.append(gj.Job(name='tail', cmd='echo must-not-run', lane='final'))
    runner = gj.Runner(jobs, [0], 1, tmp_path, poll_s=.01, out=io.StringIO(), max_seconds=.15)
    start = time.monotonic()
    assert runner.run() == 1
    assert time.monotonic() - start < 3
    assert [j.status for j in jobs] == ['timeout', 'timeout', 'skipped']
    assert all(j.note == 'session hard stop' for j in jobs)
    assert all(j.proc.poll() is not None for j in jobs[:2])
    assert not (tmp_path / 'tail.log').exists()
    assert [j['status'] for j in json.loads((tmp_path / 'status.json').read_text())] == [
        'timeout', 'timeout', 'skipped']


@pytest.mark.skipif(not os.path.isdir('/proc/self/task'), reason='descendant sweep reads /proc (Linux, the box)')
def test_hard_stop_kills_children_forked_during_the_stop(tmp_path):
    # review 09, R15: a job that keeps forking (a compiler under GNU timeout, its own process group) must leave no
    # survivor once the runner reports the session done
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
    job = gj.Job(name='forker', cmd=f'timeout --kill-after=5s 5s {sys.executable} {child}; echo after', lane='cpu',
                 cwd=str(tmp_path), timeout=5)
    runner = gj.Runner([job], [0], 1, tmp_path / 'logs', poll_s=.005, out=io.StringIO(), max_seconds=.8)
    assert runner.run() == 1
    time.sleep(2.5)
    assert job.status == 'timeout' and not list(tmp_path.glob('late-*'))


@pytest.mark.parametrize('limit', ['0', '-1', "float('inf')", "float('nan')", "'2400'"])
def test_invalid_session_budget_refused(tmp_path, monkeypatch, limit):
    monkeypatch.delenv('ETX_SESS_MAX_SECONDS', raising=False)
    path = tmp_path / 'session.py'
    path.write_text(f'JOBS = []\nMAX_SESSION_SECONDS = {limit}\n')
    with pytest.raises(SystemExit, match='finite and positive'):
        gj.load_jobs(path)


@pytest.mark.parametrize('override,expected', [(None, 8400), ('6000', 6000)])
def test_session_budget_override(tmp_path, monkeypatch, override, expected):
    monkeypatch.delenv('ETX_SESS_MAX_SECONDS', raising=False)
    if override is not None:
        monkeypatch.setenv('ETX_SESS_MAX_SECONDS', override)
    path = tmp_path / 'session.py'
    path.write_text('JOBS = []\nMAX_SESSION_SECONDS = 8400\n')
    assert gj.load_session(path)[1] == expected


@pytest.mark.parametrize('override', ['nope', '', '0', '-1', 'inf', 'nan'])
def test_invalid_session_budget_override_refused(tmp_path, monkeypatch, override):
    monkeypatch.setenv('ETX_SESS_MAX_SECONDS', override)
    path = tmp_path / 'session.py'
    path.write_text('JOBS = []\nMAX_SESSION_SECONDS = 8400\n')
    with pytest.raises(SystemExit) as exc:
        gj.main([str(path), '--dry-run'])
    assert exc.value.code == 'MAX_SESSION_SECONDS must be finite and positive'
