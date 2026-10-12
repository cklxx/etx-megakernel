"""etx.tools.gpu_jobs: two GPU lanes run in parallel, pins hold, failed dependencies skip, gates decide, timeouts
kill, every job's full output lands in its log."""
import io
import json
import os
from pathlib import Path
import shlex
import sys
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
        dict(name="pin1", cmd="echo GPU=$GPU", gpu=1, after=["a", "b"]),
        dict(name="build", cmd="echo built; echo GATE_OK", lane="cpu"),
        dict(name="bad", cmd="echo oops; exit 3", lane="cpu"),
        dict(name="needs_bad", cmd="echo never", after=["bad"]),
        dict(name="gated_yes", cmd="echo ran", gate=("build", "GATE_OK")),
        dict(name="gated_no", cmd="echo ran", gate=("build", "NOPE")),
        dict(name="slow", cmd="echo start; sleep 30", timeout=0.6, summary=["start"]),
        dict(name="any", cmd="echo done", after_any=["bad"]),
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


def test_load_and_dry_run(tmp_path, capsys):
    f = tmp_path / "s.py"
    f.write_text('JOBS = [dict(name="x", cmd="true"), dict(name="y", cmd="true", after="x", machine="B")]\n')
    jobs = gj.load_jobs(f)
    assert jobs[1].after == ["x"]
    assert gj.main([str(f), "--dry-run", "--machine", "B"]) == 0
    assert "y" in capsys.readouterr().out


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
        assert args == ['bash', '-c', cmd]
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
        dict(name="skip", lane="cpu", after=["bad"], cmd="exit 99"),
        dict(name="slow", lane="cpu", cmd="sleep 10", timeout=.1),
        dict(name="report", lane="cpu", after_any=["bad", "skip", "slow"],
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


def test_follow_on_session_preserves_parent_logs(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    session = tmp_path / "session.py"
    logdir = tmp_path / "logs"
    env = dict(os.environ, PYTHONPATH=str(Path(gj.__file__).resolve().parents[2]))

    def start(marker):
        # Same job name deliberately exercises job-log preservation as well.
        session.write_text(f'JOBS = [dict(name="probe", lane="cpu", cmd="echo {marker}")]\n')
        result = subprocess.run(
            [sys.executable, "-m", "etx.tools.gpu_jobs", str(session),
             "--gpus", "", "--logdir", str(logdir)],
            env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    start("parent")
    originals = {name: (logdir / name).read_bytes()
                 for name in ("runner.log", "status.json", "probe.log")}
    output = start("follow_on")
    for name, contents in originals.items():
        assert (logdir / name).read_bytes() == contents
    children = list(logdir.glob("session-*/runner.log"))
    assert len(children) == 1
    child = children[0].parent
    assert str(child) in output
    assert "session done" in children[0].read_text()
    assert "follow_on" in (child / "probe.log").read_text()
    assert json.loads((child / "status.json").read_text())[0]["status"] == "ok"


@pytest.mark.parametrize('reuse', [False, True])
@pytest.mark.parametrize('lane', ['cpu', 'gpu'])
def test_job_environment_uses_actual_logdir(tmp_path, monkeypatch, reuse, lane):
    logdir = tmp_path / 'logs'
    logdir.mkdir()
    if reuse:
        (logdir / 'status.json').write_text('[]')
    monkeypatch.setenv('ETX_SESS_LOGDIR', '/stale/parent')
    job = gj.Job(name='probe', lane=lane, cwd=str(tmp_path),
                 env={'ETX_SESS_LOGDIR': '/stale/job'},
                 cmd='printf "%s\\n" "$ETX_SESS_LOGDIR"; test -f "$ETX_SESS_LOGDIR/status.json"')
    runner = gj.Runner([job], [0], 1, logdir, out=io.StringIO(), poll_s=.01)
    assert runner.run() == 0
    assert job.log.read_text().splitlines()[-1] == str(runner.logdir)
    if reuse:
        assert runner.logdir.parent == logdir
        assert runner.logdir.name.startswith('session-')
        assert (logdir / 'status.json').read_text() == '[]'
    else:
        assert runner.logdir == logdir


def test_session_hard_stop_kills_all_lanes_and_skips_tail(tmp_path):
    jobs = [gj.Job(name=name, cmd='echo started; sleep 30', lane=lane, cwd=str(tmp_path))
            for name, lane in [('cpu', 'cpu'), ('gpu', 'gpu')]]
    jobs.append(gj.Job(name='tail', cmd='echo must-not-run', after_any=['cpu', 'gpu']))
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
