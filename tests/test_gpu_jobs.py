"""etx.tools.gpu_jobs: two GPU lanes run in parallel, pins hold, failed dependencies skip, gates decide, timeouts
kill, every job's full output lands in its log."""
import io
import json
import time

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


def test_pin_folds_onto_one_gpu(tmp_path):
    rc, j, out, _ = _run(tmp_path, [dict(name="p1", cmd="echo GPU=$GPU", gpu=1), dict(name="p0", cmd="echo GPU=$GPU", gpu=0)], gpus=(0,))
    assert rc == 0 and j["p1"].on_gpu == 0 and "absent" in j["p1"].note
