"""Small synthetic gpu_jobs sessions, following test_tile_ab.py's tmp_path convention."""
import json
import subprocess
import pytest

from examples.vllm_llm import nvref_report as nr
from etx.experiments.jobs import Runner, load_jobs



def write(d, name, body, gpu="0"):
    (d / f"{name}.log").write_text(f"# job {name} gpu={gpu} cwd=/x\n"
                                  "# echo 'RESULT matched=32 gen=32'; echo 'CHECK-DSL OK'\n" + body + "\n")


def etx(ms, arm="fused", matched=32, count=31):
    mode = "ETX megakernel (vLLM kernels imported)" if arm == "fused" else "unfused (same imported kernels, one launch per grid, HIP graph)"
    # A faster warm-up must not influence the final measured run.
    steps = "  step 0: 1.000 ms wall\n" + "".join(f"  step {t}: 1.000 ms wall\n" for t in range(1024, 1055))
    steps += "run 0: 32/32 generated tokens match HF greedy\n  step 0: 2.000 ms wall\n"
    steps += "".join(f"  step {t}: {ms:.3f} ms wall\n" for t in range(1024, 1024 + count))
    return ("ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=\n" + steps
            + f"run 1: {matched}/32 generated tokens match HF greedy\nmode: {mode}\n"
            + f"RESULT model=/m requests=1 matched={matched} gen=32 median_ms={ms - .15:.3f} wall_ms={ms:.3f} matched_each={matched}")


def vllm(ms=5.0, aiter=0):
    return (f"VLLM label=default aiter={aiter} custom_ops=None eager=0 cudagraph=None compile_mode=None "
            f"batch=1 version=0.27.1 context=1024 median_ms={ms:.3f} per_token=[{ms}] first_divergence=32/32")


@pytest.fixture
def session(tmp_path):
    write(tmp_path, "gate_all", etx(4.8) + "\nCHECK-DSL OK: 0 values beyond 2 bf16 ulps\nGATE-OK")
    write(tmp_path, "gate_unf", etx(5.2, "graph") + "\nGATE-OK")
    write(tmp_path, "check_imp", "CHECK OK: 0 launch(es) differ")
    write(tmp_path, "golden_sha", "/models/golden_c1024.txt: OK", "None")
    write(tmp_path, "record", "MI300X VF\n7.2.4\nLinux test\nHF_REV abcdef", "None")
    write(tmp_path, "docker_pull", "IMAGE vllm/image@sha256:abcdef", "None")
    for r in range(1, 6):
        write(tmp_path, f"vllm_r{r}", vllm(5 + r * .001))
        write(tmp_path, f"etx_r{r}", etx(4.8 + r * .001))
        write(tmp_path, f"etxu_r{r}", etx(5.2 + r * .001, "graph"))
    return tmp_path






















@pytest.mark.parametrize("deltas, expected", [([-0.05]*5, "LEAD"), ([0.05]*5, "LOSS"),
    ([0.01]*5, "TIE"), ([-.2, -.1, -.1, -.1, .1], "INSUFFICIENT")])
def test_verdict(deltas, expected):
    assert nr.verdict(deltas) == expected




def test_session_commands_and_order():
    jobs = load_jobs("examples/sessions/nvref.py")
    original = load_jobs("examples/sessions/tiles.py")
    assert [j.__dict__ for j in jobs[:5]] == [j.__dict__ for j in original[:5]]
    for j in jobs:
        assert subprocess.run(["bash", "-n", "-c", j.cmd], capture_output=True).returncode == 0
        assert j.lane == "cpu" or j.gpu == 0
    names = [j.name for j in jobs]
    by = {j.name: j for j in jobs}
    preceding = ["gate_all", "gate_unf", "check_imp", "check_imp1"]
    for r in range(1, 6):
        for arm in (["vllm", "etx", "etxu"] if r % 2 else ["etxu", "etx", "vllm"]):
            j = next(j for j in jobs if j.name == f"{arm}_r{r}")
            assert j.lane == "gpu" and all(by[d].lane == "gpu" for d in preceding)   # the GPU lane keeps the order
            assert "golden_sha" in j.deps()
            assert all(names.index(dep) < names.index(j.name) for dep in preceding)
            expected = {"vllm": ["golden_sha", "docker_pull"],
                        "etx": ["golden_sha", "build_all", "gate_all"],
                        "etxu": ["golden_sha", "build_all", "gate_all", "gate_unf"]}
            assert j.deps() == expected[arm]
            preceding.append(j.name)














def test_check_build_and_skipped_neighbour_barrier(tmp_path):
    jobs = load_jobs("examples/sessions/nvref.py")
    by_name = {j.name: j for j in jobs}
    assert "ETX_VL_CHECK=1 ETX_VL_XCD=0" in by_name["build_chk1"].cmd
    assert "--check 2" in by_name["check_imp1"].cmd
    assert "CHECK OK: 0 launch(es) differ" in by_name["check_imp1"].cmd
    runner = Runner(jobs, [0], 2, tmp_path)
    for job in jobs:
        job.status = "ok"
    by_name["gate_unf"].status = "fail"
    for r in range(1, 6):
        assert runner._state(by_name[f"etxu_r{r}"]) == "skip"
        by_name[f"etxu_r{r}"].status = "skipped"
    # the barrier between neighbours is the GPU lane: one GPU, strictly in list order, a skip advances it
    assert by_name["etx_r1"].gpu == by_name["etx_r2"].gpu == 0 and jobs.index(by_name["etx_r1"]) < jobs.index(by_name["etx_r2"])
    for name in ("hostprof", "etx_noprog", "vllm_aiter_r1", "vllm_aiter_r2", "vllm_aiter_r3", "report"):
        assert runner._state(by_name[name]) == "ready"
        by_name[name].status = "fail"













def test_tail_needs_builds_not_gates(tmp_path):
    jobs = load_jobs("examples/sessions/nvref.py")
    runner = Runner(jobs, [0], 2, tmp_path)
    by_name = runner.by_name
    for j in jobs:
        j.status = "ok"
    for name in ("gate_all", "gate_unf"):
        by_name[name].status = "fail"
    for name in ("hostprof", "etx_noprog", "vllm_aiter_r1", "vllm_aiter_r2", "vllm_aiter_r3", "report"):
        j = by_name[name]
        assert runner._state(j) == "ready"
        assert "gate_all" not in j.deps() and "gate_unf" not in j.deps()
        for dep in j.deps():
            by_name[dep].status = "fail"
            assert runner._state(j) == "skip"
            by_name[dep].status = "ok"


