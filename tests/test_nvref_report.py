"""Small synthetic gpu_jobs sessions, following test_tile_ab.py's tmp_path convention."""
import json
import subprocess
import pytest

from examples.vllm_llm import nvref_report as nr
from etx.tools.gpu_jobs import Runner, load_jobs


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
    write(tmp_path, "golden_sha", "/home/llm/golden_c1024.txt: OK", "None")
    write(tmp_path, "record", "MI300X VF\n7.2.4\nLinux test\nHF_REV abcdef", "None")
    write(tmp_path, "docker_pull", "IMAGE vllm/image@sha256:abcdef", "None")
    for r in range(1, 6):
        write(tmp_path, f"vllm_r{r}", vllm(5 + r * .001))
        write(tmp_path, f"etx_r{r}", etx(4.8 + r * .001))
        write(tmp_path, f"etxu_r{r}", etx(5.2 + r * .001, "graph"))
    return tmp_path


def test_passing(session, capsys):
    assert nr.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "VERDICT LEAD | import-check OK | image vllm/image@sha256:abcdef" in out
    assert "median=-0.200 spread=0.000" in out
    assert "FUSION (unfused - fused) median=+0.400" in out
    assert "IMPORT-CHECK OK" in out
    assert "4.801, 4.802, 4.803, 4.804, 4.805" in out
    assert out.count("third party eliovp.com 2025-09-26, not reproduced") == 4
    assert "BASIS-MISMATCH" not in out


@pytest.mark.parametrize("arm", ["etx", "etxu", "vllm"])
def test_31_of_32_refuses(session, capsys, arm):
    write(session, f"{arm}_r3", vllm().replace("32/32", "31/32") if arm == "vllm"
          else etx(4.8, "graph" if arm == "etxu" else "fused", matched=31))
    if arm == "etxu":
        assert nr.main([str(session)]) == 0
        out = capsys.readouterr().out
        assert "FUSION unavailable (etxu_r3:" in out and "32/32" in out
        assert "VERDICT LEAD" in out and "FUSION (unfused" not in out
        return
    assert nr.main([str(session)]) == 2
    out = capsys.readouterr().out
    assert "32/32" in out and "VERDICT NOT-COMPLETE" in out
    assert "| source |" not in out


def test_missing_run_refuses(session, capsys):
    (session / "vllm_r5.log").unlink()
    assert nr.main([str(session)]) == 2
    assert "vllm_r5: missing run/job log" in capsys.readouterr().out


def test_last_repeat_without_progress_refuses(session, capsys):
    write(session, "etx_r5", etx(4.8, count=0))
    assert nr.main([str(session)]) == 2
    assert "got n=0" in capsys.readouterr().out


def test_comment_is_not_gate_evidence(session, capsys):
    write(session, "gate_all", etx(4.8) + "\nGATE-OK")
    assert nr.main([str(session)]) == 2
    assert "missing CHECK-DSL OK" in capsys.readouterr().out


def test_graph_off_refuses(session, capsys):
    write(session, "etxu_r1", etx(5.2, "graph").replace("ETX_UNFUSED_NOGRAPH=", "ETX_UNFUSED_NOGRAPH=1"))
    assert nr.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "graph-off" in out and "FUSION unavailable" in out
    assert "VERDICT LEAD" in out and "FUSION (unfused" not in out


@pytest.mark.parametrize("count", [30, 32])
def test_wrong_step_count_refuses(session, capsys, count):
    write(session, "etx_r5", etx(4.8, count=count))
    assert nr.main([str(session)]) == 2
    out = capsys.readouterr().out
    assert "expected 31 measured steps" in out and "| source |" not in out


@pytest.mark.parametrize("gate", ["gate_all", "gate_unf", "check_imp"])
def test_failed_gate(session, capsys, gate):
    write(session, gate, "CHECK FAILED: 1 launch(es) differ")
    if gate == "gate_unf":
        assert nr.main([str(session)]) == 0
        out = capsys.readouterr().out
        assert "FUSION unavailable (gate_unf:" in out
        assert "VERDICT LEAD" in out and "FUSION (unfused" not in out
        return
    assert nr.main([str(session)]) == 2
    assert "VERDICT NOT-COMPLETE" in capsys.readouterr().out


def test_status_failure_after_result(session, capsys):
    (session / "status.json").write_text(json.dumps([dict(name="etx_r5", status="timeout")]))
    assert nr.main([str(session)]) == 2
    assert "etx_r5: job timeout" in capsys.readouterr().out


def test_import_na_and_basis_mismatch(session, capsys):
    write(session, "check_imp", "ERROR: --check needs a plan without per-XCD replicas (build without ETX_VL_XCD)")
    write(session, "etx_r1", etx(4.8).replace("wall_ms=4.800", "wall_ms=4.850"))
    assert nr.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "IMPORT-CHECK n/a (host.hip:515)" in out
    assert "BASIS-MISMATCH etx_r1 mean_ms=4.800 RESULT wall_ms=4.850" in out


@pytest.mark.parametrize("deltas, expected", [([-0.05]*5, "LEAD"), ([0.05]*5, "LOSS"),
    ([0.01]*5, "TIE"), ([-.2, -.1, -.1, -.1, .1], "INSUFFICIENT")])
def test_verdict(deltas, expected):
    assert nr.verdict(deltas) == expected


def test_aiter_recompute(session, capsys):
    for r in range(1, 4):
        write(session, f"vllm_aiter_r{r}", vllm(4.6, aiter=1))
    assert nr.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "VERDICT LEAD |" in out
    assert "VERDICT LOSS against AITER (unpaired median reference); default=LEAD" in out


def test_session_commands_and_order():
    jobs = load_jobs("examples/sessions/nvref.py")
    original = load_jobs("examples/sessions/tiles.py")
    assert [j.__dict__ for j in jobs[:5]] == [j.__dict__ for j in original[:5]]
    for j in jobs:
        assert subprocess.run(["bash", "-n", "-c", j.cmd], capture_output=True).returncode == 0
        assert j.lane == "cpu" or j.gpu == 0
    names = [j.name for j in jobs]
    preceding = ["gate_all", "gate_unf", "check_imp", "check_imp1"]
    for r in range(1, 6):
        for arm in (["vllm", "etx", "etxu"] if r % 2 else ["etxu", "etx", "vllm"]):
            j = next(j for j in jobs if j.name == f"{arm}_r{r}")
            assert j.after_any == preceding
            assert "golden_sha" in j.after
            assert all(names.index(dep) < names.index(j.name) for dep in preceding)
            expected = {"vllm": ["golden_sha", "docker_pull"],
                        "etx": ["golden_sha", "build_all", "gate_all"],
                        "etxu": ["golden_sha", "build_all", "gate_all", "gate_unf"]}
            assert j.after == expected[arm]
            preceding.append(j.name)


@pytest.mark.parametrize("missing", ["gate_unf", "etxu_r1", "etxu_r5", "all"])
def test_fusion_missing_still_verdict(session, capsys, missing):
    names = [f"etxu_r{r}" for r in range(1, 6)] if missing == "all" else [missing]
    for name in names:
        (session / f"{name}.log").unlink()
    assert nr.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "| vLLM 0.27.1 default |" in out and "| ETX fused, 4 tile flags |" in out
    assert "FUSION unavailable (" in out and "missing run/job log" in out
    assert "FUSION (unfused" not in out and "| ETX unfused, HIP graph |" not in out
    assert "DELTA r1..r5" in out and "VERDICT LEAD" in out


def test_check_imp1_ok_wins_over_legacy_na(session, capsys):
    """The single-XCD check supplies the evidence; the legacy per-XCD refusal is not reported."""
    write(session, "check_imp1", "CHECK OK: 0 launch(es) differ")
    write(session, "check_imp", "ERROR: --check needs a plan without per-XCD replicas (build without ETX_VL_XCD)")
    assert nr.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert "IMPORT-CHECK OK" in out


def test_legacy_check_failed_refuses_even_when_new_check_passed(session, capsys):
    """A differing launch in either build is a correctness failure, not an unavailable check."""
    write(session, "check_imp1", "CHECK OK: 0 launch(es) differ")
    write(session, "check_imp", "CHECK FAILED: 1 launch(es) differ")
    assert nr.main([str(session)]) == 2
    out = capsys.readouterr().out
    assert "REFUSED: check_imp: CHECK FAILED" in out
    assert "DELTA" not in out


def test_check_imp1_failure_refuses(session, capsys):
    write(session, "check_imp1", "CHECK FAILED: 1 launch(es) differ")
    write(session, "check_imp", "ERROR: --check needs a plan without per-XCD replicas (build without ETX_VL_XCD)")
    (session / "status.json").write_text(json.dumps([dict(name="check_imp1", status="fail")]))
    assert nr.main([str(session)]) == 2
    out = capsys.readouterr().out
    assert out.startswith("REFUSED: check_imp1:")
    assert out.endswith("No performance numbers.\nVERDICT NOT-COMPLETE\n")
    assert "| source |" not in out and "DELTA" not in out
    assert "CHECK FAILED" in out


@pytest.mark.parametrize("body,status", [
    ("CHECK OK: 0 launch(es) differ", "timeout"),
    ("ERROR: --check needs a plan without per-XCD replicas (build without ETX_VL_XCD)", "fail"),
    ("", "skipped"),
])
def test_check_imp1_incomplete_proceeds_with_not_met(session, capsys, body, status):
    """An unavailable check keeps the numbers and marks the gate NOT MET on the verdict line."""
    write(session, "check_imp1", body)
    write(session, "check_imp", "ERROR: --check needs a plan without per-XCD replicas (build without ETX_VL_XCD)")
    (session / "status.json").write_text(json.dumps([dict(name="check_imp1", status=status)]))
    assert nr.main([str(session)]) == 0
    out = capsys.readouterr().out
    assert f"IMPORT-CHECK NOT MET (check_imp1: {status})" in out
    assert "DELTA r1..r5" in out
    assert f"VERDICT LEAD | import-check NOT MET (check_imp1: {status})" in out


@pytest.mark.parametrize("arm", ["vllm", "etx"])
def test_missing_required_with_missing_fusion_refuses(session, capsys, arm):
    (session / "etxu_r1.log").unlink()
    (session / f"{arm}_r5.log").unlink()
    assert nr.main([str(session)]) == 2
    out = capsys.readouterr().out
    assert "No performance numbers." in out and "VERDICT NOT-COMPLETE" in out
    assert "| source |" not in out and "DELTA" not in out


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
        assert runner._deps_state(by_name[f"etxu_r{r}"]) == "skip"
        by_name[f"etxu_r{r}"].status = "skipped"
    by_name["etx_r1"].status = "running"
    assert runner._deps_state(by_name["etx_r2"]) == "wait"
    by_name["etx_r1"].status = "ok"
    assert runner._deps_state(by_name["etx_r2"]) == "ready"
    for name in ("hostprof", "etx_noprog", "vllm_aiter_r1", "vllm_aiter_r2", "vllm_aiter_r3", "report"):
        assert runner._deps_state(by_name[name]) == "ready"
        by_name[name].status = "fail"
