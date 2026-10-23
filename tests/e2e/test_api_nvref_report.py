"""Command-level cases moved from tests/test_nvref_report.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
_LAST = []


def nr_main(argv=None):
    """`nr.main(argv)` as the command `python -m examples.vllm_llm.nvref_report`."""
    result = run_command([sys.executable, "-m", "examples.vllm_llm.nvref_report", *map(str, argv or [])], cwd=ROOT)
    _LAST.append(result.stdout)
    return result.returncode


def last_output():
    return _LAST[-1] if _LAST else ""


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


def test_passing(session, capsys):
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "VERDICT LEAD | import-check INCOMPLETE (uncertified legacy check) | image vllm/image@sha256:abcdef" in out
    assert "median=-0.200 spread=0.000" in out
    assert "FUSION (unfused - fused) median=+0.400" in out
    assert "IMPORT-CHECK OK" not in out
    assert "FULL-CLAIM NOT ELIGIBLE" in out
    assert "4.801, 4.802, 4.803, 4.804, 4.805" in out
    assert out.count("third party eliovp.com 2025-09-26, not reproduced") == 4
    assert "BASIS-MISMATCH" not in out


@pytest.mark.parametrize("arm", ["etx", "etxu", "vllm"])
def test_31_of_32_refuses(session, capsys, arm):
    write(session, f"{arm}_r3", vllm().replace("32/32", "31/32") if arm == "vllm"
          else etx(4.8, "graph" if arm == "etxu" else "fused", matched=31))
    if arm == "etxu":
        assert nr_main([str(session)]) == 0
        out = last_output()
        assert "FUSION unavailable (etxu_r3:" in out and "32/32" in out
        assert "VERDICT LEAD" in out and "FUSION (unfused" not in out
        return
    assert nr_main([str(session)]) == 2
    out = last_output()
    assert "32/32" in out and "VERDICT NOT-COMPLETE" in out
    assert "| source |" not in out


def test_missing_run_refuses(session, capsys):
    (session / "vllm_r5.log").unlink()
    assert nr_main([str(session)]) == 2
    assert "vllm_r5: missing run/job log" in last_output()


def test_last_repeat_without_progress_refuses(session, capsys):
    write(session, "etx_r5", etx(4.8, count=0))
    assert nr_main([str(session)]) == 2
    assert "got n=0" in last_output()


def test_comment_is_not_gate_evidence(session, capsys):
    write(session, "gate_all", etx(4.8) + "\nGATE-OK")
    assert nr_main([str(session)]) == 2
    assert "missing CHECK-DSL OK" in last_output()


def test_graph_off_refuses(session, capsys):
    write(session, "etxu_r1", etx(5.2, "graph").replace("ETX_UNFUSED_NOGRAPH=", "ETX_UNFUSED_NOGRAPH=1"))
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "graph-off" in out and "FUSION unavailable" in out
    assert "VERDICT LEAD" in out and "FUSION (unfused" not in out


@pytest.mark.parametrize("count", [30, 32])
def test_wrong_step_count_refuses(session, capsys, count):
    write(session, "etx_r5", etx(4.8, count=count))
    assert nr_main([str(session)]) == 2
    out = last_output()
    assert "expected 31 measured steps" in out and "| source |" not in out


@pytest.mark.parametrize("gate", ["gate_all", "gate_unf", "check_imp"])
def test_failed_gate(session, capsys, gate):
    write(session, gate, "CHECK FAILED: 1 launch(es) differ")
    if gate == "gate_unf":
        assert nr_main([str(session)]) == 0
        out = last_output()
        assert "FUSION unavailable (gate_unf:" in out
        assert "VERDICT LEAD" in out and "FUSION (unfused" not in out
        return
    assert nr_main([str(session)]) == 2
    assert "VERDICT NOT-COMPLETE" in last_output()


def test_status_failure_after_result(session, capsys):
    (session / "status.json").write_text(json.dumps([dict(name="etx_r5", status="timeout")]))
    assert nr_main([str(session)]) == 2
    assert "etx_r5: job timeout" in last_output()


def test_import_na_and_basis_mismatch(session, capsys):
    write(session, "check_imp", "ERROR: --check needs a plan without per-XCD replicas (build without ETX_VL_XCD)")
    write(session, "etx_r1", etx(4.8).replace("wall_ms=4.800", "wall_ms=4.850"))
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "IMPORT-CHECK n/a (host.hip:515)" in out
    assert "BASIS-MISMATCH etx_r1 mean_ms=4.800 RESULT wall_ms=4.850" in out


def test_aiter_recompute(session, capsys):
    for r in range(1, 4):
        write(session, f"vllm_aiter_r{r}", vllm(4.6, aiter=1))
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "VERDICT LEAD |" in out
    assert "VERDICT LOSS against AITER (unpaired median reference); default=LEAD" in out


@pytest.mark.parametrize("missing", ["gate_unf", "etxu_r1", "etxu_r5", "all"])
def test_fusion_missing_still_verdict(session, capsys, missing):
    names = [f"etxu_r{r}" for r in range(1, 6)] if missing == "all" else [missing]
    for name in names:
        (session / f"{name}.log").unlink()
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "| vLLM 0.27.1 default |" in out and "| ETX fused, 4 tile flags |" in out
    assert "FUSION unavailable (" in out and "missing run/job log" in out
    assert "FUSION (unfused" not in out and "| ETX unfused, HIP graph |" not in out
    assert "DELTA r1..r5" in out and "VERDICT LEAD" in out


def test_check_imp1_ok_cannot_mask_current_refusal(session, capsys):
    """A legacy different-build pass cannot certify the refused timed build."""
    write(session, "check_imp1", "  in_norm L0 rms_norm_2d identical  original 1.0 us  imported 1.0 us\n" * 5
          + "CHECK OK: 0 launch(es) differ")
    write(session, "check_imp", "ERROR: --check needs a plan without per-XCD replicas (build without ETX_VL_XCD)")
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "IMPORT-CHECK INCOMPLETE 5/97, different build" in out
    assert "IMPORT-CHECK OK" not in out
    assert "FULL-CLAIM NOT ELIGIBLE" in out


def test_legacy_check_failed_refuses_even_when_new_check_passed(session, capsys):
    """A differing launch in either build is a correctness failure, not an unavailable check."""
    write(session, "check_imp1", "CHECK OK: 0 launch(es) differ")
    write(session, "check_imp", "CHECK FAILED: 1 launch(es) differ")
    assert nr_main([str(session)]) == 2
    out = last_output()
    assert "REFUSED: check_imp: CHECK FAILED" in out
    assert "DELTA" not in out


def test_check_imp1_failure_refuses(session, capsys):
    write(session, "check_imp1", "CHECK FAILED: 1 launch(es) differ")
    write(session, "check_imp", "ERROR: --check needs a plan without per-XCD replicas (build without ETX_VL_XCD)")
    (session / "status.json").write_text(json.dumps([dict(name="check_imp1", status="fail")]))
    assert nr_main([str(session)]) == 2
    out = last_output()
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
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert f"IMPORT-CHECK NOT MET (check_imp1: {status})" in out
    assert "DELTA r1..r5" in out
    assert f"VERDICT LEAD | import-check NOT MET (check_imp1: {status})" in out


@pytest.mark.parametrize("arm", ["vllm", "etx"])
def test_missing_required_with_missing_fusion_refuses(session, capsys, arm):
    (session / "etxu_r1.log").unlink()
    (session / f"{arm}_r5.log").unlink()
    assert nr_main([str(session)]) == 2
    out = last_output()
    assert "No performance numbers." in out and "VERDICT NOT-COMPLETE" in out
    assert "| source |" not in out and "DELTA" not in out


def test_skipped_check_without_log(session, capsys):
    (session / "status.json").write_text(json.dumps([dict(name="check_imp1", status="skipped")]))
    assert nr_main([str(session)]) == 0
    assert "IMPORT-CHECK NOT MET (check_imp1: skipped)" in last_output()


@pytest.mark.parametrize("name", ["hostprof", "etx_noprog", "vllm_aiter_r1"])
@pytest.mark.parametrize("failure", ["incorrect", "timeout", "missing", "malformed"])
def test_optional_failure_never_refuses(session, capsys, name, failure):
    body = vllm(4.6, 1) if name.startswith("vllm") else etx(4.8)
    if failure == "incorrect":
        body = body.replace("32/32", "31/32").replace("matched=32", "matched=31")
    if failure == "malformed":
        body = "invalid output"
    if failure != "missing":
        write(session, name, body)
    if failure == "timeout":
        (session / "status.json").write_text(json.dumps([dict(name=name, status="timeout")]))
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "VERDICT LEAD" in out and f"{name} unavailable (" in out


@pytest.mark.parametrize("values, expected", [
    # Deltas -0.203, -0.053, -0.203, -0.053, -0.203 against a ~5.003 ms baseline: spread 0.150 ms, but every
    # round has the same sign and the median (-0.203) sits outside the spread, so the verdict stands.
    ([4.80, 4.95, 4.80, 4.95, 4.80], "LEAD"),
    # Median (-0.053) inside the spread (0.150): the sign is not stable, so no verdict may rest on it.
    ([4.80, 4.95, 4.80, 4.95, 4.95], "INSUFFICIENT"),
])
def test_resolution(session, capsys, values, expected):
    for r, value in enumerate(values, 1):
        write(session, f"etx_r{r}", etx(value))
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "RESOLUTION: paired delta spread" in out and "0.050 ms threshold" in out
    assert f"VERDICT {expected} |" in out


@pytest.mark.parametrize("unfused, expected", [([5.2]*5, "met"), ([4.85]*5, "not met"), ([5., 5., 5., 5., 5.4], "not met")])
def test_fusion_rule(session, capsys, unfused, expected):
    for r, value in enumerate(unfused, 1):
        write(session, f"etxu_r{r}", etx(value, "graph"))
    assert nr_main([str(session)]) == 0
    line = next(s for s in last_output().splitlines() if s.startswith("FUSION "))
    assert f"rule {expected} (" in line


def test_host_excess_uses_result_wall(session, capsys):
    for r in range(1, 6):
        write(session, f"etx_r{r}", etx(4.8).replace("median_ms=4.650 wall_ms=4.800", "median_ms=4.650 wall_ms=4.847"))
        write(session, f"etxu_r{r}", etx(5.2, "graph").replace("median_ms=5.050", "median_ms=5.164"))
    assert nr_main([str(session)]) == 0
    assert "HOST-EXCESS fused median=0.197 unfused median=0.036 excess(fused-unfused)=0.161 ms" in last_output()


def test_certificate_required_and_revalidated(session, capsys):
    from etx.testing import certified_import_session as fixture_session
    fixture_session(session)
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "IMPORT-CHECK OK (certified 97/97, positions 0,1040)" in out
    assert "FULL-CLAIM ELIGIBLE" in out
    (session / "check_p1040.txt").unlink()
    assert nr_main([str(session)]) == 0
    out = last_output()
    assert "IMPORT-CHECK OK" not in out
    assert "FULL-CLAIM NOT ELIGIBLE" in out
    assert "FUSION (unfused" in out
