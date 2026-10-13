"""G6 on Qwen (docs/INVENTORY-SYSTEM-2026-10-13.md 4.1): the registry arm, its pre-registered rule at the boundaries,
the timed-invocation gate, the session's job graph, and a stored verdict re-derived from the raw logs."""
import json
from pathlib import Path

import pytest

from etx.session import experiment
from examples.vllm_llm import arms
from examples.vllm_llm import g6_session as g6

ROOT = Path(__file__).resolve().parents[1]


def host_log(device="4.471", wall="4.505", matched=32, env="ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=", runs=1):
    one = (f"{env}\nmode: ETX megakernel (vLLM kernels imported)\nrun 0: {matched}/32 generated tokens match HF greedy\n"
           f"RESULT model=qwen requests=1 matched={matched} gen=32 median_ms={device} wall_ms={wall} tokens_per_s=223 "
           f"matched_each={matched}\n")
    return one * runs


def test_kept_l1_is_the_kept_build_with_only_the_compiler_switch():
    kept, l1 = (arms.build_env(arms.ARMS[n]) for n in ("kept", "kept_l1"))
    assert {k: v for k, v in l1.items() if kept.get(k) != v} == {"ETX_COHERENCE": "l1"}
    assert set(kept) == set(l1) - {"ETX_COHERENCE"}


def test_the_rule_is_g6_as_registered():
    r = arms.ARMS["kept_l1"].rule
    assert (r.basis, r.reference, r.min_median_us, r.spread_factor, r.strict, r.labels) == \
           ("device_ms", "kept", 30, 2, True, ("KEEP", "NO-GO"))


def rounds(gains_us, ref="4.500"):
    base = experiment.ns(ref)
    return [dict(kept=dict(device_ms=ref, wall_ms=ref),
                 kept_l1=dict(device_ms=f"{(base - g * 1000) / 1e6:.3f}", wall_ms=ref)) for g in gains_us]


@pytest.mark.parametrize("gains, keep", [
    ([30, 30, 30, 30, 30], True),       # median exactly 30 us (>= is inclusive), spread 0
    ([30, 30, 30, 30, 45], False),      # median 30 = 2 x spread 15: "median > 2 x spread" is strict
    ([31, 31, 31, 31, 46], True),       # median 31 > 2 x 15
    ([29, 29, 29, 29, 29], False),      # below 30 us
    ([100, 100, 100, 100, -1], False),  # one pair slower
])
def test_rule_boundaries_in_integer_nanoseconds(gains, keep):
    d = experiment.decide(arms.ARMS["kept_l1"].rule, rounds(gains), "kept_l1")
    assert d["passed"] is keep


def test_timed_invocation_gate():
    assert arms.parse(host_log()) == dict(device_ms="4.471", wall_ms="4.505", tokens=32)
    for bad in (host_log(matched=31),                                                  # not 32/32
                host_log(runs=2),                                                       # not exactly one run
                host_log(env="ETX_ENV ETX_UNFUSED_SYNC=1 ETX_UNFUSED_NOGRAPH="),         # debug synchronisation
                host_log(device="0.000")):                                              # zero time
        with pytest.raises(Exception):
            arms.parse(bad)


def test_job_graph_gates_every_gpu_job_on_what_it_runs():
    from etx.tools.gpu_jobs import load_session
    jobs, limit = load_session(ROOT / "examples/sessions/qwen_g6.py")
    by = {j.name: j for j in jobs}
    assert [j.name for j in jobs if j.lane == "gpu"] == ["certify_kept", "certify_kept_l1", "pairs_kept_l1"]
    assert {"certificate_kept/gate.log", "certificate_kept_l1/gate.log", "inputs.json"} <= set(by["pairs_kept_l1"].reads)
    assert by["certify_kept_l1"].reads == ["inputs.json", "b_kept_l1/arm.json"]
    assert limit == 3600


def test_pairs_writes_a_verdict_that_re_derives_from_the_raw_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("ETX_SESS_LOGDIR", str(tmp_path))
    for b in ("kept", "kept_l1"):
        (tmp_path / f"certificate_{b}").mkdir()
    device = {"kept": "4.500", "kept_l1": "4.440"}                     # 60 us every round

    def fake_host(arm, name, args):
        text = host_log(device=device[arm])
        (tmp_path / f"{name}.raw.log").write_text(text)
        return text
    monkeypatch.setattr(g6, "host", fake_host)
    g6.pairs("kept_l1")
    stored = json.loads((tmp_path / "pairs_kept_l1.json").read_text())
    assert stored["verdict"]["outcome"] == "KEEP" and stored["decision"]["median_us"] == 60
    arm = arms.ARMS["kept_l1"]
    again = experiment.report(arm, experiment.measure(arm, lambda b, name: arms.parse(
        (tmp_path / f"{name}.raw.log").read_text())), arms.CONDITION)
    assert experiment.same(stored, again, arm)
    assert sorted(p.name for p in tmp_path.glob("*.raw.log"))[:2] == ["kept_l1_r0_kept.raw.log", "kept_l1_r0_kept_l1.raw.log"]


def test_pairs_refuses_an_uncertified_build(tmp_path, monkeypatch):
    monkeypatch.setenv("ETX_SESS_LOGDIR", str(tmp_path))
    (tmp_path / "certificate_kept").mkdir()
    with pytest.raises(ValueError, match="kept_l1 has no certificate"):
        g6.pairs("kept_l1")


def test_preflight_refuses_a_non_bf16_checkpoint(tmp_path, monkeypatch):
    hf = tmp_path / "hf"; hf.mkdir()
    (hf / "config.json").write_text(json.dumps(dict(g6.QWEN3_30B_A3B, torch_dtype="bfloat16",
                                                    quantization_config={"quant_method": "fp8"})))
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(dict(model=str(tmp_path), hf=str(hf), golden_sha256="0" * 64)))
    monkeypatch.setenv("ETX_QWEN_G6_CONFIG", str(cfg))
    monkeypatch.setenv("ETX_SESS_LOGDIR", str(tmp_path))
    from examples.vllm_llm import vm_preflight
    monkeypatch.setattr(vm_preflight, "check", lambda *a, **k: 0)        # build inputs present: reach the HF check
    with pytest.raises(ValueError, match="bf16 weights required"):
        g6.preflight()


def test_preflight_refuses_a_box_without_the_build_inputs_before_anything_runs(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(dict(model=str(tmp_path / "absent"), hf=str(tmp_path / "absent/hf"), golden_sha256="0" * 64)))
    monkeypatch.setenv("ETX_QWEN_G6_CONFIG", str(cfg))
    monkeypatch.setenv("ETX_SESS_LOGDIR", str(tmp_path))
    monkeypatch.setenv("VLLM", str(tmp_path / "no-vllm"))
    with pytest.raises(ValueError, match="box inputs missing"):
        g6.preflight()
    out = capsys.readouterr().out
    assert "FAIL packed_model" in out and "missing=vl.bin,vl.manifest,hf/config.json,golden_c1024.txt" in out
    assert "FAIL vllm_source" in out and "git clone --depth 1 --branch v0.27.1" in out
    assert not (tmp_path / "inputs.json").exists()
    assert [l.split()[1].rstrip(":") for l in out.splitlines() if l.startswith(("PASS", "FAIL"))] == list(g6.BUILD_INPUTS)
