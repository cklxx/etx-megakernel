"""Fixed, out-of-sample profile; the historical fit remains the default."""
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from etx.sim import phase
from etx.tools.floors import round11_from_trace
from etx.models.vllm_llm import pmodel

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "examples/llm/configs/qwen3-30b-a3b.json").read_text())


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    fixtures = ROOT / "tests/fixtures/pmodel"
    output = tmp_path_factory.mktemp("pmodel")
    for name, info in json.loads((fixtures / "provenance.json").read_text()).items():
        data = gzip.decompress((fixtures / (name + ".gz")).read_bytes())
        assert len(data) == info["bytes"]
        assert hashlib.sha256(data).hexdigest() == info["sha256"]
        (output / name).write_bytes(data)
    return output / "plan.json", output / "tr_all.bin"


def synthetic_plan():
    widths = {"qkv_0": 256, "prep_0": 4, "attn_0": 20}
    return {"types": {n: {"id": i} for i, n in enumerate(widths)},
            "descs": [{"type": i} for i, width in enumerate(widths.values()) for _ in range(width)]}


def test_default_profile_unchanged():
    plan = synthetic_plan()
    default = pmodel.cost_fn(plan, CFG)
    explicit = pmodel.cost_fn(plan, CFG, profile="2026-09-30")
    for name in plan["types"]:
        assert default(name, {}) == explicit(name, {})


def test_new_profile_costs():
    cost = pmodel.cost_fn(synthetic_plan(), CFG, profile="2026-10-08",
                         etx_us=99, fused_us=99, coh_save_us=1.5, coh_touch=True)
    qkv = cost("qkv_0", {})
    assert qkv.etx_us == 0
    assert qkv.agg_bw == phase.BW_HBM
    assert qkv.cu_rate == 1e9
    assert qkv.touch_bytes == 0
    assert cost("prep_0", {}).chain_us == 6.04
    assert cost("attn_0", {}).chain_us == 1.9


def test_stream_bodies_against_trace(historical):
    plan, trace = historical
    measured = round11_from_trace(plan, trace)
    rb = pmodel.role_bytes(CFG)
    for role in ("qkv", "moe_up", "moe_down"):
        predicted = phase.FLOOR_US + rb[role] / phase.BW_HBM
        assert predicted == pytest.approx(measured[role]["mean_body"], rel=0.03), role


def test_current_plan_prediction(historical):
    plan, _ = historical
    fused = pmodel.step_us(plan, CFG, profile="2026-10-08")
    unfused = pmodel.step_us(plan, CFG, fused=False, profile="2026-10-08",
                             handoff_domain=1.61, handoff_device=1.61)
    assert fused == pytest.approx(4.058, abs=0.01)
    assert unfused == pytest.approx(4.246, abs=0.01)
    assert unfused - fused > 0
    assert abs(unfused - fused - 0.225) < 0.093
