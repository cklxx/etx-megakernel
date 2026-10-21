"""The ETX_VL_COH prediction of the phase model (docs/qwen/MODEL-vllm-kernels.md section 20): the coh variant lowers a
fused build within the pre-registered range, leaves the non-coh path byte-identical, and does not apply unfused."""
from pathlib import Path

import json
import pytest

from examples.vllm_llm import pmodel

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results_2026-09-30/plan_moe_dsl3.json"


@pytest.fixture(scope="module")
def cfg():
    return json.load(open(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"))


@pytest.mark.skipif(not PLAN.exists(), reason="measured plan not on this machine")
def test_coh_range(cfg):
    r = pmodel.predict_coh(PLAN, cfg)
    assert r["coh_hi"] < r["coh_lo"] < r["base"]                       # overhead share alone: monotone in the save
    assert r["coh_touch_hi"] < r["coh_hi"]                             # the touch adds on top
    assert r["base"] - r["coh_lo"] < 0.6                               # bounded: not the old model's 2 ms promises
    assert r["coh_touch_hi"] > 4.0                                     # nowhere near the bandwidth floor


@pytest.mark.skipif(not PLAN.exists(), reason="measured plan not on this machine")
def test_coh_off_is_identity(cfg):
    assert pmodel.step_us(PLAN, cfg) == pmodel.step_us(PLAN, cfg, coh_save_us=0.0, coh_touch=False)


@pytest.mark.skipif(not PLAN.exists(), reason="measured plan not on this machine")
def test_coh_not_applied_unfused(cfg):
    base = pmodel.step_us(PLAN, cfg, fused=False)
    assert pmodel.step_us(PLAN, cfg, fused=False, coh_save_us=1.5) == base
