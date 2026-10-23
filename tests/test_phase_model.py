"""The measured phase cost model (etx.sim.phase, examples/vllm_llm/pmodel): it reproduces the microbenchmark it was
built from, it removes the Infinity Cache from the rates a weight stream cannot get, and it charges a phase for how
wide it is rather than for existing -- the fact the 2026-09-30 builds established."""
import json
from pathlib import Path

import pytest

from etx.sim import phase
from etx.sim.xcd import Cost, Hw, simulate

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "examples/llm/configs/qwen3-30b-a3b.json").read_text())


def test_phase_us_reproduces_the_stream_benchmark():
    for mapping, table in phase.SPAN_US.items():
        for b, span, n in table:
            got = phase.phase_us(b, n, mapping)
            # contig is exact by construction; rows is clamped where the measurement beat HBM (see below)
            lim = 0.02 if mapping == "contig" else 0.20
            assert abs(got - span) / span < lim, (mapping, b, span, got)
    # the one narrow point: 8 workgroups x 312 KB
    assert abs(phase.phase_us(319488, 8, "contig") - 12.10) < 0.05


def test_infinity_cache_is_clamped_out_of_the_rates():
    """296 workgroups reading 28 KB each measured 4.56 TB/s with the interleaved mapping -- above what HBM can
    deliver, because 8.5 MB fits in the 256 MB MALL. A weight stream never does, so agg_bw must not report it."""
    assert phase.raw_rate(28672, "rows") * phase.N_REF > 4.4e6      # what was measured
    assert phase.agg_bw(28672, "rows") == pytest.approx(phase.BW_HBM)
    for mapping in phase.SPAN_US:
        for b, _, _ in phase.SPAN_US[mapping]:
            assert phase.agg_bw(b, mapping) <= phase.BW_HBM + 1e-9


def test_aggregate_bandwidth_grows_with_bytes_per_workgroup():
    """It is not one number: a shallow queue does not reach 3.5 TB/s."""
    assert phase.agg_bw(28672, "contig") < 2.5e6
    assert phase.agg_bw(1048576, "contig") > 3.4e6


def _step(plan_name, mapping="contig", fused=True):
    from etx.models.vllm_llm import pmodel
    plan = json.loads((ROOT / "results_2026-09-30" / plan_name).read_text())
    hw = pmodel.hw() if fused else pmodel.hw(handoff_domain=1.61, handoff_device=1.61)
    _, st = simulate(plan, pmodel.cost_fn(plan, CFG, mapping, fused=fused), hw)
    return st["makespan_us"] / 1e3


@pytest.mark.skipif(not (ROOT / "results_2026-09-30/plan_moe_dsl3.json").exists(), reason="plans not present")
def test_reproduces_the_measured_builds():
    assert abs(_step("plan_moe_dsl3.json") - 5.895) / 5.895 < 0.10
    assert abs(_step("plan_moe_dsl2.json") - 5.934) / 5.934 < 0.10


@pytest.mark.skipif(not (ROOT / "results_2026-09-30/plan_moe_dsl3.json").exists(), reason="plans not present")
def test_folding_narrow_phases_away_is_worth_almost_nothing():
    """build 2 folds nothing and runs 12 phases a layer, build 3 folds four away and runs 8; they measured 5.934
    and 5.895 ms. The old model predicted 0.9 ms for that fold. The new one must not."""
    assert abs(_step("plan_moe_dsl2.json") - _step("plan_moe_dsl3.json")) < 0.35


def test_a_phase_is_charged_for_its_width_not_for_existing():
    """The same bytes over fewer tasks costs less, because the fixed cost follows the width."""
    from etx.models.vllm_llm import pmodel
    rows = {r: (now, at_now, best, at_best) for r, _, now, at_now, best, at_best, _ in pmodel.width_sweep(CFG)}
    now, at_now, best, at_best = rows["router"]
    assert now == 304 and best < 64 and at_best < at_now / 4      # 0.5 MB does not want 304 tasks
    for role in ("qkv", "o_proj", "moe_down"):
        _, a0, nb, a1 = rows[role]
        assert nb < 304 and a1 < a0


def test_the_megakernel_costs_more_per_phase_than_separate_launches():
    """Measured: build 3 is 5.895 ms fused and 5.660 unfused although an unfused launch also pays a kernel
    boundary. The model carries that as a positive constant rather than assuming fusion is free."""
    from etx.models.vllm_llm import pmodel
    assert pmodel.FUSED_US > 0


def test_cost_fixed_us_sums_the_fixed_terms():
    c = Cost(front_us=0.5, chain_us=1.0, floor_us=3.0, etx_us=7.5)
    assert c.fixed_us == pytest.approx(12.0)


def test_agg_bw_caps_the_shared_bandwidth_in_the_simulator():
    """A task's agg_bw, not hw.bw_max, limits what concurrent streams of that shape get."""
    plan = {"types": {"t": {"id": 0, "symbol": "x"}}, "descs": [{"type": 0, "coord": [i], "domain": 0} for i in range(2)],
            "events": [], "ev_inst": [], "task_in": [[], []], "static_queues": {"0:0": [0], "0:1": [1]},
            "workers_per_domain": 2}
    hw = Hw(bw_max=10e6, n_half=0.0, handoff_us={"DOMAIN": 0.0, "DEVICE": 0.0}, lat_us=0.0)
    slow = simulate(plan, lambda n, d: Cost(stream_bytes=1e6, cu_rate=1e6, agg_bw=1e6), hw)[1]["makespan_us"]
    fast = simulate(plan, lambda n, d: Cost(stream_bytes=1e6, cu_rate=1e6, agg_bw=4e6), hw)[1]["makespan_us"]
    assert slow > fast
