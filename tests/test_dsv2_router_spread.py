"""Router order and router spread on top of the batched o_proj (B = 4, ETX_DSV2_OPROJ_XB=1):
ETX_DSV2_ROUTER_FIRST=1 (graph.py: every request's router of a layer before any request's experts) and
ETX_DSV2_ROUTER_SPREAD=1 (graph.py build_moe_router, fleet_xb.hip fleet_norm_router_spread: each
request's norm + router on 18 workers of its slot). CPU only, no GPU:

  * default off: x4xbo's plan and kernel are byte-identical to 01d08f4 (hash pins), with the switches unset or "0";
  * on: grids, worker maps, event counts, no new argument tensor, creation order, queue order, routing handoff,
    no deadlock in the protocol simulator;
  * unsupported combinations are refused;
  * the tile runs task_norm_router's steps in fleet's order with the same helpers (so the same lever edits);
  * numbers: the router logits and the top-6 of every XCD with the rows split over 18 workers against 37 (a mirror of
    gemv_rows_pipelined<2, 4>'s per-row order), on random data with DeepSeek-V2-Lite's dimensions.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from etx.codegen import emit_kernel, emit_plan_json
from etx.passes import PassOptions, compile_graph
from examples.dsv2lite import graph as dsv

ROOT = Path(__file__).resolve().parents[1]
XB_SRC = (ROOT / dsv.XB_T).read_text()
FLEET_SRC = Path(os.environ.get("ETX_FLEET_SRC", "/workspace/fleet-mi300x/src"))
N_REQ, LAYERS = 4, 3
XBO = dict(ETX_DSV2_COH="l1", ETX_DSV2_COUPLE="none", ETX_DSV2_REQUESTS="4", ETX_DSV2_QKV_XB="1", ETX_DSV2_OPROJ_XB="1")
# 01d08f4 (PR #23), 3 layers, PassOptions(inline_tiles=True, relay="on"): (plan.json sha256, megakernel sha256)
PINS = {
    "shared": ("aa45fe1d960b6e7540f8d812303aae8e7f9bba01aa7c6db602f24f46f88c4fe2",
               "7ea6324df95ca6011b385d3e7cc007f98000064b0d1b9c395719c3ec5165fa3a"),
    "replicated": ("7052a247d639a142b1ce0fda294f2f8d249500a9460ac1202bbd68fdbefcc8b5",
                   "8e5dc85e6801d3620d0f875ebf5a177e38e05de49488974aea14de5d69743d49"),
}


@pytest.fixture(autouse=True)
def clean_graph_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("ETX_DSV2_"):
            monkeypatch.delenv(key)


def _compile(env: dict, layers=LAYERS):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return compile_graph(dsv.build(layers=layers), "gfx942", {}, {}, PassOptions(inline_tiles=True, relay="on"))
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ---------------------------------------------------------------- default off
@pytest.mark.parametrize("kva", list(PINS))
@pytest.mark.parametrize("switch", [None, "0"])
def test_default_off_x4xbo_is_byte_identical_to_01d08f4(kva, switch):
    env = dict(XBO, ETX_DSV2_KVA=kva)
    if switch is not None:
        env.update(ETX_DSV2_ROUTER_FIRST=switch, ETX_DSV2_ROUTER_SPREAD=switch)
    plan = _compile(env)
    assert (hashlib.sha256(emit_plan_json(plan).encode()).hexdigest(),
            hashlib.sha256(emit_kernel(plan).encode()).hexdigest()) == PINS[kva]
    assert not any(g.body.symbol == "fleet_norm_router_spread" for g in plan.graph.grids)


def test_spread_tile_is_compiled_only_with_its_define():
    i = XB_SRC.index('extern "C" __device__ void fleet_norm_router_spread(')
    head = XB_SRC[:i]
    assert "#if defined(ETX_DSV2_ROUTER_SPREAD)" in head and "namespace etx_xb {" in head[head.rindex("#if defined(ETX_DSV2_ROUTER_SPREAD)"):]
    assert "#if defined(ETX_DSV2_ROUTER_SPREAD) && ETX_DSV2_ROUTER_SPREAD" in head[head.rindex("#endif"):]
    assert XB_SRC.rstrip().endswith("#endif")


# ---------------------------------------------------------------- on
MODES = {"first": dict(ETX_DSV2_ROUTER_FIRST="1"), "spread": dict(ETX_DSV2_ROUTER_SPREAD="1"),
         "both": dict(ETX_DSV2_ROUTER_FIRST="1", ETX_DSV2_ROUTER_SPREAD="1"),
         "bothlm": dict(ETX_DSV2_ROUTER_FIRST="1", ETX_DSV2_ROUTER_SPREAD="1", ETX_DSV2_LMHEAD_XB="1")}


@pytest.fixture(scope="module", params=[(m, k) for m in MODES for k in ("shared", "replicated")],
                ids=lambda p: f"{p[0]}-{p[1]}")
def plan_on(request):
    mode, kva = request.param
    return mode, _compile(dict(XBO, ETX_DSV2_KVA=kva, **MODES[mode]))


def test_router_grids_workers_and_arguments(plan_on):
    mode, plan = plan_on
    g = plan.graph
    spread = mode != "first"
    for L in range(dsv.FIRST_DENSE, LAYERS):
        used = {}
        for r in range(N_REQ):
            gr = g.grids[[x.name for x in g.grids].index(f"r{r}_router_{L}")]
            assert gr.args == [f"fleet{r}", f"layer_{L}"]           # the router's own arguments: no new base
            if not spread:
                assert gr.body.symbol == "fleet_norm_router" and tuple(gr.grid) == (dsv.XCDS, dsv.W)
                assert g.events[f"r{r}_E_norm_{L}"].wait_count == dsv.W
                continue
            assert gr.body.symbol == "fleet_norm_router_spread" and gr.body.source == dsv.XB_T
            assert tuple(gr.grid) == (dsv.XCDS, dsv.ROUTER_SPREAD_W) and tuple(gr.consts) == (L, r, dsv.ROUTER_SPREAD_W)
            assert g.events[f"r{r}_E_norm_{L}"].wait_count == dsv.ROUTER_SPREAD_W
            assert gr.duration_us == pytest.approx(dsv.router_spread_us())
            for t in (t for t in plan.tasks if t.grid == gr.name):
                used.setdefault((t.domain, t.worker), []).append(r)
        if spread:   # the 4 routers of a layer on 4 x 18 distinct workers: 0..17 / 18..35 of each slot
            assert len(used) == N_REQ * dsv.XCDS * dsv.ROUTER_SPREAD_W and all(len(v) == 1 for v in used.values())
            per = plan.workers_per_domain
            for (d, w), (r,) in used.items():
                local = w - d * per - (r % 2) * dsv.CUS
                assert (r // 2) * dsv.ROUTER_SPREAD_W <= local < (r // 2 + 1) * dsv.ROUTER_SPREAD_W
    assert set(g.tensors) == {"layer_-1", *(f"layer_{L}" for L in range(LAYERS)), *(f"fleet{r}" for r in range(N_REQ))}
    assert plan.wg_per_cu == 2


def test_norm_events_get_their_wait_count_from_the_router_only(plan_on):
    _, plan = plan_on
    inst, g = plan.inst, plan.graph
    for L in range(dsv.FIRST_DENSE, LAYERS):
        for r in range(N_REQ):
            ev = f"r{r}_E_norm_{L}"
            for x in range(dsv.XCDS):
                prods = inst.producers[(ev, (x,))]
                assert len(prods) == g.events[ev].wait_count
                assert {plan.tasks[plan.task_index[p]].grid for p in prods} == {f"r{r}_router_{L}"}


def test_creation_order_and_queue_order(plan_on):
    mode, plan = plan_on
    g = plan.graph
    order = {gr.name: i for i, gr in enumerate(g.grids)}
    tasks = {t.id: t for t in plan.tasks}
    for queue in plan.static_queues.values():
        idx = [order[tasks[i].grid] for i in queue]
        assert idx == sorted(idx)
    for L in range(dsv.FIRST_DENSE, LAYERS):
        x = order[f"xb_oproj_{L}"]
        routers = [order[f"r{r}_router_{L}"] for r in range(N_REQ)]
        experts = [order[f"r{r}_gate_up_{L}"] for r in range(N_REQ)]
        assert x < min(routers)
        if mode == "spread":    # today's order: r's router, gate_up, down, then r + 1's
            assert all(routers[r] < experts[r] < routers[r + 1] for r in range(N_REQ - 1))
        else:
            assert max(routers) < min(experts)


def test_routing_handoff_still_adjacent(plan_on):
    _, plan = plan_on
    tasks = {t.id: t for t in plan.tasks}
    count = 0
    for queue in plan.static_queues.values():
        for i, tid in enumerate(queue):
            if "_gate_up_" in tasks[tid].grid:
                down = tasks[queue[i + 1]]
                assert down.grid == tasks[tid].grid.replace("_gate_up_", "_down_") and down.coord == tasks[tid].coord
                count += 1
    assert count == N_REQ * (LAYERS - dsv.FIRST_DENSE) * dsv.XCDS * dsv.W


def test_protocol_simulation_has_no_deadlock(plan_on):
    from etx.sim.protocol import simulate
    _, plan = plan_on
    for seed in range(3):
        r = simulate(plan, seed=seed)
        assert not r.deadlock, r.blocked[:8]
    assert len({s.task for s in r.segments if s.kind == "task"}) == len(plan.tasks)


@pytest.mark.parametrize("sw", ["ETX_DSV2_ROUTER_FIRST", "ETX_DSV2_ROUTER_SPREAD"])
@pytest.mark.parametrize("env,match", [
    (dict(ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1"), "need ETX_DSV2_OPROJ_XB=1"),
    (dict(ETX_DSV2_REQUESTS="2", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1"), "need ETX_DSV2_OPROJ_XB=1"),
    (dict(ETX_DSV2_REQUESTS="1"), "need ETX_DSV2_OPROJ_XB=1"),
    (dict(ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none"), "need ETX_DSV2_OPROJ_XB=1"),
])
def test_unsupported_combinations_are_refused(monkeypatch, sw, env, match):
    monkeypatch.setenv(sw, "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match=match):
        dsv.build(layers=2)


@pytest.mark.parametrize("sw", ["ETX_DSV2_ROUTER_FIRST", "ETX_DSV2_ROUTER_SPREAD"])
def test_switch_value_is_checked(monkeypatch, sw):
    for k, v in XBO.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(sw, "2")
    with pytest.raises(ValueError, match="0 \\| 1"):
        dsv.build(layers=2)


def test_arms():
    from examples.dsv2lite.arms import ARMS
    base, q, r = ARMS["x4xbl"], ARMS["x4xboq"], ARMS["x4xbor"]
    assert q.env == dict(base.env, ETX_DSV2_ROUTER_FIRST="1") and q.levers == base.levers and q.flags == base.flags
    assert r.env == dict(base.env, ETX_DSV2_ROUTER_FIRST="1", ETX_DSV2_ROUTER_SPREAD="1")
    assert r.levers == base.levers and r.flags == (*base.flags, "-DETX_DSV2_ROUTER_SPREAD=1")
    assert {n for n, a in ARMS.items() if "ETX_DSV2_ROUTER_SPREAD" in a.env} == {"x4xbor"}


def test_duration_model():
    # 64 router rows: 2 per worker at 37 workers, 4 at 18; fold + norm unchanged
    assert dsv.router_spread_us() == pytest.approx(dsv.DUR["router"] + dsv.ROUTER_GEMV_US)


# ---------------------------------------------------------------- the tile follows fleet's task_norm_router
@pytest.mark.skipif(not (FLEET_SRC / "kernels/fleet_kernel.hip").exists(), reason="pinned fleet source not present")
def test_spread_tile_mirrors_task_norm_router():
    from examples.dsv2lite.patch_fleet import _function
    fleet = _function((FLEET_SRC / "kernels/fleet_kernel.hip").read_text(), "__device__ void task_norm_router(")
    body = XB_SRC[XB_SRC.index("void norm_router_spread("):]
    body = body[:body.index("\n}\n")]
    # the same steps in the same order; only the row split's worker count differs
    steps_fleet = ["stage_residual_f32(d, act.x[0], act.oproj_partial, act.x[1], s.vec, xcd, worker, coh);",
                   "rms_norm_lds(d, w.layer_norm_post[layer], s.vec, s.smem);", "if (worker == 0) {",
                   "1.f, d.n_routed, d.hidden, 0, 1, worker, kWorkersPerXCD);"]
    steps_tile = ["fleet::stage_residual_f32(d, act.x[0], act.oproj_partial, act.x[1], s.vec, xcd, worker, coh);",
                  "input_norm(rank2{}, d, fp->w.layer_norm_post[L], s.vec, s.smem);", "if (worker == 0) {",
                  "1.f, d.n_routed, d.hidden, 0, 1, worker, nw);"]
    pos = [fleet.index(x) for x in steps_fleet]
    assert pos == sorted(pos)
    pos = [body.index(x) for x in steps_tile]
    assert pos == sorted(pos)
    assert "__shared__" not in body and "args[" not in body and "run_task_coh(fp->rt)" in body
    # the router uses no row-split o_proj in these arms (fleet's FLAG_OPROJ_ROW_SPLIT branch is the only other path)
    assert "FLAG_OPROJ_ROW_SPLIT" not in XB_SRC[XB_SRC.index("void norm_router_spread("):]


# ---------------------------------------------------------------- numbers
WAVE, D, HIDDEN, N_ROUTED, TOP_K = 64, 4, 2048, 64, 6


def _wave_sum(v: np.ndarray) -> np.float32:
    v = v.astype(np.float32).copy()
    off = WAVE // 2
    while off:                      # xor butterfly; every lane ends with the same value, lane 0 read
        v = (v + v[np.arange(WAVE) ^ off]).astype(np.float32)
        off //= 2
    return v[0]


def _row(w_row: np.ndarray, x: np.ndarray) -> np.float32:
    """One router row as gemv_rows_pipelined<2, 4> reduces it: lane l takes 16-byte chunks l + 64 d (d < 4) in d order,
    8 fp32 FMAs each, then one wave_sum. Nothing here depends on the worker or on n_workers."""
    acc = np.zeros(WAVE, np.float32)
    for d in range(D):
        for e in range(8):
            cols = (np.arange(WAVE) + d * WAVE) * 8 + e
            acc = (acc + w_row[cols] * x[cols]).astype(np.float32)
    return _wave_sum(acc)


def _logits(w: np.ndarray, x: np.ndarray, nw: int) -> tuple[np.ndarray, np.ndarray]:
    """gemv_rows(..., N = 64, xcd = 0, n_xcds = 1, worker, n_workers = nw): worker k owns rows k, k + nw, ..."""
    out = np.full(N_ROUTED, np.nan, np.float32)
    owner = np.full(N_ROUTED, -1)
    for k in range(nw):
        first = k
        if first >= N_ROUTED:
            continue
        cnt = (N_ROUTED - first + nw - 1) // nw
        for j in range(cnt):
            row = first + j * nw
            assert owner[row] == -1
            owner[row] = k
            out[row] = _row(w[row], x)
    return out, owner


def _topk(logits: np.ndarray) -> list[int]:
    p = np.exp(logits - logits.max())
    return [int(i) for i in np.argsort(-p, kind="stable")[:TOP_K]]


def test_router_logits_and_topk_bitwise_with_18_workers():
    rng = np.random.default_rng(23)
    w = (rng.standard_normal((N_ROUTED, HIDDEN)) * 0.02).astype(np.float32)
    w = w.astype(np.float32)
    for trial in range(4):
        x = rng.standard_normal(HIDDEN).astype(np.float32)
        ref, own37 = _logits(w, x, dsv.W)
        new, own18 = _logits(w, x, dsv.ROUTER_SPREAD_W)
        assert (own37 >= 0).all() and (own18 >= 0).all()           # every row written exactly once
        assert np.array_equal(ref.view(np.uint32), new.view(np.uint32))
        assert _topk(ref) == _topk(new)
        assert np.bincount(own18).max() == 4 and np.bincount(own37).max() == 2


def test_gemv_rows_selects_its_path_by_k_only():
    """The bitwise claim rests on gemv_rows choosing the GEMV path from K alone (not from cnt or n_workers)."""
    gemv = FLEET_SRC / "kernels/gemv.h"
    if not gemv.exists():
        pytest.skip("pinned fleet source not present")
    text = gemv.read_text()
    body = text[text.index("__device__ inline void gemv_rows("):]
    body = body[:body.index("\n}\n")]
    conds = re.findall(r"if \((n4 <= [^)]*)\)", body)
    assert conds and all("cnt" not in c and "n_workers" not in c for c in conds)
