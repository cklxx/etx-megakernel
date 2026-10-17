"""Stage B2, the token tail on top of o_proj: the batched lm_head of the 4-request DeepSeek graph (ETX_DSV2_LMHEAD_XB=1
with ETX_DSV2_OPROJ_XB=1; graph.py build_xb_lm, examples/dsv2lite/fleet_xb.hip fleet_xb_final_norm / fleet_lm_head_xb /
fleet_xb_lm_cand). CPU only, no GPU:

  * default off: the plans and kernels of every existing arm shape are byte-identical to 01d08f4 (hash pins);
  * on: per request a final-norm grid, ONE lm_head grid, per request a candidate grid on fleet's 37 workers and fleet's
    argmax; argument tensors of existing grids only; full wait counts, creation order, no deadlock;
  * the row split of the batched grid writes every logit of every request exactly once;
  * numbers: final norm (bf16 copy exact), the batched logits against fleet's single-column path, and the greedy
    token through fleet's candidate + argmax rule, bitwise in the numpy model.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from etx.codegen import emit_kernel, emit_plan_json
from etx.passes import PassOptions, compile_graph
from examples.dsv2lite import graph as dsv
from tests.test_dsv2_qkv_xb import _bf16, _rms_norm, _rows_written, _wave_rows

ROOT = Path(__file__).resolve().parents[1]
XB_SRC = (ROOT / dsv.XB_T).read_text()

BASE = dict(ETX_DSV2_COH="l1", ETX_DSV2_COUPLE="none", ETX_DSV2_KVA="shared")
X4 = dict(BASE, ETX_DSV2_REQUESTS="4", ETX_DSV2_QKV_XB="1")
# 01d08f4 (PR #23 head), 3 layers, PassOptions(inline_tiles=True, relay="on"): (plan.json sha256, megakernel sha256)
PINS = {
    "x1": ({}, "02d85386eaedfcda77b2121da14ab8255cef352531734621ce75a608e2d97450",
           "37f938c1a78378cb6fe2e66785dab9ec6be3165f85509e77d39892d358cfcd20"),
    "x2": (dict(BASE, ETX_DSV2_REQUESTS="2"), "26bc57c6236a9542071fe1c2b69608859191b4a9eddecfed27d80502fde459d4",
           "b8213a370d0b51eecb0aeb1d75fbe052fc620277120c5ec6dddea819b2366bee"),
    "x4": (dict(BASE, ETX_DSV2_REQUESTS="4"), "910dad04c3e94e71f60abc56a0e662d63891335935393292f4fa0e5fb9c5f3e7",
           "60b67e9c8f029bd4cdfb60423539d600fca76179fed159502bd7bcf36c385949"),
    "x2xb": (dict(BASE, ETX_DSV2_REQUESTS="2", ETX_DSV2_QKV_XB="1"),
             "8873edb92ce9c91a9531fd6e379c67fb8fd33971eeaf46f1f2a468a15de1fe05",
             "52cb4f06a7060a4701ac9799ee8a7fb8588771f6392f5094f096d042a14c218f"),
    "x4xb": (X4, "9e8c37bc11a4fe97535615be3c99b381adaaa0240f9b0fde8627d09ccb4b72bc",
             "f711d266334884bd62c554cca1e170e922669db68a23cd0e370100a5e57a50d3"),
    "x4xb-replicated": (dict(X4, ETX_DSV2_KVA="replicated"),
                        "d8281403bd2083cc3d60901679a468d07ead2a1e168427da33b58f5ee16d4c77",
                        "f41d0dddf25c4fb2da3993bec4fc33995e2cbb6bc670508f8dab0612d3794995"),
    "x4xbo": (dict(X4, ETX_DSV2_OPROJ_XB="1"), "aa45fe1d960b6e7540f8d812303aae8e7f9bba01aa7c6db602f24f46f88c4fe2",
              "7ea6324df95ca6011b385d3e7cc007f98000064b0d1b9c395719c3ec5165fa3a"),
    "x4xbo-replicated": (dict(X4, ETX_DSV2_OPROJ_XB="1", ETX_DSV2_KVA="replicated"),
                         "7052a247d639a142b1ce0fda294f2f8d249500a9460ac1202bbd68fdbefcc8b5",
                         "8e5dc85e6801d3620d0f875ebf5a177e38e05de49488974aea14de5d69743d49"),
}
N_REQ = 4


@pytest.fixture(autouse=True)
def clean_graph_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("ETX_DSV2_", "ETX_FLEET_PATCH")):
            monkeypatch.delenv(key)


def _compile(layers=3):
    return compile_graph(dsv.build(layers=layers), "gfx942", {}, {}, PassOptions(inline_tiles=True, relay="on"))


# ---------------------------------------------------------------- default off: byte-identical
@pytest.mark.parametrize("arm", list(PINS))
@pytest.mark.parametrize("switch", [None, "0"])
def test_default_off_plan_and_kernel_are_byte_identical_to_01d08f4(monkeypatch, arm, switch):
    env, plan_sha, kernel_sha = PINS[arm]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if switch is not None:
        monkeypatch.setenv("ETX_DSV2_LMHEAD_XB", switch)
    plan = _compile()
    assert hashlib.sha256(emit_plan_json(plan).encode()).hexdigest() == plan_sha
    assert hashlib.sha256(emit_kernel(plan).encode()).hexdigest() == kernel_sha
    assert not any(g.name == "xb_lm" or g.name.endswith(("_lmn", "_lmc")) for g in plan.graph.grids)


# ---------------------------------------------------------------- on: the lowered plan
@pytest.fixture(scope="module", params=["shared", "replicated"])
def xbl_plan(request):
    env = dict(BASE, ETX_DSV2_REQUESTS=str(N_REQ), ETX_DSV2_KVA=request.param, ETX_DSV2_QKV_XB="1",
               ETX_DSV2_OPROJ_XB="1", ETX_DSV2_LMHEAD_XB="1")
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        plan = _compile(layers=3)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return request.param, plan


def test_token_tail_grids_with_existing_arguments(xbl_plan):
    _, plan = xbl_plan
    g = plan.graph
    by = {gr.name: gr for gr in g.grids}
    assert not any(gr.name.endswith("_lm_head") for gr in g.grids)        # no per-request lm_head grid
    lm = by["xb_lm"]
    assert lm.body.symbol == "fleet_lm_head_xb" and tuple(lm.grid) == (dsv.XCDS, dsv.XB_W)
    assert tuple(lm.consts) == (-1, dsv.XB_W, 0) and lm.args == ["fleet0", "layer_-1"]
    assert lm.worker_map == by["xb_qkv_0"].worker_map and lm.duration_us == pytest.approx(dsv.xb_lm_us(N_REQ))
    for r in range(N_REQ):
        n, c, a = by[f"r{r}_lmn"], by[f"r{r}_lmc"], by[f"r{r}_argmax"]
        assert n.body.symbol == "fleet_xb_final_norm" and tuple(n.grid) == (dsv.XCDS,)
        assert tuple(n.consts) == (-1, r, dsv.XB_FOLD) and n.args == [f"fleet{r}", "layer_-1"]   # the last layer is MoE
        assert n.worker_map == f"x->({dsv.xb_norm_worker(r)})"
        assert c.body.symbol == "fleet_xb_lm_cand" and tuple(c.grid) == (dsv.XCDS, dsv.W)
        assert tuple(c.consts) == (-1, r) and c.args == [f"fleet{r}", "layer_-1"]
        assert c.worker_map == dsv._Req(r, N_REQ).wmap_chiplet()
        assert a.body.symbol == "fleet_argmax" and a.args == [f"fleet{r}", "layer_-1"]
    # no new argument tensor: the switch adds none
    assert set(g.tensors) == {"layer_-1", *(f"layer_{L}" for L in range(3)), *(f"fleet{r}" for r in range(N_REQ))}
    assert all(gr.body.source == dsv.XB_T for gr in g.grids)


def test_events_and_wait_counts(xbl_plan):
    _, plan = xbl_plan
    inst, g = plan.inst, plan.graph

    def producers(ev):
        out = {}
        for k, prods in inst.producers.items():
            if k[0] == ev:
                out[k] = [plan.tasks[plan.task_index[p]].grid for p in prods]
        return out

    def consumers(ev):
        return {t.grid for t in plan.tasks if any(e[0] == ev for e in inst.task_in[(t.grid, t.coord)])}
    for t in (t for t in plan.tasks if t.grid == "xb_lm"):
        assert set(inst.task_in[(t.grid, t.coord)]) == {(f"r{r}_E_lmn", (t.coord[0],)) for r in range(N_REQ)}
    for r in range(N_REQ):
        for ev, grid, n_idx, want_cons in ((f"r{r}_E_lmn", f"r{r}_lmn", dsv.XCDS, {"xb_lm"}),
                                           (f"r{r}_E_lmx", "xb_lm", dsv.XCDS, {f"r{r}_lmc"}),
                                           (f"r{r}_E_lm", f"r{r}_lmc", 1, {f"r{r}_argmax"})):
            p = producers(ev)
            assert len(p) == n_idx
            assert all(len(v) == g.events[ev].wait_count and set(v) == {grid} for v in p.values()), ev
            assert consumers(ev) == want_cons, ev
        assert g.events[f"r{r}_E_lmx"].wait_count == dsv.XB_W and g.events[f"r{r}_E_lm"].wait_count == dsv.XCDS * dsv.W
        # the final norm waits for the request's last layer
        for t in (t for t in plan.tasks if t.grid == f"r{r}_lmn"):
            assert {e[0] for e in inst.task_in[(t.grid, t.coord)]} == {f"r{r}_E_down_2"}
        # the candidate task on XCD x waits for the batched grid's XCD-x logits only
        for t in (t for t in plan.tasks if t.grid == f"r{r}_lmc"):
            assert set(inst.task_in[(t.grid, t.coord)]) == {(f"r{r}_E_lmx", (t.coord[0],))}


def test_candidate_grid_keeps_the_acquire_invalidate(xbl_plan):
    """fleet::lm_head_candidate reads act.logits with plain loads: its grid is not marked coherent_l1 (it keeps the
    acquire invalidate, as dense_down); the other new grids are, like every fleet tile in this build."""
    _, plan = xbl_plan
    by = {gr.name: gr for gr in plan.graph.grids}
    assert "fleet_xb_lm_cand" in dsv.COH_PLAIN_XBL
    for r in range(N_REQ):
        assert by[f"r{r}_lmc"].body.coherence == "imported" and by[f"r{r}_lmc"].body.coherence_caps == ()
        assert by[f"r{r}_lmn"].body.coherence == "coherent_l1"
    assert by["xb_lm"].body.coherence == "coherent_l1" == by["xb_qkv_0"].body.coherence


def test_queues_keep_creation_order_and_no_deadlock(xbl_plan):
    from etx.sim.protocol import simulate
    _, plan = xbl_plan
    order = {gr.name: i for i, gr in enumerate(plan.graph.grids)}
    tasks = {t.id: t for t in plan.tasks}
    for queue in plan.static_queues.values():
        idx = [order[tasks[i].grid] for i in queue]
        assert idx == sorted(idx)
    last = max(order[f"r{r}_down_2"] for r in range(N_REQ))
    assert last < min(order[f"r{r}_lmn"] for r in range(N_REQ))
    assert max(order[f"r{r}_lmn"] for r in range(N_REQ)) < order["xb_lm"] < min(order[f"r{r}_lmc"] for r in range(N_REQ))
    assert max(order[f"r{r}_lmc"] for r in range(N_REQ)) < min(order[f"r{r}_argmax"] for r in range(N_REQ))
    r = simulate(plan)
    assert not r.deadlock, r.blocked[:8]
    assert len({s.task for s in r.segments if s.kind == "task"}) == len(plan.tasks)


@pytest.mark.parametrize("env,match", [
    (dict(ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1"), "needs ETX_DSV2_OPROJ_XB=1"),
    (dict(ETX_DSV2_REQUESTS="2", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1"), "requests"),
    (dict(ETX_DSV2_REQUESTS="1"), "needs ETX_DSV2_OPROJ_XB=1"),
    (dict(ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1", ETX_DSV2_OPROJ_XB="1",
          ETX_FLEET_PATCH="norm_tree l1_loads"), "norm_tree_all"),
    (dict(ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1", ETX_DSV2_OPROJ_XB="1",
          ETX_FLEET_PATCH="norm_gamma_early"), "norm_tree_all"),
])
def test_unsupported_combinations_are_refused(monkeypatch, env, match):
    monkeypatch.setenv("ETX_DSV2_LMHEAD_XB", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match=match):
        dsv.build(layers=2)


def test_switch_value_is_checked(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_LMHEAD_XB", "yes")
    with pytest.raises(ValueError, match="0 \\| 1"):
        dsv.build(layers=2)


def test_arm_differs_from_x4xbo_only_by_the_switch(monkeypatch):
    from examples.dsv2lite import inplace_session as sess
    from examples.dsv2lite.arms import ARMS, build_env
    a, b = ARMS["x4xbo"], ARMS["x4xbl"]
    assert (b.levers, b.flags, b.scratch_allow) == (a.levers, a.flags, a.scratch_allow)
    assert build_env(b) == {**build_env(a), "ETX_DSV2_LMHEAD_XB": "1"}
    assert "norm_tree_all" in b.levers and b.rule is None and sess.required_wg_per_cu(b) == 2
    assert {n for n in ARMS if "ETX_DSV2_LMHEAD_XB" in ARMS[n].env} == {"x4xbl", "x4xboq", "x4xbor", "x4xbora"}   # x4xbo[qr], x4xbora: stacked on x4xbl
    # the arm's own build environment passes the lever guard
    for k, v in build_env(b).items():
        if k.startswith("ETX_DSV2_") or k == "ETX_FLEET_PATCH":
            monkeypatch.setenv(k, v)
    dsv.build(layers=2)


def test_source_lines():
    for line in ("fleet::stage_residual(d, act.x[1], fold ? act.expert_out : nullptr, nullptr, s.vec, xcd, 0, run_task_coh(fp->rt));",
                 "fleet::dump_layer(d, act, d.layers - 1, s.vec, xcd, 0, tc);",
                 "input_norm(rank2{}, d, fp->w.final_norm, s.vec, s.smem);",
                 "tab[threadIdx.x] = p.act.logits;",
                 "tab[2 * kMaxB + threadIdx.x] = p.act.x_norm + (int64_t)xcd * d.hidden;",
                 "gemv_rows_xb(wl, d.hidden, s, reinterpret_cast<float* const*>(tab), d.vocab, d.hidden, xcd, fleet::kNumXCDs, worker, nw);",
                 "fleet::lm_head_candidate(fp->d, act, (int)c->domain, (int)c->coord[1], fleet::kWorkersPerXCD, s.smem, s.ids_blk,",
                 "if (p.w.lm_head != wl || p.d.hidden != d.hidden || p.d.vocab != d.vocab) __builtin_trap();",
                 'extern "C" __device__ void fleet_lm_head_xb(const etx_ctx* c) { etx_xb::lm_head(c); }'):
        assert line in XB_SRC, line


# ---------------------------------------------------------------- row split and numbers
VOCAB, HIDDEN = 102400, 2048


def test_row_split_writes_every_logit_exactly_once():
    hits = np.zeros(VOCAB, dtype=np.int32)
    for xcd in range(dsv.XCDS):
        for w in range(dsv.XB_W):
            hits[_rows_written(VOCAB, xcd, dsv.XCDS, w, dsv.XB_W)] += 1
    assert (hits == 1).all()


def _rows_no_terminal(N, xcd, n_xcds, worker, nw, waves=4):
    """Python mirror of gemv_rows_xb's loop (kRPI = 1): which rows it reduces, and how many groups it issues."""
    per = (N + n_xcds - 1) // n_xcds
    begin, end = xcd * per, min(N, xcd * per + per)
    first = begin + worker
    if first >= end:
        return [], 0
    cnt = (end - first + nw - 1) // nw
    out, issued = [], 0
    for wave in range(waves):
        k = wave
        if k >= cnt:
            continue
        issued += 1
        while k + 2 * waves < cnt:
            issued += 2
            out += [k, k + waves]
            k += 2 * waves
        if k + waves < cnt:
            issued += 1
            out += [k, k + waves]
        else:
            out += [k]
    return [first + kk * nw for kk in out if kk < cnt], issued


def test_no_terminal_loop_reduces_the_same_rows_and_issues_no_dead_group():
    """gemv_rows_xb (lever gemv_no_terminal_all's loop, for the qkv and the lm_head tile since the merge of
    feat/ds-qkv-noterm) reduces exactly the rows of the row split, and every issued group is reduced."""
    for line in ("while (k + 2 * stride < cnt) {", "if (k + stride < cnt) {"):
        assert line in XB_SRC, line
    assert "kNoTerminal" not in XB_SRC and "gemv_rows_xb<" not in XB_SRC   # one loop, no opt-in variant
    for xcd in (0, 7):
        for w in (0, 1, 36, 73):
            rows, issued = _rows_no_terminal(VOCAB, xcd, dsv.XCDS, w, dsv.XB_W)
            assert sorted(rows) == sorted(_rows_written(VOCAB, xcd, dsv.XCDS, w, dsv.XB_W))
            assert issued == len(rows)                  # kRPI = 1: one row per group, no dead group


def _fleet_candidates(logits):
    """fleet::lm_head_candidate for every (xcd, worker) of fleet's 37 workers: max, lowest index on ties."""
    per = VOCAB // dsv.XCDS
    vals, idxs = [], []
    for x in range(dsv.XCDS):
        for w in range(dsv.W):
            rows = np.arange(x * per + w, (x + 1) * per, dsv.W)
            v = logits[rows]
            k = int(np.flatnonzero(v == v.max())[0])          # rows ascend: the first max is the lowest index
            vals.append(v[k]); idxs.append(int(rows[k]))
    return np.array(vals, dtype=np.float32), np.array(idxs)


def _fleet_argmax(vals, idxs):
    best = vals.max()
    return int(idxs[vals == best].min())


def test_token_tail_bitwise_against_the_per_request_path():
    rng = np.random.default_rng(20261018)
    rows = 4096                                           # a slice of W_lm for the arithmetic (full vocab below)
    Wt = _bf16(rng.standard_normal((rows, HIDDEN), dtype=np.float32) * np.float32(0.02))
    gamma = _bf16(1.0 + 0.1 * rng.standard_normal(HIDDEN, dtype=np.float32))
    x1 = [_bf16(rng.standard_normal(HIDDEN, dtype=np.float32) * np.float32(s)) for s in (0.5, 1.0, 2.0, 8.0)]
    x32 = [_rms_norm(v, gamma) for v in x1]               # fleet's lm_head prologue output in s.vec (fp32 storage)
    x16 = [_bf16(v) for v in x32]                         # fleet_xb_final_norm's bf16 copy in act.x_norm
    assert all(np.array_equal(a, b) for a, b in zip(x32, x16))
    single = [_bf16(_wave_rows(Wt, v)) for v in x32]       # fleet's EPI_BF16 logits
    batched = [_bf16(_wave_rows(Wt, v.astype(np.float32))) for v in x16]
    assert all(np.array_equal(a, b) for a, b in zip(single, batched))


def test_greedy_token_through_fleet_candidates_and_argmax():
    """The batched grid writes the same logits (test above), and the candidate grid is fleet's own code on fleet's
    37-worker rows: the 296 candidates and the token are the per-request path's. Includes bf16 ties."""
    rng = np.random.default_rng(7)
    for trial in range(4):
        logits = _bf16(rng.standard_normal(VOCAB, dtype=np.float32) * np.float32(2.0))
        top = logits.max()
        if trial:                                         # plant ties of the max at random rows
            logits[rng.choice(VOCAB, size=3 * trial, replace=False)] = top
        vals, idxs = _fleet_candidates(logits)
        assert vals.size == dsv.XCDS * dsv.W
        assert _fleet_argmax(vals, idxs) == int(np.flatnonzero(logits == logits.max())[0])   # torch.argmax


def test_duration_model():
    """Per slot at the token end: x4xbo runs two lm_head tasks (140 us each); x4xbl one final-norm task (7.4 us), one
    batched task (2 us stage + max(132.6 x 37 / 74, 80.7 us HBM floor)) and two candidate tasks (2 us each)."""
    assert 2 * dsv.DUR["lm_head"] == pytest.approx(280.0)
    assert dsv.XBL_HBM_US == pytest.approx(80.66, abs=0.01)
    assert dsv.xb_lm_us(4) == pytest.approx(82.66, abs=0.01)
    assert dsv.XBL_NORM_US + dsv.xb_lm_us(4) + 2 * dsv.XBL_CAND_US == pytest.approx(94.06, abs=0.01)
