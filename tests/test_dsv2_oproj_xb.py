"""Stage B2, o_proj on top of qkv: the column-batched o_proj tile of the 4-request DeepSeek graph
(ETX_DSV2_OPROJ_XB=1 with ETX_DSV2_QKV_XB=1; graph.py build_xb_oproj, examples/dsv2lite/fleet_xb.hip fleet_oproj_xb).
CPU only, no GPU:

  * default off: the plans and kernels of x1, x2, x4, x2xb and x4xb are byte-identical to a4bb267 (hash pins);
  * on: per layer one o_proj grid for all 4 requests, with the arguments of an existing grid (no new dispatch base);
    every request's E_oproj gets exactly its wait_count of signals, its consumers are the request's own router (or
    dense norm), the static queues keep creation order and the protocol simulator finds no deadlock;
  * B = 2, and the switch without ETX_DSV2_QKV_XB=1, are refused;
  * the tile's row split (a Python mirror of gemv_rows_half_xb, pinned to its source lines) writes every o_proj row
    of every request exactly once per XCD;
  * numbers: the batched partials against fleet's per-request half-wave path, then the router's fold, on random data
    with DeepSeek-V2-Lite's dimensions.
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

ROOT = Path(__file__).resolve().parents[1]
XB_SRC = (ROOT / dsv.XB_T).read_text()

# a4bb267 (PR #21), 3 layers, PassOptions(inline_tiles=True, relay="on"): (plan.json sha256, megakernel sha256)
BASE = dict(ETX_DSV2_COH="l1", ETX_DSV2_COUPLE="none", ETX_DSV2_KVA="shared")
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
    "x4xb": (dict(BASE, ETX_DSV2_REQUESTS="4", ETX_DSV2_QKV_XB="1"),
             "9e8c37bc11a4fe97535615be3c99b381adaaa0240f9b0fde8627d09ccb4b72bc",
             "f711d266334884bd62c554cca1e170e922669db68a23cd0e370100a5e57a50d3"),
    "x4xb-replicated": (dict(BASE, ETX_DSV2_REQUESTS="4", ETX_DSV2_QKV_XB="1", ETX_DSV2_KVA="replicated"),
                        "d8281403bd2083cc3d60901679a468d07ead2a1e168427da33b58f5ee16d4c77",
                        "f41d0dddf25c4fb2da3993bec4fc33995e2cbb6bc670508f8dab0612d3794995"),
}


@pytest.fixture(autouse=True)
def clean_graph_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("ETX_DSV2_"):
            monkeypatch.delenv(key)


def _compile(layers=3):
    return compile_graph(dsv.build(layers=layers), "gfx942", {}, {}, PassOptions(inline_tiles=True, relay="on"))


# ---------------------------------------------------------------- default off: byte-identical
@pytest.mark.parametrize("arm", list(PINS))
@pytest.mark.parametrize("switch", [None, "0"])
def test_default_off_plan_and_kernel_are_byte_identical_to_a4bb267(monkeypatch, arm, switch):
    env, plan_sha, kernel_sha = PINS[arm]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if switch is not None:
        monkeypatch.setenv("ETX_DSV2_OPROJ_XB", switch)
    plan = _compile()
    assert hashlib.sha256(emit_plan_json(plan).encode()).hexdigest() == plan_sha
    assert hashlib.sha256(emit_kernel(plan).encode()).hexdigest() == kernel_sha
    assert not any(g.name.startswith("xb_oproj") or g.body.symbol == "fleet_oproj_xb" for g in plan.graph.grids)


# ---------------------------------------------------------------- on: the lowered plan
N_REQ = 4


@pytest.fixture(scope="module", params=["shared", "replicated"])
def xbo_plan(request):
    env = dict(ETX_DSV2_COH="l1", ETX_DSV2_COUPLE="none", ETX_DSV2_REQUESTS=str(N_REQ), ETX_DSV2_KVA=request.param,
               ETX_DSV2_QKV_XB="1", ETX_DSV2_OPROJ_XB="1")
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        plan = _compile(layers=3)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return request.param, plan


def test_one_batched_oproj_grid_per_layer_with_existing_arguments(xbo_plan):
    _, plan = xbo_plan
    g = plan.graph
    assert not any("_oproj_" in gr.name for gr in g.grids if gr.name.startswith("r"))   # no per-request o_proj grid
    xbo = [gr for gr in g.grids if gr.name.startswith("xb_oproj_")]
    assert [gr.name for gr in xbo] == [f"xb_oproj_{L}" for L in range(3)]
    qkv = {gr.consts[0]: gr for gr in g.grids if gr.name.startswith("xb_qkv_")}
    for L, gr in enumerate(xbo):
        assert gr.body.symbol == "fleet_oproj_xb" and gr.body.source == dsv.XB_T
        assert tuple(gr.grid) == (dsv.XCDS, dsv.XB_W) and tuple(gr.consts) == (L, dsv.XB_W, 0)
        # no new dispatch base: the same argument tensors as the layer's batched qkv grid (request 0's FleetParams and
        # the layer index), which already exist in the x4xb plan
        assert gr.args == ["fleet0", f"layer_{L}"] == qkv[L].args
        assert gr.worker_map == qkv[L].worker_map and gr.resource == qkv[L].resource
        assert gr.duration_us == pytest.approx(dsv.xb_oproj_us(N_REQ))
        assert sum(1 for x in g.grids if x.body.symbol in ("fleet_o_proj", "fleet_oproj_xb") and x.consts[0] == L) == 1
    # the argument tensors of the whole plan are the x4xb plan's: the switch adds none
    assert set(g.tensors) == {"layer_-1", *(f"layer_{L}" for L in range(3)), *(f"fleet{r}" for r in range(N_REQ))}


def test_every_request_oproj_event_gets_its_full_wait_count(xbo_plan):
    _, plan = xbo_plan
    inst, g = plan.inst, plan.graph
    for L in range(3):
        tasks = [t for t in plan.tasks if t.grid == f"xb_oproj_{L}"]
        assert len(tasks) == dsv.XCDS * dsv.XB_W
        for t in tasks:          # every request's XCD-local merge event of this task's XCD
            assert set(inst.task_in[(t.grid, t.coord)]) == {(f"r{r}_E_merge_{L}", (t.coord[0],)) for r in range(N_REQ)}
        for r in range(N_REQ):
            ev = f"r{r}_E_oproj_{L}"
            assert g.events[ev].wait_count == dsv.XCDS * dsv.XB_W
            idxs = [k for k in inst.producers if k[0] == ev]
            assert len(idxs) == 1
            prods = inst.producers[idxs[0]]
            assert len(prods) == dsv.XCDS * dsv.XB_W
            assert {plan.tasks[plan.task_index[p]].grid for p in prods} == {f"xb_oproj_{L}"}
            consumers = {t.grid for t in plan.tasks if any(e[0] == ev for e in inst.task_in[(t.grid, t.coord)])}
            assert consumers == {f"r{r}_norm_{L}" if L < dsv.FIRST_DENSE else f"r{r}_router_{L}"}


def test_both_slots_work_on_it_and_queues_keep_creation_order(xbo_plan):
    _, plan = xbo_plan
    g = plan.graph
    assert plan.wg_per_cu == 2
    per = plan.workers_per_domain
    order = {gr.name: i for i, gr in enumerate(g.grids)}
    for t in (t for t in plan.tasks if t.grid.startswith("xb_oproj_")):
        w = t.coord[1]
        assert t.domain == t.coord[0] and t.worker - t.domain * per == w % dsv.W + (w // dsv.W) * dsv.CUS
    tasks = {t.id: t for t in plan.tasks}
    for queue in plan.static_queues.values():
        idx = [order[tasks[i].grid] for i in queue]
        assert idx == sorted(idx)
    # creation order: every request's merge of L before the batched grid, every request's router / norm of L after it
    for L in range(3):
        x = order[f"xb_oproj_{L}"]
        tail = "norm" if L < dsv.FIRST_DENSE else "router"
        assert max(order[f"r{r}_merge_{L}"] for r in range(N_REQ)) < x < min(order[f"r{r}_{tail}_{L}"] for r in range(N_REQ))
        assert order[f"xb_qkv_{L}"] < x


def test_protocol_simulation_has_no_deadlock_and_runs_every_task(xbo_plan):
    from etx.sim.protocol import simulate
    _, plan = xbo_plan
    r = simulate(plan)
    assert not r.deadlock, r.blocked[:8]
    assert len({s.task for s in r.segments if s.kind == "task"}) == len(plan.tasks)


def test_routing_handoff_still_adjacent(xbo_plan):
    _, plan = xbo_plan
    tasks = {t.id: t for t in plan.tasks}
    count = 0
    for queue in plan.static_queues.values():
        for i, tid in enumerate(queue):
            gu = tasks[tid]
            if "_gate_up_" in gu.grid:
                down = tasks[queue[i + 1]]
                assert down.grid == gu.grid.replace("_gate_up_", "_down_") and down.coord == gu.coord
                count += 1
    assert count == N_REQ * 2 * dsv.XCDS * dsv.W          # MoE layers 1 and 2 (layer 0 is dense)


@pytest.mark.parametrize("env,match", [
    (dict(ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none"), "needs ETX_DSV2_QKV_XB=1"),
    (dict(ETX_DSV2_REQUESTS="2", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1"), "needs 4 requests"),
    (dict(ETX_DSV2_REQUESTS="3", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1"), "needs 4 requests"),
    (dict(ETX_DSV2_REQUESTS="1"), "needs ETX_DSV2_QKV_XB=1"),
])
def test_unsupported_combinations_are_refused(monkeypatch, env, match):
    monkeypatch.setenv("ETX_DSV2_OPROJ_XB", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match=match):
        dsv.build(layers=2)


def test_switch_value_is_checked(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_OPROJ_XB", "2")
    with pytest.raises(ValueError, match="0 \\| 1"):
        dsv.build(layers=2)


def test_arm_differs_from_x4xb_only_by_the_switch():
    from examples.dsv2lite import inplace_session as sess
    from examples.dsv2lite.arms import ARMS, build_env
    a, b = ARMS["x4xb"], ARMS["x4xbo"]
    assert (b.levers, b.flags, b.scratch_allow) == (a.levers, a.flags, a.scratch_allow) == (a.levers, a.flags, 80)
    assert build_env(b) == {**build_env(a), "ETX_DSV2_OPROJ_XB": "1"}
    assert "-DETX_DS_REQUESTS=4" in b.flags and b.rule is None and b.order == ()
    assert sess.required_wg_per_cu(b) == 2
    assert all("ETX_DSV2_OPROJ_XB" not in ARMS[n].env for n in ARMS if not n.startswith("x4xbo"))


# ---------------------------------------------------------------- the row split of gemv_rows_half_xb
def _half_rows_written(N, worker, nw, waves=4, oh=4, issued=None):
    """Python mirror of gemv_rows_half_xb's row loop (= fleet's gemv_rows_pipelined_half<2 oh> with the
    gemv_no_terminal_all lever): first / cnt, groups of 2 oh rows per wave, row k + 2 j + half on half-wave `half`, the
    two-buffer pipeline with the last group peeled. issued: if a list, gets (wave, k) for each group of loads."""
    first = worker
    if first >= N:
        return []
    cnt = (N - first + nw - 1) // nw
    stride = 2 * oh * waves
    out = []
    for wave in range(waves):
        k = 2 * oh * wave
        if k >= cnt:
            continue

        def issue(kk):
            if issued is not None:
                issued.append((wave, kk))

        def reduce(kk):
            for half in (0, 1):
                out.extend(first + (kk + 2 * j + half) * nw for j in range(oh) if kk + 2 * j + half < cnt)
        issue(k)
        while k + 2 * stride < cnt:
            k2 = k + stride
            issue(k2)
            reduce(k)
            k3 = k2 + stride
            issue(k3)
            reduce(k2)
            k = k3
        if k + stride < cnt:
            k2 = k + stride
            issue(k2)
            reduce(k)
            reduce(k2)
        else:
            reduce(k)
    return out


def test_mirror_matches_the_source_lines():
    for line in ("const int first = worker;",
                 "const int cnt = (N - first + n_workers - 1) / n_workers;",
                 "const int half = lane >> 5, hl = lane & 31;",
                 "const int stride = 2 * kOH * fleet::kWaves;",
                 "int k = 2 * kOH * wave;",
                 "constexpr int kOH = 4;",
                 "const int r = (KK) + 2 * j + half;",
                 "if (r < cnt) fleet::epi_st(ys[b] + first + r * n_workers, acc[j], coh);",
                 "const float* x8 = xs + b * kOprojK + hl * 8;",
                 "BUF[j] = fleet::load_weight(reinterpret_cast<const uint4*>(XBO_ROW(KK, j)) + hl);",
                 "#define XBO_ROW(KK, J) (w + (int64_t)(first + min((KK) + 2 * (J) + half, cnt - 1) * n_workers) * ld)",
                 "tab[threadIdx.x] = p.act.oproj_partial + (int64_t)xcd * d.hidden;",
                 "tab[kMaxB + threadIdx.x] = p.act.o + (int64_t)xcd * kc;",
                 "reinterpret_cast<uint4*>(s.vec)[t] = ld_act16(reinterpret_cast<const uint4*>(tab[kMaxB + j]) + i, coh);",
                 "gemv_rows_half_xb(wo + (int64_t)xcd * kc, hv, s.vec, reinterpret_cast<float* const*>(tab), d.hidden, worker, nw, coh);",
                 "const bool coh = run_task_coh(fq->rt);",
                 'extern "C" __device__ void fleet_oproj_xb(const etx_ctx* c) { etx_xb::oproj(c); }',
                 "while (k + 2 * stride < cnt) {",
                 "if (k + stride < cnt) {"):
        assert line in XB_SRC, line
    # the unpatched loop (a dead group of loads at the end of each wave) is gone
    xbo = XB_SRC[XB_SRC.index("void gemv_rows_half_xb("):XB_SRC.index("#undef XBO_ROW")]
    assert "if (k2 >= cnt) break;" not in xbo and "if (k3 >= cnt) break;" not in xbo


def test_no_wave_issues_a_group_it_does_not_reduce():
    """gemv_no_terminal_all: every group of loads is a group of rows < cnt. At 74 workers each wave issues one group."""
    hidden = 2048
    for nw in (dsv.W, dsv.XB_W):
        for w in range(nw):
            cnt = (hidden - w + nw - 1) // nw
            issued = []
            _half_rows_written(hidden, w, nw, issued=issued)
            assert issued and all(k < cnt for _, k in issued), (nw, w)
            assert len(issued) == len(set(issued))
            if nw == dsv.XB_W:
                assert len(issued) == len({wave for wave, _ in issued})


def test_trace_stamps_are_in_stamped_builds_only():
    """The stamp after the staging barrier (and the start / end stamps) compile only with ETX_DS_PRO or
    ETX_DSV2_STAMPS2, so the device code of an unstamped build does not change."""
    body = XB_SRC[XB_SRC.index("__device__ __forceinline__ void oproj("):XB_SRC.index("}  // namespace etx_xb")]
    stack, stamps = [], 0
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("#if"):
            stack.append(s)
        elif s.startswith("#endif"):
            stack.pop()
        elif "__builtin_amdgcn_s_memrealtime()" in s:
            stamps += 1
            assert "#if ETX_DS_PRO || ETX_DSV2_STAMPS2" in stack, s
    assert stamps == 3 and stack == []
    staging = body.index("reinterpret_cast<uint4*>(s.vec)[t] = ld_act16(")
    barrier = body.index("__syncthreads();", staging)
    assert barrier < body.index("s.stamp_stage = __builtin_amdgcn_s_memrealtime();") < body.index("gemv_rows_half_xb(")
    assert "constexpr int kXboProKind = 31;" in XB_SRC
    # 8 XCDs x 128 coordinates fit one kind's 1024 slots; 74 workers fit 128
    assert 8 * 128 == 1024 and dsv.XB_W <= 128


def test_row_split_writes_every_row_exactly_once_per_xcd():
    hidden = 2048
    for nw in (dsv.W, dsv.XB_W):              # fleet's 37 workers and the batched grid's 74
        hits = np.zeros(hidden, dtype=int)
        for w in range(nw):
            for n in _half_rows_written(hidden, w, nw):
                hits[n] += 1
        assert (hits == 1).all(), nw
    # the batched grid halves the rows per task: 55 or 56 -> 27 or 28, one group per wave
    assert max(len(_half_rows_written(hidden, w, dsv.XB_W)) for w in range(dsv.XB_W)) == 28


# ---------------------------------------------------------------- numbers
def _bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def _half_row(wrow, x):
    """fleet's half-wave arithmetic for one K = 256 row: lane hl owns 16-byte chunk hl (8 values), fma8's left-to-right
    fp32 adds, then the xor butterfly over 32 lanes (offsets 16 .. 1). Returns lane 0's fp32 sum."""
    w = wrow.reshape(32, 8)
    xv = x.reshape(32, 8)
    acc = np.zeros(32, dtype=np.float32)
    for e in range(0, 8, 2):
        acc = acc + (w[:, e] * xv[:, e] + w[:, e + 1] * xv[:, e + 1])
    for off in (16, 8, 4, 2, 1):
        acc = acc + acc[np.arange(32) ^ off]
    return acc[0]


def _fleet_partials(Wo, o, nw):
    """fleet's K-split TASK_O_PROJ for one request: XCD x, its 256 columns, rows over nw workers -> [8, 2048] fp32."""
    hidden, hv = Wo.shape
    kc = hv // dsv.XCDS
    out = np.full((dsv.XCDS, hidden), np.nan, dtype=np.float32)
    for x in range(dsv.XCDS):
        for w in range(nw):
            for n in _half_rows_written(hidden, w, nw):
                out[x, n] = _half_row(Wo[n, x * kc:(x + 1) * kc], o[x * kc:(x + 1) * kc])
    return out


def _xb_partials(Wo, os_, nw):
    """fleet_oproj_xb: for every XCD the B o slices side by side in s.vec (slice b at b * 256), one pass over the rows,
    each row's weight chunk used for all B slices."""
    hidden, hv = Wo.shape
    kc = hv // dsv.XCDS
    B = len(os_)
    out = np.full((B, dsv.XCDS, hidden), np.nan, dtype=np.float32)
    for x in range(dsv.XCDS):
        vec = np.concatenate([o[x * kc:(x + 1) * kc] for o in os_])          # s.vec after the staging
        for w in range(nw):
            for n in _half_rows_written(hidden, w, nw):
                wrow = Wo[n, x * kc:(x + 1) * kc]
                for b in range(B):
                    out[b, x, n] = _half_row(wrow, vec[b * kc:(b + 1) * kc])
    return out


def _router_fold(x0, partials):
    """fleet::stage_residual_f32 (coherent path): acc = sum of the 8 partials in XCD order, x1 = bf16(x0 + bf16(acc))."""
    acc = np.zeros(partials.shape[1], dtype=np.float32)
    for x in range(dsv.XCDS):
        acc = acc + partials[x]
    return _bf16(x0 + _bf16(acc))


def test_batched_partials_and_fold_against_the_per_request_path():
    rng = np.random.default_rng(20261018)
    hidden, hv, B = 2048, 16 * 128, N_REQ
    Wo = _bf16(rng.standard_normal((hidden, hv), dtype=np.float32) * np.float32(0.02))
    os_ = [_bf16(rng.standard_normal(hv, dtype=np.float32) * np.float32(s)) for s in (0.25, 1.0, 3.0, 10.0)]
    x0 = [_bf16(rng.standard_normal(hidden, dtype=np.float32)) for _ in range(B)]
    single = np.stack([_fleet_partials(Wo, o, dsv.W) for o in os_])          # today: one task grid per request
    batched = _xb_partials(Wo, os_, dsv.XB_W)
    assert not np.isnan(single).any() and not np.isnan(batched).any()        # every row written
    # 1. same per-lane order and butterfly -> bitwise equal fp32 partials, so the router's fold is bitwise equal
    assert np.array_equal(single, batched)
    x1_single = [_router_fold(x0[b], single[b]) for b in range(B)]
    x1_batched = [_router_fold(x0[b], batched[b]) for b in range(B)]
    assert all(np.array_equal(a, c) for a, c in zip(x1_single, x1_batched))
    # 2. against fp64 (the shared fp32 error; context for the token gate)
    for b in range(B):
        ref = Wo.astype(np.float64) @ os_[b].astype(np.float64)
        err = np.abs(batched[b].sum(axis=0, dtype=np.float64) - ref)
        assert err.max() < 1e-4 * max(1.0, np.abs(ref).max())
    # 3. the check is not vacuous: request b must use its own slice (a swapped slice changes the partials)
    swapped = _xb_partials(Wo, [os_[1], os_[0], os_[2], os_[3]], dsv.XB_W)
    assert not np.array_equal(swapped[0], single[0])


def test_lds_layout_is_checked_at_compile_time():
    """fleet_oproj_xb reuses fleet::Scratch (no new LDS): the B fp32 slices in s.vec, the 2 x B pointers in s.logits."""
    for text in ('static_assert(sizeof(fleet::Scratch::vec) >= (size_t)kMaxB * kOprojK * sizeof(float)',
                 'static_assert(sizeof(fleet::Scratch::logits) >= 2 * kMaxB * sizeof(void*)',
                 'static_assert(kMaxB * kOprojK / 4 <= fleet::kBlockThreads',
                 'static_assert(kOprojK / 8 == fleet::kWaveLanes / 2',
                 'if (p.w.layer_o_proj[L] != wo || p.d.hidden != d.hidden || kc != kOprojK) __builtin_trap();'):
        assert text in XB_SRC, text
    assert dsv.XBO_REQUESTS == 4


# ---------------------------------------------------------------- the duration model
def test_duration_model_per_slot_oproj_time():
    """Per layer and slot: x4xb runs two requests' o_proj tasks (12.0 us each); x4xbo runs one batched task with half
    the rows plus a 0.5 us stage per request."""
    assert 2 * dsv.DUR["oproj"] == pytest.approx(24.0)
    assert dsv.xb_oproj_us(4) == pytest.approx(8.0)


def test_keep_rule_range_and_gate_compiler_are_written_in_arms_py():
    """The rule was written before any GPU run: WIN (6/6 rounds, p <= 0.05) and median saving >= 0.15 ms, token gates
    pass; the expected task time and saving are ranges; ROCm 7.14 is the gate compiler, 7.2.4 is recorded only."""
    src = (ROOT / "examples/dsv2lite/arms.py").read_text()
    block = src[src.index("# Per-slot timing model per layer (graph.py xb_oproj_us"):src.index('Arm("x4xbo", ')]
    text = " ".join(line.strip().lstrip("#").strip() for line in block.splitlines())
    for s in ("8 to 13 us per slot", "0.36 to 0.52 ms", "simulator value x 1.055",
              "gives verdict WIN AND the median step saving is >= 0.15 ms; otherwise DROP",
              "A = x4xb, B = x4xbo, 6 rounds", "--identity-cmd", "p = 1/64 = 0.0156",
              "ROCm 7.14 (the compiler of the GPU machine) is the GATE compiler", "7.2.4 numbers are recorded only"):
        assert s in text, s
    assert ">= 0.31 ms" not in text


def test_keep_rule_statistics_match_ab_same_box():
    """6 rounds: 6/6 faster gives p = 1/64 <= 0.05 (WIN possible); 5/6 gives p = 7/64 > 0.05 (no WIN, so DROP)."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import ab_same_box as ab
    assert ab.ALPHA == 0.05
    assert ab.sign_test_p([0.03] * 6, positive=True) == pytest.approx(1 / 64)
    assert ab.sign_test_p([0.03] * 5 + [-0.01], positive=True) == pytest.approx(7 / 64)
    assert ab.verdict([0.03] * 6, [0.002], 0.01)[0] == "WIN"
    assert ab.verdict([0.03] * 5 + [-0.01], [0.002], 0.01)[0] == "NO-DIFF"
