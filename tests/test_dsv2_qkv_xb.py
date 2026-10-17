"""Stage B2, qkv only (docs/BATCH-2026-10-14.md section 5): the column-batched qkv tile of the multi-request DeepSeek
graph (ETX_DSV2_QKV_XB=1; graph.py build_xb_qkv, examples/dsv2lite/fleet_xb.hip). CPU only, no GPU:

  * default off: the plans and kernels of x1, x2 and x4 are byte-identical to origin/main b573664 (hash pins);
  * on: per layer one norm grid per request (8 tasks) and one qkv grid for all requests, every request's E_xbn / E_qkv
    gets exactly its wait_count of signals, the static queues keep creation order and the protocol simulator finds no
    deadlock;
  * the session refuses a paired build that fits fewer than 2 workgroups per CU (registers or LDS);
  * the tile's row split (a Python mirror of gemv_rows_xb, pinned to its source lines) writes every q / kv_a row of
    every request exactly once;
  * numbers: the batched math (bf16 copy of the normalized vector, fleet's per-lane fp32 order, bf16 output) against
    fleet's single-column path with the fp32 vector, on random data with DeepSeek-V2-Lite's dimensions.
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

# origin/main b573664, 3 layers, PassOptions(inline_tiles=True, relay="on"): (plan.json sha256, megakernel sha256)
BASE = dict(ETX_DSV2_COH="l1", ETX_DSV2_COUPLE="none", ETX_DSV2_KVA="shared")
DEFAULT_PINS = {
    "x1": ({}, "02d85386eaedfcda77b2121da14ab8255cef352531734621ce75a608e2d97450",
           "37f938c1a78378cb6fe2e66785dab9ec6be3165f85509e77d39892d358cfcd20"),
    "x2": (dict(BASE, ETX_DSV2_REQUESTS="2"), "26bc57c6236a9542071fe1c2b69608859191b4a9eddecfed27d80502fde459d4",
           "b8213a370d0b51eecb0aeb1d75fbe052fc620277120c5ec6dddea819b2366bee"),
    "x4": (dict(BASE, ETX_DSV2_REQUESTS="4"), "910dad04c3e94e71f60abc56a0e662d63891335935393292f4fa0e5fb9c5f3e7",
           "60b67e9c8f029bd4cdfb60423539d600fca76179fed159502bd7bcf36c385949"),
}


@pytest.fixture(autouse=True)
def clean_graph_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("ETX_DSV2_"):
            monkeypatch.delenv(key)


def _compile(layers=3):
    return compile_graph(dsv.build(layers=layers), "gfx942", {}, {}, PassOptions(inline_tiles=True, relay="on"))


# ---------------------------------------------------------------- default off: byte-identical
@pytest.mark.parametrize("arm", list(DEFAULT_PINS))
@pytest.mark.parametrize("switch", [None, "0"])
def test_default_off_plan_and_kernel_are_byte_identical_to_main(monkeypatch, arm, switch):
    env, plan_sha, kernel_sha = DEFAULT_PINS[arm]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if switch is not None:
        monkeypatch.setenv("ETX_DSV2_QKV_XB", switch)
    plan = _compile()
    assert hashlib.sha256(emit_plan_json(plan).encode()).hexdigest() == plan_sha
    source = emit_kernel(plan)
    assert hashlib.sha256(source.encode()).hexdigest() == kernel_sha
    assert "fleet_xb" not in source and not any(g.name.startswith("xb_") for g in plan.graph.grids)


# ---------------------------------------------------------------- on: the lowered plan
@pytest.fixture(scope="module", params=[(2, "replicated"), (4, "replicated"), (2, "shared"), (4, "shared")],
                ids=lambda p: f"x{p[0]}-{p[1]}")
def xb_plan(request):
    n, kva = request.param
    env = dict(ETX_DSV2_COH="l1", ETX_DSV2_COUPLE="none", ETX_DSV2_REQUESTS=str(n), ETX_DSV2_KVA=kva,
               ETX_DSV2_QKV_XB="1")
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        plan = _compile(layers=3)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return n, kva, plan


def test_one_batched_qkv_grid_per_layer_reads_the_weight_once(xb_plan):
    n, kva, plan = xb_plan
    g = plan.graph
    assert all("_qkv_" not in gr.name for gr in g.grids if gr.name.startswith("r"))      # no per-request qkv grid
    xb = [gr for gr in g.grids if gr.name.startswith("xb_qkv_")]
    assert [gr.name for gr in xb] == [f"xb_qkv_{L}" for L in range(3)]
    kvs = dsv.XB_KVS if kva == "shared" else 0
    assert {gr.body.symbol for gr in xb} == {"fleet_qkv_xb"}
    by_name = {gr.name: gr for gr in g.grids}
    for L, gr in enumerate(xb):
        assert gr.body.source == dsv.XB_T and tuple(gr.grid) == (dsv.XCDS, dsv.XB_W)
        assert tuple(gr.consts) == (L, dsv.XB_W, 0, kvs)
        assert gr.args == ["fleet0", f"layer_{L}"]
        # the per-request norm grids: one task per XCD, fleet's prologue with the request's own FleetParams;
        # layer 0: input from embed; layer 1: after the dense layer (no expert partials); layer 2: after MoE (fold)
        for r in range(n):
            nm = by_name[f"r{r}_xbn_{L}"]
            assert nm.body.symbol == "fleet_xb_norm" and tuple(nm.grid) == (dsv.XCDS,)
            assert tuple(nm.consts) == (L, r, dsv.XB_FOLD if L == 2 else 0) and nm.args == [f"fleet{r}", f"layer_{L}"]
        # one grid per layer = one stream of the fused [3072 + 576, 2048] weight per layer (it was n)
        assert sum(1 for x in g.grids if x.body.symbol.startswith("fleet_qkv") and x.consts[0] == L) == 1
    assert all(gr.body.source == dsv.XB_T for gr in g.grids)       # one TU: fleet_xb.hip includes fleet_shim.hip
    assert '#include "examples/dsv2lite/fleet_shim.hip"' in XB_SRC


def test_every_request_qkv_event_gets_its_full_wait_count(xb_plan):
    n, kva, plan = xb_plan
    inst, g = plan.inst, plan.graph
    for L in range(3):
        tasks = [t for t in plan.tasks if t.grid == f"xb_qkv_{L}"]
        assert len(tasks) == dsv.XCDS * dsv.XB_W
        for r in range(n):
            ev = f"r{r}_E_qkv_{L}"
            et = g.events[ev]
            idxs = [k for k in inst.producers if k[0] == ev]
            assert len(idxs) == (1 if kva == "shared" else dsv.XCDS)
            for k in idxs:
                prods = inst.producers[k]
                assert len(prods) == et.wait_count, (ev, k, len(prods), et.wait_count)
                assert {plan.tasks[plan.task_index[p]].grid for p in prods} == {f"xb_qkv_{L}"}
            # its consumers are exactly the request's own qabs and attention grids, as in the default graph
            consumers = {t.grid for t in plan.tasks if any(e[0] == ev for e in inst.task_in[(t.grid, t.coord)])}
            assert consumers == {f"r{r}_qabs_{L}", f"r{r}_attn_{L}"}
        # it waits for every request's normalized vector on its own XCD
        for t in tasks:
            assert {e[0] for e in inst.task_in[(t.grid, t.coord)]} == {f"r{r}_E_xbn_{L}" for r in range(n)}
        for r in range(n):
            # the norm event: one signal per XCD index, from that XCD's norm task
            ev = f"r{r}_E_xbn_{L}"
            assert g.events[ev].wait_count == 1
            idxs = [k for k in inst.producers if k[0] == ev]
            assert len(idxs) == dsv.XCDS
            for k in idxs:
                assert [plan.tasks[plan.task_index[p]].grid for p in inst.producers[k]] == [f"r{r}_xbn_{L}"]
            # the norm grid waits for the request's layer input
            want = f"r{r}_E_embed" if L == 0 else f"r{r}_E_dense_0" if L == 1 else f"r{r}_E_down_{L - 1}"
            for t in (t for t in plan.tasks if t.grid == f"r{r}_xbn_{L}"):
                assert {e[0] for e in inst.task_in[(t.grid, t.coord)]} == {want}


def test_both_slots_work_on_it_and_queues_keep_creation_order(xb_plan):
    n, kva, plan = xb_plan
    g = plan.graph
    assert plan.wg_per_cu == 2
    per = plan.workers_per_domain
    order = {gr.name: i for i, gr in enumerate(g.grids)}
    for t in plan.tasks:
        local = t.worker - t.domain * per
        if t.grid.startswith("xb_qkv_"):
            w = t.coord[1]
            assert t.domain == t.coord[0] and local == w % dsv.W + (w // dsv.W) * dsv.CUS
        elif "_xbn_" in t.grid:
            r = int(t.grid.split("_", 1)[0][1:])
            assert t.domain == t.coord[0] and local == dsv.xb_norm_worker(r)
            assert (r % 2) * dsv.CUS <= local < (r % 2) * dsv.CUS + dsv.W
        else:
            r = int(t.grid.split("_", 1)[0][1:])
            assert (r % 2) * dsv.CUS <= local < (r % 2) * dsv.CUS + dsv.W
    tasks = {t.id: t for t in plan.tasks}
    for queue in plan.static_queues.values():
        idx = [order[tasks[i].grid] for i in queue]
        assert idx == sorted(idx)
    # creation order: layer L's norm grids after every request's layer L-1 grids, then the batched grid, then every
    # other layer-L grid
    for L in range(3):
        x = order[f"xb_qkv_{L}"]
        norms = [order[f"r{r}_xbn_{L}"] for r in range(n)]
        assert max(norms) < x
        for name, i in order.items():
            if name.startswith("r") and name.rsplit("_", 1)[-1].lstrip("-").isdigit():
                layer = int(name.rsplit("_", 1)[-1])
                if layer == L and "_xbn_" not in name:
                    assert i > x, name
                elif 0 <= layer < L:
                    assert i < min(norms), name


def test_protocol_simulation_has_no_deadlock_and_runs_every_task(xb_plan):
    from etx.sim.protocol import simulate
    n, kva, plan = xb_plan
    r = simulate(plan)
    assert not r.deadlock, r.blocked[:8]
    assert len({s.task for s in r.segments if s.kind == "task"}) == len(plan.tasks)


def test_routing_handoff_still_adjacent(xb_plan):
    n, kva, plan = xb_plan
    tasks = {t.id: t for t in plan.tasks}
    count = 0
    for queue in plan.static_queues.values():
        for i, tid in enumerate(queue):
            gu = tasks[tid]
            if "_gate_up_" in gu.grid:
                down = tasks[queue[i + 1]]
                assert down.grid == gu.grid.replace("_gate_up_", "_down_") and down.coord == gu.coord
                count += 1
    assert count == n * 2 * dsv.XCDS * dsv.W          # MoE layers 1 and 2 (layer 0 is dense)


@pytest.mark.parametrize("env,match", [
    (dict(ETX_DSV2_REQUESTS="1"), "2..4 requests"),
    (dict(ETX_DSV2_REQUESTS="8", ETX_DSV2_COUPLE="none"), "2..4 requests"),
    (dict(ETX_DSV2_REQUESTS="2", ETX_DSV2_COUPLE="router"), "ETX_DSV2_COUPLE=none"),
    (dict(ETX_DSV2_REQUESTS="2", ETX_DSV2_COUPLE="none", ETX_DSV2_D5="1"), "ETX_DSV2_D5"),
    (dict(ETX_DSV2_REQUESTS="2", ETX_DSV2_COUPLE="none", ETX_DSV2_ISOLATE="0:qkv_1"), "ISOLATE"),
    (dict(ETX_DSV2_REQUESTS="2", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="2"), "0 \\| 1"),
])
def test_unsupported_combinations_are_refused(monkeypatch, env, match):
    monkeypatch.setenv("ETX_DSV2_QKV_XB", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match=match):
        dsv.build(layers=2)


def test_arms_differ_from_their_reference_only_by_the_switch():
    from examples.dsv2lite.arms import ARMS, build_env
    for n in (2, 4):
        a, b = ARMS[f"x{n}"], ARMS[f"x{n}xb"]
        assert (b.levers, b.flags, b.scratch_allow) == (a.levers, a.flags, a.scratch_allow)
        assert build_env(b) == {**build_env(a), "ETX_DSV2_QKV_XB": "1"}
        assert f"-DETX_DS_REQUESTS={n}" in b.flags and b.rule is None and b.order == ()
    assert "ETX_DSV2_QKV_XB" not in ARMS["x4"].env


# ---------------------------------------------------------------- the row split of gemv_rows_xb
def _rows_written(N, xcd, n_xcds, worker, nw, waves=4, rpi=1):
    """Python mirror of gemv_rows_xb's row loop (fleet's xcd_rows, first / cnt, the two-buffer pipeline)."""
    per = (N + n_xcds - 1) // n_xcds
    begin, end = xcd * per, min(N, xcd * per + per)
    first = begin + worker
    if first >= end:
        return []
    cnt = (end - first + nw - 1) // nw
    stride = rpi * waves
    out = []
    for wave in range(waves):
        k = rpi * wave
        if k >= cnt:
            continue

        def reduce(kk):
            out.extend(first + (kk + r) * nw for r in range(rpi) if kk + r < cnt)
        while True:
            k2 = k + stride
            reduce(k)
            if k2 >= cnt:
                break
            k3 = k2 + stride
            reduce(k2)
            if k3 >= cnt:
                break
            k = k3
    return out


def test_mirror_matches_the_source_lines():
    for line in ("const int first = sl.begin + worker;",
                 "const int cnt = (sl.end - first + n_workers - 1) / n_workers;",
                 "const int stride = kRPI * fleet::kWaves;",
                 "int k = kRPI * wave;",
                 "if ((KK) + r < cnt) {",
                 "const int n = first + ((KK) + r) * n_workers;",
                 "constexpr int kRPI = 1;",
                 "for (int seg = 0; seg < 2; ++seg) {",
                 "const bool kv = seg != 0;",
                 "gemv_rows_xb(wq + (int64_t)(kv ? kv0 : q0) * d.hidden, d.hidden, s, out + (kv ? kMaxB : 0), kv ? kv_rows : q_rows,",
                 "d.hidden, kv && kvs ? xcd : 0, kv && kvs ? fleet::kNumXCDs : 1, worker, nw);",
                 "tab[kMaxB + threadIdx.x] = p.act.kv_a + (kvs ? 0 : (int64_t)xcd * kv_rows);",
                 "constexpr int kFold = 1, kKvs = 2;",
                 "const bool kvs = (c->cst[3] & kKvs) != 0;",
                 "const bool fold = (c->cst[2] & kFold) != 0;",
                 "tab[threadIdx.x] = p.act.q + q0;",
                 "fleet::epi_st(ys[j] + n, fleet::bf16_round(1.f * fleet::bf16_round(acc[r][j])), false);"):
        assert line in XB_SRC, line


@pytest.mark.parametrize("kvs", [False, True])
def test_row_split_writes_every_row_exactly_once_per_request(kvs):
    heads, q_head_dim, kv_rows = 16, 192, 512 + 64
    q_rows = heads // dsv.XCDS * q_head_dim            # 384 per XCD
    q_hits = np.zeros(heads * q_head_dim, dtype=int)
    kv_hits = np.zeros((1 if kvs else dsv.XCDS, kv_rows), dtype=int)
    for xcd in range(dsv.XCDS):
        for w in range(dsv.XB_W):
            for n in _rows_written(q_rows, 0, 1, w, dsv.XB_W):
                q_hits[xcd * q_rows + n] += 1
            if kvs:
                for n in _rows_written(kv_rows, xcd, dsv.XCDS, w, dsv.XB_W):
                    kv_hits[0, n] += 1
            else:
                for n in _rows_written(kv_rows, 0, 1, w, dsv.XB_W):
                    kv_hits[xcd, n] += 1
    # the store loop writes all B requests at each row, so per request the counts are these
    assert (q_hits == 1).all() and (kv_hits == 1).all()


# ---------------------------------------------------------------- numbers
def _bf16(x):
    """float32 -> nearest bf16 (round to nearest even), returned as float32 (fleet's bf16_round / __float2bfloat16)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def _rms_norm(v, gamma, eps=np.float32(1e-6)):
    """fleet::rms_norm_lds: fp32 sum of squares, then bf16(bf16(v * inv) * gamma)."""
    ssq = np.sum(v.astype(np.float32) * v.astype(np.float32), dtype=np.float32)
    inv = np.float32(1.0) / np.sqrt(ssq / np.float32(v.size) + eps, dtype=np.float32)
    return _bf16(_bf16(v * inv) * gamma)


def _wave_rows(Wt, x):
    """fleet's per-row arithmetic: lane l owns 16-byte chunks l + 64 d (d < 4), fma8's left-to-right fp32 adds, then the
    fixed xor butterfly over 64 lanes. Wt: [rows, 2048] bf16-valued float32, x: [2048] float32. Returns fp32 sums."""
    rows = Wt.shape[0]
    w = Wt.reshape(rows, 4, 64, 8)                    # [row, d, lane, element]
    xv = x.reshape(4, 64, 8)
    acc = np.zeros((rows, 64), dtype=np.float32)
    for d in range(4):
        for e in range(0, 8, 2):
            acc = acc + (w[:, d, :, e] * xv[d, :, e] + w[:, d, :, e + 1] * xv[d, :, e + 1])
    for off in (32, 16, 8, 4, 2, 1):
        acc = acc + acc[:, np.arange(64) ^ off]
    return acc[:, 0]


def _xb_rows(Wt, xs_bf16):
    """gemv_rows_xb: the same per-lane order for each of the B vectors, from the bf16 copies."""
    return np.stack([_wave_rows(Wt, x.astype(np.float32)) for x in xs_bf16])


def test_batched_math_against_the_single_column_path():
    rng = np.random.default_rng(20261017)
    hidden, rows, B = 2048, 3072 + 576, 4
    Wt = _bf16(rng.standard_normal((rows, hidden), dtype=np.float32) * np.float32(0.02))
    gamma = _bf16(1.0 + 0.1 * rng.standard_normal(hidden, dtype=np.float32))
    residual = [_bf16(rng.standard_normal(hidden, dtype=np.float32) * np.float32(s)) for s in (0.5, 1.0, 2.0, 8.0)]
    x32 = [_rms_norm(r, gamma) for r in residual]                       # fleet's s.vec after the norm (fp32 storage)
    x16 = [_bf16(x) for x in x32]                                       # fleet_xb.hip's LDS slot (bf16 storage)
    # 1. the bf16 copy is exact: the norm's output is already bf16-valued
    assert all(np.array_equal(a, b) for a, b in zip(x32, x16))
    single = np.stack([_wave_rows(Wt, x) for x in x32])                # fleet: one request at a time, fp32 vector
    batched = _xb_rows(Wt, x16)
    out_single, out_batched = _bf16(single), _bf16(batched)             # EPI_BF16
    # 2. same arithmetic order -> bitwise equal pre-rounding sums and outputs
    assert np.array_equal(single, batched) and np.array_equal(out_single, out_batched)
    # 3. both against fp64 (the shared fp32 accumulation error; context for the gate)
    ref = np.stack([Wt.astype(np.float64) @ x.astype(np.float64) for x in x32])
    abs_err = np.abs(batched - ref)
    rel = abs_err / np.maximum(np.abs(ref), 1e-3)
    assert abs_err.max() < 2e-5 and np.median(rel) < 1e-5
    # 4. sensitivity: a vector that is NOT bf16-valued would be rounded by the bf16 slot (this is why 1. matters)
    raw = residual[1] / np.float32(np.sqrt(np.mean(residual[1] ** 2)) + 1e-3) + np.float32(1e-3)
    lossy = _bf16(_wave_rows(Wt, _bf16(raw))) - _bf16(_wave_rows(Wt, raw.astype(np.float32)))
    assert np.abs(lossy).max() > 0                                      # the check above is not vacuous


def test_bf16_slot_layout_is_checked_at_compile_time():
    """fleet_xb.hip reuses fleet::Scratch (no new LDS): the static_asserts pin the slot regions; graph.py keeps the
    tile's declared LDS at SCRATCH_BYTES like every other tile."""
    for text in ('static_assert(sizeof(fleet::Scratch::part) >= 2 * kVecBytes',
                 'sizeof(fleet::Scratch::row) + sizeof(fleet::Scratch::q_c) + sizeof(fleet::Scratch::q_pe) >= kVecBytes',
                 'static_assert(sizeof(fleet::Scratch::vec) >= kVecBytes',
                 'static_assert(sizeof(fleet::Scratch::logits) >= 3 * kMaxB * sizeof(void*)',
                 'static_assert(kB >= 2 && kB <= kMaxB'):
        assert text in XB_SRC
    assert dsv.XB_MAX_REQUESTS == 4 and "constexpr int kMaxB = 4;" in XB_SRC


def test_norm_tile_hands_over_the_exact_bf16_vector_through_x_norm():
    """fleet_xb_norm writes this XCD's slice of act.x_norm (bf16, 16 bytes per thread) and the batched tile reads the
    same slice into LDS with the build's activation loads; the bf16 copy is exact (test above)."""
    for text in ("uint4* dst = reinterpret_cast<uint4*>(act.x_norm + (int64_t)xcd * d.hidden);",
                 "tab[2 * kMaxB + threadIdx.x] = p.act.x_norm + (int64_t)xcd * d.hidden;",
                 "slot(s, j)[threadIdx.x] = ld_act16(reinterpret_cast<const uint4*>(tab[2 * kMaxB + j]) + threadIdx.x, coh);",
                 "fleet::stage_residual(d, src, parts, (L == 0) ? nullptr : act.x[0], s.vec, xcd, 0, run_task_coh(fp->rt));",
                 "return fleet::ld_l1_u4(p);"):
        assert text in XB_SRC, text


# ---------------------------------------------------------------- the duration model
def test_duration_model_per_slot_qkv_time():
    """Per layer and slot: x4 runs two requests' qkv tasks (prologue 7.4 + GEMV 12.0 us each); x4xb runs one norm task
    (only on the slot's norm workers) and one batched task with half the rows."""
    x4 = 2 * dsv.DUR["qkv"]
    xb4 = dsv.XB_NORM_US + dsv.xb_qkv_us(4)
    xb2 = dsv.XB_NORM_US + dsv.xb_qkv_us(2)
    assert x4 == pytest.approx(38.8) and dsv.DUR["qkv"] == pytest.approx(19.4)
    assert dsv.xb_qkv_us(4) == pytest.approx(8.0) and xb4 == pytest.approx(15.4)
    assert dsv.xb_qkv_us(2) == pytest.approx(7.0) and xb2 == pytest.approx(14.4)


# ---------------------------------------------------------------- the occupancy gate
@pytest.mark.parametrize("res,want", [
    (dict(vgpr=248, agpr=6, lds=26668, occupancy=2), 2),    # x4 / x4xb, ROCm 7.2.4
    (dict(vgpr=256, agpr=0, lds=26668, occupancy=2), 2),    # x4 / x4xb, ROCm 7.14
    (dict(vgpr=250, agpr=6, lds=26668, occupancy=1), 1),    # an x4xb variant: 252 + 6 rounds to 264
    (dict(vgpr=256, agpr=6, lds=26668, occupancy=1), 1),    # PR #20 head 9ee4d40, ROCm 7.2.4
    (dict(vgpr=128, agpr=0, lds=40000, occupancy=4), 1),    # LDS-bound
    (dict(vgpr=250, agpr=6, lds=26668, occupancy=2), 1),    # registers decide even if the record says 2
])
def test_wg_per_cu_from_registers_and_lds(res, want):
    from examples.dsv2lite.build_resources import wg_per_cu
    assert wg_per_cu(dict(res, sgpr=108, scratch=0)) == want


def test_session_refuses_a_paired_arm_at_one_workgroup_per_cu(tmp_path):
    from examples.dsv2lite import inplace_session as sess
    from examples.dsv2lite.arms import ARMS
    assert sess.required_wg_per_cu(ARMS["x4xb"]) == 2 and sess.required_wg_per_cu(ARMS["x2xb"]) == 2
    assert sess.required_wg_per_cu(ARMS["x4"]) == 2 and sess.required_wg_per_cu(ARMS["default"]) == 1

    def record(vgpr, agpr, occ):
        return ("remark: Function Name: etx_megakernel_d0\n"
                + "".join(f"remark:     {k}: {v}\n" for k, v in (
                    ("VGPRs", vgpr), ("AGPRs", agpr), ("TotalSGPRs", 108), ("ScratchSize [bytes/lane]", 0),
                    ("Occupancy [waves/SIMD]", occ), ("LDS Size [bytes/block]", 26668))))
    s = object.__new__(sess.Session)
    s.out = lambda arm, stamped=False: tmp_path / arm
    s.fleet = tmp_path
    for arm in ("x4xb", "default"):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "plan.json").write_text("{}")
    s.run = lambda *a, **k: record(256, 6, 1)
    with pytest.raises(RuntimeError, match="needs 2"):
        sess.Session.build(s, "x4xb")
    sess.Session.build(s, "default")                      # one workgroup per CU is a single-request arm's normal
    s.run = lambda *a, **k: record(248, 6, 2)
    sess.Session.build(s, "x4xb")
