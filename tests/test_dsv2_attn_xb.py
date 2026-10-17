"""Attention blocking on top of x4xbor (ETX_DSV2_ATTN_XB=1, B = 4; graph.py build_attention_block, fleet_xb.hip
fleet_attention8 / fleet_attention8_kvs / fleet_merge_uv8; docs/ATTN-XB-BOUND-2026-10-18.md). CPU only, no GPU:

  * default off: the plan and the kernel of every ETX arm are byte-identical to 0c641ec (hash pins), and x4xbor with
    ETX_DSV2_ATTN_XB=0 too;
  * the new tiles are compiled only with -DETX_DSV2_ATTN_XB=1 and pass n_split = 8 as a compile-time constant;
  * on: grids (16, 8), symbols, arguments (no new tensor), wait counts E_attn 8 / E_merge 16, producers and consumers,
    the 16-worker block of each request, queue order (qabs before attention before merge on every worker), the 2
    requests of a slot on disjoint workers, no deadlock in the protocol simulator;
  * unsupported combinations and bad switch values are refused; x4xbora is x4xbor plus the switch and the define;
  * levers: merge_uv_waves and gemv_rows cover the 16 W_UV rows of a merge task once each; attn_prologue_live does not
    touch the chunk split or the cache writer;
  * numbers (a numpy mirror of fleet's attention_chunk and merge_and_uv, fp32): the task path with 8 chunks (partial
    layout head * 8 + chunk, 16 W_UV rows per merge task) is bitwise equal to a direct 8-chunk reference; against the
    16-chunk path the result is within the stated fp32 tolerance (not bitwise: the merge adds 8 rescaled partials, not
    16); the greedy token of a test case does not change.
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
N_REQ, LAYERS, CH = 4, 3, 8

# 0c641ec (PR #25 head), 3 layers, PassOptions(inline_tiles=True, relay="on"), env = arms.build_env(arm):
# (arms, plan.json sha256, megakernel sha256). Every ETX arm except x4xbora is listed (explicit set, checked below).
PINS = [
    (('default', 'Q', 'Qall', 'N', 'Nall', 'argmax', 'topk', 'prep', 'relaxed', 'all', 'allrelaxed', 'allnoQ'),
     "02d85386eaedfcda77b2121da14ab8255cef352531734621ce75a608e2d97450",
     "37f938c1a78378cb6fe2e66785dab9ec6be3165f85509e77d39892d358cfcd20"),
    (('coh',),
     "e488de7bf350bd9223b8d7ff1e1f44c2ef1466b453329c5f857d935311b50b2f",
     "0ee9761331f27f0a567f371cd7a060db68626d663eed66d05e12d9a47a0d06e8"),
    (('d4', 'alld4s'),
     "be8e1b9e237c26a26a9c452f96768cec070624dac4671ff0d616576a3c31368d",
     "40b0a0cff2cbdd8fe6b113d26ef9adbc2230d3b48b1196ea31fb4f7afa48efc4"),
    (('d4r0', 'alld4r0'),
     "be8e1b9e237c26a26a9c452f96768cec070624dac4671ff0d616576a3c31368d",
     "e404ac9ebc5569f0f8eb80af44211c350e0a866cb9f6d3db6f7e2c62cd84e146"),
    (('l1',),
     "d5be7a4bede1fb2ebfb16461562cd6afa70d3e6d2b737da917f5a818f11d71ac",
     "340642d89b30ad06bdd05c7ba4dcb4ce57c52de3c4d6db9fe570c79f9c8b90b2"),
    (('alld4r0l1', 'alld4r0l1noQ', 'alld4r0l1noA', 'alld4r0l1mw'),
     "0ba0fd1178793dda3b0a6549e8d0c2d0ab2ac6c21c27d9c8034b4e791b80011c",
     "3b8bbae027523ad7b77501ac6b7be758e6dff4afbc00874d3ec2b1a86d8adfd8"),
    (('alld3',),
     "d94ba3ccd2281655e71cdd968f2ef23f68f273b72d5de6567157b51d0e4b0f68",
     "bb365e4cdb77f277e3ad37f24ba4bdcccbf9e2332eac3d65286df2472668247d"),
    (('alld5',),
     "2dddec7c19714c8abb07624f2dea95c57d38a102a8e30f19e8c01d51637f4649",
     "5e4205144d9a80f0397fe3d8f027c7c39807f5e0a7fa3eece8786c02d23fe4f8"),
    (('alld4r0l1kvs', 'alld4r0l1kvsmw'),
     "bca1227d22e5f1da3959165b4883c9bd56335fe5137cc799051dc7856438c5e5",
     "66350715e757cb0fc2e93032793e4c0e4be84bb749e78822c17487de3d44ea9e"),
    (('x1km',),
     "99f7c9d2288341023b3c6cf20e4f8cd9cc6a25b084591a4863ce244d174c7012",
     "26c896e5282ce744a517bc8e58d7e67f6d8ea9d1c384e7957cf0df23cfd3627b"),
    (('x2',),
     "26bc57c6236a9542071fe1c2b69608859191b4a9eddecfed27d80502fde459d4",
     "b8213a370d0b51eecb0aeb1d75fbe052fc620277120c5ec6dddea819b2366bee"),
    (('x4',),
     "910dad04c3e94e71f60abc56a0e662d63891335935393292f4fa0e5fb9c5f3e7",
     "60b67e9c8f029bd4cdfb60423539d600fca76179fed159502bd7bcf36c385949"),
    (('x2xb',),
     "8873edb92ce9c91a9531fd6e379c67fb8fd33971eeaf46f1f2a468a15de1fe05",
     "52cb4f06a7060a4701ac9799ee8a7fb8588771f6392f5094f096d042a14c218f"),
    (('x4xb',),
     "9e8c37bc11a4fe97535615be3c99b381adaaa0240f9b0fde8627d09ccb4b72bc",
     "f711d266334884bd62c554cca1e170e922669db68a23cd0e370100a5e57a50d3"),
    (('x4xbo',),
     "aa45fe1d960b6e7540f8d812303aae8e7f9bba01aa7c6db602f24f46f88c4fe2",
     "7ea6324df95ca6011b385d3e7cc007f98000064b0d1b9c395719c3ec5165fa3a"),
    (('x4xbl',),
     "7022f073ff4d80640fabb01e54918034f07678f1bd028c08141908b31e394836",
     "12012ceaabbf22a3d0686b63ec057d138b9781b337f06399fccb35ebb5ae1acd"),
    (('x4xboq',),
     "69269ebbcf8ab7eed6555baa69b9876c30dfbf1ab3d2cd54c48bbd4820838ffa",
     "6efa4d24bddfea4cb4f66f8c8a5ef34f37d4a776894b445006b97458721b324f"),
    (('x4xbor',),
     "79238918574f40755638480778df0da9fe1485c14be0d28ba8ca4e75fd2a9b4a",
     "2ab468053d3c15976759c071e28c228c1747a19a310ce6dbf6b975249edf0b3d"),
    (('x8p', 'x8m'),
     "bf4928715c957e8598ae887f20dbd1edea710d14c6a9d25e09dd2607baa86bcf",
     "052d29fcd48240b0505be78f4d78c6e4a49080b5a876d0bac501b2e734af758e"),
    (('x8k', 'x8'),
     "7d5e3084650aae0edaf7e41da691e8bb69ea50cf307d98b97f39b50d16ce2fe6",
     "cad795fe1baaa5d99ab423e0ebf9d2197282ff893e556bf8b73d91b3b2b7580f"),
]
PIN_OF = {a: (p, k) for names, p, k in PINS for a in names}


@pytest.fixture(autouse=True)
def clean_graph_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("ETX_DSV2_") or key in ("ETX_FLEET_PATCH", "ETX_CFLAGS"):
            monkeypatch.delenv(key)


def _compile(env: dict, layers=LAYERS):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return compile_graph(dsv.build(layers=layers), "gfx942", {}, {}, PassOptions(inline_tiles=True, relay="on"))
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _hashes(plan):
    return (hashlib.sha256(emit_plan_json(plan).encode()).hexdigest(),
            hashlib.sha256(emit_kernel(plan).encode()).hexdigest())


def _env(arm):
    from examples.dsv2lite.arms import build_env
    return build_env(arm)


# ---------------------------------------------------------------- default off: byte-identical
def test_pins_list_every_etx_arm_but_x4xbora():
    from examples.dsv2lite.arms import ARMS
    assert {n for n, a in ARMS.items() if a.builder == "etx"} == set(PIN_OF) | {"x4xbora"}
    assert not set(PIN_OF) & {"x4xbora"}


@pytest.mark.parametrize("arm", sorted(PIN_OF))
def test_default_off_every_arm_is_byte_identical_to_0c641ec(arm, _cache={}):
    from examples.dsv2lite.arms import ARMS
    env = _env(ARMS[arm])
    key = tuple(sorted(env.items()))
    if key not in _cache:
        _cache[key] = _hashes(_compile(env))
    assert _cache[key] == PIN_OF[arm]


def test_switch_zero_keeps_x4xbor_byte_identical():
    from examples.dsv2lite.arms import ARMS
    plan = _compile(dict(_env(ARMS["x4xbor"]), ETX_DSV2_ATTN_XB="0"))
    assert _hashes(plan) == PIN_OF["x4xbor"]
    assert not any(g.body.symbol in ("fleet_attention8", "fleet_attention8_kvs", "fleet_merge_uv8") for g in plan.graph.grids)


def test_tiles_are_compiled_only_with_their_define():
    i = XB_SRC.index("#if defined(ETX_DSV2_ATTN_XB) && ETX_DSV2_ATTN_XB")
    block = XB_SRC[i:]
    assert XB_SRC.rstrip().endswith("#endif") and "#endif" in XB_SRC[:i]
    for sym in ("fleet_attention8", "fleet_attention8_kvs", "fleet_merge_uv8"):
        assert f'extern "C" __device__ void {sym}(' in block and f"void {sym}(" not in XB_SRC[:i]
    kvs = block.index('void fleet_attention8_kvs(')
    assert "#if defined(ETX_DSV2_KVA_SHARED) && ETX_DSV2_KVA_SHARED" in block[:kvs]
    assert "constexpr int kAttnXbChunks = 8;" in block and dsv.ATTN_XB_CHUNKS == 8
    # n_split is the constant, in both bodies; the rest of the attention descriptor is fleet_attention's
    assert "t.n_split = kAttnXbChunks;" in block
    assert "fleet_call(c, fleet::TASK_MERGE_UV, 0, c->coord[0], c->coord[1], etx_xb::kAttnXbChunks, -1, 0);" in block
    shim = (ROOT / "examples/dsv2lite/fleet_shim.hip").read_text()
    body = shim[shim.index('extern "C" __device__ void fleet_attention(const etx_ctx* c) {'):]
    body = body[:body.index("\n}\n")]
    for line in body.splitlines()[1:]:
        line = line.strip().replace("t.n_split = fleet::kMaxKvChunks;", "t.n_split = kAttnXbChunks;")
        line = line.replace("t.flags = fleet::FLAG_PUB_WAIT;", "t.flags = (decltype(t.flags))kFlags;")
        assert line in block, line


# ---------------------------------------------------------------- on: the lowered plan
@pytest.fixture(scope="module", params=["shared", "replicated"])
def plan_on(request):
    from examples.dsv2lite.arms import ARMS
    env = {k: v for k, v in _env(ARMS["x4xbora"]).items() if k.startswith("ETX_DSV2_")}
    return request.param, _compile(dict(env, ETX_DSV2_KVA=request.param))


def _local(plan, t):
    return t.worker - t.domain * plan.workers_per_domain


def test_grids_symbols_and_arguments(plan_on):
    kva, plan = plan_on
    g = plan.graph
    by = {gr.name: gr for gr in g.grids}
    sfx = "_kvs" if kva == "shared" else ""
    for L in range(LAYERS):
        for r in range(N_REQ):
            a, m, q = by[f"r{r}_attn_{L}"], by[f"r{r}_merge_{L}"], by[f"r{r}_qabs_{L}"]
            assert tuple(a.grid) == (dsv.HEADS, CH) and a.body.symbol == "fleet_attention8" + sfx
            assert tuple(m.grid) == (dsv.HEADS, CH) and m.body.symbol == "fleet_merge_uv8"
            assert tuple(q.grid) == (dsv.HEADS, dsv.QABS) and q.body.symbol == "fleet_q_absorb"
            for gr in (a, m, q):
                assert gr.body.source == dsv.XB_T and gr.args == [f"fleet{r}", f"layer_{L}"] and tuple(gr.consts) == (L, r)
            assert a.duration_us == dsv.ATTN_XB_DUR["attn"] and m.duration_us == dsv.ATTN_XB_DUR["merge"]
    assert not any(gr.body.symbol in ("fleet_attention", "fleet_attention_kvs", "fleet_merge_uv") for gr in g.grids)
    # no new argument tensor: the x4xbor plan's set
    assert set(g.tensors) == {"layer_-1", *(f"layer_{L}" for L in range(LAYERS)), *(f"fleet{r}" for r in range(N_REQ))}
    assert plan.wg_per_cu == 2


def test_wait_counts_producers_and_consumers(plan_on):
    _, plan = plan_on
    inst, g = plan.inst, plan.graph
    for L in range(LAYERS):
        for r in range(N_REQ):
            ea, em = f"r{r}_E_attn_{L}", f"r{r}_E_merge_{L}"
            assert g.events[ea].wait_count == CH and g.events[em].wait_count == (dsv.HEADS // dsv.XCDS) * CH == 16
            for h in range(dsv.HEADS):
                prods = inst.producers[(ea, (h,))]
                assert sorted(plan.tasks[plan.task_index[p]].coord for p in prods) == [(h, c) for c in range(CH)]
                assert {plan.tasks[plan.task_index[p]].grid for p in prods} == {f"r{r}_attn_{L}"}
            for x in range(dsv.XCDS):
                prods = inst.producers[(em, (x,))]
                assert len(prods) == 16 and {plan.tasks[plan.task_index[p]].grid for p in prods} == {f"r{r}_merge_{L}"}
            cons_a = {t.grid for t in plan.tasks if any(e[0] == ea for e in inst.task_in[(t.grid, t.coord)])}
            cons_m = {t.grid for t in plan.tasks if any(e[0] == em for e in inst.task_in[(t.grid, t.coord)])}
            assert cons_a == {f"r{r}_merge_{L}"} and cons_m == {f"xb_oproj_{L}"}
            for t in (t for t in plan.tasks if t.grid == f"r{r}_merge_{L}"):   # merge (h, c) waits for head h only
                assert set(inst.task_in[(t.grid, t.coord)]) == {(ea, (t.coord[0],))}


def test_each_request_on_its_16_worker_block(plan_on):
    _, plan = plan_on
    for L in range(LAYERS):
        used = {}
        for r in range(N_REQ):
            lo = (r % 2) * dsv.CUS + (r // 2) * 16
            for kind in ("qabs", "attn", "merge"):
                for t in (t for t in plan.tasks if t.grid == f"r{r}_{kind}_{L}"):
                    h, c = t.coord
                    assert t.domain == h // 2
                    want = (h % 2) * (2 if kind == "qabs" else CH) + c
                    assert _local(plan, t) == lo + want
                    if kind == "qabs":
                        assert want < 4
                    used.setdefault((t.domain, _local(plan, t)), set()).add(r)
        # the 2 requests of a slot on disjoint workers: r and r + 2 never share one
        assert all(len(v) == 1 for v in used.values())
        assert len(used) == dsv.XCDS * 16 * N_REQ


def test_queue_order_qabs_attention_merge(plan_on):
    _, plan = plan_on
    g = plan.graph
    order = {gr.name: i for i, gr in enumerate(g.grids)}
    tasks = {t.id: t for t in plan.tasks}
    for queue in plan.static_queues.values():
        idx = [order[tasks[i].grid] for i in queue]
        assert idx == sorted(idx)
        for L in range(LAYERS):
            for r in range(N_REQ):
                pos = {k: [i for i, tid in enumerate(queue) if tasks[tid].grid == f"r{r}_{k}_{L}"] for k in ("qabs", "attn", "merge")}
                if pos["qabs"] and pos["attn"]:
                    assert max(pos["qabs"]) < min(pos["attn"])     # the in-body q_c wait cannot wait on its own worker
                if pos["attn"] and pos["merge"]:
                    assert max(pos["attn"]) < min(pos["merge"])
    for L in range(LAYERS):
        assert order[f"xb_qkv_{L}"] < min(order[f"r{r}_qabs_{L}"] for r in range(N_REQ))
        assert max(order[f"r{r}_merge_{L}"] for r in range(N_REQ)) < order[f"xb_oproj_{L}"]


def test_protocol_simulation_has_no_deadlock(plan_on):
    from etx.sim.protocol import simulate
    _, plan = plan_on
    for seed in range(3):
        r = simulate(plan, seed=seed)
        assert not r.deadlock, r.blocked[:8]
    assert len({s.task for s in r.segments if s.kind == "task"}) == len(plan.tasks)


@pytest.mark.parametrize("env", [
    dict(ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1"),
    dict(ETX_DSV2_REQUESTS="2", ETX_DSV2_COUPLE="none", ETX_DSV2_QKV_XB="1"),
    dict(ETX_DSV2_REQUESTS="4", ETX_DSV2_COUPLE="none"),
    dict(ETX_DSV2_REQUESTS="1"),
])
def test_unsupported_combinations_are_refused(monkeypatch, env):
    monkeypatch.setenv("ETX_DSV2_ATTN_XB", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match="ETX_DSV2_ATTN_XB=1 needs ETX_DSV2_OPROJ_XB=1"):
        dsv.build(layers=2)


def test_switch_value_is_checked(monkeypatch):
    monkeypatch.setenv("ETX_DSV2_ATTN_XB", "2")
    with pytest.raises(ValueError, match="0 \\| 1"):
        dsv.build(layers=2)


def test_arm():
    from examples.dsv2lite.arms import ARMS
    base, a = ARMS["x4xbor"], ARMS["x4xbora"]
    assert a.env == dict(base.env, ETX_DSV2_ATTN_XB="1") and a.levers == base.levers
    assert a.flags == (*base.flags, "-DETX_DSV2_ATTN_XB=1") and a.scratch_allow == base.scratch_allow
    assert {n for n, x in ARMS.items() if "ETX_DSV2_ATTN_XB" in x.env} == {"x4xbora"}
    assert {n for n, x in ARMS.items() if "-DETX_DSV2_ATTN_XB=1" in x.flags} == {"x4xbora"}


# ---------------------------------------------------------------- levers with 8 chunks
needs_fleet = pytest.mark.skipif(not (FLEET_SRC / "kernels/fleet_kernel.hip").exists(), reason="pinned fleet source not present")


def _rows_of_pipelined(rpi, cnt, waves=4):
    """Rows that gemv_rows_pipelined<RPI, D>(.., first = 0, cnt, n_workers = 1) reduces, per wave (its loop)."""
    out = []
    for wave in range(waves):
        k = rpi * wave
        if k >= cnt:
            continue
        while True:
            out += [k + r for r in range(rpi) if k + r < cnt]
            k2 = k + rpi * waves
            if k2 >= cnt:
                break
            out += [k2 + r for r in range(rpi) if k2 + r < cnt]
            k = k2 + rpi * waves
            if k >= cnt:
                break
    return out


@needs_fleet
def test_merge_rows_are_covered_once_with_and_without_merge_uv_waves(tmp_path):
    gemv = (FLEET_SRC / "kernels/gemv.h").read_text()
    body = gemv[gemv.index("__device__ __forceinline__ void gemv_rows_pipelined("):]
    body = body[:body.index("#undef FLEET_ROW") if "#undef FLEET_ROW" in body else len(body)]
    for line in ("const int stride = RPI * kWaves;", "int k = RPI * wave;", "if (k >= cnt) return;",
                 "const int k2 = k + stride;", "if (k2 >= cnt) break;", "const int k3 = k2 + stride;",
                 "FLEET_REDUCE_GROUP(bufB, k2)", "if (k3 >= cnt) break;", "k = k3;"):
        assert line in body, line
    # fleet's merge_and_uv: rows = v_head / n_split per task, from row kv_chunk * rows
    k = (FLEET_SRC / "kernels/fleet_kernel.hip").read_text()
    assert "const int rows = d.v_head / chunks;" in k and "t.kv_chunk * rows, rows);" in k
    rows = 128 // CH
    for rpi in (2, 8):     # merge_uv_waves' <2, 1>; gemv_rows' <8, 1> for K = 512 (lever off)
        assert sorted(_rows_of_pipelined(rpi, rows)) == list(range(rows))
        assert sorted(_rows_of_pipelined(rpi, 128 // 16)) == list(range(8))
    from examples.dsv2lite.levers import merge_uv_waves
    assert "/*cnt=*/n_rows" in merge_uv_waves.NEW
    src = tmp_path / "src"
    shutil.copytree(FLEET_SRC, src)
    merge_uv_waves.apply(src)
    a = (src / "kernels/attention.h").read_text()
    assert "/*first=*/0, /*cnt=*/n_rows, /*n_workers=*/1" in a


@needs_fleet
def test_attn_prologue_live_keeps_the_chunk_split_and_the_cache_writer(tmp_path):
    from examples.dsv2lite.levers import attn_prologue_live
    src = tmp_path / "src"
    shutil.copytree(FLEET_SRC, src)
    attn_prologue_live.apply(src)
    k = (src / "kernels/fleet_kernel.hip").read_text()
    task = k[k.index("        case TASK_ATTENTION: {\n            __hip_bfloat16* cache_layer"):]
    task = task[:task.index("        case TASK_O_PROJ:")]
    assert "/*write_cache=*/t.head == 0 && t.kv_chunk == 0,   // one writer per layer" in task
    assert "float* partials = act.attn_partial + (int64_t)t.head * chunks * kPartialStride;" in task
    assert "tc.seq_len, kv.row, t.kv_chunk, chunks," in task
    assert "partials + (int64_t)t.kv_chunk * kPartialStride, s.part);" in task
    assert "const int chunks = max(1, (int)t.n_split);" in k
    lever = Path(attn_prologue_live.__file__).read_text()
    assert not re.search(r"\bkv_chunk\b|\bn_split\b|\bn_chunks\b", lever)


# ---------------------------------------------------------------- numbers: numpy mirror of fleet's attention
LANES, WAVES, KV_LORA, ROPE, V_HEAD, BATCH = 64, 4, 512, 64, 128, 8
f32 = np.float32


def _bf16(x):
    """Round fp32 to bf16 (nearest even), as fp32 values."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


def _wave_sum(v):
    """xor butterfly over the 64 lanes (last axis): every lane ends with the same sum (fp32 add is commutative)."""
    v = v.copy()
    off = LANES // 2
    while off:
        idx = np.arange(LANES) ^ off
        v = (v + v[..., idx]).astype(f32)
        off //= 2
    return v[..., 0]


def _lane_scores(qc, qp, rows):
    """attention_chunk's per-lane part: sum_j qc[j] * c[j] + qp[j] * kp[j] (qp = 0 off the 8 rope lanes), then the
    butterfly. rows: [P, 576] fp32 (bf16 values)."""
    c = rows[:, :KV_LORA].reshape(-1, LANES, 8)
    kp = np.tile(rows[:, KV_LORA:KV_LORA + ROPE].reshape(-1, 8, 8), (1, LANES // 8, 1))
    qcl, qpl = qc.reshape(LANES, 8), np.zeros((LANES, 8), f32)
    qpl[:8] = qp.reshape(8, 8)
    part = np.zeros(c.shape[:2], f32)
    for j in range(8):
        part = (part + (qcl[:, j] * c[:, :, j]).astype(f32) + (qpl[:, j] * kp[:, :, j]).astype(f32)).astype(f32)
    return _wave_sum(part)


def attention_chunk(cache, qc, qp, row_new, scale, seq_len, chunk, n_chunks):
    """fleet's attention_chunk: the chunk's positions over 4 waves (batches of 8, interleaved), online softmax per wave,
    wave 0 takes this token's row, then the wave partials in wave order. Returns [2 + 512]: m, l, acc."""
    per = (seq_len + n_chunks - 1) // n_chunks
    tb, te, tn = chunk * per, min(seq_len, chunk * per + per), seq_len - 1
    tc = min(te, tn)
    parts = []
    for wave in range(WAVES):
        m, l, acc = f32(-np.inf), f32(0), np.zeros(KV_LORA, f32)
        pos = [b + k for b in range(tb + wave * BATCH, tc, WAVES * BATCH) for k in range(BATCH) if b + k < tc]
        rows = [cache[p] for p in pos]
        if wave == 0 and tb <= tn < te:
            pos.append(tn); rows.append(row_new)
        if rows:
            s = _lane_scores(qc, qp, np.stack(rows))
            for i in range(len(rows)):
                si = f32(s[i] * scale)
                mn = max(m, si)
                rs, p = f32(np.exp(f32(m - mn))), f32(np.exp(f32(si - mn)))
                l, m = f32(l * rs + p), mn
                acc = (acc * rs + p * rows[i][:KV_LORA]).astype(f32)
        parts.append((m, l, acc))
    gm = max(p[0] for p in parts)
    sw = [f32(0) if p[0] == -np.inf else f32(np.exp(f32(p[0] - gm))) for p in parts]
    gl, v = f32(0), np.zeros(KV_LORA, f32)
    for (m, l, acc), w in zip(parts, sw):
        gl = f32(gl + l * w)
        v = (v + acc * w).astype(f32)
    return np.concatenate([[gm, gl], v]).astype(f32)


def merge_oc(partials, n_chunks, max_chunks=16):
    """merge_and_uv's merge: global max, rescale, the sum over the (predicated, max_chunks-unrolled) chunks."""
    m = [partials[c][0] if c < n_chunks else f32(-np.inf) for c in range(max_chunks)]
    l = [partials[c][1] if c < n_chunks else f32(0) for c in range(max_chunks)]
    gm = max(m)
    sc = [f32(np.exp(f32(m[c] - gm))) if c < n_chunks else f32(0) for c in range(max_chunks)]
    gl = f32(0)
    for c in range(max_chunks):
        gl = f32(gl + l[c] * sc[c])
    inv = f32(f32(1) / gl)
    v = np.zeros(KV_LORA, f32)
    for c in range(max_chunks):
        a = partials[c][2:] if c < n_chunks else np.zeros(KV_LORA, f32)
        v = (v + a * sc[c]).astype(f32)
    return (v * inv).astype(f32)


def uv_rows(w_uv, oc):
    """fleet's per-row GEMV (fma8 of the lane's 16-byte chunk, wave_sum, EPI_BF16) for K = 512 = 64 lanes x 8."""
    w = w_uv.reshape(-1, LANES, 8)
    x = oc.reshape(LANES, 8)
    acc = np.zeros(w.shape[:2], f32)
    for j in range(8):
        acc = (acc + w[:, :, j] * x[:, j]).astype(f32)
    return _bf16(_wave_sum(acc))


def task_path(case, n_chunks):
    """The plan's tasks: attention (h, c) writes partial h * n_chunks + c of one attn_partial buffer, merge (h, c)
    re-merges head h and writes its v_head / n_chunks rows from row c * rows."""
    cache, qc, qp, row_new, w_uv, scale, seq = case
    buf = np.full((dsv.HEADS * 16, 2 + KV_LORA), np.nan, f32)        # attn_partial, sized for kMaxKvChunks
    for h in range(dsv.HEADS):
        for c in range(n_chunks):
            buf[h * n_chunks + c] = attention_chunk(cache, qc[h], qp[h], row_new, scale, seq, c, n_chunks)
    o = np.full(dsv.HEADS * V_HEAD, np.nan, f32)
    rows = V_HEAD // n_chunks
    for h in range(dsv.HEADS):
        for c in range(n_chunks):
            oc = merge_oc(buf[h * n_chunks:(h + 1) * n_chunks], n_chunks)
            o[h * V_HEAD + c * rows:h * V_HEAD + (c + 1) * rows] = uv_rows(w_uv[h][c * rows:(c + 1) * rows], oc)
    return o


def reference(case, n_chunks):
    """Direct: per head, the n_chunks partials in a list, one merge, all 128 W_UV rows at once."""
    cache, qc, qp, row_new, w_uv, scale, seq = case
    o, ocs = [], []
    for h in range(dsv.HEADS):
        parts = [attention_chunk(cache, qc[h], qp[h], row_new, scale, seq, c, n_chunks) for c in range(n_chunks)]
        oc = merge_oc(parts, n_chunks)
        ocs.append(oc)
        o.append(uv_rows(w_uv[h], oc))
    return np.concatenate(o), np.stack(ocs)


def _case(seed, seq):
    rng = np.random.default_rng(seed)
    cache = _bf16(rng.standard_normal((max(seq - 1, 1), KV_LORA + ROPE)).astype(f32))
    row_new = _bf16(rng.standard_normal(KV_LORA + ROPE).astype(f32))
    qc = rng.standard_normal((dsv.HEADS, KV_LORA)).astype(f32) * f32(0.5)
    qp = rng.standard_normal((dsv.HEADS, ROPE)).astype(f32)
    w_uv = _bf16(rng.standard_normal((dsv.HEADS, V_HEAD, KV_LORA)).astype(f32) * f32(0.05))
    return cache, qc, qp, row_new, w_uv, f32(0.1352), seq


SEQS = (1, 7, 9, 130, 1024, 1025)


@pytest.mark.parametrize("seq", SEQS)
def test_8_chunk_task_path_is_bitwise_the_8_chunk_reference(seq):
    case = _case(seq, seq)
    o, _ = reference(case, CH)
    assert np.array_equal(task_path(case, CH), o)
    assert np.array_equal(task_path(case, 16), reference(case, 16)[0])     # the mirror on x4xbor's 16 chunks too


# Tolerance (8 against 16 chunks). The positions are the same; what changes is how they are grouped into chunks and
# waves, so the online-softmax rescaling and the final merge run in a different fp32 order. Bounds:
#   o_c (fp32, per head): max |o_c8 - o_c16| <= OC_REL x max |o_c16|; measured max 4.7e-7 (about 4 fp32 eps) on
#   these cases, bound 2e-6 (about 17 eps);
#   o (bf16 values after EPI_BF16): every element at most O_BF16_STEPS bf16 step of |o16|'s binade from the 16-chunk
#   value, and at most O_NEQ_FRAC of the elements differ at all; measured 1 step, 0.1 % (seq 130), 0.05 % (seq 1024).
# With seq_len <= 8 every chunk holds at most one position in both splits: the partials are the same and the extra
# empty chunks add exact zeros, so the result is bitwise equal.
OC_REL, O_BF16_STEPS, O_NEQ_FRAC = 2e-6, 1, 0.01


@pytest.mark.parametrize("seq", SEQS)
def test_8_chunks_against_16_chunks_within_the_fp32_tolerance(seq):
    case = _case(seq, seq)
    o8, oc8 = reference(case, CH)
    o16, oc16 = reference(case, 16)
    for h in range(dsv.HEADS):
        assert np.max(np.abs(oc8[h] - oc16[h])) <= OC_REL * np.max(np.abs(oc16[h]))
    step = np.exp2(np.floor(np.log2(np.maximum(np.abs(o16), np.float32(2.0 ** -126)))) - 7)
    assert np.all(np.abs(o8 - o16) <= O_BF16_STEPS * step)
    assert np.mean(o8 != o16) <= O_NEQ_FRAC
    if seq <= CH:
        assert np.array_equal(o8, o16) and np.array_equal(oc8, oc16)


@pytest.mark.parametrize("seed,seq", [(11, 130), (12, 1024), (13, 1025)])
def test_greedy_token_does_not_change(seed, seq):
    """A test case: the 16 heads' o (2048) through a random bf16 W_o and a random bf16 lm_head (8192 rows); the
    argmax with 8 chunks is the argmax with 16."""
    case = _case(seed, seq)
    o8, _ = reference(case, CH)
    o16, _ = reference(case, 16)
    rng = np.random.default_rng(seed + 100)
    w_o = _bf16(rng.standard_normal((2048, 2048)).astype(f32) * f32(0.02))
    w_lm = _bf16(rng.standard_normal((8192, 2048)).astype(f32) * f32(0.02))
    lg8, lg16 = w_lm @ _bf16(w_o @ o8), w_lm @ _bf16(w_o @ o16)
    assert int(np.argmax(lg8)) == int(np.argmax(lg16))
