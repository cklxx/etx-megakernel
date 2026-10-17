"""DeepSeek-Coder-V2-Lite-Base, batch-1 decode, one MI300X: fleet-mi300x's task
graph expressed in ETX (design §15.1, milestones M1 and M2).

The graph mirrors `fleet-mi300x/src/host/taskgraph.py` (default config: kv_a
replicated per XCD, q_c published by 2 slices per head, 16 KV chunks, one
K chunk, no worker split) one task kind at a time:

  fleet TaskKind        ETX grid (per layer L)              placement     event out
  QKV_FUSED            qkv_L (8 xcd, 37 w)                 xw->x         E_qkv_L (8,) xcd-local, 37
  Q_ABSORB             qabs_L (16 h, 2 slices)             hs->(h/2)     E_qabs_L (16,) local, 2
  ATTENTION            attn_L (16 h, 16 kv chunks)         hc->(h/2)     E_attn_L (16,) local, 16
  MERGE_UV             merge_L (16 h, 16 chunks)           hc->(h/2)     E_merge_L (8,) local, 32
  O_PROJ (K-split)     oproj_L (8, 37)                     xw->x         E_oproj_L (1,) global, 296
  NORM_ROUTER          router_L (8, 37)                    xw->x         E_norm_L (8,) local, 37
  EXPERT_GATE_UP       gate_up_L (8, 37)                   xw->x         E_gu_L (8,) local, 37
  EXPERT_DOWN          down_L (8, 37)                      xw->x         E_down_L (1,) global, 296
  DENSE_GATE_UP/DOWN   layer 0 only, global boundaries
  EMBED / LM_HEAD / ARGMAX  once per token

Fleet's Chiplet-task (one logical task per XCD, one descriptor per worker) is
an ETX grid over (xcd, worker) pinned with `domain_map="xw->x"`; fleet's
XCD-local events become DOMAIN-scope events, its global events DEVICE scope;
fleet's per-XCD scheduler mirror is replaced by ETX's inv-L1 polling; fleet's
prologue folds (FOLD_PARTIALS, kv_post, q-absorb reads) stay inside the tile
bodies exactly as in fleet.  Tile bodies are fleet's own, exported by
`fleet_shim.hip`; every grid's argument 0 is the FleetParams struct and
argument 1 a device int holding the layer index.  Durations are fleet's
measured per-task times (docs/STATUS.md).
"""
import os

from etx.frontends import hip_link
from etx.ir import Graph, Resource

XCDS, W = 8, 37            # fleet reserves one CU per XCD for its scheduler; ETX keeps 37 workers for a like-for-like port
HEADS, KV_CHUNKS, QABS = 16, 16, 2
LAYERS, FIRST_DENSE = 27, 1
SCRATCH_BYTES = 24576      # >= sizeof(fleet::Scratch) (23,408 B); checked by the host against fleet_scratch_bytes()
T = "examples/dsv2lite/fleet_shim.hip"
D5_T = "examples/dsv2lite/fleet_d5.hip"
# ETX_DSV2_QKV_XB=1 (docs/BATCH-2026-10-14.md section 5, stage B2, qkv only): this source replaces fleet_shim.hip ONLY in
# that graph; it includes fleet_shim.hip and adds the column-batched qkv tile. The default TU does not include it.
XB_T = "examples/dsv2lite/fleet_xb.hip"
XB_MAX_REQUESTS = 4        # fleet_xb.hip keeps one bf16 hidden vector per request in fleet::Scratch regions (4 slots)
# D5 uses Pass 7's existing l2_warm hook, not fleet's retained-register prefetch loop.
D5_PREFETCH = {
    "fleet_qkv_fused": "fleet_d5_qkv", "fleet_qkv_fused_fold": "fleet_d5_qkv",
    "fleet_merge_uv": "fleet_d5_merge", "fleet_o_proj": "fleet_d5_oproj",
    "fleet_norm_router": "fleet_d5_router",
    "fleet_expert_gate_up": "fleet_d5_shared_gate_up",
    "fleet_expert_down": "fleet_d5_shared_down", "fleet_expert_down_cached": "fleet_d5_shared_down",
    "fleet_lm_head": "fleet_d5_lm_head", "fleet_lm_head_fold": "fleet_d5_lm_head",
}

# fleet per-task durations in microseconds (STATUS.md, v0.14-v0.18 traces)
DUR = {"qkv": 7.4 + 12.0, "qabs": 4.0, "attn": 16.7, "merge": 7.6, "oproj": 12.0, "router": 9.0,
       "gate_up": 33.0, "down": 30.0, "dense_gu": 45.0, "dense_dn": 45.0, "embed": 2.0, "lm_head": 140.0, "argmax": 5.0}


CUS = 38                   # CUs per XCD: the interleaved build pins request r to worker slots r * CUS + w (ETX_SLOT_BY_CU pairs them per CU)

# D2 (docs/DSV2-D1-D5.md): ETX_DSV2_COH=1 marks fleet's tiles "coherent_loads" (no acquire invalidate; their releases
# write the L2 back). Correct ONLY with the coherent_loads lever patched into the tiles (examples/dsv2lite/levers/
# coherent_loads.py); examples/dsv2lite/coh_check.py refuses a build that has one without the other. dense_down reads the
# dense h from global inside gemv_rows, which the lever does not cover: it stays plain and keeps its acquire.
COH_PLAIN = frozenset({"fleet_dense_down"})


# ETX_DSV2_COH=l1 (D2b): the same tiles marked "coherent_l1" (agent-scope payload loads, lever l1_loads).
COH_MODES = {"1": "coherent_loads", "l1": "coherent_l1"}


def _tile(symbol: str):
    mode = COH_MODES.get(os.environ.get("ETX_DSV2_COH", "0"))
    d5 = os.environ.get("ETX_DSV2_D5", "0") == "1"
    kw = {"prefetch": D5_PREFETCH.get(symbol)} if d5 else {}
    source = D5_T if d5 else XB_T if _xb() else T
    # the fleet tile family implements "coherent_l1" through lever l1_loads, which examples/dsv2lite/build.sh adds when
    # the build-wide switch ETX_COHERENCE=l1 is set (etx/passes/plan.py); dense_down reads h inside gemv_rows: not capable
    kw["coherence_caps"] = () if symbol in COH_PLAIN else ("coherent_l1",)
    if mode and symbol not in COH_PLAIN:
        return hip_link(source, symbol, coherence=mode, **kw)
    return hip_link(source, symbol, **kw)


def _xb() -> bool:
    v = os.environ.get("ETX_DSV2_QKV_XB", "0")
    if v not in ("0", "1"):
        raise ValueError(f"ETX_DSV2_QKV_XB={v!r}: 0 | 1")
    return v == "1"


def _res(lds: int = SCRATCH_BYTES, paired: bool = False) -> Resource:
    # paired: two workgroups per CU (one per request) -> at most 256 registers per lane (measured cost 0.8%, design 15.12)
    return Resource(threads=256, vgpr=256, agpr=0 if paired else 72, lds_bytes=lds)


class _Req:
    """Names, arguments and placement of one request (r = 0 for the single-request graph, which keeps the old names)."""
    def __init__(self, r: int, requests: int):
        self.r, self.paired = r, requests > 1
        self.p = f"r{r}_" if self.paired else ""
        self.fleet = f"fleet{r}" if self.paired else "fleet"
        # two workgroups per CU is the residency limit of the paired build (docs/BATCH-2026-10-14.md section 3): request
        # r takes slot r mod 2, so with 4 or 8 requests a slot's requests take turns in its queue in creation order
        self.off = (r % 2) * CUS

    def n(self, name: str) -> str:
        return self.p + name

    def wmap_chiplet(self) -> str:
        return f"xw->(w+{self.off})" if self.paired else "xw->w"

    def wmap(self, text: str) -> str | None:           # head-indexed grids: pinned only when paired
        return text.replace("OFF", str(self.off)) if self.paired else None


def _chiplet(g: Graph, q: _Req, name: str, symbol: str, L: int, in_edges: dict, out_edges: dict, dur: float, **kw):
    return g.call_device(q.n(name), (XCDS, W), _tile(symbol), resource=_res(paired=q.paired), args=[q.fleet, f"layer_{L}"],
                         consts=(L, q.r), domain_map="xw->x", worker_map=q.wmap_chiplet(), in_edges=in_edges, out_edges=out_edges,
                         duration_us=dur, duration_cv=0.05, **kw)


def _layer_tensor(g: Graph, L: int) -> None:
    if f"layer_{L}" not in g.tensors:
        g.tensor(f"layer_{L}", (1,), role="runtime", bytes_per_elem=4)


def build_attention_block(g: Graph, q: _Req, L: int, e_in: str, e_in_map: str, fold: bool, extra_in: dict | None = None,
                          xb: bool = False) -> str:
    """qkv -> (qabs, attention) -> merge -> o_proj; returns the global o_proj event name. extra_in: more in-edges of qkv
    (the interleave's cross-request ordering). xb: the layer's column-batched qkv grid (build_xb_qkv) was created
    before this request's block and already created and signals E_qkv; this block adds everything after qkv."""
    _layer_tensor(g, L)
    n = q.n
    # K1 (docs/DSV2-D1-D5.md), ETX_DSV2_KVA=shared: kv_a is computed once, 1/8 per XCD (fleet's FLAG_KVA_SHARED), so the
    # qkv event is global -- every attention/q_absorb task waits for all 296 qkv tasks and reads the one kv_a buffer.
    # Default `replicated`: every XCD computes its own full kv_a, the event stays XCD-local (today's graph).
    kvs = os.environ.get("ETX_DSV2_KVA", "replicated") == "shared"
    sfx = "_kvs" if kvs else ""
    if not xb:
        g.etensor(n(f"E_qkv_{L}"), (1,) if kvs else (XCDS,), wait_count=XCDS * W if kvs else W)
    g.etensor(n(f"E_qabs_{L}"), (HEADS,), wait_count=QABS)
    g.etensor(n(f"E_attn_{L}"), (HEADS,), wait_count=KV_CHUNKS)
    g.etensor(n(f"E_merge_{L}"), (XCDS,), wait_count=(HEADS // XCDS) * KV_CHUNKS)
    g.etensor(n(f"E_oproj_{L}"), (1,), wait_count=XCDS * W)
    if not xb:
        qkv_in = {e_in: e_in_map, **(extra_in or {})}
        _chiplet(g, q, f"qkv_{L}", ("fleet_qkv_fused_fold" if fold else "fleet_qkv_fused") + sfx, L, qkv_in,
                 {n(f"E_qkv_{L}"): "xw->(0)" if kvs else "xw->x"}, DUR["qkv"])
    g.call_device(n(f"qabs_{L}"), (HEADS, QABS), _tile("fleet_q_absorb"), resource=_res(paired=q.paired), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r), domain_map="hs->(h/2)", worker_map=q.wmap("hs->((h%2)*2+s+OFF)"),
                  in_edges={n(f"E_qkv_{L}"): "hs->(0)" if kvs else "hs->(h/2)"}, out_edges={n(f"E_qabs_{L}"): "hs->h"},
                  duration_us=DUR["qabs"])
    g.call_device(n(f"attn_{L}"), (HEADS, KV_CHUNKS), _tile("fleet_attention" + sfx), resource=_res(paired=q.paired), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r),
                  # E_qabs is consumed INSIDE the attention body (fleet's PUB_WAIT, kept for overlap; see fleet_shim.hip);
                  # ETX still orders attention after E_qkv, and qabs after E_qkv, so the in-body wait is deadlock-free.
                  domain_map="hc->(h/2)", worker_map=q.wmap("hc->((h%2)*16+c+OFF)"),
                  in_edges={n(f"E_qkv_{L}"): "hc->(0)" if kvs else "hc->(h/2)"},
                  out_edges={n(f"E_attn_{L}"): "hc->h"}, duration_us=DUR["attn"], duration_cv=0.1)
    g.call_device(n(f"merge_{L}"), (HEADS, KV_CHUNKS), _tile("fleet_merge_uv"), resource=_res(paired=q.paired), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r), domain_map="hc->(h/2)", worker_map=q.wmap("hc->((h%2)*16+c+OFF)"),
                  in_edges={n(f"E_attn_{L}"): "hc->h"}, out_edges={n(f"E_merge_{L}"): "hc->(h/2)"}, duration_us=DUR["merge"])
    _chiplet(g, q, f"oproj_{L}", "fleet_o_proj", L, {n(f"E_merge_{L}"): "xw->x"}, {n(f"E_oproj_{L}"): "xw->(0)"}, DUR["oproj"])
    return n(f"E_oproj_{L}")


def build_moe_layer(g: Graph, q: _Req, L: int, e_in: str, e_in_map: str, fold: bool, routing_cached: bool, extra_in=None,
                    extra_gu=None, xb: bool = False) -> str:
    n = q.n
    e_oproj = build_attention_block(g, q, L, e_in, e_in_map, fold, extra_in, xb=xb)
    kchunks = os.environ.get("ETX_DSV2_KCHUNKS", "1") == "4"
    g.etensor(n(f"E_norm_{L}"), (XCDS,), wait_count=W)
    if not kchunks:                     # event creation order is part of the plan: unchanged for the default path
        g.etensor(n(f"E_gu_{L}"), (XCDS,), wait_count=W)
    g.etensor(n(f"E_down_{L}"), (1,), wait_count=XCDS * W)
    _chiplet(g, q, f"router_{L}", "fleet_norm_router", L, {e_oproj: "xw->(0)"}, {n(f"E_norm_{L}"): "xw->x"}, DUR["router"])
    if kchunks:
        # D3 (docs/DSV2-D1-D5.md): no E_gu -- down waits for the routing like gate_up, then for each of the 4 K-chunks
        # of h inside its body (fleet_shim.hip ETX_DSV2_KCHUNKS); the worker's own gate_up precedes it in its queue
        _chiplet(g, q, f"gate_up_{L}", "fleet_expert_gate_up_k", L, {n(f"E_norm_{L}"): "xw->x"}, {}, DUR["gate_up"])
        _chiplet(g, q, f"down_{L}", "fleet_expert_down_k", L, {n(f"E_norm_{L}"): "xw->x"}, {n(f"E_down_{L}"): "xw->(0)"},
                 DUR["down"])
        return n(f"E_down_{L}")
    _chiplet(g, q, f"gate_up_{L}", "fleet_expert_gate_up", L, {n(f"E_norm_{L}"): "xw->x", **(extra_gu or {})}, {n(f"E_gu_{L}"): "xw->x"},
             DUR["gate_up"])
    _chiplet(g, q, f"down_{L}", "fleet_expert_down_cached" if routing_cached else "fleet_expert_down", L,
             {n(f"E_gu_{L}"): "xw->x"}, {n(f"E_down_{L}"): "xw->(0)"}, DUR["down"])
    return n(f"E_down_{L}")


def build_moe_layer_d4(g: Graph, q: _Req, L: int, e_in: str, e_in_map: str, fold: bool, rows: int) -> str:
    """D4 (docs/DSV2-D1-D5.md, examples/dsv2lite/d4_placement.py): routed XCDs 0..5 route and run rows [rows, 1408) of
    their slot; shared XCDs 6, 7 run no router -- their gate_up folds and norms by itself and waits on o_proj only --
    then, after an in-body wait for the routed XCDs' published routing, rows [0, rows) of three slots each, and one down
    for the shared half and those slices. Every slice's gate_up and down on one XCD (checked below, refused otherwise)."""
    from examples.dsv2lite import d4_placement as d4
    n = q.n
    e_oproj = build_attention_block(g, q, L, e_in, e_in_map, fold)
    R, S = d4.TOP_K, d4.SHARED
    g.etensor(n(f"E_norm_{L}"), (R,), wait_count=W)
    g.etensor(n(f"E_shgu_{L}"), (S,), wait_count=W)
    g.etensor(n(f"E_gu_{L}"), (R,), wait_count=W)
    g.etensor(n(f"E_down_{L}"), (1,), wait_count=XCDS * W)
    consts = (L, q.r, rows, d4.SPX)
    shared_map = f"xw->(x+{R})"

    def grid(name, shape, symbol, dmap, ins, outs, dur):
        g.call_device(n(f"{name}_{L}"), shape, _tile(symbol), resource=_res(), args=[q.fleet, f"layer_{L}"], consts=consts,
                      domain_map=dmap, worker_map="xw->w", in_edges=ins, out_edges=outs, duration_us=dur, duration_cv=0.05)
    # creation order = each worker's queue order: routed workers router, gate_up, down (down reuses gate_up's routing in
    # LDS); shared workers shgu, sdown
    grid("router", (R, W), "fleet_norm_router", "xw->x", {e_oproj: "xw->(0)"}, {n(f"E_norm_{L}"): "xw->x"}, DUR["router"])
    grid("shgu", (S, W), "fleet_d4_shared_gate_up", shared_map, {e_oproj: "xw->(0)"}, {n(f"E_shgu_{L}"): "xw->x"},
         DUR["router"] - 1.6 + DUR["gate_up"] - 5.5)
    grid("gate_up", (R, W), "fleet_d4_routed_gate_up", "xw->x", {n(f"E_norm_{L}"): "xw->x"}, {n(f"E_gu_{L}"): "xw->x"},
         DUR["gate_up"])
    grid("down", (R, W), "fleet_d4_routed_down", "xw->x", {n(f"E_gu_{L}"): "xw->x"}, {n(f"E_down_{L}"): "xw->(0)"},
         DUR["down"])
    grid("sdown", (S, W), "fleet_d4_shared_down", shared_map, {n(f"E_shgu_{L}"): "xw->x"}, {n(f"E_down_{L}"): "xw->(0)"},
         DUR["down"])
    stages = d4.layout(rows)
    d4.check(stages)
    d4.verify_graph(g, L, stages, router_events={n(f"E_norm_{L}")})
    return n(f"E_down_{L}")


def build_dense_layer(g: Graph, q: _Req, L: int, e_in: str, e_in_map: str, fold: bool, extra_in=None, xb: bool = False) -> str:
    n = q.n
    e_oproj = build_attention_block(g, q, L, e_in, e_in_map, fold, extra_in, xb=xb)
    g.etensor(n(f"E_norm_{L}"), (XCDS,), wait_count=W)
    g.etensor(n(f"E_dgu_{L}"), (1,), wait_count=XCDS * W)
    g.etensor(n(f"E_dense_{L}"), (1,), wait_count=XCDS * W)
    _chiplet(g, q, f"norm_{L}", "fleet_norm_router", L, {e_oproj: "xw->(0)"}, {n(f"E_norm_{L}"): "xw->x"}, DUR["router"])
    _chiplet(g, q, f"dense_gu_{L}", "fleet_dense_gate_up", L, {n(f"E_norm_{L}"): "xw->x"}, {n(f"E_dgu_{L}"): "xw->(0)"}, DUR["dense_gu"])
    _chiplet(g, q, f"dense_dn_{L}", "fleet_dense_down", L, {n(f"E_dgu_{L}"): "xw->(0)"}, {n(f"E_dense_{L}"): "xw->(0)"}, DUR["dense_dn"])
    return n(f"E_dense_{L}")


XB_FOLD, XB_KVS = 1, 2    # fleet_xb.hip kFold, kKvs (grid constant 3)
XB_W = 2 * W              # the batched qkv grid takes both slots' 37 workers of every XCD (74 per XCD)


def build_xb_qkv(g: Graph, qs: list, L: int, ins: dict, fold: bool) -> None:
    """ETX_DSV2_QKV_XB=1: ONE qkv grid for all requests of layer L, created before any request's layer-L grids.

    Each task stages and norms every request's layer input (fleet's prologue, per request), keeps the normalized
    vectors as bf16 in LDS, then streams its weight rows of the fused [3072 + 576, 2048] matrix once and forms all B dot
    products per row (fleet_xb.hip). The grid is (8 XCDs, 74): workers w < 37 are slot 0's, w >= 37 slot 1's
    (local worker w % 37 + (w / 37) * 38), so both workgroups of every CU work on it. It waits on every request's
    layer input event and signals every request's own E_qkv_L, so everything after qkv is the unbatched graph.

    Deadlock freedom: the grid is created after every request's layer L-1 grids (its producers) and before every
    request's layer-L grids (its consumers); every static queue runs in creation order, so the earliest unfinished task
    always has its inputs."""
    _layer_tensor(g, L)
    kvs = os.environ.get("ETX_DSV2_KVA", "replicated") == "shared"
    for q in qs:
        g.etensor(q.n(f"E_qkv_{L}"), (1,) if kvs else (XCDS,), wait_count=XCDS * XB_W if kvs else XB_W)
    # consts: (layer, workers per XCD of this grid, first request = 0, flags: 1 = fold the expert partials, 2 = shared
    # kv_a); the body loops over the ETX_DS_REQUESTS requests itself. One symbol for every case: one copy of the body
    # in the megakernel (fleet_xb.hip)
    flags = (XB_FOLD if fold else 0) | (XB_KVS if kvs else 0)
    g.call_device(f"xb_qkv_{L}", (XCDS, XB_W), _tile("fleet_qkv_xb"), resource=_res(paired=True), args=[qs[0].fleet, f"layer_{L}"],
                  consts=(L, XB_W, 0, flags), domain_map="xw->x", worker_map=f"xw->(w%{W}+(w/{W})*{CUS})", in_edges=ins,
                  out_edges={q.n(f"E_qkv_{L}"): "xw->(0)" if kvs else "xw->x" for q in qs},
                  # model only (no measurement yet): B prologues of 7.4 us, then half the rows per worker
                  duration_us=7.4 * len(qs) + DUR["qkv"] - 7.4, duration_cv=0.05)


def build(layers: int = LAYERS, routing_cached: bool | None = None, requests: int | None = None) -> Graph:
    # ETX_DSV2_ROUTING_CACHED=0: expert_down recomputes the top-k instead of reusing its worker's LDS
    # (needed for the unfused baseline, where LDS does not survive the kernel boundary)
    # ETX_DSV2_REQUESTS=2: two requests interleaved (docs/PLAN-v3.md 4.4): one workgroup per request per CU, request 1's
    # layer L starts when request 0's router of layer L is done and request 0's layer L+1 when request 1's router of
    # layer L is done, so one request's latency-bound phases run beside the other's expert stream
    if routing_cached is None:
        routing_cached = os.environ.get("ETX_DSV2_ROUTING_CACHED", "1") != "0"
    if requests is None:
        requests = int(os.environ.get("ETX_DSV2_REQUESTS", "1"))
    # ETX_DSV2_COUPLE (two requests): router = each request starts a layer when the other's router of it is done (strict
    # half-layer offset); experts = only the expert streams alternate (r1's gate_up of L after r0's down of L, r0's of L+1
    # after r1's of L), the latency phases run freely; none = no cross-request edges
    couple = os.environ.get("ETX_DSV2_COUPLE", "router")
    # ETX_DSV2_KCHUNKS=4: D3, gate_up -> down in four 352-row K-chunks (build_moe_layer); 1 = today's single piece
    if os.environ.get("ETX_DSV2_KCHUNKS", "1") not in ("1", "4"):
        raise ValueError("ETX_DSV2_KCHUNKS must be 1 or 4")
    if os.environ.get("ETX_DSV2_KCHUNKS", "1") == "4" and (requests != 1 or not routing_cached):
        raise ValueError("ETX_DSV2_KCHUNKS=4 supports one request with cached routing only")
    # ETX_DSV2_D4=1: D4's expert placement (build_moe_layer_d4); ETX_DSV2_D4_ROWS: rows of each routed expert moved to the
    # shared XCDs (d4_placement.ROWS by default). Single request, routing cached; the tiles need -DETX_DSV2_D4=1.
    d4_rows = None
    if os.environ.get("ETX_DSV2_D4", "0") == "1":
        from examples.dsv2lite import d4_placement
        d4_rows = int(os.environ.get("ETX_DSV2_D4_ROWS", str(d4_placement.ROWS)))
        if requests != 1 or not routing_cached:
            raise ValueError("ETX_DSV2_D4 supports one request with cached routing only")
        if os.environ.get("ETX_DSV2_KCHUNKS", "1") != "1":    # D4's expert bodies do not chunk
            raise ValueError("ETX_DSV2_D4 with ETX_DSV2_KCHUNKS is not supported")
        coh = os.environ.get("ETX_DSV2_COH", "0")
        if os.environ.get("ETX_COHERENCE", "declared") == "l1":
            coh = "l1"                                      # the build-wide switch reaches the same tiles
        if coh == "1" or (coh == "l1" and d4_rows):     # the slices' h reads are plain; at 0 rows every read is patched
            raise ValueError("ETX_DSV2_D4 with ETX_DSV2_COH is not supported: the d4 tiles read h plainly "
                             "(only ETX_DSV2_COH=l1 with ETX_DSV2_D4_ROWS=0)")
    if couple not in ("router", "experts", "none"):
        raise ValueError(f"ETX_DSV2_COUPLE={couple!r}: router | experts | none")
    if requests < 1:
        raise ValueError(f"ETX_DSV2_REQUESTS={requests}: at least one request")
    if requests > 2 and couple != "none":   # the coupling edges name two requests only
        raise ValueError("more than two requests need ETX_DSV2_COUPLE=none")
    # K1: ETX_DSV2_KVA=shared (build_attention_block). One request only, and not with D5, whose prefetch map has no
    # shared-kv_a qkv symbol: neither combination is part of K1's experiment.
    kva = os.environ.get("ETX_DSV2_KVA", "replicated")
    if kva not in ("replicated", "shared"):
        raise ValueError(f"ETX_DSV2_KVA={kva!r}: replicated | shared")
    if kva == "shared" and os.environ.get("ETX_DSV2_D5", "0") == "1":
        raise ValueError("ETX_DSV2_KVA=shared supports one request without ETX_DSV2_D5 only")
    # ETX_DSV2_QKV_XB=1: one column-batched qkv grid per layer for all requests (build_xb_qkv, fleet_xb.hip)
    xb = _xb()
    if xb:
        if not 2 <= requests <= XB_MAX_REQUESTS:
            raise ValueError(f"ETX_DSV2_QKV_XB=1 needs 2..{XB_MAX_REQUESTS} requests (fleet_xb.hip LDS slots), not {requests}")
        if couple != "none":
            raise ValueError("ETX_DSV2_QKV_XB=1 needs ETX_DSV2_COUPLE=none: the coupling edges are qkv in-edges")
        if os.environ.get("ETX_DSV2_D5", "0") == "1":
            raise ValueError("ETX_DSV2_QKV_XB=1 with ETX_DSV2_D5 is not supported: no prefetch sibling for the batched qkv")
        if os.environ.get("ETX_DSV2_ISOLATE", "") or os.environ.get("ETX_DSV2_SNAPSHOT", ""):
            raise ValueError("ETX_DSV2_QKV_XB=1 with ETX_DSV2_ISOLATE / ETX_DSV2_SNAPSHOT is not supported: no per-request qkv grid")
    g = Graph("dsv2lite_decode" if requests == 1 else f"dsv2lite_decode_x{requests}")
    g.tensor("layer_-1", (1,), role="runtime", bytes_per_elem=4)
    qs = [_Req(r, requests) for r in range(requests)]
    state = {}
    for q in qs:
        n = q.n
        g.tensor(q.fleet, (1,), role="weight", bytes_per_elem=8)   # FleetParams*, set by the host
        g.etensor(n("E_embed"), (1,), wait_count=1)
        g.call_device(n("embed"), (1,), _tile("fleet_embed"), resource=_res(paired=q.paired), args=[q.fleet, "layer_-1"],
                      consts=(-1, q.r), out_edges={n("E_embed"): "i->i"}, duration_us=DUR["embed"], worker_map=q.wmap("i->(OFF)"))
        state[q.r] = (n("E_embed"), "xw->(0)", False)
    # layers outer, requests inner: every cross-request producer is created before its consumer (program order is the
    # static queue order, and each worker only runs its own request's tasks)
    for L in range(layers):
        if xb:
            build_xb_qkv(g, qs, L, {state[q.r][0]: state[q.r][1] for q in qs}, fold=state[qs[0].r][2])
        for q in qs:
            e, m, prev_moe = state[q.r]
            extra = extra_gu = None
            if requests == 2 and couple == "router":
                if q.r == 1:
                    extra = {f"r0_E_norm_{L}": "xw->x"}                # r1 layer L after r0's router L
                elif L > 0:
                    extra = {f"r1_E_norm_{L - 1}": "xw->x"}            # r0 layer L after r1's router L-1
            elif requests == 2 and couple == "experts" and L >= FIRST_DENSE:
                if q.r == 1:
                    extra_gu = {f"r0_E_down_{L}": "xw->(0)"}           # r1's expert stream of L after r0's
                elif L > FIRST_DENSE:
                    extra_gu = {f"r1_E_down_{L - 1}": "xw->(0)"}       # r0's of L after r1's of L-1
            if L < FIRST_DENSE:
                e = build_dense_layer(g, q, L, e, m, fold=prev_moe, extra_in=extra, xb=xb); prev_moe = False
            elif d4_rows is not None:
                e = build_moe_layer_d4(g, q, L, e, m, fold=prev_moe, rows=d4_rows)
                prev_moe = True
            else:
                e = build_moe_layer(g, q, L, e, m, fold=prev_moe, routing_cached=routing_cached, extra_in=extra, extra_gu=extra_gu,
                                    xb=xb)
                prev_moe = True
            state[q.r] = (e, "xw->(0)", prev_moe)
    for q in qs:
        n = q.n
        e, _, prev_moe = state[q.r]
        g.etensor(n("E_lm"), (1,), wait_count=XCDS * W)
        g.etensor(n("E_argmax"), (1,), wait_count=1)
        _chiplet(g, q, "lm_head", "fleet_lm_head_fold" if prev_moe else "fleet_lm_head", -1, {e: "xw->(0)"}, {n("E_lm"): "xw->(0)"}, DUR["lm_head"])
        g.call_device(n("argmax"), (1,), _tile("fleet_argmax"), resource=_res(paired=q.paired), args=[q.fleet, "layer_-1"],
                      consts=(-1, q.r), in_edges={n("E_lm"): "i->i"}, out_edges={n("E_argmax"): "i->i"}, duration_us=DUR["argmax"],
                      worker_map=q.wmap("i->(OFF)"))
    spec = os.environ.get("ETX_DSV2_ISOLATE", "")
    snapshot = os.environ.get("ETX_DSV2_SNAPSHOT", "")
    if snapshot:
        if spec or requests != 1:
            raise ValueError("snapshot requires one request and no appended ETX_DSV2_ISOLATE")
        _snapshot(g, snapshot)
    if spec:
        _isolate(g, spec, int(os.environ.get("ETX_DSV2_ISOLATE_REPS", "8")), qs)
    if os.environ.get("ETX_DSV2_D5", "0") == "1":
        g.codegen["prefetch_dispatch"] = True
        for grid in g.grids:
            if grid.body.prefetch:
                # FleetParams is an opaque container: this annotation lets P7 record the hook, but is NOT an
                # address-safety proof. fleet_d5.hip passes only fp->w (never fp->act) to the weight planner.
                grid.weight_args = [grid.args[0]]
    return g


def _snapshot(g: Graph, spec: str) -> None:
    """Measurement only: save the native phase inputs, then replay one independently mutating phase.

    ETX_DSV2_SNAPSHOT=capture:0:qkv_5 or replay:0:qkv_5 (router_5 also).
    host.hip's -DETX_DSV2_SNAPSHOT path supplies the saved inputs; no token correctness
    claim comes from these truncated graphs. The unisolated arms keep the full gate.
    """
    if spec not in {f"{mode}:0:{phase}" for mode in ("capture", "replay") for phase in ("qkv_5", "router_5")}:
        raise ValueError("snapshot must be capture|replay:0:qkv_5|router_5")
    mode, _, phase = spec.split(":")
    src = g.grid(phase)
    if mode == "capture":
        g.grids = g.grids[:g.grids.index(src)]
        used = {e for grid in g.grids for e in (*grid.in_edges, *grid.out_edges)}
        g.events = {k: v for k, v in g.events.items() if k in used}
    else:
        src.name = f"iso0x0_r0_{phase}"
        src.in_edges = {}
        src.out_edges = {}
        g.grids = [src]
        g.events = {}
    used_args = {a for grid in g.grids for a in grid.args}
    g.tensors = {k: v for k, v in g.tensors.items() if k in used_args}


def _all_map(grid) -> str:
    return "".join("xyzw"[i] for i in range(len(grid))) + "->(0)"


def _isolate(g: Graph, spec: str, reps: int, qs: list) -> None:
    """In-situ microbenchmark (docs/MODEL-dsv2.md section 4): after the whole token (so every activation the tiles read
    is real), run the phases in `spec` again. spec = blocks separated by ';', run one after the other; a block = groups
    separated by ',', started together; a group = "<request>:<phase>[+<phase>...]" (phases named as in this graph without
    the request prefix, run in order). Each block is repeated `reps` times, every repetition after the previous one has
    finished (device-wide). E.g. "0:qkv_5;0:gate_up_5+down_5;0:attn_5,1:gate_up_5". Every copy keeps its original tile,
    constants and worker pins; grids are named iso<b>x<k>_r<request>_<phase>."""
    prev = [q.n("E_argmax") for q in qs]
    for b, block in enumerate(spec.split(";")):
        groups = []
        for grp in block.split(","):
            r, chain = grp.split(":")
            groups.append((int(r), chain.split("+")))
        for k in range(reps):
            ends = []
            for r, chain in groups:
                q = qs[r]
                prev_ev = None
                for ph in chain:
                    src = g.grid(q.n(ph))
                    name = f"iso{b}x{k}_r{r}_{ph}"
                    n = 1
                    for d in src.grid:
                        n *= int(d)
                    g.etensor(f"E_{name}", (1,), wait_count=n)
                    waits = [prev_ev] if prev_ev else prev
                    g.call_device(name, src.grid, src.body, resource=src.resource, args=list(src.args), consts=src.consts,
                                  domain_map=src.domain_map, worker_map=src.worker_map,
                                  in_edges={e: _all_map(src.grid) for e in waits}, out_edges={f"E_{name}": _all_map(src.grid)},
                                  duration_us=src.duration_us)
                    prev_ev = f"E_{name}"
                ends.append(prev_ev)
            prev = ends


def bindings(**overrides) -> dict:
    if overrides:
        raise ValueError(f"DeepSeek graph does not accept --bind overrides: {', '.join(sorted(overrides))}; "
                         "use ETX_DSV2_REQUESTS for request count")
    return {}


def runtime(bindings, rng=None):
    return {}
