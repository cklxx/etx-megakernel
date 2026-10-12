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

# fleet per-task durations in microseconds (STATUS.md, v0.14-v0.18 traces)
DUR = {"qkv": 7.4 + 12.0, "qabs": 4.0, "attn": 16.7, "merge": 7.6, "oproj": 12.0, "router": 9.0,
       "gate_up": 33.0, "down": 30.0, "dense_gu": 45.0, "dense_dn": 45.0, "embed": 2.0, "lm_head": 140.0, "argmax": 5.0}


CUS = 38                   # CUs per XCD: the interleaved build pins request r to worker slots r * CUS + w (ETX_SLOT_BY_CU pairs them per CU)


def _res(lds: int = SCRATCH_BYTES, paired: bool = False) -> Resource:
    # paired: two workgroups per CU (one per request) -> at most 256 registers per lane (measured cost 0.8%, design 15.12)
    return Resource(threads=256, vgpr=256, agpr=0 if paired else 72, lds_bytes=lds)


class _Req:
    """Names, arguments and placement of one request (r = 0 for the single-request graph, which keeps the old names)."""
    def __init__(self, r: int, requests: int):
        self.r, self.paired = r, requests > 1
        self.p = f"r{r}_" if self.paired else ""
        self.fleet = f"fleet{r}" if self.paired else "fleet"
        self.off = r * CUS

    def n(self, name: str) -> str:
        return self.p + name

    def wmap_chiplet(self) -> str:
        return f"xw->(w+{self.off})" if self.paired else "xw->w"

    def wmap(self, text: str) -> str | None:           # head-indexed grids: pinned only when paired
        return text.replace("OFF", str(self.off)) if self.paired else None


def _chiplet(g: Graph, q: _Req, name: str, symbol: str, L: int, in_edges: dict, out_edges: dict, dur: float, **kw):
    return g.call_device(q.n(name), (XCDS, W), hip_link(T, symbol), resource=_res(paired=q.paired), args=[q.fleet, f"layer_{L}"],
                         consts=(L, q.r), domain_map="xw->x", worker_map=q.wmap_chiplet(), in_edges=in_edges, out_edges=out_edges,
                         duration_us=dur, duration_cv=0.05, **kw)


def _layer_tensor(g: Graph, L: int) -> None:
    if f"layer_{L}" not in g.tensors:
        g.tensor(f"layer_{L}", (1,), role="runtime", bytes_per_elem=4)


def build_attention_block(g: Graph, q: _Req, L: int, e_in: str, e_in_map: str, fold: bool, extra_in: dict | None = None) -> str:
    """qkv -> (qabs, attention) -> merge -> o_proj; returns the global o_proj event name. extra_in: more in-edges of qkv
    (the interleave's cross-request ordering)."""
    _layer_tensor(g, L)
    n = q.n
    g.etensor(n(f"E_qkv_{L}"), (XCDS,), wait_count=W)
    g.etensor(n(f"E_qabs_{L}"), (HEADS,), wait_count=QABS)
    g.etensor(n(f"E_attn_{L}"), (HEADS,), wait_count=KV_CHUNKS)
    g.etensor(n(f"E_merge_{L}"), (XCDS,), wait_count=(HEADS // XCDS) * KV_CHUNKS)
    g.etensor(n(f"E_oproj_{L}"), (1,), wait_count=XCDS * W)
    qkv_in = {e_in: e_in_map, **(extra_in or {})}
    _chiplet(g, q, f"qkv_{L}", "fleet_qkv_fused_fold" if fold else "fleet_qkv_fused", L, qkv_in, {n(f"E_qkv_{L}"): "xw->x"}, DUR["qkv"])
    g.call_device(n(f"qabs_{L}"), (HEADS, QABS), hip_link(T, "fleet_q_absorb"), resource=_res(paired=q.paired), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r), domain_map="hs->(h/2)", worker_map=q.wmap("hs->((h%2)*2+s+OFF)"),
                  in_edges={n(f"E_qkv_{L}"): "hs->(h/2)"}, out_edges={n(f"E_qabs_{L}"): "hs->h"}, duration_us=DUR["qabs"])
    g.call_device(n(f"attn_{L}"), (HEADS, KV_CHUNKS), hip_link(T, "fleet_attention"), resource=_res(paired=q.paired), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r),
                  # E_qabs is consumed INSIDE the attention body (fleet's PUB_WAIT, kept for overlap; see fleet_shim.hip);
                  # ETX still orders attention after E_qkv, and qabs after E_qkv, so the in-body wait is deadlock-free.
                  domain_map="hc->(h/2)", worker_map=q.wmap("hc->((h%2)*16+c+OFF)"), in_edges={n(f"E_qkv_{L}"): "hc->(h/2)"},
                  out_edges={n(f"E_attn_{L}"): "hc->h"}, duration_us=DUR["attn"], duration_cv=0.1)
    g.call_device(n(f"merge_{L}"), (HEADS, KV_CHUNKS), hip_link(T, "fleet_merge_uv"), resource=_res(paired=q.paired), args=[q.fleet, f"layer_{L}"],
                  consts=(L, q.r), domain_map="hc->(h/2)", worker_map=q.wmap("hc->((h%2)*16+c+OFF)"),
                  in_edges={n(f"E_attn_{L}"): "hc->h"}, out_edges={n(f"E_merge_{L}"): "hc->(h/2)"}, duration_us=DUR["merge"])
    _chiplet(g, q, f"oproj_{L}", "fleet_o_proj", L, {n(f"E_merge_{L}"): "xw->x"}, {n(f"E_oproj_{L}"): "xw->(0)"}, DUR["oproj"])
    return n(f"E_oproj_{L}")


def build_moe_layer(g: Graph, q: _Req, L: int, e_in: str, e_in_map: str, fold: bool, routing_cached: bool, extra_in=None,
                    extra_gu=None) -> str:
    n = q.n
    e_oproj = build_attention_block(g, q, L, e_in, e_in_map, fold, extra_in)
    g.etensor(n(f"E_norm_{L}"), (XCDS,), wait_count=W)
    g.etensor(n(f"E_gu_{L}"), (XCDS,), wait_count=W)
    g.etensor(n(f"E_down_{L}"), (1,), wait_count=XCDS * W)
    _chiplet(g, q, f"router_{L}", "fleet_norm_router", L, {e_oproj: "xw->(0)"}, {n(f"E_norm_{L}"): "xw->x"}, DUR["router"])
    _chiplet(g, q, f"gate_up_{L}", "fleet_expert_gate_up", L, {n(f"E_norm_{L}"): "xw->x", **(extra_gu or {})}, {n(f"E_gu_{L}"): "xw->x"},
             DUR["gate_up"])
    _chiplet(g, q, f"down_{L}", "fleet_expert_down_cached" if routing_cached else "fleet_expert_down", L,
             {n(f"E_gu_{L}"): "xw->x"}, {n(f"E_down_{L}"): "xw->(0)"}, DUR["down"])
    return n(f"E_down_{L}")


def build_dense_layer(g: Graph, q: _Req, L: int, e_in: str, e_in_map: str, fold: bool, extra_in=None) -> str:
    n = q.n
    e_oproj = build_attention_block(g, q, L, e_in, e_in_map, fold, extra_in)
    g.etensor(n(f"E_norm_{L}"), (XCDS,), wait_count=W)
    g.etensor(n(f"E_dgu_{L}"), (1,), wait_count=XCDS * W)
    g.etensor(n(f"E_dense_{L}"), (1,), wait_count=XCDS * W)
    _chiplet(g, q, f"norm_{L}", "fleet_norm_router", L, {e_oproj: "xw->(0)"}, {n(f"E_norm_{L}"): "xw->x"}, DUR["router"])
    _chiplet(g, q, f"dense_gu_{L}", "fleet_dense_gate_up", L, {n(f"E_norm_{L}"): "xw->x"}, {n(f"E_dgu_{L}"): "xw->(0)"}, DUR["dense_gu"])
    _chiplet(g, q, f"dense_dn_{L}", "fleet_dense_down", L, {n(f"E_dgu_{L}"): "xw->(0)"}, {n(f"E_dense_{L}"): "xw->(0)"}, DUR["dense_dn"])
    return n(f"E_dense_{L}")


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
    if couple not in ("router", "experts", "none"):
        raise ValueError(f"ETX_DSV2_COUPLE={couple!r}: router | experts | none")
    g = Graph("dsv2lite_decode" if requests == 1 else f"dsv2lite_decode_x{requests}")
    g.tensor("layer_-1", (1,), role="runtime", bytes_per_elem=4)
    qs = [_Req(r, requests) for r in range(requests)]
    state = {}
    for q in qs:
        n = q.n
        g.tensor(q.fleet, (1,), role="weight", bytes_per_elem=8)   # FleetParams*, set by the host
        g.etensor(n("E_embed"), (1,), wait_count=1)
        g.call_device(n("embed"), (1,), hip_link(T, "fleet_embed"), resource=_res(paired=q.paired), args=[q.fleet, "layer_-1"],
                      consts=(-1, q.r), out_edges={n("E_embed"): "i->i"}, duration_us=DUR["embed"], worker_map=q.wmap("i->(OFF)"))
        state[q.r] = (n("E_embed"), "xw->(0)", False)
    # layers outer, requests inner: every cross-request producer is created before its consumer (program order is the
    # static queue order, and each worker only runs its own request's tasks)
    for L in range(layers):
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
                e = build_dense_layer(g, q, L, e, m, fold=prev_moe, extra_in=extra); prev_moe = False
            else:
                e = build_moe_layer(g, q, L, e, m, fold=prev_moe, routing_cached=routing_cached, extra_in=extra, extra_gu=extra_gu)
                prev_moe = True
            state[q.r] = (e, "xw->(0)", prev_moe)
    for q in qs:
        n = q.n
        e, _, prev_moe = state[q.r]
        g.etensor(n("E_lm"), (1,), wait_count=XCDS * W)
        g.etensor(n("E_argmax"), (1,), wait_count=1)
        _chiplet(g, q, "lm_head", "fleet_lm_head_fold" if prev_moe else "fleet_lm_head", -1, {e: "xw->(0)"}, {n("E_lm"): "xw->(0)"}, DUR["lm_head"])
        g.call_device(n("argmax"), (1,), hip_link(T, "fleet_argmax"), resource=_res(paired=q.paired), args=[q.fleet, "layer_-1"],
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
    return {}


def runtime(bindings, rng=None):
    return {}
