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


def _res(lds: int = SCRATCH_BYTES) -> Resource:
    return Resource(threads=256, vgpr=256, agpr=72, lds_bytes=lds)


def _chiplet(g: Graph, name: str, symbol: str, L: int, in_edges: dict, out_edges: dict, dur: float, **kw):
    return g.call_device(name, (XCDS, W), hip_link(T, symbol), resource=_res(), args=["fleet", f"layer_{L}"],
                         domain_map="xw->x", worker_map="xw->w", in_edges=in_edges, out_edges=out_edges, duration_us=dur, duration_cv=0.05, **kw)


def build_attention_block(g: Graph, L: int, e_in: str, e_in_map: str, fold: bool) -> str:
    """qkv -> (qabs, attention) -> merge -> o_proj; returns the global o_proj event name."""
    g.tensor(f"layer_{L}", (1,), role="runtime", bytes_per_elem=4)
    g.etensor(f"E_qkv_{L}", (XCDS,), wait_count=W)
    g.etensor(f"E_qabs_{L}", (HEADS,), wait_count=QABS)
    g.etensor(f"E_attn_{L}", (HEADS,), wait_count=KV_CHUNKS)
    g.etensor(f"E_merge_{L}", (XCDS,), wait_count=(HEADS // XCDS) * KV_CHUNKS)
    g.etensor(f"E_oproj_{L}", (1,), wait_count=XCDS * W)
    _chiplet(g, f"qkv_{L}", "fleet_qkv_fused_fold" if fold else "fleet_qkv_fused", L, {e_in: e_in_map}, {f"E_qkv_{L}": "xw->x"}, DUR["qkv"])
    g.call_device(f"qabs_{L}", (HEADS, QABS), hip_link(T, "fleet_q_absorb"), resource=_res(), args=["fleet", f"layer_{L}"],
                  domain_map="hs->(h/2)", in_edges={f"E_qkv_{L}": "hs->(h/2)"}, out_edges={f"E_qabs_{L}": "hs->h"}, duration_us=DUR["qabs"])
    g.call_device(f"attn_{L}", (HEADS, KV_CHUNKS), hip_link(T, "fleet_attention"), resource=_res(), args=["fleet", f"layer_{L}"],
                  domain_map="hc->(h/2)", in_edges={f"E_qkv_{L}": "hc->(h/2)", f"E_qabs_{L}": "hc->h"},
                  out_edges={f"E_attn_{L}": "hc->h"}, duration_us=DUR["attn"], duration_cv=0.1)
    g.call_device(f"merge_{L}", (HEADS, KV_CHUNKS), hip_link(T, "fleet_merge_uv"), resource=_res(), args=["fleet", f"layer_{L}"],
                  domain_map="hc->(h/2)", in_edges={f"E_attn_{L}": "hc->h"}, out_edges={f"E_merge_{L}": "hc->(h/2)"}, duration_us=DUR["merge"])
    _chiplet(g, f"oproj_{L}", "fleet_o_proj", L, {f"E_merge_{L}": "xw->x"}, {f"E_oproj_{L}": "xw->(0)"}, DUR["oproj"])
    return f"E_oproj_{L}"


def build_moe_layer(g: Graph, L: int, e_in: str, e_in_map: str, fold: bool, routing_cached: bool) -> str:
    e_oproj = build_attention_block(g, L, e_in, e_in_map, fold)
    g.etensor(f"E_norm_{L}", (XCDS,), wait_count=W)
    g.etensor(f"E_gu_{L}", (XCDS,), wait_count=W)
    g.etensor(f"E_down_{L}", (1,), wait_count=XCDS * W)
    _chiplet(g, f"router_{L}", "fleet_norm_router", L, {e_oproj: "xw->(0)"}, {f"E_norm_{L}": "xw->x"}, DUR["router"])
    _chiplet(g, f"gate_up_{L}", "fleet_expert_gate_up", L, {f"E_norm_{L}": "xw->x"}, {f"E_gu_{L}": "xw->x"}, DUR["gate_up"])
    _chiplet(g, f"down_{L}", "fleet_expert_down_cached" if routing_cached else "fleet_expert_down", L,
             {f"E_gu_{L}": "xw->x"}, {f"E_down_{L}": "xw->(0)"}, DUR["down"])
    return f"E_down_{L}"


def build_dense_layer(g: Graph, L: int, e_in: str, e_in_map: str, fold: bool) -> str:
    e_oproj = build_attention_block(g, L, e_in, e_in_map, fold)
    g.etensor(f"E_norm_{L}", (XCDS,), wait_count=W)
    g.etensor(f"E_dgu_{L}", (1,), wait_count=XCDS * W)
    g.etensor(f"E_dense_{L}", (1,), wait_count=XCDS * W)
    _chiplet(g, f"norm_{L}", "fleet_norm_router", L, {e_oproj: "xw->(0)"}, {f"E_norm_{L}": "xw->x"}, DUR["router"])
    _chiplet(g, f"dense_gu_{L}", "fleet_dense_gate_up", L, {f"E_norm_{L}": "xw->x"}, {f"E_dgu_{L}": "xw->(0)"}, DUR["dense_gu"])
    _chiplet(g, f"dense_dn_{L}", "fleet_dense_down", L, {f"E_dgu_{L}": "xw->(0)"}, {f"E_dense_{L}": "xw->(0)"}, DUR["dense_dn"])
    return f"E_dense_{L}"


def build(layers: int = LAYERS, routing_cached: bool = True) -> Graph:
    g = Graph("dsv2lite_decode")
    g.tensor("fleet", (1,), role="weight", bytes_per_elem=8)   # FleetParams*, set by the host
    g.tensor("layer_-1", (1,), role="runtime", bytes_per_elem=4)
    g.etensor("E_embed", (1,), wait_count=1)
    g.call_device("embed", (1,), hip_link(T, "fleet_embed"), resource=_res(), args=["fleet", "layer_-1"],
                  out_edges={"E_embed": "i->i"}, duration_us=DUR["embed"])
    e, m = "E_embed", "xw->(0)"
    prev_moe = False
    for L in range(layers):
        if L < FIRST_DENSE:
            e = build_dense_layer(g, L, e, m, fold=prev_moe); prev_moe = False
        else:
            e = build_moe_layer(g, L, e, m, fold=prev_moe, routing_cached=routing_cached); prev_moe = True
    g.etensor("E_lm", (1,), wait_count=XCDS * W)
    g.etensor("E_argmax", (1,), wait_count=1)
    _chiplet(g, "lm_head", "fleet_lm_head_fold" if prev_moe else "fleet_lm_head", -1, {e: "xw->(0)"}, {"E_lm": "xw->(0)"}, DUR["lm_head"])
    g.call_device("argmax", (1,), hip_link(T, "fleet_argmax"), resource=_res(), args=["fleet", "layer_-1"],
                  in_edges={"E_lm": "i->i"}, out_edges={"E_argmax": "i->i"}, duration_us=DUR["argmax"])
    return g


def bindings(**overrides) -> dict:
    return {}


def runtime(bindings, rng=None):
    return {}
