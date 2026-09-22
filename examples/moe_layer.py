"""One MoE transformer layer at decode time, with both kinds of dynamism.

Shape dynamism:   B (tokens this step), n_gg (GroupGEMM tiles this step)
Data dependence:  routing writes topk / indptr / tile_expert at runtime;
                  E_expert counts are initialised by `grouping` (init_kind RUNTIME)
                  gather tiles notify "b->topk[b,:]", GroupGEMM tiles wait on
                  "t->tile_expert[t,:]" and down tiles notify per expert.

Phase chain (decode): norm -> qkv -> attention -> o_proj -> router -> grouping
                      -> gather -> group_gemm -> down -> combine
`norm` is pure with out_to_in given, so Pass 3 inlines it into qkv (the
RMSNorm example of design §09 Pass 3).
"""
import random

from etx.frontends import hip_link
from etx.ir import Graph, Resource

NE = 16          # experts
K = 2            # top-k
TOK_PER_TILE = 8 # tokens per GroupGEMM tile
H = 8            # attention heads (one task per (token, head))

T = "examples/tiles/moe.hip"


def build() -> Graph:
    g = Graph("moe_layer")
    g.tensor("x", ("B", 2048))
    g.tensor("x_n", ("B", 2048), role="scratch")
    g.tensor("W_qkv", (2048, 3 * 2048), role="weight")
    g.tensor("qkv", ("B", 3 * 2048), role="scratch")
    g.tensor("kv_cache", ("S", 2048), role="activation")
    g.tensor("attn", ("B", 2048), role="scratch")
    g.tensor("W_o", (2048, 2048), role="weight")
    g.tensor("h", ("B", 2048), role="scratch")
    g.tensor("W_router", (2048, NE), role="weight")
    g.tensor("topk", ("B", K), role="runtime", bytes_per_elem=4)
    g.tensor("indptr", (NE + 1,), role="runtime", bytes_per_elem=4)
    g.tensor("tile_expert", ("n_gg", 1), role="runtime", bytes_per_elem=4)
    g.tensor("grouped", ("B*K", 2048), role="scratch")
    g.tensor("W_up", (NE, 2048, 1408), role="weight")
    g.tensor("W_down", (NE, 1408, 2048), role="weight")
    g.tensor("act", ("B*K", 1408), role="scratch")
    g.tensor("partial", ("B*K", 2048), role="scratch")
    g.tensor("y", ("B", 2048))

    g.etensor("E_norm", (1,), wait_count=1)
    g.etensor("E_qkv", ("B",), wait_count=3)
    g.etensor("E_attn", ("B",), wait_count=H)
    g.etensor("E_o", ("B",), wait_count=1)
    g.etensor("E_route", (1,), wait_count="B")
    g.etensor("E_group", (1,), wait_count=1)
    g.etensor("E_expert", (NE,), wait_count="runtime", runtime_init_by="grouping")
    g.etensor("E_gg", ("n_gg",), wait_count=1)
    g.etensor("E_down", (NE,), wait_count="runtime", runtime_init_by="grouping")

    g.call_device("norm", (1,), hip_link(T, "etx_tile_rmsnorm"), resource=Resource(threads=256, vgpr=32),
                  args=["x", "x_n"], reads=["x"], writes=["x_n"], out_edges={"E_norm": "i->i"},
                  pure=True, out_to_in="i->()", duration_us=1.0, bytes_per_tile=4096)
    g.call_device("qkv_proj", ("B", 3), hip_link(T, "etx_tile_gemv_qkv"), resource=Resource(threads=256, vgpr=128, lds_bytes=16384),
                  args=["x_n", "W_qkv", "qkv"], reads=["x_n", "W_qkv"], writes=["qkv"], weight_args=["W_qkv"],
                  in_edges={"E_norm": "bj->(0)"}, out_edges={"E_qkv": "bj->b"}, duration_us=6.0, duration_cv=0.05,
                  bytes_per_tile=2048 * 2048 * 2)
    g.call_device("attention", ("B", H), hip_link(T, "etx_tile_attention"), resource=Resource(threads=256, vgpr=160, lds_bytes=32768),
                  args=["qkv", "kv_cache", "attn"], reads=["qkv", "kv_cache"], writes=["attn"],
                  in_edges={"E_qkv": "bh->b"}, out_edges={"E_attn": "bh->b"}, duration_us=8.0, duration_cv=0.3)
    g.call_device("o_proj", ("B",), hip_link(T, "etx_tile_gemv_o"), resource=Resource(threads=256, vgpr=128, lds_bytes=16384),
                  args=["attn", "W_o", "h"], reads=["attn", "W_o"], writes=["h"], weight_args=["W_o"],
                  in_edges={"E_attn": "b->b"}, out_edges={"E_o": "b->b"}, duration_us=4.0, duration_cv=0.05)
    g.call_device("router", ("B",), hip_link(T, "etx_tile_router"), resource=Resource(threads=256, vgpr=48),
                  args=["h", "W_router", "topk"], reads=["h", "W_router"], writes=["topk"],
                  in_edges={"E_o": "b->b"}, out_edges={"E_route": "b->(0)"}, duration_us=1.5)
    g.call_device("grouping", (1,), hip_link(T, "etx_tile_grouping"), resource=Resource(threads=256, vgpr=48),
                  args=["topk", "indptr", "tile_expert"], reads=["topk"], writes=["indptr", "tile_expert"],
                  in_edges={"E_route": "i->i"}, out_edges={"E_group": "i->i"}, duration_us=2.0)
    g.call_device("gather", ("B",), hip_link(T, "etx_tile_gather"), resource=Resource(threads=256, vgpr=48),
                  args=["h", "topk", "indptr", "grouped"], reads=["h", "topk"], writes=["grouped"],
                  in_edges={"E_group": "b->(0)"}, out_edges={"E_expert": "b->topk[b,:]"}, duration_us=1.0)
    g.call_device("group_gemm", ("n_gg",), hip_link(T, "etx_tile_group_gemm_up"), resource=Resource(threads=256, vgpr=192, lds_bytes=32768),
                  args=["grouped", "W_up", "act", "tile_expert"], reads=["grouped", "W_up"], writes=["act"], weight_args=["W_up"],
                  in_edges={"E_expert": "t->tile_expert[t,:]"}, out_edges={"E_gg": "t->t"}, duration_us=5.0, duration_cv=0.4)
    g.call_device("down", ("n_gg",), hip_link(T, "etx_tile_group_gemm_down"), resource=Resource(threads=256, vgpr=192, lds_bytes=32768),
                  args=["act", "W_down", "partial", "tile_expert"], reads=["act", "W_down"], writes=["partial"], weight_args=["W_down"],
                  in_edges={"E_gg": "t->t"}, out_edges={"E_down": "t->tile_expert[t,:]"}, duration_us=5.0, duration_cv=0.4)
    g.call_device("combine", ("B",), hip_link(T, "etx_tile_combine"), resource=Resource(threads=256, vgpr=48),
                  args=["partial", "topk", "h", "y"], reads=["partial", "topk", "h"], writes=["y"],
                  in_edges={"E_down": "b->topk[b,:]"}, duration_us=1.0)
    return g


def runtime(bindings: dict, rng: random.Random | None = None) -> dict:
    """Sample a routing outcome: topk per token, expert counts, tile ranges."""
    rng = rng or random.Random(bindings.get("seed", 0))
    B = bindings["B"]
    topk = [sorted(rng.sample(range(NE), K)) for _ in range(B)]
    counts = [0] * NE
    for row in topk:
        for e in row:
            counts[e] += 1
    tiles = [-(-c // TOK_PER_TILE) for c in counts]
    indptr = [0]
    for t in tiles:
        indptr.append(indptr[-1] + t)
    tile_expert = [[e] for e in range(NE) for _ in range(tiles[e])]
    return {"topk": topk, "indptr": indptr, "tile_expert": tile_expert, "counts": counts}


def bindings(**overrides) -> dict:
    b = {"B": 32, "S": 4096, "seed": 0}
    b.update(overrides)
    rt = runtime(b)
    b["n_gg"] = rt["indptr"][-1]
    return b
