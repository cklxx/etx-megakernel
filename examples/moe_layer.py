"""One MoE transformer layer at decode time, with both kinds of dynamism, and
real tile bodies (examples/tiles/moe.hip) plus a CPU reference host
(examples/hosts/moe.hip).

Dimensions are small so the CPU reference is instant; the structure is the
real one:  norm -> qkv -> attention (over a KV cache) -> o_proj (+residual)
-> router (top-2) -> grouping -> gather -> group_gemm (SiLU) -> down -> combine.

Shape dynamism:   B (tokens this step), n_gg (GroupGEMM tiles this step)
Data dependence:  the router writes topk; grouping writes expert_counts,
                  expert_tiles, tok_indptr, indptr, tok_slot, tile_expert and
                  initialises E_expert / E_down from those tensors
                  (ETensor.runtime_count); gather notifies "b->topk[b,:]";
                  group tiles wait on "t->tile_expert[t,:]".
`norm` is pure with out_to_in "bj->b": Pass 3 inlines it into qkv_proj and
the generated kernel recomputes the token's norm in the consumer's prologue.

Routing is a function of the raw input x (top-2 of x[b][0..NE)), so the
Python side and the C++ host derive the same routing from the same LCG-
generated input without a reference forward pass; ties break to the lower
expert id on both sides.
"""
from etx.frontends import hip_link
from etx.ir import Graph, Resource

import os
_SMALL = os.environ.get("ETX_MOE_SMALL") == "1"      # short tasks: build tiles with -DD=256 -DH=2 -DS=128 -DFF=128
D, H, HD, S, NE, K, FF, TPT = (256, 2, 128, 128, 8, 2, 128, 4) if _SMALL else (1024, 8, 128, 512, 8, 2, 512, 4)
T = "examples/tiles/moe.hip"


def build() -> Graph:
    g = Graph("moe_layer")
    # inputs / weights (bf16) -----------------------------------------------------------
    g.tensor("x", ("B", D))
    g.tensor("W_qkv", (3 * D, D), role="weight")
    g.tensor("W_o", (D, D), role="weight")
    g.tensor("k_cache", (S, D), role="weight")          # treated as static data for this step
    g.tensor("v_cache", (S, D), role="weight")
    g.tensor("W_up", (NE, FF, D), role="weight")
    g.tensor("W_down", (NE, D, FF), role="weight")
    # fp32 activations ------------------------------------------------------------------
    for name, shape in (("x_n", ("B", D)), ("qkv", ("B", 3 * D)), ("attn", ("B", D)), ("h", ("B", D)),
                        ("grouped", (f"B*{K}", D)), ("act", (f"B*{K}", FF)), ("partial", (f"B*{K}", D))):
        g.tensor(name, shape, role="scratch", bytes_per_elem=4)
    g.tensor("y", ("B", D), role="activation", bytes_per_elem=4)
    g.tensor("topk_w", ("B", K), role="scratch", bytes_per_elem=4)
    # runtime int32 tensors -------------------------------------------------------------
    for name, shape in (("topk", ("B", K)), ("expert_counts", (NE,)), ("expert_tiles", (NE,)),
                        ("tok_indptr", (NE + 1,)), ("indptr", (NE + 1,)), ("tok_slot", ("B", K)),
                        ("tile_expert", ("n_gg", 1))):
        g.tensor(name, shape, role="runtime", bytes_per_elem=4)

    g.etensor("E_norm", ("B",), wait_count=1)
    g.etensor("E_qkv", ("B",), wait_count=3)
    g.etensor("E_attn", ("B",), wait_count=H)
    g.etensor("E_o", ("B",), wait_count=1)
    g.etensor("E_route", (1,), wait_count="B")
    g.etensor("E_group", (1,), wait_count=1)
    g.etensor("E_expert", (NE,), runtime_count="expert_counts[i]", runtime_init_by="grouping")
    g.etensor("E_gg", ("n_gg",), wait_count=1)
    g.etensor("E_down", (NE,), runtime_count="expert_tiles[i]", runtime_init_by="grouping")

    g.call_device("norm", ("B",), hip_link(T, "etx_tile_rmsnorm"), resource=Resource(threads=256, vgpr=32, lds_bytes=64),
                  args=["x", "x_n"], reads=["x"], writes=["x_n"], out_edges={"E_norm": "b->b"},
                  pure=True, out_to_in="bj->b", duration_us=1.0, bytes_per_tile=D * 2)
    g.call_device("qkv_proj", ("B", 3), hip_link(T, "etx_tile_gemv_qkv", prefetch="etx_prefetch_gemv_qkv"),
                  resource=Resource(threads=256, vgpr=64),
                  args=["x_n", "W_qkv", "qkv"], reads=["x_n", "W_qkv"], writes=["qkv"], weight_args=["W_qkv"],
                  in_edges={"E_norm": "bj->b"}, out_edges={"E_qkv": "bj->b"}, duration_us=6.0, duration_cv=0.05,
                  bytes_per_tile=D * D * 2)
    g.call_device("attention", ("B", H), hip_link(T, "etx_tile_attention", prefetch="etx_prefetch_attention"),
                  resource=Resource(threads=256, vgpr=64, lds_bytes=S * 4 + 64),
                  args=["qkv", "k_cache", "v_cache", "attn"], reads=["qkv", "k_cache", "v_cache"], writes=["attn"],
                  in_edges={"E_qkv": "bh->b"}, out_edges={"E_attn": "bh->b"}, duration_us=8.0, duration_cv=0.3)
    g.call_device("o_proj", ("B",), hip_link(T, "etx_tile_gemv_o", prefetch="etx_prefetch_gemv_o"),
                  resource=Resource(threads=256, vgpr=64),
                  args=["attn", "x", "W_o", "h"], reads=["attn", "x", "W_o"], writes=["h"], weight_args=["W_o"],
                  in_edges={"E_attn": "b->b"}, out_edges={"E_o": "b->b"}, duration_us=4.0, duration_cv=0.05)
    g.call_device("router", ("B",), hip_link(T, "etx_tile_router"), resource=Resource(threads=256, vgpr=32),
                  args=["x", "topk", "topk_w"], reads=["x"], writes=["topk", "topk_w"],
                  in_edges={"E_o": "b->b"}, out_edges={"E_route": "b->(0)"}, duration_us=1.0)
    g.call_device("grouping", (1,), hip_link(T, "etx_tile_grouping"), resource=Resource(threads=256, vgpr=32),
                  args=["topk", "expert_counts", "expert_tiles", "tok_indptr", "indptr", "tok_slot", "tile_expert"],
                  reads=["topk"], writes=["expert_counts", "expert_tiles", "tok_indptr", "indptr", "tok_slot", "tile_expert"],
                  in_edges={"E_route": "i->i"}, out_edges={"E_group": "i->i"}, duration_us=2.0)
    g.call_device("gather", ("B",), hip_link(T, "etx_tile_gather"), resource=Resource(threads=256, vgpr=32),
                  args=["h", "tok_slot", "grouped"], reads=["h", "tok_slot"], writes=["grouped"],
                  in_edges={"E_group": "b->(0)"}, out_edges={"E_expert": "b->topk[b,:]"}, duration_us=1.0)
    g.call_device("group_gemm", ("n_gg",), hip_link(T, "etx_tile_group_gemm_up", prefetch="etx_prefetch_group_gemm_up"),
                  resource=Resource(threads=256, vgpr=96, lds_bytes=TPT * D * 4),
                  args=["grouped", "W_up", "act", "tile_expert", "indptr", "tok_indptr"],
                  reads=["grouped", "W_up", "tile_expert", "indptr", "tok_indptr"], writes=["act"], weight_args=["W_up"],
                  in_edges={"E_expert": "t->tile_expert[t,:]"}, out_edges={"E_gg": "t->t"}, duration_us=5.0, duration_cv=0.4)
    g.call_device("down", ("n_gg",), hip_link(T, "etx_tile_group_gemm_down", prefetch="etx_prefetch_group_gemm_down"),
                  resource=Resource(threads=256, vgpr=96, lds_bytes=TPT * FF * 4),
                  args=["act", "W_down", "partial", "tile_expert", "indptr", "tok_indptr"],
                  reads=["act", "W_down", "tile_expert", "indptr", "tok_indptr"], writes=["partial"], weight_args=["W_down"],
                  in_edges={"E_gg": "t->t"}, out_edges={"E_down": "t->tile_expert[t,:]"}, duration_us=5.0, duration_cv=0.4)
    g.call_device("combine", ("B",), hip_link(T, "etx_tile_combine"), resource=Resource(threads=256, vgpr=32),
                  args=["h", "partial", "topk_w", "tok_slot", "y"], reads=["h", "partial", "topk_w", "tok_slot"], writes=["y"],
                  in_edges={"E_down": "b->topk[b,:]"}, duration_us=1.0)
    return g


# ---------------------------------------------------------------- deterministic input + routing
def lcg_x(B: int, seed: int = 12345) -> list[list[float]]:
    """x[b][i] in bf16-exact values (integer/128, |integer| <= 128); the C++ host uses the same generator."""
    s = seed & 0xFFFFFFFF
    x = []
    for _ in range(B):
        row = []
        for _ in range(D):
            s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
            row.append(((s >> 8) % 257 - 128) / 128.0)
        x.append(row)
    return x


def routing(x_rows: list[list[float]]) -> list[list[int]]:
    out = []
    for row in x_rows:
        logits = row[:NE]
        order = sorted(range(NE), key=lambda e: (-logits[e], e))        # larger value first, tie -> lower id
        out.append(sorted(order[:K]))                                    # stored ascending by expert id
    return out


def runtime(bindings: dict, rng=None) -> dict:
    B = bindings["B"]
    topk = routing(lcg_x(B, bindings.get("seed", 12345)))
    counts = [0] * NE
    for row in topk:
        for e in row:
            counts[e] += 1
    tiles = [-(-c // TPT) for c in counts]
    tok_indptr, indptr = [0], [0]
    for e in range(NE):
        tok_indptr.append(tok_indptr[-1] + counts[e])
        indptr.append(indptr[-1] + tiles[e])
    fill = [0] * NE
    tok_slot = []
    for row in topk:
        slots = []
        for e in row:
            slots.append(tok_indptr[e] + fill[e])
            fill[e] += 1
        tok_slot.append(slots)
    tile_expert = [[e] for e in range(NE) for _ in range(tiles[e])]
    return {"topk": topk, "expert_counts": counts, "expert_tiles": tiles, "tok_indptr": tok_indptr,
            "indptr": indptr, "tok_slot": tok_slot, "tile_expert": tile_expert}


def bindings(**overrides) -> dict:
    b = {"B": 8, "seed": 12345}
    b.update(overrides)
    b["n_gg"] = runtime(b)["indptr"][-1]
    return b
