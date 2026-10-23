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
from etx.operators.moe_layer import provider as _provider

# Compiler graph-file loading executes this wrapper fresh for each invocation.
# Bind this import independently; earlier providers retain their own dimensions.
_SMALL = os.environ.get("ETX_MOE_SMALL") == "1"
_bound = _provider(small=_SMALL)
D, H, HD, S, NE, K, FF, TPT = (_bound.D, _bound.H, _bound.HD, _bound.S,
                              _bound.NE, _bound.K, _bound.FF, _bound.TPT)
T = _bound.T
build, lcg_x, routing, runtime, bindings = (_bound.build, _bound.lcg_x, _bound.routing,
                                           _bound.runtime, _bound.bindings)
