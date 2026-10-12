"""Lever topk_rank: expert selection by rank instead of six serial butterfly argmax passes.

fleet's select_experts_from (fleet_kernel.hip @3fb67a6) runs, on wave 0, a 6-step max, a 6-step sum and six 6-step
butterfly argmax passes with two shuffles each: 84 dependent bpermutes, 4.7 us per gate_up workgroup (fleet's own
stamp, docs/MODEL-dsv2.md:157). This lever keeps the max / exp / sum / normalise code byte for byte, then gives every lane
its rank under fleet's exact order (other beats self if larger, or equal with a lower lane index) from 64 independent
readlanes, and the lane of rank k < top_k writes slot k with its own p: the same ids and weights in the same slots.

Exact for every NaN-free probability vector: iterated argmax-and-remove under a strict total order is the sorted order,
and removing by p = -inf is exact because every valid p >= 0. Applies when n_routed == 64 (every lane a real expert,
distinct indices); any other shape takes the original code. Host proof in the test.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _function, _once

ORIGINAL_SIGNATURE = "__device__ inline void select_experts_from(const ModelDims& d, float v,"

RANK_BODY = """    if (d.n_routed != kWaveLanes) {               // etx lever topk_rank: only the 64-expert shape, else fleet's code
        select_experts_from_ref(d, v, ids, ws);
        return;
    }
    if (threadIdx.x < kWaveLanes) {
        const int lane = threadIdx.x;
        float mx = v;
#pragma unroll
        for (int off = kWaveLanes / 2; off; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, kWaveLanes));
        float p = __expf(v - mx);
        const float inv = 1.f / wave_sum(p);
        p *= inv;
        // rank = how many experts precede this one in fleet's order: larger p, or equal p and a lower index
        int rank = 0;
#pragma unroll
        for (int j = 0; j < kWaveLanes; ++j) {
            const float pj = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(p), j));
            rank += (pj > p || (pj == p && j < lane)) ? 1 : 0;
        }
        if (rank < d.top_k) { ids[rank] = lane; ws[rank] = p; }   // unnormalised, as fleet writes it
    }
    __syncthreads();
}
"""


def apply(src: Path) -> None:
    name = "topk_rank"
    kernel = src / "kernels/fleet_kernel.hip"
    k = kernel.read_text()
    original = _function(k, ORIGINAL_SIGNATURE)
    template = "template <typename I>\n"
    if k.count(template + original) != 1:
        raise SystemExit(f"patch {name}: select_experts_from is not the pinned template")
    ref = original.replace("void select_experts_from(", "void select_experts_from_ref(", 1)
    body_start = original.index("{") + 1
    rank = original[:body_start] + "\n" + RANK_BODY
    # The original survives as select_experts_from_ref (the fallback); the public name now selects by rank.
    k = _once(k, template + original, template + ref + "\n\n" + template + rank, name)
    kernel.write_text(k)
