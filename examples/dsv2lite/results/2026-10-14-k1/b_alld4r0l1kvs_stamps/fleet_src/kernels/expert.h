// MoE expert task bodies — gate_up then down, one balanced unit per XCD.
//
// docs/design.md §3: the 8 XCDs each stream exactly one unit, where a unit is
// either one of the 6 routed experts (17.3 MB) or one half of the shared expert
// pair (also 17.3 MB). That is a perfect 8-way split with no remainder, which
// is why the expert phase has no straggler XCD.
//
// Two details make this work inside an immutable task queue:
//
//   * Indirect expert index (§3): the descriptor stores the routing *slot* k,
//     not an expert id. The kernel reads topk_ids[k] at run time and forms the
//     weight pointer as base + id * stride. The queue never changes, so the
//     ahead-of-time launch stays valid even though routing is data-dependent.
//
//   * gate_up -> down is XCD-local (§3): h[1408] is written by the XCD's 37
//     gate_up workers and read by its 37 down workers after one XCD-local
//     event per K-chunk of h, never crossing a chiplet (§12 tile granularity:
//     the down of chunk c overlaps the gate_up of chunk c+1).
#pragma once

#include <hip/hip_runtime.h>
#include "gemv.h"
#include "../runtime/fleet_runtime.h"

namespace fleet {

// A unit is either a routed expert (slot >= 0) or a shared half (slot < 0).
struct ExpertUnit {
    const __hip_bfloat16* gate_up;   // rows [gate; up] interleaved, [2 * inter, hidden]
    const __hip_bfloat16* down;      // [hidden, inter] at leading dimension down_ld
    int down_ld;                     // routed: inter; shared half: the full 2816
    float weight;                    // routing weight (x routed_scaling), 1 for shared
    int inter;                       // rows this unit owns
};

// Resolve the descriptor's slot into actual pointers. Routed experts index
// through topk_ids; shared halves split the merged shared expert in two so all
// 8 XCDs carry the same 17.3 MB.
//
// Shared layout (pack_weights.py): gate_up interleaved [2 * shared_inter,
// hidden] then down [hidden, shared_inter]. Half h owns gate/up row pairs
// [h * rows, (h+1) * rows) — element offset h * 2 * rows * hidden — and
// columns [h * rows, (h+1) * rows) of every down row, so its down rows are
// strided by shared_inter, not by rows.
__device__ inline ExpertUnit resolve_unit(
        int expert_slot, const __hip_bfloat16* __restrict__ experts_base,
        const __hip_bfloat16* __restrict__ shared_base,
        const int32_t* __restrict__ topk_ids, const float* __restrict__ topk_w,
        float routed_scaling, int hidden, int moe_inter, int shared_inter,
        int64_t expert_stride) {
    ExpertUnit u;
    if (expert_slot >= 0) {
        const int id = topk_ids[expert_slot];           // indirect, read at run time
        const __hip_bfloat16* base = experts_base + (int64_t)id * expert_stride;
        u.gate_up = base;
        u.down = base + (int64_t)2 * moe_inter * hidden;
        u.down_ld = moe_inter;
        u.weight = topk_w[expert_slot] * routed_scaling;
        u.inter = moe_inter;
    } else {
        const int half = -expert_slot - 1;              // 0 or 1
        const int rows = shared_inter / 2;
        u.gate_up = shared_base + (int64_t)half * 2 * rows * hidden;
        u.down = shared_base + (int64_t)2 * shared_inter * hidden
                 + (int64_t)half * rows;
        u.down_ld = shared_inter;
        u.weight = 1.f;
        u.inter = rows;
    }
    return u;
}

// h[n] = SiLU(gate_n . x) * (up_n . x) over this worker's share of the unit's
// rows; x is the post-attention normed vector, staged in LDS by the caller.
// chunk_counters (or nullptr): the XCD-local event counters of the unit's
// n_chunks K-chunks of chunk_rows rows, signalled per wave as the rows of
// each chunk land (gemv_gate_up_rows).
__device__ inline void expert_gate_up(
        const ExpertUnit& u, const float* __restrict__ x_lds,
        float* __restrict__ h, int hidden, int worker, int n_workers,
        uint32_t* __restrict__ chunk_counters = nullptr, int chunk_rows = 0,
        int n_chunks = 0) {
    gemv_gate_up_rows(u.gate_up, x_lds, h, u.inter, hidden,
                      /*xcd=*/0, /*n_xcds=*/1, worker, n_workers,
                      chunk_counters, chunk_rows, n_chunks);
}

// out[n] = bf16(weight * bf16(down_n . h)): this XCD's partial of the layer
// output, with HF's roundings (down_proj yields bf16, then `mul_(weight)` in
// bf16). The layer's last down worker sums the 8 partials in a fixed order —
// no float atomics, so the result is bitwise reproducible (§4).
//
// K-chunked: one call per chunk [col0, col0 + kc) of h, in order, with the
// chunk staged in h_lds and fp32 partial sums kept in acc_lds[hidden] between
// calls; the last call applies the bf16 epilogue into out. The chunk
// boundaries are multiples of 8 (16-byte weight loads).
__device__ inline void expert_down_chunk(
        const ExpertUnit& u, const float* __restrict__ h_lds,
        float* __restrict__ acc_lds, __hip_bfloat16* __restrict__ out, int hidden,
        int worker, int n_workers, int col0, int kc, bool first, bool last,
        bool coh = false) {
    const GemvEpilogue epi = first ? (last ? EPI_BF16 : EPI_NONE)
                                   : (last ? EPI_BF16_ACC : EPI_ACC);
    gemv_rows(u.down + col0, u.down_ld, h_lds, last ? nullptr : acc_lds,
              (last && !first) ? acc_lds : nullptr,   // residual = the accumulator
              epi, u.weight, hidden, kc, /*xcd=*/0, /*n_xcds=*/1, worker, n_workers,
              last && coh,    // the partial crosses XCDs; the accumulator is LDS
              last ? out : nullptr);
}

__device__ inline void expert_down(
        const ExpertUnit& u, const float* __restrict__ h_lds,
        __hip_bfloat16* __restrict__ out, int hidden, int worker, int n_workers) {
    expert_down_chunk(u, h_lds, nullptr, out, hidden, worker, n_workers,
                      0, u.inter, true, true);
}

// Layer 0's dense MLP reuses the same machinery, but h is 10944 wide and does
// not fit an XCD-local buffer, so gate_up -> down costs one extra global event
// (§3). Keeping the same helpers means layer 0 needs no separate kernel path.
__device__ inline void dense_gate_up(
        const __hip_bfloat16* __restrict__ gate_up, const float* __restrict__ x_lds,
        float* __restrict__ h, int hidden, int inter, int xcd, int n_xcds,
        int worker, int n_workers) {
    gemv_gate_up_rows(gate_up, x_lds, h, inter, hidden, xcd, n_xcds,
                      worker, n_workers);
}

}  // namespace fleet
