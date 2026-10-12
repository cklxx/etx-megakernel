"""Lever argmax_tree: lm_head's candidate pick and the argmax task reduce in 2 barriers instead of 9.

fleet's two argmax reductions (fleet_kernel.hip @3fb67a6, lm_head_candidate and task_argmax) run a 256-wide LDS tree with
a barrier per level. The pick (larger value, lower index on ties; fleet's comparison, unchanged) is a selection, and this
lever keeps every original pairing: lanes 0-63 combine levels 128 and 64 from LDS exactly as the tree does
(pick(pick(x[t], x[t+128]), pick(x[t+64], x[t+192]))), levels 32..1 run as shuffles in wave 0 with the same
(self, other) order. Thread 0 then holds the winner in registers; a final barrier keeps the LDS slots from being reused
before wave 0 has read them. Same result for every input, NaN and ties included (host proof in the test).
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _once

HELPER = """// etx lever argmax_tree: the 256-wide pick tree's exact pairings, levels 128 and 64 from LDS, 32..1 by shuffle.
// Same comparison as the tree (other wins if larger, or equal with a lower index). Valid in thread 0.
static_assert(kBlock == 256 && kBlockThreads == 256, "etx argmax_tree pairs a 256-wide tree");
__device__ __forceinline__ void etx_argmax_pick(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
}
__device__ __forceinline__ void etx_argmax_tree(const float* vals, const int* idxs, float& v, int& i) {
    const int t = threadIdx.x;
    float a = vals[t], b = vals[t + 64];
    int ai = idxs[t], bi = idxs[t + 64];
    etx_argmax_pick(a, ai, vals[t + 128], idxs[t + 128]);
    etx_argmax_pick(b, bi, vals[t + 192], idxs[t + 192]);
    etx_argmax_pick(a, ai, b, bi);
#pragma unroll
    for (int s = 32; s; s >>= 1) {
        const float ov = __shfl_down(a, s, 64);
        const int oi = __shfl_down(ai, s, 64);
        if (t < s) etx_argmax_pick(a, ai, ov, oi);
    }
    v = a; i = ai;
}

"""

CAND_OLD = """    __syncthreads();
    for (int st = kBlockThreads >> 1; st; st >>= 1) {
        if ((int)threadIdx.x < st) {
            const float ov = smem_val[threadIdx.x + st];
            const int oi = smem_idx[threadIdx.x + st];
            if (ov > smem_val[threadIdx.x] || (ov == smem_val[threadIdx.x] && oi < smem_idx[threadIdx.x])) {
                smem_val[threadIdx.x] = ov; smem_idx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        st_coh(act.argmax_val + xcd * n_workers + worker, smem_val[0], coh);
        st_coh(act.argmax_idx + xcd * n_workers + worker, smem_idx[0], coh);
    }
"""
CAND_NEW = """    __syncthreads();
    if (threadIdx.x < 64) {
        float v; int i;
        etx_argmax_tree(smem_val, smem_idx, v, i);
        if (threadIdx.x == 0) {
            st_coh(act.argmax_val + xcd * n_workers + worker, v, coh);
            st_coh(act.argmax_idx + xcd * n_workers + worker, i, coh);
        }
    }
    __syncthreads();   // wave 0 has read every slot before the next task may reuse the LDS
"""

TASK_OLD = """    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s; s >>= 1) {
        if ((int)threadIdx.x < s) {
            const bool take = best_val[threadIdx.x + s] > best_val[threadIdx.x] ||
                              (best_val[threadIdx.x + s] == best_val[threadIdx.x] &&
                               best_idx[threadIdx.x + s] < best_idx[threadIdx.x]);
            if (take) {
                best_val[threadIdx.x] = best_val[threadIdx.x + s];
                best_idx[threadIdx.x] = best_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        act.token_out[tc.tok] = best_idx[0];
        if (!rt.teacher) st_coh(act.token_in + tc.tok + 1, best_idx[0], coh);
"""
TASK_NEW = """    __syncthreads();
    int best = 0;
    if (threadIdx.x < 64) {
        float v;
        etx_argmax_tree(best_val, best_idx, v, best);
    }
    __syncthreads();   // wave 0 has read every slot before the next task may reuse the LDS
    if (threadIdx.x == 0) {
        act.token_out[tc.tok] = best;
        if (!rt.teacher) st_coh(act.token_in + tc.tok + 1, best, coh);
"""

ANCHOR = "__device__ void lm_head_candidate("


def apply(src: Path) -> None:
    name = "argmax_tree"
    kernel = src / "kernels/fleet_kernel.hip"
    k = kernel.read_text()
    # The comment block above lm_head_candidate stays where it is; the helpers go right before the function.
    k = _once(k, ANCHOR, HELPER + ANCHOR, name)
    k = _once(k, CAND_OLD, CAND_NEW, name)
    k = _once(k, TASK_OLD, TASK_NEW, name)
    kernel.write_text(k)
