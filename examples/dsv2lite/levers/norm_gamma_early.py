"""Lever N, norm_gamma_early: qkv's input norm and the MoE router's post-attention norm load gamma before the reduction.

fleet's rms_norm_lds (fleet_kernel.hip @3fb67a6) loads gamma after its 10-barrier ordered tree, so the gamma round
trip is exposed on every qkv and router WG. This lever loads gamma into registers first (as ETX_DSL_NORM_HOIST does for
Qwen, etx/ops/gemv.py:602-631) with the same values and the same epilogue math. norm_tree (levers/norm_tree.py) adds the
same add DAG in 3 barriers instead of 10. Dense post-norm and lm_head keep the original helper.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _function, _once

ORDERED_TREE = """// Same expression DAG as block_reduce_ordered at blockDim.x == 256:
// lane t's first two levels are (v[t]+v[t+128])+(v[t+64]+v[t+192]).
// Never reduce each wave first: that changes the floating-point add order.
__device__ __forceinline__ float block_reduce_ordered_etx_tree(float v, float* smem) {
    if (blockDim.x != 256) return block_reduce_ordered(v, smem);
    smem[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x < 64) {
        const int lane = threadIdx.x;
        const float lo = smem[lane] + smem[lane + 128];
        const float hi = smem[lane + 64] + smem[lane + 192];
        float root = lo + hi;
#pragma unroll
        for (int stride = 32; stride; stride >>= 1) {
            const float rhs = __shfl_down(root, stride, 64);
            if (lane < stride) root += rhs;
        }
        if (lane == 0) smem[0] = root;
    }
    __syncthreads();
    const float out = smem[0];
    __syncthreads();  // retain the original read-before-reuse contract
    return out;
}

"""

GAMMA_EARLY = """    // Keep the hidden dimension dynamic for the original square-sum and
    // inverse: an equality specialization changed ROCm's FMA contractions.
    // Clamp unused gamma slots instead of branching per load (which inserted
    // eight serial waitcnts). Every gamma that the epilogue uses is unchanged.
    if (d.hidden <= 0 || d.hidden > kMaxHidden || blockDim.x != kBlockThreads) {
        rms_norm_lds(d, gamma, v, smem);
        return;
    }
    // The memory-only compiler fence prevents sinking immutable gamma loads
    // through the tree, without adding a device fence or an explicit wait.
    constexpr int gamma_slots = (kMaxHidden + kBlockThreads - 1) / kBlockThreads;
    float gamma_early[gamma_slots];
#pragma unroll
    for (int u = 0; u < gamma_slots; ++u) {
        const int i = min((int)threadIdx.x + u * kBlockThreads, d.hidden - 1);
        gamma_early[u] = __bfloat162float(gamma[i]);
    }
    asm volatile("" ::: "memory");
"""


def norm_early(src: Path, tree: bool, all_sites: bool = False) -> None:
    """all_sites: also the dense layer's post-norm and lm_head's final norm (norm_tree_all), not only qkv + MoE router."""
    name = "norm_tree_all" if all_sites else "norm_tree" if tree else "norm_gamma_early"
    kernel = src / "kernels/fleet_kernel.hip"
    k = kernel.read_text()
    if "void rms_norm_lds_gamma_early(" in k or "void rms_norm_lds_tree_early(" in k:
        raise SystemExit("norm_gamma_early and norm_tree are alternative patches; select only one")
    original = _function(k, "__device__ inline void rms_norm_lds(")
    helper = "rms_norm_lds_tree_early" if tree else "rms_norm_lds_gamma_early"
    new = _once(original, "void rms_norm_lds(", f"void {helper}(", name)
    new = _once(new, "    float local = 0.f;", GAMMA_EARLY + "    float local = 0.f;", name)
    # Match the epilogue, not the hoisted gamma load itself.
    new = _once(new, "bf16_round(v[i] * inv) * __bfloat162float(gamma[i])",
                "bf16_round(v[i] * inv) * gamma_early[(i - (int)threadIdx.x) / kBlockThreads]", name)
    if tree:
        new = _once(new, "block_reduce_ordered(local, smem)", "block_reduce_ordered_etx_tree(local, smem)", name)
    k = _once(k, original, original + "\n\n" + (ORDERED_TREE if tree else "") + new, name)
    k = _once(k, "rms_norm_lds(d, w.layer_norm_in[L], s.vec, s.smem);",
                f"{helper}(d, w.layer_norm_in[L], s.vec, s.smem);", name)
    router = "rms_norm_lds(d, w.layer_norm_post[layer], s.vec, s.smem);"
    if all_sites:
        k = _once(k, router, f"{helper}(d, w.layer_norm_post[layer], s.vec, s.smem);", name)
        k = _once(k, "rms_norm_lds(d, w.final_norm, s.vec, s.smem);", f"{helper}(d, w.final_norm, s.vec, s.smem);", name)
    else:
        k = _once(k, router, f"if (moe) {helper}(d, w.layer_norm_post[layer], s.vec, s.smem);\n    else {router}", name)
    kernel.write_text(k)


def apply(src: Path) -> None:
    norm_early(src, tree=False)
