"""Build-time patches for a COPY of fleet-mi300x's sources (the fleet repository itself is never modified).

  python -m examples.dsv2lite.patch_fleet <fleet src dir copy> <patch> [<patch> ...]

gemv_noinline: fleet's gemv_rows dispatches on K at run time between five GEMV paths, so every call site instantiates
  all five and the register allocator sizes the union: 262 registers per lane (VGPR + AGPR) where each path alone needs
  74-118 (measured with the local gfx942 compile, 2026-09-27). Two workgroups per CU (the two-request interleave) need
  <= 256, so the paths become separate (noinline) functions: each call pays one function call, and the megakernel's
  heaviest tile becomes attention (240).

gemv_no_terminal: qkv's q and kv_a GEMVs only; peel the terminal group in a
  private copy of the pipelined implementation. Consumed padded rows stay.
norm_gamma_early: qkv and MoE router only; cache immutable gamma before the
  unchanged square-sum/reduction. Dense post-norm and lm_head stay unchanged.
norm_tree: same gamma hoist plus the exact stride-128/64/32/.../1 add DAG,
  using LDS for the inter-wave inputs and ordered shuffles inside wave zero.
  Alternative to norm_gamma_early, not an additional patch.
"""
from __future__ import annotations

import sys
from pathlib import Path

PATCHES = {
    "gemv_noinline": ("kernels/gemv.h", [
        ("template <int RPI, int D>\n__device__ __forceinline__ void gemv_rows_pipelined(",
         "template <int RPI, int D>\n__device__ __noinline__ void gemv_rows_pipelined(   // etx patch gemv_noinline"),
        ("template <int RPI>   // rows per wave per group (even); each half-wave takes RPI/2\n__device__ __forceinline__ void gemv_rows_pipelined_half(",
         "template <int RPI>   // rows per wave per group (even); each half-wave takes RPI/2\n__device__ __noinline__ void gemv_rows_pipelined_half(   // etx patch"),
        ("template <int RPI, int D>\n__device__ __forceinline__ void gemv_rows_impl(",
         "template <int RPI, int D>\n__device__ __noinline__ void gemv_rows_impl(   // etx patch gemv_noinline"),
    ]),
}


PIPELINE_OLD = """    FLEET_ISSUE_GROUP(bufA, k)
    while (true) {
        const int k2 = k + stride;
        FLEET_ISSUE_GROUP(bufB, k2)          // unconditional; clamped past the end
        FLEET_REDUCE_GROUP(bufA, k)
        if (k2 >= cnt) break;
        const int k3 = k2 + stride;
        FLEET_ISSUE_GROUP(bufA, k3)
        FLEET_REDUCE_GROUP(bufB, k2)
        if (k3 >= cnt) break;
        k = k3;
    }
"""

PIPELINE_NO_TERMINAL = """    FLEET_ISSUE_GROUP(bufA, k)
    // Complete pairs retain unconditional issue-before-reduce in each basic
    // block. Peel the one/two-group tail, never predicate individual loads.
    while (k + 2 * stride < cnt) {
        const int k2 = k + stride;
        FLEET_ISSUE_GROUP(bufB, k2)
        FLEET_REDUCE_GROUP(bufA, k)
        const int k3 = k2 + stride;
        FLEET_ISSUE_GROUP(bufA, k3)
        FLEET_REDUCE_GROUP(bufB, k2)
        k = k3;
    }
    if (k + stride < cnt) {
        const int k2 = k + stride;
        FLEET_ISSUE_GROUP(bufB, k2)
        FLEET_REDUCE_GROUP(bufA, k)
        FLEET_REDUCE_GROUP(bufB, k2)
    } else {
        FLEET_REDUCE_GROUP(bufA, k)
    }
"""

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


def _once(text: str, old: str, new: str, name: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"patch {name}: expected exactly one match for:\n{old}")
    return text.replace(old, new)


def _function(text: str, signature: str) -> str:
    """Extract a pinned C++ helper, including its balanced macro bodies."""
    if text.count(signature) != 1:
        raise SystemExit(f"expected one helper signature: {signature}")
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise SystemExit(f"unbalanced helper: {signature}")
    return text[start:end]


def _gemv_no_terminal(src: Path) -> None:
    name = "gemv_no_terminal"
    gemv = src / "kernels/gemv.h"
    kernel = src / "kernels/fleet_kernel.hip"
    g, k = gemv.read_text(), kernel.read_text()
    # Keep the original helper/dispatch byte-for-byte: router, expert_down,
    # merge, o_proj, dense_down and lm_head still bind to those originals.
    piped = _function(g, "template <int RPI, int D>\n__device__ __forceinline__ void gemv_rows_pipelined(")
    new_piped = _once(piped, "void gemv_rows_pipelined(", "void gemv_rows_pipelined_no_terminal(", name)
    new_piped = _once(new_piped, PIPELINE_OLD, PIPELINE_NO_TERMINAL, name)
    g = _once(g, piped, piped + "\n\n" + new_piped, name)
    dispatch = _function(g, "__device__ inline void gemv_rows(")
    new_dispatch = _once(dispatch, "void gemv_rows(", "void gemv_rows_no_terminal(", name)
    if new_dispatch.count("gemv_rows_pipelined<") != 3:
        raise SystemExit(f"patch {name}: expected three full-wave dispatch paths")
    new_dispatch = new_dispatch.replace("gemv_rows_pipelined<", "gemv_rows_pipelined_no_terminal<")
    g = _once(g, dispatch, dispatch + "\n\n" + new_dispatch, name)
    qkv = _function(k, "        case TASK_QKV_FUSED: {\n            // layer input:")
    if qkv.count("gemv_rows(") != 3:
        raise SystemExit(f"patch {name}: expected q plus both kv_a branches")
    k = _once(k, qkv, qkv.replace("gemv_rows(", "gemv_rows_no_terminal("), name)
    gemv.write_text(g)
    kernel.write_text(k)


def _norm_early(src: Path, tree: bool) -> None:
    name = "norm_tree" if tree else "norm_gamma_early"
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
    k = _once(k, router, f"if (moe) {helper}(d, w.layer_norm_post[layer], s.vec, s.smem);\n    else {router}", name)
    kernel.write_text(k)


def apply(src: Path, name: str) -> None:
    if name in {"gemv_no_terminal", "norm_gamma_early", "norm_tree"}:
        if name == "gemv_no_terminal":
            _gemv_no_terminal(src)
        else:
            _norm_early(src, tree=name == "norm_tree")
        print(f"patched {src} ({name}; qkv / MoE-router scope)")
        return
    rel, subs = PATCHES[name]
    f = src / rel
    text = f.read_text()
    for old, new in subs:
        if text.count(old) != 1:
            raise SystemExit(f"patch {name}: expected exactly one match in {f} for:\n{old}")
        text = text.replace(old, new)
    f.write_text(text)
    print(f"patched {f} ({name}, {len(subs)} edits)")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    for name in sys.argv[2:]:
        apply(src, name)


if __name__ == "__main__":
    main()
