"""Lever coherent_loads (D2): fleet's tiles read every payload another task wrote with system-scope coherent loads, so
the megakernel can drop the per-task acquire `buffer_inv sc1` while every producer keeps its L2 write-back.

Qwen's cache fix (ETX_VL_COH, etx/ops/coh.py) ported to fleet's tiles. Memory model (gfx942, docs/deepseek/DSV2-D1-D5.md D2):
  producer  plain stores (or fleet's agent-scope st_coh), then the megakernel's release writes the XCD's L2 back
            (ETX_RELEASE_DEVICE / the DEVICE last-arriver flush; graph.py marks the tiles "coherent_loads", which the
            compiler treats as a PLAIN producer: fleet measured the write-back necessary, fleet_runtime.h:247-255);
  consumer  sc0 sc1 loads (relaxed system-scope atomics through address_space(1), the lowering Qwen's coherent builds
            proved 32/32 against plain write-back producers), past L1 and L2, so no stale line can be read and no
            invalidate is needed.
The patched copy is only built by the `coh` arm; every original helper stays byte for byte otherwise.

Covered payloads (graph.py, ETX_DSV2_COH=1), consumer side:
  x / x_norm / o / expert_h      stage_vector (every call site stages an activation) and the two residual folds
  expert_out (bf16), o_proj partials   the vector fold paths, which fleet's own coherent mode left plain for bf16
  router_logits                  load_logit
  q, q_absorbed, kv_a            q_absorb_rows, q_absorb_gather, kv_post
  attn_partial                   merge_and_uv
  token_in, argmax candidates    fleet's own coh path (agent-scope ld_coh / st_coh), switched on at compile time
Not covered, and why that is safe: the KV cache rows a task reads were written in earlier tokens, i.e. earlier kernel
launches (host.hip launches once per token); dense_down reads the dense h from global inside gemv_rows, so graph.py
leaves dense_down plain (it keeps its acquire); q_absorb (no PUB_WAIT) is unreachable, fleet_shim.hip sets
FLAG_PUB_WAIT on every attention task.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _once

NAME = "coherent_loads"

HELPERS_ANCHOR = "constexpr int kWaves = 256 / kWaveLanes;   // waves per workgroup (= fleet_runtime.h's)\n"
HELPERS = """
// etx lever coherent_loads (D2): system-scope relaxed atomic loads, `global_load ... sc0 sc1` (address_space(1), so not
// flat_), past this XCD's L1 and L2. For payloads written by another task: the producer's release wrote its L2 back.
#define FLEET_COH_LOADS 1
#define FLEET_COH_G __attribute__((address_space(1)))
__device__ __forceinline__ float ld_sys(const float* p) {
    return __hip_atomic_load((const FLEET_COH_G float*)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint64_t ld_sys64(const void* p) {
    return __hip_atomic_load((const FLEET_COH_G uint64_t*)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// 16 B as two 8 B coherent loads (no 16 B atomic form), both in flight
__device__ __forceinline__ uint4 ld_sys_u4(const uint4* p) {
    const uint64_t lo = ld_sys64(p), hi = ld_sys64(reinterpret_cast<const uint64_t*>(p) + 1);
    uint4 r;
    r.x = (uint32_t)lo; r.y = (uint32_t)(lo >> 32); r.z = (uint32_t)hi; r.w = (uint32_t)(hi >> 32);
    return r;
}
__device__ __forceinline__ float4 ld_sys_f4(const float4* p) {
    const uint4 u = ld_sys_u4(reinterpret_cast<const uint4*>(p));
    float4 r;
    r.x = __uint_as_float(u.x); r.y = __uint_as_float(u.y); r.z = __uint_as_float(u.z); r.w = __uint_as_float(u.w);
    return r;
}
"""

# (file, old, new): every edit is an exact single match in the pinned 3fb67a6 source
EDITS = [
    ("kernels/gemv.h", HELPERS_ANCHOR, HELPERS_ANCHOR + HELPERS),
    # stage_vector: every call site stages an activation another task wrote (x_norm, o, expert_h, x)
    ("kernels/gemv.h",
     "    for (int i = threadIdx.x; i < n4; i += kBlockThreads) d4[i] = s4[i];\n"
     "    for (int i = (n4 << 2) + threadIdx.x; i < n; i += kBlockThreads) dst[i] = src[i];\n",
     "    for (int i = threadIdx.x; i < n4; i += kBlockThreads) d4[i] = ld_sys_f4(s4 + i);   // etx lever coherent_loads\n"
     "    for (int i = (n4 << 2) + threadIdx.x; i < n; i += kBlockThreads) dst[i] = ld_sys(src + i);\n"),
    # fleet's coherent switch, at compile time: x / partials / argmax / token_in go through its coherent paths
    ("kernels/fleet_kernel.hip",
     "    const bool coh = rt.coherent_acts != 0;   // cross-XCD activations via atomics\n",
     "    const bool coh = true;   // etx lever coherent_loads: fleet's coherent activation paths, always\n"),
    ("kernels/fleet_kernel.hip",
     "    const bool coh = rt.coherent_acts != 0;\n    for (int i = threadIdx.x; i < n_cand; i += blockDim.x) {\n",
     "    const bool coh = true;   // etx lever coherent_loads\n    for (int i = threadIdx.x; i < n_cand; i += blockDim.x) {\n"),
    # o_proj partial fold (router prologue): fleet's coherent path is per element (72 dependent-free 4 B requests per
    # thread); the vector path with coherent 16 B loads keeps the plain build's request count and summation order
    ("kernels/fleet_kernel.hip",
     "    if (coh) {\n#pragma unroll 4\n        for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {\n"
     "            float acc = 0.f;\n#pragma unroll\n            for (int x = 0; x < kNumXCDs; ++x) acc += ld_coh(partials",
     "    if (false) {   // etx lever coherent_loads: the vector path below loads coherently\n#pragma unroll 4\n"
     "        for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {\n"
     "            float acc = 0.f;\n#pragma unroll\n            for (int x = 0; x < kNumXCDs; ++x) acc += ld_coh(partials"),
    ("kernels/fleet_kernel.hip", "            float4 acc = p4[i];\n", "            float4 acc = ld_sys_f4(p4 + i);\n"),
    ("kernels/fleet_kernel.hip", "                const float4 q = p4[x * n4 + i];\n",
     "                const float4 q = ld_sys_f4(p4 + x * n4 + i);\n"),
    ("kernels/fleet_kernel.hip", "            const float4 xs = s4[i];\n", "            const float4 xs = ld_sys_f4(s4 + i);\n"),
    # expert partial fold (qkv / lm_head prologue): fleet's coherent path read the bf16 partials PLAIN
    ("kernels/fleet_kernel.hip",
     "    if (coh) {\n        // coherent mode: per-element agent-scope atomic loads (no 16-byte form)\n",
     "    if (false) {   // etx lever coherent_loads: the vector path below loads coherently (fleet's read bf16 plain)\n"),
    ("kernels/fleet_kernel.hip", "            for (int x = 0; x < kNumXCDs; ++x) v[x] = p4[x * n8 + i];\n",
     "            for (int x = 0; x < kNumXCDs; ++x) v[x] = ld_sys_u4(p4 + x * n8 + i);\n"),
    ("kernels/fleet_kernel.hip", "            const float4 lo = s4[2 * i], hi = s4[2 * i + 1];\n",
     "            const float4 lo = ld_sys_f4(s4 + 2 * i), hi = ld_sys_f4(s4 + 2 * i + 1);\n"),
    # router logits (gate_up's per-task top-k; down without cached routing)
    ("kernels/fleet_kernel.hip",
     "    return (lane < kWaveLanes && lane < d.n_routed) ? logits[lane] : -INFINITY;\n",
     "    return (lane < kWaveLanes && lane < d.n_routed) ? ld_sys(logits + lane) : -INFINITY;   // etx lever coherent_loads\n"),
    # kv_post: this XCD's kv_a copy (the qkv task's output)
    ("kernels/attention.h", "        const float v = kv_a_raw[i];\n", "        const float v = ld_sys(kv_a_raw + i);   // etx lever coherent_loads\n"),
    ("kernels/attention.h",
     "        row_lds[i] = bf16_round(bf16_round(kv_a_raw[i] * inv) * __bfloat162float(kv_a_norm_w[i]));\n",
     "        row_lds[i] = bf16_round(bf16_round(ld_sys(kv_a_raw + i) * inv) * __bfloat162float(kv_a_norm_w[i]));\n"),
    ("kernels/attention.h", "        row_lds[kv_lora + i] = kv_a_raw[kv_lora + i];\n",
     "        row_lds[kv_lora + i] = ld_sys(kv_a_raw + kv_lora + i);\n"),
    # q_absorb_rows: q (wave-uniform per row) as one coherent load per lane, broadcast by shuffle
    ("kernels/attention.h",
     "    constexpr int kRowBatch = 8;\n"
     "    for (int i0 = row0 + wave * kRowBatch; i0 < row0 + rows; i0 += kWaves * kRowBatch) {\n"
     "        uint4 v[kRowBatch];\n#pragma unroll\n"
     "        for (int k = 0; k < kRowBatch; ++k) v[k] = w_uk4[(int64_t)(i0 + k) * row4 + lane];\n#pragma unroll\n"
     "        for (int k = 0; k < kRowBatch; ++k) {\n"
     "            const float q = q_head[i0 + k];\n",
     "    constexpr int kRowBatch = 8;\n"
     "    // etx lever coherent_loads: q is the qkv task's output. Lane r loads row0 + r once (coherent loads have no L1 to\n"
     "    // share 8 wave-uniform loads per batch); each batch row takes it by shuffle. The loop covers rows exactly.\n"
     "    if (rows > kWaveLanes || rows % (kWaves * kRowBatch) != 0) __builtin_trap();\n"
     "    const float q_lane = lane < rows ? ld_sys(q_head + row0 + lane) : 0.f;\n"
     "    for (int i0 = row0 + wave * kRowBatch; i0 < row0 + rows; i0 += kWaves * kRowBatch) {\n"
     "        uint4 v[kRowBatch];\n#pragma unroll\n"
     "        for (int k = 0; k < kRowBatch; ++k) v[k] = w_uk4[(int64_t)(i0 + k) * row4 + lane];\n#pragma unroll\n"
     "        for (int k = 0; k < kRowBatch; ++k) {\n"
     "            const float q = __shfl(q_lane, i0 + k - row0, kWaveLanes);\n"),
    # q_absorb_gather: the published q_c partials and q_pe
    ("kernels/attention.h",
     "        for (int s = 0; s < n_slices; ++s) v += partials[s * kv_lora + j];   // fixed order\n"
     "        q_c_lds[j] = v;\n    }\n"
     "    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) {\n"
     "        q_pe_lds[i] = q_head[qk_nope + i];\n",
     "        for (int s = 0; s < n_slices; ++s) v += ld_sys(partials + s * kv_lora + j);   // fixed order (etx lever coherent_loads)\n"
     "        q_c_lds[j] = v;\n    }\n"
     "    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) {\n"
     "        q_pe_lds[i] = ld_sys(q_head + qk_nope + i);\n"),
    # merge_and_uv: (m, l) of every chunk (wave-uniform) by one lane each, then the accumulators
    ("kernels/attention.h",
     "#pragma unroll\n    for (int c = 0; c < kMaxChunks; ++c) {\n"
     "        m_c[c] = c < n_chunks ? partials[c * kPartialStride] : -INFINITY;\n"
     "        l_c[c] = c < n_chunks ? partials[c * kPartialStride + 1] : 0.f;\n    }\n",
     "    // etx lever coherent_loads: lane c loads chunk c's (m, l) once, every lane takes them by shuffle (the same\n"
     "    // values: lanes past n_chunks hold -inf / 0 exactly as the predicated loads did)\n"
     "    static_assert(kMaxChunks <= kWaveLanes, \"one lane per chunk\");\n"
     "    const int lane_c = threadIdx.x % kWaveLanes;\n"
     "    const float m_lane = lane_c < n_chunks ? ld_sys(partials + lane_c * kPartialStride) : -INFINITY;\n"
     "    const float l_lane = lane_c < n_chunks ? ld_sys(partials + lane_c * kPartialStride + 1) : 0.f;\n"
     "#pragma unroll\n    for (int c = 0; c < kMaxChunks; ++c) {\n"
     "        m_c[c] = __shfl(m_lane, c, kWaveLanes);\n"
     "        l_c[c] = __shfl(l_lane, c, kWaveLanes);\n    }\n"),
    ("kernels/attention.h",
     "                a[c] = c0 + c < n_chunks ? partials[(c0 + c) * kPartialStride + 2 + j] : 0.f;\n",
     "                a[c] = c0 + c < n_chunks ? ld_sys(partials + (c0 + c) * kPartialStride + 2 + j) : 0.f;\n"),
]


def apply(src: Path) -> None:
    texts: dict[str, str] = {}
    for rel, old, new in EDITS:
        text = texts.get(rel)
        if text is None:
            text = (src / rel).read_text()
        texts[rel] = _once(text, old, new, NAME)
    for rel, text in texts.items():
        (src / rel).write_text(text)
