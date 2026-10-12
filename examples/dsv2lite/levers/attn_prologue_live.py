"""Lever attn_prologue_live: the attention prologue loads everything once, up front, and keeps it in registers.

fleet's kv_post (attention.h @3fb67a6) loads kv_a_raw for the sum of squares, runs the 10-barrier ordered tree, then
RELOADS kv_a_raw together with the norm weights and the rope part (one dependent round trip), and its rope loads cos/sin
in another; q_absorb_gather later reloads the same cos/sin for the query's rope after the publication wait (a third).
This lever issues all of them in one batch at entry (clamped addresses, no branches around loads), keeps them live,
runs the same ordered add DAG in 3 barriers instead of 10, and hands the four rope coefficients to the query's rope.
Every arithmetic expression and its order is fleet's; only where values come from changes. Shapes other than
kv_lora == 2 x 256 and qk_rope <= 64 take fleet's code. kv_post overlaps the q_absorb publication wait, so its share of
the saving is real only when kv_post, not that wait, is the attention task's critical path.
"""
from __future__ import annotations

from pathlib import Path

from examples.dsv2lite.patch_fleet import _function, _once

HELPERS = """
// etx lever attn_prologue_live --------------------------------------------------------------------------------------
// block_reduce_ordered's exact DAG at 256 threads in 3 barriers (lane t: (v[t]+v[t+128])+(v[t+64]+v[t+192]), then 32..1).
__device__ __forceinline__ float etx_attn_reduce_ordered(float v, float* smem) {
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
    __syncthreads();
    return out;
}

// The four rope coefficients one lane needs (pairs (i, i + half)), loaded once per attention task.
struct EtxRopeCS { float c0, s0, c1, s1; };

__device__ __forceinline__ EtxRopeCS etx_rope_cs(const float* __restrict__ cos, const float* __restrict__ sin, int d) {
    const int half = d / 2;
    const int i = min((int)threadIdx.x, half - 1);           // clamped: lanes >= half load an unused valid entry
    return EtxRopeCS{cos[i], sin[i], cos[i + half], sin[i + half]};
}

// apply_rope_interleaved with the coefficients already in registers: the same expressions, the same barriers.
__device__ __forceinline__ void etx_rope_pre(float* __restrict__ v, const EtxRopeCS& cs, int d) {
    const int half = d / 2;
    const int i = threadIdx.x;
    float a = 0.f, b = 0.f;
    if (i < half) { a = v[2 * i]; b = v[2 * i + 1]; }
    __syncthreads();
    if (i < half) {
        v[i] = a * cs.c0 - b * cs.s0;
        v[i + half] = b * cs.c1 + a * cs.s1;
    }
    __syncthreads();
}

__device__ inline void kv_post_live(
        const float* __restrict__ kv_a_raw, const __hip_bfloat16* __restrict__ kv_a_norm_w,
        const float* __restrict__ cos, const float* __restrict__ sin,
        float* __restrict__ row_lds, int kv_lora, int qk_rope, float eps,
        float* smem, bool write_cache, __hip_bfloat16* __restrict__ cache_row, EtxRopeCS& cs) {
    cs = etx_rope_cs(cos, sin, qk_rope);
    if (kv_lora != 2 * kBlockThreads || qk_rope > 64 || (int)blockDim.x != kBlockThreads) {
        kv_post(kv_a_raw, kv_a_norm_w, cos, sin, row_lds, kv_lora, qk_rope, eps, smem, write_cache, cache_row);
        return;
    }
    const int t = threadIdx.x;
    const float raw0 = kv_a_raw[t], raw1 = kv_a_raw[t + kBlockThreads];
    const float g0 = __bfloat162float(kv_a_norm_w[t]), g1 = __bfloat162float(kv_a_norm_w[t + kBlockThreads]);
    const float rope_raw = kv_a_raw[kv_lora + min(t, qk_rope - 1)];
    float local = 0.f;                                        // the loop's order: i = t, then t + 256
    local += raw0 * raw0;
    local += raw1 * raw1;
    const float ssq = etx_attn_reduce_ordered(local, smem);
    const float inv = rsqrtf(ssq / kv_lora + eps);
    row_lds[t] = bf16_round(bf16_round(raw0 * inv) * g0);     // HF's two roundings
    row_lds[t + kBlockThreads] = bf16_round(bf16_round(raw1 * inv) * g1);
    if (t < qk_rope) row_lds[kv_lora + t] = rope_raw;
    __syncthreads();
    etx_rope_pre(row_lds + kv_lora, cs, qk_rope);

    for (int i = threadIdx.x; i < kv_lora + qk_rope; i += blockDim.x) {
        const __hip_bfloat16 b = __float2bfloat16(row_lds[i]);
        row_lds[i] = __bfloat162float(b);
        if (write_cache) cache_row[i] = b;
    }
    __syncthreads();
}
"""

GATHER_LIVE = """
// q_absorb_gather with the rope coefficients kept from kv_post_live (no second cos/sin round trip).
__device__ inline void q_absorb_gather_live(
        const float* __restrict__ q_head, const float* __restrict__ partials, const EtxRopeCS& cs,
        float* __restrict__ q_c_lds, float* __restrict__ q_pe_lds,
        int qk_nope, int qk_rope, int kv_lora, int n_slices) {
    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < n_slices; ++s) v += partials[s * kv_lora + j];   // fixed order
        q_c_lds[j] = v;
    }
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) {
        q_pe_lds[i] = q_head[qk_nope + i];
    }
    __syncthreads();
    etx_rope_pre(q_pe_lds, cs, qk_rope);
}
"""

KV_POST_CALL = """            kv_post(kv_a, w.layer_kv_a_norm[L], tc.cos, tc.sin, s.row,
                    d.kv_lora, d.qk_rope, d.rms_eps, s.smem,"""
KV_POST_LIVE_CALL = """            EtxRopeCS etx_cs;
            kv_post_live(kv_a, w.layer_kv_a_norm[L], tc.cos, tc.sin, s.row,
                    d.kv_lora, d.qk_rope, d.rms_eps, s.smem,"""
KV_POST_TAIL = """                    cache_layer + (int64_t)tc.pos * kv.row);
            FLEET_STAMP_STAGE();   // after kv_post, before q_absorb"""
KV_POST_LIVE_TAIL = """                    cache_layer + (int64_t)tc.pos * kv.row, etx_cs);
            FLEET_STAMP_STAGE();   // after kv_post, before q_absorb"""
GATHER_CALL = """                q_absorb_gather(act.q + (int64_t)t.head * d.q_head_dim,
                                act.q_absorbed + (int64_t)t.head * kQAbsSlices * d.kv_lora,
                                tc.cos, tc.sin, s.q_c, s.q_pe,"""
GATHER_LIVE_CALL = """                q_absorb_gather_live(act.q + (int64_t)t.head * d.q_head_dim,
                                act.q_absorbed + (int64_t)t.head * kQAbsSlices * d.kv_lora,
                                etx_cs, s.q_c, s.q_pe,"""


def apply(src: Path) -> None:
    name = "attn_prologue_live"
    attn = src / "kernels/attention.h"
    kernel = src / "kernels/fleet_kernel.hip"
    a, k = attn.read_text(), kernel.read_text()
    kv_post = _function(a, "__device__ inline void kv_post(")
    a = _once(a, kv_post, kv_post + "\n" + HELPERS, name)
    gather = _function(a, "__device__ inline void q_absorb_gather(")
    a = _once(a, gather, gather + "\n" + GATHER_LIVE, name)
    k = _once(k, KV_POST_CALL, KV_POST_LIVE_CALL, name)
    k = _once(k, KV_POST_TAIL, KV_POST_LIVE_TAIL, name)
    k = _once(k, GATHER_CALL, GATHER_LIVE_CALL, name)
    attn.write_text(a)
    kernel.write_text(k)
