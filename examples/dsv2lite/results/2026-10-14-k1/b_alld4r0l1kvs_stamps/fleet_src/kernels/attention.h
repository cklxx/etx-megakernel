// Absorbed-MLA decode attention — the CU-task body.
//
// docs/design.md §2, §3. The cache holds one compressed row per position,
// `c ‖ k_pe` (576 bf16), shared by all 16 heads, so a whole layer's cache is
// 1.2 MB and sits in each XCD's 4 MB L2. Materialising K and V per head would
// read 7.1x more per position and add ~2 GFLOP, which at batch 1 is pure loss.
//
// Three things are fused into this task's prologue to remove global events (§3):
//   1. kv post-processing: RMSNorm on c, RoPE on k_pe, and the cache append
//      are re-derived here from the raw 576-vector rather than costing an event
//   2. q-absorb: q_c[h] = q_nope[h] @ W_UK[h], each task reading its own
//      131 KB W_UK slice; q_pe gets its RoPE here too
//   3. head-to-XCD affinity: XCD k owns heads 2k and 2k+1, so the merge that
//      follows waits on an XCD-local counter instead of a global event
//
// Everything the prologue produces lives in this workgroup's LDS: the cache
// row, q_c and q_pe. No two attention tasks share scratch, so the 16 (or 64)
// tasks of a layer can run in any order on any XCD.
//
// The new token's row is *not* read back from the cache: one designated task
// writes it there for later tokens, but every task scores the current
// position from its own LDS copy, so there is no cross-XCD read of a line
// another workgroup is still writing.
//
// Split-KV (D4) turns each head into kv_chunks tasks emitting flash-decoding
// partials (m, l, acc[512]); with kv_chunks == 1 the partial is the answer.
#pragma once

#include <hip/hip_runtime.h>
#include "gemv.h"
#include "../runtime/fleet_types.h"

namespace fleet {

// DeepseekV2 RoPE: interleave-transpose, then the standard rotation.
// x[0..d) becomes [x0,x2,...,x_{d-2}, x1,x3,...,x_{d-1}] before rotate_half.
// Implementing the plain form instead is silently wrong, which is why
// tests/test_absorbed_equivalence.py checks the two differ.
//
// In place on an LDS vector; d = 64 so one wave covers it, and the gather
// happens before any write because each lane holds both of its inputs.
__device__ __forceinline__ void apply_rope_interleaved(
        float* __restrict__ v, const float* __restrict__ cos,
        const float* __restrict__ sin, int d) {
    const int half = d / 2;
    const int i = threadIdx.x;
    float a = 0.f, b = 0.f;
    if (i < half) { a = v[2 * i]; b = v[2 * i + 1]; }
    __syncthreads();
    if (i < half) {
        // rotate_half pairs (i, i+half): out_i = a*cos_i - b*sin_i
        //                                out_{i+half} = b*cos_{i+half} + a*sin_{i+half}
        v[i] = a * cos[i] - b * sin[i];
        v[i + half] = b * cos[i + half] + a * sin[i + half];
    }
    __syncthreads();
}

// Prologue step 1: rebuild this token's cache row from the raw kv_a output,
// into LDS, rounded through bf16 so it is bit-identical to what later tokens
// will read back from the cache. One designated task also stores it.
__device__ inline void kv_post(
        const float* __restrict__ kv_a_raw, const __hip_bfloat16* __restrict__ kv_a_norm_w,
        const float* __restrict__ cos, const float* __restrict__ sin,
        float* __restrict__ row_lds, int kv_lora, int qk_rope, float eps,
        float* smem, bool write_cache, __hip_bfloat16* __restrict__ cache_row) {
    float local = 0.f;
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) {
        const float v = ld_l1(kv_a_raw + i);   // etx lever l1_loads
        local += v * v;
    }
    const float ssq = block_reduce_ordered(local, smem);
    const float inv = rsqrtf(ssq / kv_lora + eps);

    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) {   // HF's two roundings
        row_lds[i] = bf16_round(bf16_round(ld_l1(kv_a_raw + i) * inv) * __bfloat162float(kv_a_norm_w[i]));
    }
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) {
        row_lds[kv_lora + i] = ld_l1(kv_a_raw + kv_lora + i);
    }
    __syncthreads();
    apply_rope_interleaved(row_lds + kv_lora, cos, sin, qk_rope);

    for (int i = threadIdx.x; i < kv_lora + qk_rope; i += blockDim.x) {
        const __hip_bfloat16 b = __float2bfloat16(row_lds[i]);
        row_lds[i] = __bfloat162float(b);
        if (write_cache) cache_row[i] = b;
    }
    __syncthreads();
}

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
    const float raw0 = ld_l1(kv_a_raw + t), raw1 = ld_l1(kv_a_raw + t + kBlockThreads);   // etx lever l1_loads
    const float g0 = __bfloat162float(kv_a_norm_w[t]), g1 = __bfloat162float(kv_a_norm_w[t + kBlockThreads]);
    const float rope_raw = ld_l1(kv_a_raw + kv_lora + min(t, qk_rope - 1));
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


// Prologue step 2: absorb W_UK into the query, and RoPE the rope half.
//   q_c[h] = q_nope[h] @ W_UK[h],  W_UK[h] = kv_b_proj rows [h*256, h*256+128)
// This is a [128] x [128, 512] product per head — 131 KB of weights, read once
// per task. Streamed by rows: each wave takes 32 of the 128 rows, a lane
// holds columns [8l, 8l+8) of a row (one 16-byte load), scales by the scalar
// q_nope[i] and accumulates eight partials; the four waves' partials are
// summed through LDS in wave order. (The first version walked the 128 rows
// per output column with one dependent global load each; that prologue,
// not the position loop, was the 296 us measured per attention task.)
// `part_lds` must hold kWaves * kMaxKvLora floats.
// One publisher's row slice of q_c[head] = W_UK[head]^T q_nope, summed over
// rows [row0, row0 + rows) and written to global as an fp32 partial. The
// readers add the kQAbsSlices partials in slice order, so the result is
// deterministic; it is not the same summation order as the single-task
// q_absorb below, which sums over all 128 rows per wave.
//
// The split is by rows, not columns: a column slice would leave half of every
// wave's lanes idle on each 16-byte load and halve the publisher's bandwidth,
// which is the whole point of publishing.
__device__ inline void q_absorb_rows(
        const float* __restrict__ q_head, const __hip_bfloat16* __restrict__ kv_b,
        float* __restrict__ out_partial, float* __restrict__ part_lds,
        int head, int qk_nope, int v_head, int kv_lora, int row0, int rows) {
    const int row_stride = qk_nope + v_head;
    const uint4* w_uk4 = reinterpret_cast<const uint4*>(
        kv_b + (int64_t)head * row_stride * kv_lora);
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;
    const int row4 = kv_lora / 8;

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;

    constexpr int kRowBatch = 8;
    // etx lever l1_loads: q is the qkv task's output. Lane r loads row0 + r once (coherent loads have no L1 to
    // share 8 wave-uniform loads per batch); each batch row takes it by shuffle. The loop covers rows exactly.
    if (rows > kWaveLanes || rows % (kWaves * kRowBatch) != 0) __builtin_trap();
    const float q_lane = lane < rows ? ld_l1(q_head + row0 + lane) : 0.f;
    for (int i0 = row0 + wave * kRowBatch; i0 < row0 + rows; i0 += kWaves * kRowBatch) {
        uint4 v[kRowBatch];
#pragma unroll
        for (int k = 0; k < kRowBatch; ++k) v[k] = w_uk4[(int64_t)(i0 + k) * row4 + lane];
#pragma unroll
        for (int k = 0; k < kRowBatch; ++k) {
            const float q = __shfl(q_lane, i0 + k - row0, kWaveLanes);
            float a, b;
            unpack_bf16x2(v[k].x, a, b); acc[0] += q * a; acc[1] += q * b;
            unpack_bf16x2(v[k].y, a, b); acc[2] += q * a; acc[3] += q * b;
            unpack_bf16x2(v[k].z, a, b); acc[4] += q * a; acc[5] += q * b;
            unpack_bf16x2(v[k].w, a, b); acc[6] += q * a; acc[7] += q * b;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) part_lds[wave * kMaxKvLora + 8 * lane + j] = acc[j];
    __syncthreads();
    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
        float v = 0.f;
        for (int w = 0; w < kWaves; ++w) v += part_lds[w * kMaxKvLora + j];   // fixed order
        out_partial[j] = v;
    }
    __syncthreads();
}

// The reader side: add the published partials in slice order and take q_pe
// straight from q, which is 256 bytes and needs no publishing.
__device__ inline void q_absorb_gather(
        const float* __restrict__ q_head, const float* __restrict__ partials,
        const float* __restrict__ cos, const float* __restrict__ sin,
        float* __restrict__ q_c_lds, float* __restrict__ q_pe_lds,
        int qk_nope, int qk_rope, int kv_lora, int n_slices) {
    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < n_slices; ++s) v += ld_l1(partials + s * kv_lora + j);   // fixed order (etx lever l1_loads)
        q_c_lds[j] = v;
    }
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) {
        q_pe_lds[i] = ld_l1(q_head + qk_nope + i);
    }
    __syncthreads();
    apply_rope_interleaved(q_pe_lds, cos, sin, qk_rope);
}

// q_absorb_gather with the rope coefficients kept from kv_post_live (no second cos/sin round trip).
__device__ inline void q_absorb_gather_live(
        const float* __restrict__ q_head, const float* __restrict__ partials, const EtxRopeCS& cs,
        float* __restrict__ q_c_lds, float* __restrict__ q_pe_lds,
        int qk_nope, int qk_rope, int kv_lora, int n_slices) {
    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < n_slices; ++s) v += ld_l1(partials + s * kv_lora + j);   // fixed order (etx lever l1_loads)
        q_c_lds[j] = v;
    }
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) {
        q_pe_lds[i] = ld_l1(q_head + qk_nope + i);
    }
    __syncthreads();
    etx_rope_pre(q_pe_lds, cs, qk_rope);
}


__device__ inline void q_absorb(
        const float* __restrict__ q_head, const __hip_bfloat16* __restrict__ kv_b,
        const float* __restrict__ cos, const float* __restrict__ sin,
        float* __restrict__ q_c_lds, float* __restrict__ q_pe_lds,
        float* __restrict__ part_lds,
        int head, int qk_nope, int qk_rope, int v_head, int kv_lora) {
    const int row_stride = qk_nope + v_head;               // 256
    const uint4* w_uk4 = reinterpret_cast<const uint4*>(
        kv_b + (int64_t)head * row_stride * kv_lora);      // [qk_nope][kv_lora/8]
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;
    const int row4 = kv_lora / 8;                          // 64: one uint4 per lane

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;

    constexpr int kRowBatch = 8;
    for (int i0 = wave * kRowBatch; i0 < qk_nope; i0 += kWaves * kRowBatch) {
        uint4 v[kRowBatch];
#pragma unroll
        for (int k = 0; k < kRowBatch; ++k) v[k] = w_uk4[(int64_t)(i0 + k) * row4 + lane];
#pragma unroll
        for (int k = 0; k < kRowBatch; ++k) {
            const float q = q_head[i0 + k];
            float a, b;
            unpack_bf16x2(v[k].x, a, b); acc[0] += q * a; acc[1] += q * b;
            unpack_bf16x2(v[k].y, a, b); acc[2] += q * a; acc[3] += q * b;
            unpack_bf16x2(v[k].z, a, b); acc[4] += q * a; acc[5] += q * b;
            unpack_bf16x2(v[k].w, a, b); acc[6] += q * a; acc[7] += q * b;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) part_lds[wave * kMaxKvLora + 8 * lane + j] = acc[j];
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) {
        q_pe_lds[i] = q_head[qk_nope + i];
    }
    __syncthreads();
    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
        float v = 0.f;
        for (int w = 0; w < kWaves; ++w) v += part_lds[w * kMaxKvLora + j];   // fixed order
        q_c_lds[j] = v;
    }
    __syncthreads();
    apply_rope_interleaved(q_pe_lds, cos, sin, qk_rope);   // reference_decode.py:205
}

// Main body: flash-decoding over this task's slice of the cache.
//
//   s[t] = scale * (q_c . c[t] + q_pe . k_pe[t])
//   online softmax over t, accumulating acc += p[t] * c[t]
//
// One wave per position, eight positions in flight: lane l holds c[8l..8l+8)
// of the row (64 lanes x 8 = 512 = kv_lora) and lanes 0..7 additionally hold
// k_pe[8l..8l+8) (64 = qk_rope), so a position is one 16-byte load per lane
// plus one more for eight lanes, a shuffle reduction for the score, and eight
// FMAs into a register accumulator. The four waves keep private (m, l, acc)
// and are merged in a fixed order at the end. The lane mapping assumes
// kv_lora == 512 and qk_rope == 64; the host refuses other shapes.
__device__ inline void attention_chunk(
        const __hip_bfloat16* __restrict__ cache, const float* __restrict__ q_c,
        const float* __restrict__ q_pe, const float* __restrict__ row_new,
        float scale, int seq_len, int row_elems, int chunk, int n_chunks,
        float* __restrict__ out, float* __restrict__ part_lds) {
    const int per_chunk = (seq_len + n_chunks - 1) / n_chunks;
    const int t_begin = chunk * per_chunk;
    const int t_end = min(seq_len, t_begin + per_chunk);
    const int t_new = seq_len - 1;

    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;
    const bool rope_lane = lane < kMaxQkRope / 8;

    float qc[8], qp[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        qc[j] = q_c[8 * lane + j];
        qp[j] = rope_lane ? q_pe[8 * lane + j] : 0.f;
    }

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    float m = -INFINITY, l = 0.f;

    const uint4* cache4 = reinterpret_cast<const uint4*>(cache);
    const int row4 = row_elems / 8;   // 72 uint4 per position

    // One position's contribution to this wave's running (m, l, acc), given
    // its already-reduced score s.
    auto update = [&](const float* c, float s) {
        const float m_new = fmaxf(m, s);
        const float rescale = __expf(m - m_new);   // exp(-inf) = 0 on the first step
        const float p = __expf(s - m_new);
        l = l * rescale + p;
        m = m_new;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = acc[j] * rescale + p * c[j];
    };

    // Cached positions in batches of kAttnBatch per wave. All loads of the
    // batch are issued first (unconditional: positions past the end are
    // clamped to the last valid row and masked afterwards, so the waitcnt
    // pass never has to wait for everything), then the 8 scores are reduced
    // *together* — 8 independent shuffle trees instead of 8 dependent ones,
    // which was the serial cost per position — and only then the online
    // softmax runs over the 8 results.
    constexpr int kAttnBatch = 8;
    const int t_cached = min(t_end, t_new);        // rows that live in the cache
    for (int base = t_begin + wave * kAttnBatch; base < t_cached;
         base += kWaves * kAttnBatch) {
        uint4 vc[kAttnBatch], vr[kAttnBatch];
#pragma unroll
        for (int k = 0; k < kAttnBatch; ++k) {
            const int t = min(base + k, t_cached - 1);
            vc[k] = cache4[(int64_t)t * row4 + lane];
            vr[k] = cache4[(int64_t)t * row4 + kMaxKvLora / 8 + (lane & 7)];
        }
        float c[kAttnBatch][8];
        float s[kAttnBatch];
#pragma unroll
        for (int k = 0; k < kAttnBatch; ++k) {
            unpack_bf16x2(vc[k].x, c[k][0], c[k][1]); unpack_bf16x2(vc[k].y, c[k][2], c[k][3]);
            unpack_bf16x2(vc[k].z, c[k][4], c[k][5]); unpack_bf16x2(vc[k].w, c[k][6], c[k][7]);
            float kp[8];
            unpack_bf16x2(vr[k].x, kp[0], kp[1]); unpack_bf16x2(vr[k].y, kp[2], kp[3]);
            unpack_bf16x2(vr[k].z, kp[4], kp[5]); unpack_bf16x2(vr[k].w, kp[6], kp[7]);
            float part = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) part += qc[j] * c[k][j] + qp[j] * kp[j];   // qp = 0 off rope lanes
            s[k] = part;
        }
#pragma unroll
        for (int off = kWaveLanes / 2; off; off >>= 1) {   // 8 trees, interleaved
#pragma unroll
            for (int k = 0; k < kAttnBatch; ++k) s[k] += __shfl_xor(s[k], off, kWaveLanes);
        }
#pragma unroll
        for (int k = 0; k < kAttnBatch; ++k) {
            if (base + k < t_cached) update(c[k], s[k] * scale);   // wave-uniform
        }
    }

    // This token's row, from LDS, once, by wave 0 (the per-wave partials are
    // merged below, so which wave takes it does not matter).
    if (wave == 0 && t_new >= t_begin && t_new < t_end) {
        float c[8], kp[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            c[j] = row_new[8 * lane + j];
            kp[j] = rope_lane ? row_new[kMaxKvLora + 8 * lane + j] : 0.f;
        }
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) part += qc[j] * c[j] + qp[j] * kp[j];
        update(c, wave_sum(part) * scale);
    }

    // Publish the wave partials, then merge them in wave order.
    float* mine = part_lds + wave * kPartialStride;
    if (lane == 0) { mine[0] = m; mine[1] = l; }
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[2 + 8 * lane + j] = acc[j];
    __syncthreads();

    float gm = -INFINITY;
    for (int w = 0; w < kWaves; ++w) gm = fmaxf(gm, part_lds[w * kPartialStride]);
    float gl = 0.f, scale_w[kWaves];
    for (int w = 0; w < kWaves; ++w) {
        const float mw = part_lds[w * kPartialStride];
        scale_w[w] = (mw == -INFINITY) ? 0.f : __expf(mw - gm);   // empty wave
        gl += part_lds[w * kPartialStride + 1] * scale_w[w];
    }
    if (threadIdx.x == 0) { out[0] = gm; out[1] = gl; }
    for (int j = threadIdx.x; j < kMaxKvLora; j += blockDim.x) {
        float v = 0.f;
        for (int w = 0; w < kWaves; ++w) v += part_lds[w * kPartialStride + 2 + j] * scale_w[w];
        out[2 + j] = v;
    }
    __syncthreads();
}

// Merge task: combine the chunks' partials, then this task's rows of W_UV.
//   o_c[h] = (sum_chunks rescaled acc) / l
//   o[h]   = o_c[h] @ W_UV[h]^T,  W_UV[h] = kv_b rows [h*256+128, h*256+256)
// Runs on the same XCD as its head, so it waits on an XCD-local counter; the
// 128 W_UV rows are split over the head's merge tasks (row0, n_rows).
__device__ inline void merge_and_uv(
        const float* __restrict__ partials, const __hip_bfloat16* __restrict__ kv_b,
        float* __restrict__ o, int head, int n_chunks, int kv_lora,
        int qk_nope, int v_head, float* __restrict__ o_c_lds, int row0, int n_rows, uint64_t* etx_stamps = nullptr) {
    etx_ds_stamp(etx_stamps, 9); // merge M1 begin
    // Global max across chunks, then a rescaled sum. Fixed chunk order keeps
    // this bitwise reproducible. Every thread derives the same scalars. The
    // chunk loops are unrolled to the compile-time maximum and predicated,
    // so a thread's loads of the (up to) kMaxKvChunks partials are in flight
    // together instead of one round trip each (measured: 12.7 us per merge task).
    constexpr int kMaxChunks = kMaxKvChunks;
    float m_c[kMaxChunks], l_c[kMaxChunks];
    // etx lever l1_loads: lane c loads chunk c's (m, l) once, every lane takes them by shuffle (the same
    // values: lanes past n_chunks hold -inf / 0 exactly as the predicated loads did)
    static_assert(kMaxChunks <= kWaveLanes, "one lane per chunk");
    const int lane_c = threadIdx.x % kWaveLanes;
    const float m_lane = lane_c < n_chunks ? ld_l1(partials + lane_c * kPartialStride) : -INFINITY;
    const float l_lane = lane_c < n_chunks ? ld_l1(partials + lane_c * kPartialStride + 1) : 0.f;
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
        m_c[c] = __shfl(m_lane, c, kWaveLanes);
        l_c[c] = __shfl(l_lane, c, kWaveLanes);
    }
    float gm = -INFINITY;
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) gm = fmaxf(gm, m_c[c]);
    float scale_c[kMaxChunks], gl = 0.f;
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
        scale_c[c] = c < n_chunks ? __expf(m_c[c] - gm) : 0.f;
        gl += l_c[c] * scale_c[c];
    }
    const float inv_l = 1.f / gl;
    etx_ds_stamp(etx_stamps, 10); // M1 end / M2 begin

    // Partials in batches of 8 chunks: 16 live loads per thread put the
    // kernel into scratch (48 B/lane); two batches of 8 do not, and the
    // partials are in this XCD's L2 anyway.
    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int c0 = 0; c0 < kMaxChunks; c0 += 8) {
            float a[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                a[c] = c0 + c < n_chunks ? ld_l1(partials + (c0 + c) * kPartialStride + 2 + j) : 0.f;
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) v += a[c] * scale_c[c0 + c];
        }
        o_c_lds[j] = v * inv_l;
    }
    __syncthreads();

    etx_ds_stamp(etx_stamps, 11); // M2 end / M3 begin
    // W_UV[h] is [v_head, kv_lora] row-major: a plain row GEMV from LDS. The
    // attention output is a bf16 tensor in HF before o_proj, hence EPI_BF16.
    const int row_stride = qk_nope + v_head;
    const __hip_bfloat16* w_uv =
        kv_b + (int64_t)head * row_stride * kv_lora + (int64_t)(qk_nope + row0) * kv_lora;
    gemv_rows(w_uv, kv_lora, o_c_lds, o + head * v_head + row0, nullptr, EPI_BF16, 1.f,
              n_rows, kv_lora, /*xcd=*/0, /*n_xcds=*/1, /*worker=*/0, /*n_workers=*/1);
    etx_ds_stamp(etx_stamps, 12); // M3 end
}

}  // namespace fleet
