// Register-streamed GEMV — the task body behind every large weight read.
//
// docs/design.md §3 maps q_proj‖kv_a, o_proj, the experts and lm_head onto
// Chiplet-tasks: the N dimension is split across the 8 XCDs, then across the 37
// workers of each XCD, and accumulation is fp32. At batch 1 there is no L2
// reuse to win — m_tiles is 1 — so the gain is dispatch-count reduction and,
// above all, keeping enough loads in flight.
//
// Structure: one *wave* owns a row pair. Its 64 lanes stride the row in
// 16-byte pieces (8 bf16 each), issue kStreamDepth loads per row before
// consuming any, and reduce with a fixed-order shuffle tree. Nothing in the
// row loop touches LDS or a workgroup barrier, so the four waves of a
// workgroup stream four independent row pairs and the CU keeps
// 4 waves x 2 rows x kStreamDepth x 1 KB = 32 KB in flight — above the 18 KB
// Little's-law target in §12. (The previous version put all 256 threads on
// one row: with K = 2048 that is exactly one load per lane and a full block
// reduction per row, which never reaches streaming depth at all.)
//
// §12: weights go HBM -> VGPR directly with global_load_dwordx4; gfx942's LDS
// DMA moves only 4 B/lane, so staging through LDS would need 4x the load
// instructions for the same bytes.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

namespace fleet {

// Loads in flight per row per lane before the first s_waitcnt. Set from
// bench/microbench.hip (c) on the MI300X: depth 4 streamed 3.76 TB/s, depth 8
// 4.25 TB/s (80% of peak), depth 12 and 16 fell off again.
#ifndef FLEET_STREAM_DEPTH
#define FLEET_STREAM_DEPTH 8
#endif
constexpr int kStreamDepth = FLEET_STREAM_DEPTH;
constexpr int kWaveLanes = 64;
#ifndef FLEET_GEMV_WAVES
constexpr int kWaves = 256 / kWaveLanes;   // waves per workgroup (= fleet_runtime.h's)

// etx lever l1_loads (D2b): agent-scope relaxed atomic loads, `global_load ... sc1` (address_space(1), so not flat_):
// past this CU's L1, from the XCD's L2 (same-XCD producers drained into it; cross-XCD ones were flushed)
#define FLEET_L1_LOADS 1
#define FLEET_L1_G __attribute__((address_space(1)))
__device__ __forceinline__ float ld_l1(const float* p) {
    return __hip_atomic_load((const FLEET_L1_G float*)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint64_t ld_l164(const void* p) {
    return __hip_atomic_load((const FLEET_L1_G uint64_t*)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
// 16 B as two 8 B coherent loads (no 16 B atomic form), both in flight
__device__ __forceinline__ uint4 ld_l1_u4(const uint4* p) {
    const uint64_t lo = ld_l164(p), hi = ld_l164(reinterpret_cast<const uint64_t*>(p) + 1);
    uint4 r;
    r.x = (uint32_t)lo; r.y = (uint32_t)(lo >> 32); r.z = (uint32_t)hi; r.w = (uint32_t)(hi >> 32);
    return r;
}
__device__ __forceinline__ float4 ld_l1_f4(const float4* p) {
    const uint4 u = ld_l1_u4(reinterpret_cast<const uint4*>(p));
    float4 r;
    r.x = __uint_as_float(u.x); r.y = __uint_as_float(u.y); r.z = __uint_as_float(u.z); r.w = __uint_as_float(u.w);
    return r;
}
#endif

// bf16 pairs arrive as uint32; unpacking in registers costs no memory traffic.
// Weight streams are read exactly once per token, so they gain nothing from
// L2 residency and evict what does (KV cache, kv_b, router, norms). With
// FLEET_NT_WEIGHTS=1 every GEMV weight load is non-temporal (global_load ...
// nt: streaming allocation policy); the KV cache and kv_b keep normal loads.
// Build both binaries and measure — the MALL allocation policy under nt is
// not documented, which is exactly what the A/B decides.
#ifndef FLEET_NT_WEIGHTS
#define FLEET_NT_WEIGHTS 0
#endif
typedef unsigned int fleet_u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint4 load_weight(const uint4* __restrict__ p) {
#if FLEET_NT_WEIGHTS
    const fleet_u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const fleet_u32x4*>(p));
    uint4 r; r.x = v.x; r.y = v.y; r.z = v.z; r.w = v.w;
    return r;
#else
    return *p;
#endif
}

__device__ __forceinline__ void unpack_bf16x2(uint32_t p, float& lo, float& hi) {
    lo = __uint_as_float((p & 0xFFFFu) << 16);
    hi = __uint_as_float(p & 0xFFFF0000u);
}

__device__ __forceinline__ float bf16_round(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__device__ __forceinline__ float silu(float g) {
    return g / (1.f + __expf(-g));
}

// Fixed-order butterfly: the same tree every time, so the sum is bitwise
// reproducible (§4 forbids float atomics for exactly this reason).
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int off = kWaveLanes / 2; off; off >>= 1) v += __shfl_xor(v, off, kWaveLanes);
    return v;
}

__device__ __forceinline__ void fma8(const uint4& v, const float* __restrict__ x8,
                                     float& acc) {
    float a, b;
    unpack_bf16x2(v.x, a, b); acc += a * x8[0] + b * x8[1];
    unpack_bf16x2(v.y, a, b); acc += a * x8[2] + b * x8[3];
    unpack_bf16x2(v.z, a, b); acc += a * x8[4] + b * x8[5];
    unpack_bf16x2(v.w, a, b); acc += a * x8[6] + b * x8[7];
}

// R rows of a bf16 [*, K] matrix times an fp32 [K] vector, one wave, fp32
// accumulate, D 16-byte chunks per row in flight per lane; every lane returns
// the full sums. K is a multiple of 8 for every tensor in this model (smallest
// is 512). R x D is the number of loads in flight per lane, kept at 8 by the
// callers: for K = 2048 a row is 4 chunks per lane, so 2 rows x 4 chunks; for
// K = 512 it is 1 chunk, so 8 rows x 1.
//
// The final partial batch is issued in full before any of it is consumed
// (predicated, unrolled): the first version's plain tail loop loaded one
// chunk at a time, and with depth 8 and K = 2048 *every* row went through
// that tail, which is why raising the depth changed nothing on the MI300X.
template <int R, int D>
__device__ __forceinline__ void wave_dot(const __hip_bfloat16* const* rows,
                                         const float* __restrict__ x, int K,
                                         float* sums) {
    const int lane = threadIdx.x & (kWaveLanes - 1);
    const int n4 = K / 8;
    const uint4* r4[R];
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        r4[r] = reinterpret_cast<const uint4*>(rows[r]);
        acc[r] = 0.f;
    }

    // Double-buffered: batch i+1 is issued before batch i is consumed, so
    // consecutive batches overlap their memory latency instead of paying it
    // serially (measured on the MI300X: with one buffer the expert GEMVs
    // streamed at ~2.4 TB/s against a 4.25 TB/s ceiling).
    // Written as macros on purpose: passing the buffers to a lambda by array
    // reference made the compiler put them in scratch memory (measured:
    // 312 B/lane of scratch and every GEMV kind 1.5-3x slower), which defeats
    // the whole point. Direct array access in fully unrolled loops stays in
    // registers.
    uint4 bufA[R][D], bufB[R][D];
#define FLEET_ISSUE(BUF, I)                                                     \
    _Pragma("unroll")                                                           \
    for (int d = 0; d < D; ++d) {                                               \
        if ((I) + d * kWaveLanes < n4) {                                        \
            _Pragma("unroll")                                                   \
            for (int r = 0; r < R; ++r) BUF[r][d] = load_weight(r4[r] + (I) + d * kWaveLanes); \
        }                                                                       \
    }
#define FLEET_CONSUME(BUF, I)                                                   \
    _Pragma("unroll")                                                           \
    for (int d = 0; d < D; ++d) {                                               \
        if ((I) + d * kWaveLanes < n4) {                                        \
            const float* x8 = x + ((I) + d * kWaveLanes) * 8;                   \
            _Pragma("unroll")                                                   \
            for (int r = 0; r < R; ++r) fma8(BUF[r][d], x8, acc[r]);            \
        }                                                                       \
    }
    const int step = D * kWaveLanes;
    int i = lane;
    FLEET_ISSUE(bufA, i)
    while (true) {
        const int j = i + step;
        if (j < n4) { FLEET_ISSUE(bufB, j) }
        FLEET_CONSUME(bufA, i)
        if (j >= n4) break;
        i = j + step;
        if (i < n4) { FLEET_ISSUE(bufA, i) }
        FLEET_CONSUME(bufB, j)
        if (i >= n4) break;
    }
#undef FLEET_ISSUE
#undef FLEET_CONSUME
#pragma unroll
    for (int r = 0; r < R; ++r) sums[r] = wave_sum(acc[r]);
}

// Reduce one value across the workgroup in a fixed order. Used by the
// prologues (RMSNorm, kv-norm), never inside a GEMV row loop.
__device__ __forceinline__ float block_reduce_ordered(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s; s >>= 1) {
        if ((int)threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float out = smem[0];
    __syncthreads();   // everyone has read smem[0] before it is reused
    return out;
}

// Copy a vector into LDS so the row loop reads its operand from LDS instead
// of re-fetching it from L2 for every row. The stride is the compile-time
// block size and the loop is unrolled so the (up to 8) loads per thread are
// all in flight together instead of paying one round trip each.
constexpr int kBlockThreads = 256;

// 16-byte loads: 2048 floats are 2 float4 per thread, in flight together —
// one round trip under load instead of the 8 scalar loads' two. Every
// staged buffer here is a multiple of 4 floats at a 16-byte offset.
__device__ __forceinline__ void stage_vector(const float* __restrict__ src,
                                             float* __restrict__ dst, int n) {
    const int n4 = n >> 2;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    float4* d4 = reinterpret_cast<float4*>(dst);
#pragma unroll 4
    for (int i = threadIdx.x; i < n4; i += kBlockThreads) d4[i] = ld_l1_f4(s4 + i);   // etx lever l1_loads
    for (int i = (n4 << 2) + threadIdx.x; i < n; i += kBlockThreads) dst[i] = ld_l1(src + i);
    __syncthreads();
}

// ---------------------------------------------------------------- row ranges
//
// A Chiplet-task's rows [0, N) are split evenly over the XCDs, then over the
// workers of each XCD by interleaving (worker w owns n_begin + w, + w +
// n_workers, ...), and each wave of the worker takes every kWaves-th row
// pair. The split is a pure function of (xcd, worker, wave), so the host
// graph never has to carry row ranges.

struct RowSlice {
    int begin, end;   // this XCD's [begin, end) of N
};

__device__ __forceinline__ RowSlice xcd_rows(int N, int xcd, int n_xcds) {
    const int per = (N + n_xcds - 1) / n_xcds;
    RowSlice s;
    s.begin = xcd * per;
    s.end = min(N, s.begin + per);
    return s;
}

// bf16 boundaries (reference_decode.py with bf16=True): HuggingFace runs the
// model in bf16, so every linear layer's output is rounded to bf16 before it
// is used, and the residual add is a bf16 add. Reproducing those roundings is
// what keeps routing decisions and argmax ties on the same side as HF.
//   EPI_NONE      y = sum                       (router: HF's gate runs in fp32)
//   EPI_BF16      y = bf16(scale * sum)         (every other linear output)
//   EPI_RESIDUAL  y = bf16(residual + bf16(sum)) (o_proj, dense down)
// K-chunked GEMVs (expert down) keep fp32 partial sums across chunks:
//   EPI_ACC       y += sum                      (middle chunks; y is the accumulator)
//   EPI_BF16_ACC  y = bf16(scale * bf16(residual + sum))  (last chunk; residual = accumulator)
enum GemvEpilogue { EPI_NONE = 0, EPI_BF16 = 1, EPI_RESIDUAL = 2, EPI_ACC = 3, EPI_BF16_ACC = 4 };

// The epilogue store (and the residual load) are agent-scope atomics when
// the output crosses XCDs under coherent_acts (fleet_runtime.h).
__device__ __forceinline__ float epi_ld(const float* p, bool coh) {
    return coh ? __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : *p;
}
__device__ __forceinline__ void epi_st(float* p, float v, bool coh) {
    if (coh) __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); else *p = v;
}
// y16 != nullptr: store the bf16 result as bf16. EPI_BF16 rounds to bf16
// anyway, so this is bit-exact and halves what the consumer has to read.
#define FLEET_EPILOGUE(N, V)                                                    \
    switch (epi) {                                                              \
        case EPI_BF16:                                                          \
            if (y16 != nullptr) y16[N] = __float2bfloat16(bf16_round(scale * bf16_round(V))); \
            else epi_st(y + (N), bf16_round(scale * bf16_round(V)), coh);       \
            break;                                                              \
        case EPI_RESIDUAL: epi_st(y + (N), bf16_round(epi_ld(residual + (N), coh) + bf16_round(V)), coh); break; \
        case EPI_ACC:      y[N] += (V); break;                                  \
        case EPI_BF16_ACC:                                                      \
            if (y16 != nullptr) y16[N] = __float2bfloat16(bf16_round(scale * bf16_round(residual[N] + (V)))); \
            else epi_st(y + (N), bf16_round(scale * bf16_round(residual[N] + (V))), coh); \
            break;                                                              \
        default:           epi_st(y + (N), (V), coh);                           \
    }

// Software-pipelined row loop: the loads of the *next* group of RPI rows are
// issued before the current group is reduced, so consecutive groups overlap
// their memory latency. Preconditions: a row fits one buffer (K/8 <= D*64)
// — true for every K here except the dense down's 10944, which takes the
// unpipelined path — and every load is unconditional: out-of-range rows and
// chunks are clamped to a valid address and their contribution is masked at
// the FMA. Unconditional loads matter: a load inside an `if` splits the
// basic block and the waitcnt pass then waits for *everything* before the
// first use, which is exactly the serialisation this loop exists to remove.
template <int RPI, int D>
__device__ __forceinline__ void gemv_rows_pipelined(
        const __hip_bfloat16* __restrict__ w, int ld,
        const float* __restrict__ x, float* __restrict__ y,
        const float* __restrict__ residual, GemvEpilogue epi, float scale,
        int K, int first, int cnt, int n_workers, bool coh,
        __hip_bfloat16* __restrict__ y16) {
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;
    const int n4 = K / 8;
    const int stride = RPI * kWaves;            // groups are interleaved over waves
    int k = RPI * wave;
    if (k >= cnt) return;

    uint4 bufA[RPI][D], bufB[RPI][D];
    // chunk index per lane per d, clamped into the row; valid[d] masks the FMA
    int idx[D];
    bool valid[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
        valid[d] = lane + d * kWaveLanes < n4;
        idx[d] = valid[d] ? lane + d * kWaveLanes : 0;
    }
#define FLEET_ROW(KK, R) (w + (int64_t)(first + min((KK) + (R), cnt - 1) * n_workers) * ld)
#define FLEET_ISSUE_GROUP(BUF, KK)                                              \
    _Pragma("unroll")                                                           \
    for (int r = 0; r < RPI; ++r) {                                             \
        const uint4* r4 = reinterpret_cast<const uint4*>(FLEET_ROW(KK, r));     \
        _Pragma("unroll")                                                       \
        for (int d = 0; d < D; ++d) BUF[r][d] = load_weight(r4 + idx[d]);       \
    }
#define FLEET_REDUCE_GROUP(BUF, KK)                                             \
    {                                                                           \
        float sums[RPI];                                                        \
        _Pragma("unroll")                                                       \
        for (int r = 0; r < RPI; ++r) {                                         \
            float acc = 0.f;                                                    \
            _Pragma("unroll")                                                   \
            for (int d = 0; d < D; ++d) {                                       \
                if (valid[d]) fma8(BUF[r][d], x + idx[d] * 8, acc);             \
            }                                                                   \
            sums[r] = wave_sum(acc);                                            \
        }                                                                       \
        if (lane == 0) {                                                        \
            _Pragma("unroll")                                                   \
            for (int r = 0; r < RPI; ++r) {                                     \
                if ((KK) + r < cnt) {                                           \
                    const int n = first + ((KK) + r) * n_workers;               \
                    const float v = sums[r];                                    \
                    FLEET_EPILOGUE(n, v)                                        \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }
    FLEET_ISSUE_GROUP(bufA, k)
    // etx lever gemv_no_terminal_all: full pairs keep issue-before-reduce; the tail issues nothing past cnt.
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
#undef FLEET_ROW
#undef FLEET_ISSUE_GROUP
#undef FLEET_REDUCE_GROUP
}

// Short rows (K <= 256, 16-byte chunks <= 32: the K-split o_proj): a full
// wave per row would leave half its lanes idle on every load, so each
// half-wave takes its own row — 2 rows per load instruction, reduced within
// the half (xor offsets below 32 never cross it). Same pipelining as above.
__device__ __forceinline__ float half_wave_sum(float v) {
#pragma unroll
    for (int off = kWaveLanes / 4; off; off >>= 1) v += __shfl_xor(v, off, kWaveLanes);
    return v;
}

template <int RPI>   // rows per wave per group (even); each half-wave takes RPI/2
__device__ __forceinline__ void gemv_rows_pipelined_half(
        const __hip_bfloat16* __restrict__ w, int ld,
        const float* __restrict__ x, float* __restrict__ y,
        const float* __restrict__ residual, GemvEpilogue epi, float scale,
        int K, int first, int cnt, int n_workers, bool coh,
        __hip_bfloat16* __restrict__ y16) {
    constexpr int H = RPI / 2;
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;
    const int half = lane >> 5, hl = lane & 31;
    const int n4 = K / 8;
    const bool valid = hl < n4;
    const int idx = valid ? hl : 0;
    const int stride = RPI * kWaves;
    int k = RPI * wave;
    if (k >= cnt) return;
    uint4 bufA[H], bufB[H];
#define FLEET_HROW(KK, J) (w + (int64_t)(first + min((KK) + 2 * (J) + half, cnt - 1) * n_workers) * ld)
#define FLEET_HISSUE(BUF, KK)                                                   \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < H; ++j) {                                               \
        BUF[j] = load_weight(reinterpret_cast<const uint4*>(FLEET_HROW(KK, j)) + idx); \
    }
#define FLEET_HREDUCE(BUF, KK)                                                  \
    {                                                                           \
        float sums[H];                                                          \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < H; ++j) {                                           \
            float acc = 0.f;                                                    \
            if (valid) fma8(BUF[j], x + idx * 8, acc);                          \
            sums[j] = half_wave_sum(acc);                                       \
        }                                                                       \
        if (hl == 0) {                                                          \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < H; ++j) {                                       \
                const int r = (KK) + 2 * j + half;                              \
                if (r < cnt) {                                                  \
                    const int n = first + r * n_workers;                        \
                    const float v = sums[j];                                    \
                    FLEET_EPILOGUE(n, v)                                        \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }
    FLEET_HISSUE(bufA, k)
    // etx lever gemv_no_terminal_all: full pairs keep issue-before-reduce; the tail issues nothing past cnt.
    while (k + 2 * stride < cnt) {
        const int k2 = k + stride;
        FLEET_HISSUE(bufB, k2)
        FLEET_HREDUCE(bufA, k)
        const int k3 = k2 + stride;
        FLEET_HISSUE(bufA, k3)
        FLEET_HREDUCE(bufB, k2)
        k = k3;
    }
    if (k + stride < cnt) {
        const int k2 = k + stride;
        FLEET_HISSUE(bufB, k2)
        FLEET_HREDUCE(bufA, k)
        FLEET_HREDUCE(bufB, k2)
    } else {
        FLEET_HREDUCE(bufA, k)
    }
#undef FLEET_HROW
#undef FLEET_HISSUE
#undef FLEET_HREDUCE
}

// Unpipelined fallback for rows longer than one buffer (dense down, K = 10944).
template <int RPI, int D>
__device__ __forceinline__ void gemv_rows_impl(
        const __hip_bfloat16* __restrict__ w, int ld,
        const float* __restrict__ x, float* __restrict__ y,
        const float* __restrict__ residual, GemvEpilogue epi, float scale,
        int K, int first, int cnt, int n_workers, bool coh,
        __hip_bfloat16* __restrict__ y16) {
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;
    for (int k = RPI * wave; k < cnt; k += RPI * kWaves) {
        const int nr = min(RPI, cnt - k);
        const __hip_bfloat16* rows[RPI];
#pragma unroll
        for (int r = 0; r < RPI; ++r) {   // past the end: repeat the last row
            rows[r] = w + (int64_t)(first + (k + min(r, nr - 1)) * n_workers) * ld;
        }
        float sums[RPI];
        wave_dot<RPI, D>(rows, x, K, sums);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < RPI; ++r) {
                if (r < nr) {
                    const int n = first + (k + r) * n_workers;
                    const float v = sums[r];
                    FLEET_EPILOGUE(n, v)
                }
            }
        }
    }
}

__device__ inline void gemv_rows(
        const __hip_bfloat16* __restrict__ w, int ld,
        const float* __restrict__ x, float* __restrict__ y,
        const float* __restrict__ residual, GemvEpilogue epi, float scale,
        int N, int K, int xcd, int n_xcds, int worker, int n_workers, bool coh = false,
        __hip_bfloat16* __restrict__ y16 = nullptr) {
    const RowSlice s = xcd_rows(N, xcd, n_xcds);
    const int first = s.begin + worker;
    if (first >= s.end) return;
    const int cnt = (s.end - first + n_workers - 1) / n_workers;   // rows owned
    const int n4 = K / 8;   // 16-byte chunks per row; 64 lanes take 64 at a time
    // 8 chunks per lane per buffer, two buffers (wave_dot): 8 loads issued
    // while the previous 8 are consumed. 16 per buffer put the kernel at the
    // 256-VGPR cap and the buffers into scratch memory (measured: every GEMV
    // kind 1.5-3x slower), so the batch stays at 8 and the overlap does the
    // rest. The tasks own 7–350 rows each; they pay round trips, not bytes.
    if (n4 <= kWaveLanes / 2) {         // K <= 256:  half-wave rows (K-split o_proj)
        gemv_rows_pipelined_half<8>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers, coh, y16);
    } else if (n4 <= kWaveLanes) {      // K <= 512:  8 rows x 1 chunk  (merge W_UV)
        gemv_rows_pipelined<8, 1>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers, coh, y16);
    } else if (n4 <= 2 * kWaveLanes) {  // K <= 1024: 4 rows x 2 chunks
        gemv_rows_pipelined<4, 2>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers, coh, y16);
    } else if (n4 <= 4 * kWaveLanes) {  // K = 1408, 2048: 2 rows x 4 chunks
        gemv_rows_pipelined<2, 4>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers, coh, y16);
    } else {                            // K = 10944: rows do not fit a buffer
        gemv_rows_impl<2, 4>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers, coh, y16);
    }
}

// h[n] = bf16(bf16(SiLU(bf16(gate[n] . x))) * bf16(up[n] . x)) for this
// worker's n in [0, inter) — the roundings are HF's (gate_proj, act_fn and
// up_proj each produce a bf16 tensor, and their product is bf16).
// gate/up rows are interleaved offline (§12): row 2n is gate_n, row 2n+1 is
// up_n, so the pair is exactly what one wave streams together and the product
// is formed in registers with no shuffle or LDS round trip.
// Tile granularity (§12): with chunk_counters != nullptr, h is treated as
// n_chunks K-chunks of chunk_rows rows and every wave adds 1 to
// chunk_counters[c] (an XCD-local event counter) once its rows of chunk c
// are stored — after the wave's next row is past the chunk's end, or at the
// end of the task for whatever remains — so a chunk's consumer starts while
// the later chunks are still streaming. The wave drains its own stores first
// (gfx9 tracks loads and stores in issue order, so this also waits for the
// group of loads just issued: two such stalls per wave per task). Every wave
// signals every chunk exactly once, rows or no rows.
__device__ inline void gemv_gate_up_rows(
        const __hip_bfloat16* __restrict__ gate_up, const float* __restrict__ x,
        float* __restrict__ h, int inter, int K, int xcd, int n_xcds,
        int worker, int n_workers,
        uint32_t* __restrict__ chunk_counters = nullptr, int chunk_rows = 0,
        int n_chunks = 0) {
    const RowSlice s = xcd_rows(inter, xcd, n_xcds);
    const int first = s.begin + worker;
    const int cnt = first < s.end ? (s.end - first + n_workers - 1) / n_workers : 0;
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;
    int next_chunk = 0;
#define FLEET_CHUNKS_DONE(NEXT_J)                                                \
    if (chunk_counters != nullptr) {                                            \
        const int next_row = (NEXT_J) < cnt ? first + (NEXT_J) * n_workers : inter; \
        while (next_chunk < n_chunks &&                                         \
               next_row >= min(inter, (next_chunk + 1) * chunk_rows)) {         \
            if (lane == 0) {                                                    \
                __builtin_amdgcn_s_waitcnt(0);                                  \
                __hip_atomic_fetch_add(chunk_counters + next_chunk, 1u,         \
                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); \
            }                                                                   \
            ++next_chunk;                                                       \
        }                                                                       \
    }

    // One (gate, up) pair per wave iteration, software-pipelined across
    // pairs exactly like gemv_rows_pipelined (unconditional, clamped loads).
    const int n4 = K / 8;   // 256 for hidden = 2048: one buffer of 4 chunks
    constexpr int D = 4;
    int idx[D];
    bool valid[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
        valid[d] = lane + d * kWaveLanes < n4;
        idx[d] = valid[d] ? lane + d * kWaveLanes : 0;
    }
    int k = wave;
    if (k < cnt) {
        uint4 bufA[2][D], bufB[2][D];
#define FLEET_PAIR_ROW(KK, R) (gate_up + (int64_t)(2 * (first + min(KK, cnt - 1) * n_workers) + (R)) * K)
#define FLEET_ISSUE_PAIR(BUF, KK)                                               \
    _Pragma("unroll")                                                           \
    for (int r = 0; r < 2; ++r) {                                               \
        const uint4* r4 = reinterpret_cast<const uint4*>(FLEET_PAIR_ROW(KK, r)); \
        _Pragma("unroll")                                                       \
        for (int d = 0; d < D; ++d) BUF[r][d] = load_weight(r4 + idx[d]);       \
    }
#define FLEET_REDUCE_PAIR(BUF, KK)                                              \
    {                                                                           \
        float g = 0.f, u = 0.f;                                                 \
        _Pragma("unroll")                                                       \
        for (int d = 0; d < D; ++d) {                                           \
            if (valid[d]) { fma8(BUF[0][d], x + idx[d] * 8, g); fma8(BUF[1][d], x + idx[d] * 8, u); } \
        }                                                                       \
        g = wave_sum(g); u = wave_sum(u);                                       \
        if (lane == 0 && (KK) < cnt) {                                          \
            h[first + (KK) * n_workers] =                                       \
                bf16_round(bf16_round(silu(bf16_round(g))) * bf16_round(u));    \
        }                                                                       \
    }
        FLEET_ISSUE_PAIR(bufA, k)
        // etx lever gemv_no_terminal_all: full pairs keep issue-before-reduce; the tail issues nothing past cnt.
        while (k + 2 * kWaves < cnt) {
            const int k2 = k + kWaves;
            FLEET_ISSUE_PAIR(bufB, k2)
            FLEET_REDUCE_PAIR(bufA, k)
            FLEET_CHUNKS_DONE(k2)
            const int k3 = k2 + kWaves;
            FLEET_ISSUE_PAIR(bufA, k3)
            FLEET_REDUCE_PAIR(bufB, k2)
            FLEET_CHUNKS_DONE(k3)
            k = k3;
        }
        if (k + kWaves < cnt) {
            const int k2 = k + kWaves;
            FLEET_ISSUE_PAIR(bufB, k2)
            FLEET_REDUCE_PAIR(bufA, k)
            FLEET_CHUNKS_DONE(k2)
            FLEET_REDUCE_PAIR(bufB, k2)
            FLEET_CHUNKS_DONE(k2 + kWaves)
        } else {
            FLEET_REDUCE_PAIR(bufA, k)
            FLEET_CHUNKS_DONE(k + kWaves)
        }
#undef FLEET_PAIR_ROW
#undef FLEET_ISSUE_PAIR
#undef FLEET_REDUCE_PAIR
    }
    FLEET_CHUNKS_DONE(cnt)      // whatever is left, including waves with no rows
#undef FLEET_CHUNKS_DONE
}

}  // namespace fleet
