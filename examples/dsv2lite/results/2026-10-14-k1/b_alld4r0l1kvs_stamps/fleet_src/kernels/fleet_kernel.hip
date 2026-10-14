// The persistent kernel: one launch per token sequence, 304 co-resident
// workgroups.
//
// docs/design.md §3, §4. Every workgroup enters here, discovers which chiplet
// it landed on, and becomes either that XCD's scheduler (one CU) or one of its
// 37 workers. For each token of the launch, workers walk their pre-built
// queue: wait on the task's dependent event, run the task, signal completion.
// No host involvement, no kernel launches between operators or between
// tokens — that is the entire point.
//
// This file holds the control structure and the small tasks. The bandwidth-
// critical bodies (GEMV, attention) live in separate headers so they can be
// tuned and unit-tested without touching the scheduling logic.
//
// A Chiplet-task (§3) reaches the device as one descriptor per worker: the
// host emits 37 descriptors per XCD for it, each carrying the worker id, and
// the row split is a pure function of (xcd, worker, wave) in gemv.h.

#include "runtime/fleet_runtime.h"
#include "runtime/fleet_types.h"
#include "gemv.h"
// ETX measurement-only source boundaries: no drains or added barriers.
#if !defined(ETX_DSV2_STAMPS2) || !ETX_DSV2_STAMPS2
#error "stamped fleet copy requires -DETX_DSV2_STAMPS2"
#endif
static __device__ __forceinline__ void etx_ds_stamp(uint64_t* stamps, int index) {
    if (stamps && threadIdx.x == 0) {
        asm volatile("" ::: "memory");
        stamps[index] = __builtin_amdgcn_s_memrealtime();
        asm volatile("" ::: "memory");
    }
}
#include "attention.h"
#include "expert.h"
#include "local_assembly.h"

namespace fleet {

// Per-workgroup LDS. Every task body that needs scratch gets it here, never
// from a global buffer another workgroup might be using at the same time.
// The expert down reuses s.part as its fp32 accumulator over K-chunks.
static_assert(kWaves * kPartialStride >= kMaxHidden, "s.part must hold hidden floats");

struct __attribute__((aligned(16))) Scratch {
    float smem[kBlock];                          // block reductions
    float vec[kMaxHidden];                       // staged GEMV operand / folded x
    float row[kMaxKvLora + kMaxQkRope];          // this token's cache row
    float q_c[kMaxKvLora];                       // q_c, then the merge's o_c
    float q_pe[kMaxQkRope];
    float part[kWaves * kPartialStride];         // per-wave attention partials;
                                                 // expert down: fp32 accumulator
    float logits[kMaxRouted];
    int   ids[kMaxTopK];                         // this XCD's routing choice
    float ws[kMaxTopK];
    uint64_t stamp;                              // trace: end of the task's prologue
    uint64_t stamp_stage;                        // trace: end of the staging (before the norm)
    union { int ids_blk[kBlock]; uint64_t stamps2[13]; };                       // lm_head argmax candidate reduction
};

// Trace sub-phase: thread 0 stamps the end of a task's prologue (staging,
// norm, routing) so the report can split busy time into prologue and body.
#define FLEET_STAMP() do { if (tc.trace && threadIdx.x == 0) s.stamp = __builtin_amdgcn_s_memrealtime(); } while (0)
#define FLEET_STAMP_STAGE() do { if (tc.trace && threadIdx.x == 0) s.stamp_stage = __builtin_amdgcn_s_memrealtime(); } while (0)

// One token's loop variables, derived from the token index of the launch.
struct TokenCtx {
    int tok;            // 0 .. n_tokens-1 within this launch
    uint32_t epoch;     // rt.epoch0 + tok
    int pos;            // absolute position of this token
    int seq_len;        // rows valid after this step = pos + 1
    const float* cos;   // YaRN row for pos
    const float* sin;
    bool dump;          // copy layer outputs (first token of the launch)
    bool trace;         // per-task timeline for this token
};

// ---------------------------------------------------------------- small tasks

__device__ void task_embed(const ModelDims& d, const Weights& w,
                           Activations& act, const TokenCtx& tc, bool coh) {
    if (threadIdx.x == 0 && act.token_time) {
        act.token_time[2 * tc.tok] = __builtin_amdgcn_s_memrealtime();
    }
    const int token = ld_coh(act.token_in + tc.tok, coh);
    const __hip_bfloat16* row = w.embed + (int64_t)token * d.hidden;
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        st_coh(act.x[0] + i, __bfloat162float(row[i]), coh);
    }
}

// RMSNorm of an LDS vector in place. HF's DeepseekV2RMSNorm in bf16: variance
// and scaling in fp32, the scaled value cast to bf16, then multiplied by the
// bf16 weight — two roundings (reference_decode.py rms_norm with bf16=True).
__device__ inline void rms_norm_lds(const ModelDims& d,
                                    const __hip_bfloat16* __restrict__ gamma,
                                    float* __restrict__ v, float* smem, uint64_t* etx_stamps = nullptr) {
    etx_ds_stamp(etx_stamps, 0); // norm begin
    float local = 0.f;
#pragma unroll 8
    for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) local += v[i] * v[i];
    etx_ds_stamp(etx_stamps, 1); // tree begin
    const float ssq = block_reduce_ordered(local, smem);
    etx_ds_stamp(etx_stamps, 2); // tree end
    const float inv = rsqrtf(ssq / d.hidden + d.rms_eps);
    etx_ds_stamp(etx_stamps, 3); // gamma/apply begin
#pragma unroll 8
    for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {
        v[i] = bf16_round(bf16_round(v[i] * inv) * __bfloat162float(gamma[i]));
    }
    etx_ds_stamp(etx_stamps, 4); // gamma/apply end
    __syncthreads();
    etx_ds_stamp(etx_stamps, 5); // norm end
}

// Same expression DAG as block_reduce_ordered at blockDim.x == 256:
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

__device__ inline void rms_norm_lds_tree_early(const ModelDims& d,
                                    const __hip_bfloat16* __restrict__ gamma,
                                    float* __restrict__ v, float* smem, uint64_t* etx_stamps = nullptr) {
    etx_ds_stamp(etx_stamps, 0); // norm begin
    // Keep the hidden dimension dynamic for the original square-sum and
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
    float local = 0.f;
#pragma unroll 8
    for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) local += v[i] * v[i];
    etx_ds_stamp(etx_stamps, 1); // tree begin
    const float ssq = block_reduce_ordered_etx_tree(local, smem);
    etx_ds_stamp(etx_stamps, 2); // tree end
    const float inv = rsqrtf(ssq / d.hidden + d.rms_eps);
    etx_ds_stamp(etx_stamps, 3); // gamma/apply begin
#pragma unroll 8
    for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {
        v[i] = bf16_round(bf16_round(v[i] * inv) * gamma_early[(i - (int)threadIdx.x) / kBlockThreads]);
    }
    etx_ds_stamp(etx_stamps, 4); // gamma/apply end
    __syncthreads();
    etx_ds_stamp(etx_stamps, 5); // norm end
}

// The residual stream into LDS: vec = src, or the fold vec = bf16(src +
// bf16(sum of the 8 partials)) — HF's bf16 linear output and bf16 residual
// add. Every consumer does this for itself (8 KB + 64 KB from L2, in
// parallel with 295 others) and worker (0,0) publishes vec to dst, which is
// never the buffer being read. The partials came from other XCDs: the
// global wait that released this task acquired them.
// o_proj's partials are fp32 partial sums: they must be added before the
// single bf16 rounding, so unlike the expert partials they cannot be stored
// narrower without changing the arithmetic.
__device__ inline void stage_residual_f32(const ModelDims& d, const float* __restrict__ src,
                                          const float* __restrict__ partials,
                                          float* __restrict__ dst, float* __restrict__ vec,
                                          int xcd, int worker, bool coh) {
    if (false) {   // etx lever l1_loads: the vector path below loads coherently
#pragma unroll 4
        for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {
            float acc = 0.f;
#pragma unroll
            for (int x = 0; x < kNumXCDs; ++x) acc += ld_coh(partials + x * d.hidden + i, true);
            vec[i] = bf16_round(ld_coh(src + i, true) + bf16_round(acc));
        }
    } else {
        const int n4 = d.hidden >> 2;
        const float4* s4 = reinterpret_cast<const float4*>(src);
        const float4* p4 = reinterpret_cast<const float4*>(partials);
        float4* v4 = reinterpret_cast<float4*>(vec);
        for (int i = threadIdx.x; i < n4; i += kBlockThreads) {
            float4 acc = ld_l1_f4(p4 + i);
#pragma unroll
            for (int x = 1; x < kNumXCDs; ++x) {
                const float4 q = ld_l1_f4(p4 + x * n4 + i);
                acc.x += q.x; acc.y += q.y; acc.z += q.z; acc.w += q.w;
            }
            const float4 xs = ld_l1_f4(s4 + i);
            float4 r;
            r.x = bf16_round(xs.x + bf16_round(acc.x));
            r.y = bf16_round(xs.y + bf16_round(acc.y));
            r.z = bf16_round(xs.z + bf16_round(acc.z));
            r.w = bf16_round(xs.w + bf16_round(acc.w));
            v4[i] = r;
        }
    }
    __syncthreads();
    if (dst != nullptr && xcd == 0 && worker == 0) {
        for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) st_coh(dst + i, vec[i], coh);
    }
}

__device__ inline void stage_residual(const ModelDims& d, const float* __restrict__ src,
                                      const __hip_bfloat16* __restrict__ partials,
                                      float* __restrict__ dst, float* __restrict__ vec,
                                      int xcd, int worker, bool coh) {
    if (false) {   // etx lever l1_loads: the vector path below loads coherently (fleet's read bf16 plain)
        if (partials == nullptr) {
#pragma unroll 8
            for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) vec[i] = ld_coh(src + i, true);
        } else {
#pragma unroll 4
            for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {
                float acc = 0.f;
#pragma unroll
                for (int x = 0; x < kNumXCDs; ++x) {
                    acc += __bfloat162float(partials[x * d.hidden + i]);
                }
                vec[i] = bf16_round(ld_coh(src + i, true) + bf16_round(acc));
            }
        }
        __syncthreads();
    } else if (partials == nullptr) {
        stage_vector(src, vec, d.hidden);
    } else {
        // The partials are bf16, so one 16-byte load covers eight of this
        // thread's hidden indices per XCD: 8 loads for the 8 partials plus 2
        // for x, all in flight, and half the bytes the fp32 version moved.
        const int n8 = d.hidden >> 3;
        const uint4* p4 = reinterpret_cast<const uint4*>(partials);
        const float4* s4 = reinterpret_cast<const float4*>(src);
        for (int i = threadIdx.x; i < n8; i += kBlockThreads) {
            uint4 v[kNumXCDs];
#pragma unroll
            for (int x = 0; x < kNumXCDs; ++x) v[x] = ld_l1_u4(p4 + x * n8 + i);
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int x = 0; x < kNumXCDs; ++x) {
                float a, b;
                unpack_bf16x2(v[x].x, a, b); acc[0] += a; acc[1] += b;
                unpack_bf16x2(v[x].y, a, b); acc[2] += a; acc[3] += b;
                unpack_bf16x2(v[x].z, a, b); acc[4] += a; acc[5] += b;
                unpack_bf16x2(v[x].w, a, b); acc[6] += a; acc[7] += b;
            }
            const float4 lo = ld_l1_f4(s4 + 2 * i), hi = ld_l1_f4(s4 + 2 * i + 1);
            const float xs[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                vec[8 * i + k] = bf16_round(xs[k] + bf16_round(acc[k]));
            }
        }
        __syncthreads();
    }
    if (dst != nullptr && xcd == 0 && worker == 0) {
        for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) st_coh(dst + i, vec[i], coh);
    }
}

#if FLEET_VERIFY_LOCAL_ASSEMBLY
__device__ inline bool same_payload(float a, float b) {
    union { float f; uint32_t u; } x, y;
    x.f = a; y.f = b;
    return x.u == y.u;
}

// Diagnostic-only consumer proof, under the real decode's traffic. Read the
// assembled payload FIRST through the ordinary local path, then compare to
// agent-scope loads of the original, globally published operands. Do not add
// a global invalidate that could conceal a missing local publication/acquire.
// These builds are correctness probes and must never supply timing verdicts.
__device__ inline void verify_local_fold(const ModelDims& d, const Activations& act,
                                         const RuntimeState& rt, const float* vec,
                                         bool oproj, int task_index) {
    for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {
        float acc;
        if (oproj) {
            acc = ld_coh(act.oproj_partial + i, true);
            for (int x = 1; x < kNumXCDs; ++x)
                acc += ld_coh(act.oproj_partial + x * d.hidden + i, true);
        } else {
            acc = 0.f;
            for (int x = 0; x < kNumXCDs; ++x) {
                const uint32_t bits = __hip_atomic_load(
                    reinterpret_cast<const uint32_t*>(act.expert_out + x * d.hidden) + i / 2,
                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                float lo, hi;
                unpack_bf16x2(bits, lo, hi);
                acc += (i & 1) ? hi : lo;
            }
        }
        const float expected = bf16_round(ld_coh(act.x[oproj ? 0 : 1] + i, true)
                                           + bf16_round(acc));
        if (!same_payload(vec[i], expected))
            raise_abort(rt, kAbortLocalPayload | (uint32_t)task_index);
    }
    __syncthreads();
}
#endif

// Layer-boundary capture (§6): vec holds the output of layer `layer`.
__device__ inline void dump_layer(const ModelDims& d, const Activations& act, int layer,
                                  const float* __restrict__ vec, int xcd, int worker,
                                  const TokenCtx& tc) {
    if (tc.dump && xcd == 0 && worker == 0 && layer >= 0) {
        float* out = act.layer_dump + (int64_t)layer * d.hidden;
        for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) out[i] = vec[i];
    }
}

// Cross-task prefetch (design.md §12): the weights a task streams do not
// depend on the event it waits for (except a routed expert's), so before
// the wait every thread issues 8 x 16 B loads of the task's first rows —
// 32 KB per workgroup, this worker's own first rows in the row partition —
// keeps them in registers across the wait and consumes them after it. The
// task's first loads then hit L2 (or the Infinity Cache if the wait was
// long). Rows are described as (base, bytes per row, stride between this
// worker's rows, row count); chunk c of the 2048 touches row c / per_row.
struct PrefetchPlan { const char* base; int64_t row_bytes, stride, rows; };

__device__ inline PrefetchPlan prefetch_plan(const TaskDescriptor& t, const ModelDims& d,
                                            const Weights& w, int xcd, int worker) {
    PrefetchPlan p;
    p.base = reinterpret_cast<const char*>(w.embed);   // harmless default (0 rows)
    p.row_bytes = 16; p.stride = 0; p.rows = 0;
    const int L = t.layer;
    const int nw = kWorkersPerXCD;
    const int64_t hb = (int64_t)d.hidden * 2;          // bytes of a K = hidden row
    switch (t.kind) {
        case TASK_QKV_FUSED: {
            const int q0 = xcd * (d.heads / kNumXCDs) * d.q_head_dim;
            p.base = reinterpret_cast<const char*>(w.layer_qkv[L]) + (int64_t)(q0 + worker) * hb;
            p.row_bytes = hb; p.stride = (int64_t)nw * hb; p.rows = 8;
            break;
        }
        case TASK_ATTENTION: {   // W_UK of the head: 32 of its 128 rows
            const int64_t rs = (int64_t)(d.qk_nope + d.v_head) * d.kv_lora * 2;
            p.base = reinterpret_cast<const char*>(w.layer_kv_b[L]) + (int64_t)t.head * rs;
            p.row_bytes = (int64_t)d.kv_lora * 2; p.stride = p.row_bytes; p.rows = 32;
            break;
        }
        case TASK_MERGE_UV: {    // this task's W_UV rows
            const int rows = d.v_head / max(1, (int)t.n_split);
            const int64_t rs = (int64_t)(d.qk_nope + d.v_head) * d.kv_lora * 2;
            p.base = reinterpret_cast<const char*>(w.layer_kv_b[L]) + (int64_t)t.head * rs
                     + (int64_t)(d.qk_nope + t.kv_chunk * rows) * d.kv_lora * 2;
            p.row_bytes = (int64_t)d.kv_lora * 2; p.stride = p.row_bytes; p.rows = rows;
            break;
        }
        case TASK_O_PROJ: {
            const int64_t hvb = (int64_t)d.heads * d.v_head * 2;
            if (t.flags & FLAG_OPROJ_ROW_SPLIT) {   // 4 KB of each of ~7 whole rows
                const RowSlice sl = xcd_rows(d.hidden, xcd, kNumXCDs);
                p.base = reinterpret_cast<const char*>(w.layer_o_proj[L])
                         + (int64_t)(sl.begin + worker) * hvb;
                p.row_bytes = hvb; p.stride = (int64_t)nw * hvb; p.rows = 7;
            } else {                                // K-split: 512 B of 64 rows
                const int kc = d.heads * d.v_head / kNumXCDs;
                p.base = reinterpret_cast<const char*>(w.layer_o_proj[L])
                         + (int64_t)worker * hvb + (int64_t)xcd * kc * 2;
                p.row_bytes = (int64_t)kc * 2; p.stride = (int64_t)nw * hvb; p.rows = 64;
            }
            break;
        }
        case TASK_NORM_ROUTER: {
            if (L >= d.first_k_dense) {
                p.base = reinterpret_cast<const char*>(w.layer_router[L]) + (int64_t)worker * hb;
                p.row_bytes = hb; p.stride = (int64_t)nw * hb; p.rows = 2;
            }
            break;
        }
        case TASK_EXPERT_GATE_UP:
        case TASK_EXPERT_DOWN: {  // only the shared halves have a static address
            if (t.expert_slot < 0) {
                const int half = -t.expert_slot - 1;
                const int shared_inter = 2 * d.moe_inter, rows = d.moe_inter;
                const char* sh = reinterpret_cast<const char*>(w.layer_shared[L]);
                if (t.kind == TASK_EXPERT_GATE_UP) {
                    p.base = sh + (int64_t)half * 2 * rows * hb + (int64_t)2 * worker * hb;
                    p.row_bytes = 2 * hb; p.stride = (int64_t)2 * nw * hb; p.rows = 4;
                } else {
                    p.base = sh + (int64_t)2 * shared_inter * hb + (int64_t)half * rows * 2
                             + (int64_t)worker * shared_inter * 2;
                    p.row_bytes = (int64_t)rows * 2; p.stride = (int64_t)nw * shared_inter * 2; p.rows = 12;
                }
            }
            break;
        }
        case TASK_DENSE_GATE_UP: {
            const RowSlice sl = xcd_rows(d.dense_inter, xcd, kNumXCDs);
            p.base = reinterpret_cast<const char*>(w.layer_dense[L]) + (int64_t)2 * (sl.begin + worker) * hb;
            p.row_bytes = 2 * hb; p.stride = (int64_t)2 * nw * hb; p.rows = 4;
            break;
        }
        case TASK_DENSE_DOWN: {
            const RowSlice sl = xcd_rows(d.hidden, xcd, kNumXCDs);
            p.base = reinterpret_cast<const char*>(w.layer_dense[L]) + (int64_t)2 * d.dense_inter * hb
                     + (int64_t)(sl.begin + worker) * d.dense_inter * 2;
            p.row_bytes = 4096; p.stride = (int64_t)nw * d.dense_inter * 2; p.rows = 8;
            break;
        }
        case TASK_LM_HEAD: {
            const RowSlice sl = xcd_rows(d.vocab, xcd, kNumXCDs);
            p.base = reinterpret_cast<const char*>(w.lm_head) + (int64_t)(sl.begin + worker) * hb;
            p.row_bytes = hb; p.stride = (int64_t)nw * hb; p.rows = 8;
            break;
        }
        default: break;
    }
    return p;
}

// Issue this thread's 8 touch loads for the plan (unconditional, clamped).
__device__ __forceinline__ void prefetch_issue(const PrefetchPlan& p, uint4* pf) {
    const int64_t per_row = max((int64_t)1, p.row_bytes / 16);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int64_t c = (int64_t)u * kBlockThreads + threadIdx.x;
        int64_t row = c / per_row, col = c - row * per_row;
        if (row >= p.rows) { row = 0; col = 0; }   // clamped: touches the first chunk again
        pf[u] = *reinterpret_cast<const uint4*>(p.base + row * p.stride + col * 16);
    }
}

// Touch [p, p + bytes) with 16-byte loads, 8 in flight per thread, results
// folded into a value the asm sink keeps alive (nothing else does).
__device__ inline void prefetch_bytes(const void* p, int64_t bytes, bool nt) {
    const fleet_u32x4* v = reinterpret_cast<const fleet_u32x4*>(p);
    const int64_t n = bytes / 16;
    uint32_t acc = 0;
    int64_t i = threadIdx.x;
    for (; i + 7 * kBlockThreads < n; i += 8 * kBlockThreads) {
        fleet_u32x4 b[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            b[u] = nt ? __builtin_nontemporal_load(v + i + u * kBlockThreads)
                      : v[i + u * kBlockThreads];
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) acc ^= b[u].x;
    }
    for (; i < n; i += kBlockThreads) acc ^= v[i].x;
    asm volatile("" :: "v"(acc));
}

// RMSNorm over the full hidden vector, then this worker's rows of the router
// — a Chiplet-task per XCD, so the experts of this XCD wait on an XCD-local
// event and one CU is not asked to stream 256 KB alone. Every worker norms
// the same x in the same order; worker 0 publishes it for the experts.
// defined below, used by the publication step
template <typename I>
__device__ inline void select_experts(const ModelDims& d, const float* __restrict__ logits,
                                      I* __restrict__ ids, float* __restrict__ ws);

__device__ void task_norm_router(const ModelDims& d, const Weights& w,
                                 Activations& act, Scratch& s, int layer,
                                 int xcd, int worker, bool moe, const TokenCtx& tc, bool coh,
                                 const TaskDescriptor& t, const RuntimeState& rt) {
    // x[1] = bf16(x[0] + bf16(o_proj)). K-split leaves 8 fp32 partials for
    // this prologue to fold; row-split already wrote x[1] through o_proj's own
    // EPI_RESIDUAL epilogue, so there is nothing to fold and we only stage.
#if FLEET_LOCAL_FOLD
    if (t.flags & kLocalFoldFlag) {
        stage_vector(act.x_norm + (int64_t)(kNumXCDs + xcd) * d.hidden, s.vec, d.hidden);
#if FLEET_VERIFY_LOCAL_ASSEMBLY
        verify_local_fold(d, act, rt, s.vec, true, t.index);
#endif
    } else
#endif
    if (t.flags & FLAG_OPROJ_ROW_SPLIT) {
        if (coh) {
            for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads)
                s.vec[i] = ld_coh(act.x[1] + i, true);
            __syncthreads();
        } else {
            stage_vector(act.x[1], s.vec, d.hidden);
        }
    } else {
        stage_residual_f32(d, act.x[0], act.oproj_partial, act.x[1], s.vec, xcd, worker, coh);
    }
    FLEET_STAMP_STAGE();
    rms_norm_lds_tree_early(d, w.layer_norm_post[layer], s.vec, s.smem, tc.trace ? s.stamps2 : nullptr);
    FLEET_STAMP();
    if (worker == 0) {
        float* x_norm = act.x_norm + (int64_t)xcd * d.hidden;
        for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) x_norm[i] = s.vec[i];
    }
    if (!moe) return;
    // Router logits: fp32 throughout, matching HF's fp32 gate.
    gemv_rows(w.layer_router[layer], d.hidden, s.vec,
              act.router_logits + (int64_t)xcd * d.n_routed, nullptr, EPI_NONE,
              1.f, d.n_routed, d.hidden, 0, 1, worker, kWorkersPerXCD);

    // The softmax + 6-pass top-k is 36 dependent shuffles on one wave while
    // the other three wait; every expert task used to repeat it (measured
    // ~5 us of the gate_up prologue). Here the last of the XCD's 37 router
    // workers — the only one that can see all 64 logits — does it once and
    // publishes 8 ids + 8 weights. It signals its own event afterwards, so
    // a consumer that sees the event sees the publication.
    if (t.flags & FLAG_TOPK_PUBLISH) {
        __shared__ int last_router;
        if (threadIdx.x == 0) {
            last_router = arrive_last(rt, tc.epoch, t.local_event, xcd, kWorkersPerXCD) ? 1 : 0;
        }
        __syncthreads();
        if (last_router) {
            if (threadIdx.x == 0) fence_acquire_local();
            __syncthreads();
            select_experts(d, act.router_logits + (int64_t)xcd * d.n_routed,
                           act.topk_ids + xcd * kMaxTopK, act.topk_w + xcd * kMaxTopK);
        }
    }
}

// Softmax over the router logits and top-k, on wave 0 with one expert per
// lane (max, sum and each argmax pick are shuffle reductions), into LDS.
// Run by every expert task of the XCD: 64 floats, deterministic, and it
// saves a task and an event. Ties go to the lower index (torch.topk order).
__device__ inline float load_logit(const ModelDims& d, const float* __restrict__ logits) {
    const int lane = threadIdx.x;
    return (lane < kWaveLanes && lane < d.n_routed) ? ld_l1(logits + lane) : -INFINITY;   // etx lever l1_loads
}

// `v`: this lane's logit (wave 0), from load_logit.
template <typename I>
__device__ inline void select_experts_from_ref(const ModelDims& d, float v,
                                           I* __restrict__ ids, float* __restrict__ ws) {
    if (threadIdx.x < kWaveLanes) {
        const int lane = threadIdx.x;
        const bool valid = lane < d.n_routed;
        float mx = v;
#pragma unroll
        for (int off = kWaveLanes / 2; off; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, kWaveLanes));
        float p = valid ? __expf(v - mx) : 0.f;
        const float inv = 1.f / wave_sum(p);
        p *= inv;
        for (int k = 0; k < d.top_k; ++k) {
            float bv = p;
            int bi = valid ? lane : 0x7fffffff;
#pragma unroll
            for (int off = kWaveLanes / 2; off; off >>= 1) {
                const float ov = __shfl_xor(bv, off, kWaveLanes);
                const int oi = __shfl_xor(bi, off, kWaveLanes);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { ids[k] = bi; ws[k] = bv; }   // unnormalised
            if (lane == bi) p = -INFINITY;                // taken
        }
    }
    __syncthreads();
}

template <typename I>
__device__ inline void select_experts_from(const ModelDims& d, float v,
                                           I* __restrict__ ids, float* __restrict__ ws) {
    if (d.n_routed != kWaveLanes) {               // etx lever topk_rank: only the 64-expert shape, else fleet's code
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


template <typename I>
__device__ inline void select_experts(const ModelDims& d, const float* __restrict__ logits,
                                      I* __restrict__ ids, float* __restrict__ ws) {
    select_experts_from(d, load_logit(d, logits), ids, ws);
}

// Greedy pick over the bf16 logits (lm_head writes them bf16-rounded, as HF's
// lm_head does); ties go to the lower index, as torch.argmax does. Feeds the
// next token unless the host pre-filled token_in (teacher forcing).
// lm_head epilogue: this worker's best logit (max, lowest index on ties —
// torch.argmax's rule) over the rows it just wrote, from its own L2, so the
// argmax task reduces 296 candidates instead of scanning 102400 logits on
// one CU (measured 109 us for the scan: a dependent load per compare).
// etx lever argmax_tree: the 256-wide pick tree's exact pairings, levels 128 and 64 from LDS, 32..1 by shuffle.
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

__device__ void lm_head_candidate(const ModelDims& d, Activations& act, int xcd,
                                  int worker, int n_workers, float* smem_val, int* smem_idx,
                                  bool coh) {
    const RowSlice sl = xcd_rows(d.vocab, xcd, kNumXCDs);
    const int first = sl.begin + worker;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    // rows first, first + n_workers, ...: independent loads, 8 per batch
    for (int j0 = threadIdx.x; ; j0 += 8 * kBlockThreads) {
        int n[8]; float v[8];
        bool any = false;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            n[u] = first + (j0 + u * kBlockThreads) * n_workers;
            const bool ok = n[u] < sl.end;
            any |= ok;
            v[u] = ok ? act.logits[n[u]] : -INFINITY;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (v[u] > bv || (v[u] == bv && n[u] < bi)) { bv = v[u]; bi = n[u]; }
        }
        if (!any) break;
    }
    smem_val[threadIdx.x] = bv;
    smem_idx[threadIdx.x] = bi;
    __syncthreads();
    if (threadIdx.x < 64) {
        float v; int i;
        etx_argmax_tree(smem_val, smem_idx, v, i);
        if (threadIdx.x == 0) {
            st_coh(act.argmax_val + xcd * n_workers + worker, v, coh);
            st_coh(act.argmax_idx + xcd * n_workers + worker, i, coh);
        }
    }
    __syncthreads();   // wave 0 has read every slot before the next task may reuse the LDS
}

__device__ void task_argmax(const ModelDims& d, Activations& act, const RuntimeState& rt,
                            const TokenCtx& tc) {
    __shared__ float best_val[kBlock];
    __shared__ int best_idx[kBlock];
    const int n_cand = kNumXCDs * kWorkersPerXCD;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    const bool coh = true;   // etx lever l1_loads
    for (int i = threadIdx.x; i < n_cand; i += blockDim.x) {
        const float v = ld_coh(act.argmax_val + i, coh);
        const int ix = ld_coh(act.argmax_idx + i, coh);
        if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
    }
    best_val[threadIdx.x] = bv;
    best_idx[threadIdx.x] = bi;
    __syncthreads();
    int best = 0;
    if (threadIdx.x < 64) {
        float v;
        etx_argmax_tree(best_val, best_idx, v, best);
    }
    __syncthreads();   // wave 0 has read every slot before the next task may reuse the LDS
    if (threadIdx.x == 0) {
        act.token_out[tc.tok] = best;
        if (!rt.teacher) st_coh(act.token_in + tc.tok + 1, best, coh);
        if (act.token_time) act.token_time[2 * tc.tok + 1] = __builtin_amdgcn_s_memrealtime();
    }
}

__device__ bool run_task(const TaskDescriptor& t, const RuntimeState& rt,
                         const ModelDims& d, const Weights& w, Activations& act,
                         KVCache& kv, Scratch& s, const TokenCtx& tc, int xcd,
                         int worker) {
    const int L = t.layer;
    // WORKER_GROUP: this task's rows are split over a sub-range of the XCD's
    // workers (expert gate_up / down on disjoint groups, so they overlap)
    const int nw = (t.flags & FLAG_WORKER_GROUP) ? (int)t.n_split : kWorkersPerXCD;
    worker = (t.flags & FLAG_WORKER_GROUP) ? worker - (int)t.head : worker;
    const int chunks = max(1, (int)t.n_split);
    const bool coh = true;   // etx lever l1_loads: fleet's coherent activation paths, always
    const int heads_per_xcd = d.heads / kNumXCDs;

    switch (t.kind) {
        case TASK_EMBED:       task_embed(d, w, act, tc, coh); break;
        case TASK_NORM_ROUTER: task_norm_router(d, w, act, s, L, xcd, worker,
                                                L >= d.first_k_dense, tc, coh, t, rt); break;
        case TASK_ARGMAX:      task_argmax(d, act, rt, tc); break;
#if FLEET_LOCAL_FOLD || FLEET_LOCAL_KVA
        case TASK_REDUCE: {
#if FLEET_LOCAL_FOLD
            if (t.flags & kLocalFoldFlag) {
                // Each consumer XCD acquires the ORIGINAL global producer
                // event before assembling. Reuse the exact ordered fold and
                // its bf16 roundings; only XCD 0 publishes the canonical x.
                // The local copy is separate from x_norm, x and all partials.
                if (t.head == kFoldOproj) {
                    stage_residual_f32(d, act.x[0], act.oproj_partial, act.x[1],
                                       s.vec, xcd, 0, coh);
                } else {
                    stage_residual(d, act.x[1], act.expert_out,
                                   L < 0 ? nullptr : act.x[0], s.vec, xcd, 0, coh);
                }
                float* local = act.x_norm + (int64_t)(kNumXCDs + xcd) * d.hidden;
                for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) local[i] = s.vec[i];
            }
#endif
#if FLEET_LOCAL_KVA
            if (t.flags & kLocalKvaFlag) {
                // Projection still writes ONE 576-row buffer (72 rows/XCD).
                // This workgroup, physically resident on the consumer XCD,
                // gathers ALL shards after the global acquire and republishes
                // into its own nonaliasing slot. Even XCD 0 has a separate copy.
                const int rows = d.kv_lora + d.qk_rope;
                float* local = act.kv_a + (int64_t)(xcd + 1) * rows;
                for (int i = threadIdx.x; i < rows; i += kBlockThreads) local[i] = act.kv_a[i];
            }
#endif
            // The existing task epilogue joins all waves, drains stores and
            // signals the XCD-local event; consumers take its L1 acquire.
            break;
        }
#else
        case TASK_REDUCE:      break;   // retired: consumers fold in their prologue
#endif

        // Publish one row slice of this head's absorbed q_c. Runs on a worker
        // that would otherwise idle through the attention phase (32 of the
        // XCD's 37 workers take an attention task), so the 131 KB of W_UK is
        // streamed once per head rather than once per KV chunk.
        case TASK_Q_ABSORB: {
            const int rows = d.qk_nope / kQAbsSlices;
            q_absorb_rows(act.q + (int64_t)t.head * d.q_head_dim, w.layer_kv_b[L],
                          act.q_absorbed + ((int64_t)t.head * kQAbsSlices + t.kv_chunk) * d.kv_lora,
                          s.part, t.head, d.qk_nope, d.v_head, d.kv_lora,
                          t.kv_chunk * rows, rows);
            break;
        }

        // Idle-worker prefetch (§12): stream a slice of the weights the
        // routing does not decide — this layer's shared expert and the next
        // layer's q/kv_a, o_proj and kv_b — during attention, so they sit in
        // the Infinity Cache when their phase comes. Part [kv_chunk, +head)
        // of n_split equal slices of the concatenated byte ranges.
        case TASK_PREFETCH: {
            const void* base[4]; int64_t bytes[4]; int n = 0;
            if (L >= d.first_k_dense) {
                const int shared_inter = 2 * d.moe_inter;
                base[n] = w.layer_shared[L];
                bytes[n++] = (int64_t)3 * shared_inter * d.hidden * 2;
            }
            if (L + 1 < d.layers) {
                base[n] = w.layer_qkv[L + 1];
                bytes[n++] = (int64_t)(d.heads * d.q_head_dim + d.kv_lora + d.qk_rope) * d.hidden * 2;
                base[n] = w.layer_o_proj[L + 1];
                bytes[n++] = (int64_t)d.hidden * d.heads * d.v_head * 2;
                base[n] = w.layer_kv_b[L + 1];
                bytes[n++] = (int64_t)d.heads * (d.qk_nope + d.v_head) * d.kv_lora * 2;
            }
            int64_t total = 0;
            for (int i = 0; i < n; ++i) total += bytes[i];
            const int parts = max(1, (int)t.n_split);
            const int64_t begin = (total * t.kv_chunk / parts) & ~(int64_t)15;
            const int64_t end = (total * (t.kv_chunk + max(1, (int)t.head)) / parts) & ~(int64_t)15;
            int64_t off = 0;
            for (int i = 0; i < n; ++i) {
                const int64_t lo = max(begin, off), hi = min(end, off + bytes[i]);
                if (lo < hi) {
                    prefetch_bytes((const char*)base[i] + (lo - off), hi - lo, rt.prefetch_nt != 0);
                }
                off += bytes[i];
            }
            break;
        }

        // Merge the head's kv_chunks partials (every merge task of the head
        // recomputes the 16 KB merge) and apply this task's slice of W_UV.
        case TASK_MERGE_UV: {
            const float* partials = act.attn_partial + (int64_t)t.head * chunks * kPartialStride;
            const int rows = d.v_head / chunks;
            merge_and_uv(partials, w.layer_kv_b[L], act.o, t.head, chunks,
                         d.kv_lora, d.qk_nope, d.v_head, s.q_c,
                         t.kv_chunk * rows, rows, tc.trace ? s.stamps2 : nullptr);
            break;
        }

        // This XCD's heads' q rows plus its own copy of kv_a, from the fused
        // [3072 + 576, hidden] matrix. RMSNorm (and the fold of the previous
        // layer's expert partials) is recomputed per worker in the prologue.
        case TASK_QKV_FUSED: {
            // layer input: embed's x[0] for layer 0; else the previous layer's
            // x[1] (+ its expert partials if it was MoE), published to x[0]
            const float* src = (L == 0) ? act.x[0] : act.x[1];
            const __hip_bfloat16* parts = (t.flags & FLAG_FOLD_PARTIALS) ? act.expert_out : nullptr;
#if FLEET_LOCAL_FOLD
            if (t.flags & kLocalFoldFlag) {
                stage_vector(act.x_norm + (int64_t)(kNumXCDs + xcd) * d.hidden, s.vec, d.hidden);
#if FLEET_VERIFY_LOCAL_ASSEMBLY
                verify_local_fold(d, act, rt, s.vec, false, t.index);
#endif
            } else
#endif
            stage_residual(d, src, parts, (L == 0) ? nullptr : act.x[0], s.vec, xcd, worker, coh);
            dump_layer(d, act, L - 1, s.vec, xcd, worker, tc);
            FLEET_STAMP_STAGE();
            rms_norm_lds_tree_early(d, w.layer_norm_in[L], s.vec, s.smem, tc.trace ? s.stamps2 : nullptr);
            FLEET_STAMP();
            etx_ds_stamp(tc.trace ? s.stamps2 : nullptr, 6); // q GEMV begin
            const int q_rows = heads_per_xcd * d.q_head_dim;                // 384
            const int q0 = xcd * q_rows;
            gemv_rows(w.layer_qkv[L] + (int64_t)q0 * d.hidden, d.hidden, s.vec,
                      act.q + q0, nullptr, EPI_BF16, 1.f, q_rows, d.hidden,
                      0, 1, worker, nw);
            etx_ds_stamp(tc.trace ? s.stamps2 : nullptr, 7); // q end / kv_a begin
            const int kv_rows = d.kv_lora + d.qk_rope;                      // 576
            const int kv0 = d.heads * d.q_head_dim;
            if (t.flags & FLAG_KVA_SHARED) {   // this XCD's 1/8 of the rows, one buffer
                gemv_rows(w.layer_qkv[L] + (int64_t)kv0 * d.hidden, d.hidden, s.vec,
                          act.kv_a, nullptr, EPI_BF16, 1.f, kv_rows, d.hidden,
                          xcd, kNumXCDs, worker, nw);
            } else {                            // this XCD's own full copy
                gemv_rows(w.layer_qkv[L] + (int64_t)kv0 * d.hidden, d.hidden, s.vec,
                          act.kv_a + (int64_t)xcd * kv_rows, nullptr, EPI_BF16, 1.f,
                          kv_rows, d.hidden, 0, 1, worker, nw);
            }
            etx_ds_stamp(tc.trace ? s.stamps2 : nullptr, 8); // kv_a end
            break;
        }

        // Attention: kv post + q-absorb (+ q_pe RoPE) in the prologue, all
        // into LDS, then this task's slice of the cache into a partial.
        case TASK_ATTENTION: {
            __hip_bfloat16* cache_layer = kv.data + (int64_t)L * kv.max_pos * kv.row;
            const float* kv_a = act.kv_a + ((t.flags & FLAG_KVA_SHARED) ? 0
                                            : (int64_t)xcd * (d.kv_lora + d.qk_rope));
#if FLEET_LOCAL_KVA
            if (t.flags & kLocalKvaFlag) {
                kv_a = act.kv_a + (int64_t)(xcd + 1) * (d.kv_lora + d.qk_rope);
#if FLEET_VERIFY_LOCAL_ASSEMBLY
                stage_vector(kv_a, s.part, d.kv_lora + d.qk_rope);
                for (int i = threadIdx.x; i < d.kv_lora + d.qk_rope; i += kBlockThreads)
                    if (!same_payload(s.part[i], ld_coh(act.kv_a + i, true)))
                        raise_abort(rt, kAbortLocalPayload | (uint32_t)t.index);
                __syncthreads();
#endif
            }
#endif
            EtxRopeCS etx_cs;
            kv_post_live(kv_a, w.layer_kv_a_norm[L], tc.cos, tc.sin, s.row,
                    d.kv_lora, d.qk_rope, d.rms_eps, s.smem,
                    /*write_cache=*/t.head == 0 && t.kv_chunk == 0,   // one writer per layer
                    cache_layer + (int64_t)tc.pos * kv.row, etx_cs);
            FLEET_STAMP_STAGE();   // after kv_post, before q_absorb
            if (rt.coherent_acts && t.head == 0 && t.kv_chunk == 0) {
                // The cache row is the one cached cross-XCD payload: with the
                // global fences gone, its writer flushes it to HBM itself.
                __syncthreads();
                if (threadIdx.x == 0) fence_release();
            }
            if (t.flags & FLAG_PUB_WAIT) {
                // q_c was published for this head while this task was doing
                // kv_post, so the wait is paid against work already done and
                // 131 KB of W_UK is read once per head instead of per chunk.
                __shared__ int qabs_ok;
                if (threadIdx.x == 0) {
                    qabs_ok = wait_event(rt, tc.epoch, t.local_event, SCOPE_XCD_LOCAL,
                                         xcd, kQAbsSlices) ? 1 : 0;
                }
                __syncthreads();
                if (!qabs_ok) return false;
                q_absorb_gather_live(act.q + (int64_t)t.head * d.q_head_dim,
                                act.q_absorbed + (int64_t)t.head * kQAbsSlices * d.kv_lora,
                                etx_cs, s.q_c, s.q_pe,
                                d.qk_nope, d.qk_rope, d.kv_lora, kQAbsSlices);
            } else {
                q_absorb(act.q + (int64_t)t.head * d.q_head_dim, w.layer_kv_b[L],
                         tc.cos, tc.sin, s.q_c, s.q_pe, s.part, t.head,
                         d.qk_nope, d.qk_rope, d.v_head, d.kv_lora);
            }
            float* partials = act.attn_partial + (int64_t)t.head * chunks * kPartialStride;
            FLEET_STAMP();
            attention_chunk(cache_layer, s.q_c, s.q_pe, s.row, d.softmax_scale,
                            tc.seq_len, kv.row, t.kv_chunk, chunks,
                            partials + (int64_t)t.kv_chunk * kPartialStride, s.part);
            break;
        }

        // o_proj needs all 16 heads, so it is the layer's one unavoidable
        // global event; the residual add is folded into its epilogue.
        // K-split (§12): this XCD's heads' 256 columns of W_o against its own
        // o slice, all 2048 output rows over the 37 workers, fp32 partial into
        // oproj_partial[xcd]; the router prologue folds the 8 partials.
        case TASK_O_PROJ: {
            if (t.flags & FLAG_OPROJ_ROW_SPLIT) {
                // Row-split: this XCD's 256 of the 2048 output rows against the
                // whole o[2048] (the merge that released us was global), and the
                // residual add rides the epilogue, so nothing is left to fold.
                const int hv = d.heads * d.v_head;
                stage_vector(act.o, s.vec, hv);
                FLEET_STAMP();
                gemv_rows(w.layer_o_proj[L], hv, s.vec, act.x[1], act.x[0],
                          EPI_RESIDUAL, 1.f, d.hidden, hv, xcd, kNumXCDs, worker, nw, coh);
            } else {
                const int kc = d.heads * d.v_head / kNumXCDs;
                stage_vector(act.o + (int64_t)xcd * kc, s.vec, kc);
                FLEET_STAMP();
                gemv_rows(w.layer_o_proj[L] + (int64_t)xcd * kc, d.heads * d.v_head, s.vec,
                          act.oproj_partial + (int64_t)xcd * d.hidden, nullptr, EPI_NONE, 1.f,
                          d.hidden, kc, /*xcd=*/0, /*n_xcds=*/1, worker, nw, coh);
            }
            break;
        }

        case TASK_EXPERT_GATE_UP: {
            const int32_t* ids = act.topk_ids + xcd * kMaxTopK;
            const float* ws = act.topk_w + xcd * kMaxTopK;
            if (!(t.flags & FLAG_TOPK_READ)) {   // per-task softmax + top-k
                select_experts(d, act.router_logits + (int64_t)xcd * d.n_routed, s.ids, s.ws);
                ids = s.ids; ws = s.ws;
            }
            FLEET_STAMP_STAGE();
            const ExpertUnit u = resolve_unit(
                t.expert_slot, w.layer_experts[L], w.layer_shared[L],
                ids, ws, d.routed_scaling, d.hidden, d.moe_inter, d.moe_inter * 2,
                w.expert_stride);
            stage_vector(act.x_norm + (int64_t)xcd * d.hidden, s.vec, d.hidden);
            FLEET_STAMP();
            uint32_t* chunk_counters = (t.flags & FLAG_CHUNK_SIGNAL)
                ? rt.xcd_counters + (int64_t)xcd * kMaxEvents + t.local_event : nullptr;
            expert_gate_up(u, s.vec, act.expert_h + (int64_t)xcd * d.moe_inter,
                           d.hidden, worker, nw, chunk_counters, kExpertKChunk,
                           t.kv_chunk);
            break;
        }

        case TASK_EXPERT_DOWN: {
            // the routing (64 logits from L2 under full streaming load, then
            // softmax + top-k on one wave) measured 4.9 us per task. Either
            // the router published it (64 B), or this worker's previous
            // descriptor was the layer's gate_up and it is still in LDS.
            const int32_t* ids = act.topk_ids + xcd * kMaxTopK;
            const float* ws = act.topk_w + xcd * kMaxTopK;
            if (!(t.flags & FLAG_TOPK_READ)) {
                if (!(t.flags & FLAG_ROUTING_CACHED)) {
                    select_experts(d, act.router_logits + (int64_t)xcd * d.n_routed, s.ids, s.ws);
                }
                ids = s.ids; ws = s.ws;
            }
            const ExpertUnit u = resolve_unit(
                t.expert_slot, w.layer_experts[L], w.layer_shared[L],
                ids, ws, d.routed_scaling, d.hidden, d.moe_inter, d.moe_inter * 2,
                w.expert_stride);
            // K-chunked (§12): chunk c of h is consumed as soon as this XCD's
            // gate_up waves have stored it. Chunk 0 was the descriptor's
            // wait; chunks 1.. are waited here. Partial sums live in s.part.
            const int n_chunks = (t.flags & FLAG_CHUNK_WAIT) ? max(1, (int)t.kv_chunk) : 1;
            const float* h = act.expert_h + (int64_t)xcd * d.moe_inter;
            __hip_bfloat16* out = act.expert_out + (int64_t)xcd * d.hidden;
            __shared__ int chunk_ok;
            for (int c = 0; c < n_chunks; ++c) {
                const int col0 = c * kExpertKChunk;
                const int kc = n_chunks == 1 ? u.inter : min(kExpertKChunk, u.inter - col0);
                if (c > 0) {
                    if (threadIdx.x == 0) {
                        chunk_ok = wait_event(rt, tc.epoch, t.local_event + c, SCOPE_XCD_LOCAL,
                                              xcd, t.wait_count) ? 1 : 0;
                    }
                    __syncthreads();              // also: everyone is done with s.vec
                    if (!chunk_ok) return false;
                }
                stage_vector(h + col0, s.vec, kc);
                if (c == 0) FLEET_STAMP();
                expert_down_chunk(u, s.vec, s.part, out, d.hidden, worker, nw,
                                  col0, kc, c == 0, c + 1 == n_chunks, coh);
            }
            break;
        }

        case TASK_DENSE_GATE_UP:
            stage_vector(act.x_norm + (int64_t)xcd * d.hidden, s.vec, d.hidden);
            dense_gate_up(w.layer_dense[L], s.vec, act.expert_h, d.hidden,
                          d.dense_inter, xcd, kNumXCDs, worker, nw);
            break;

        // h is 10944 wide here: read from L2 rather than staged.
        case TASK_DENSE_DOWN:
            gemv_rows(w.layer_dense[L] + (int64_t)2 * d.dense_inter * d.hidden,
                      d.dense_inter, act.expert_h, act.x[1], act.x[1],
                      EPI_RESIDUAL, 1.f, d.hidden, d.dense_inter, xcd, kNumXCDs,
                      worker, nw, coh);
            break;

        case TASK_LM_HEAD:
#if FLEET_LOCAL_FOLD
            if (t.flags & kLocalFoldFlag) {
                stage_vector(act.x_norm + (int64_t)(kNumXCDs + xcd) * d.hidden, s.vec, d.hidden);
#if FLEET_VERIFY_LOCAL_ASSEMBLY
                verify_local_fold(d, act, rt, s.vec, false, t.index);
#endif
            } else
#endif
            stage_residual(d, act.x[1], (t.flags & FLAG_FOLD_PARTIALS) ? act.expert_out : nullptr,
                           nullptr, s.vec, xcd, worker, coh);
            dump_layer(d, act, d.layers - 1, s.vec, xcd, worker, tc);
            rms_norm_lds_tree_early(d, w.final_norm, s.vec, s.smem);
            FLEET_STAMP();
            gemv_rows(w.lm_head, d.hidden, s.vec, act.logits, nullptr, EPI_BF16,
                      1.f, d.vocab, d.hidden, xcd, kNumXCDs, worker, nw);
            __syncthreads();   // every wave's logits are written (and in L2)
            lm_head_candidate(d, act, xcd, worker, nw, s.smem, s.ids_blk, coh);
            break;

        default:
            break;
    }
    return true;
}

// ---------------------------------------------------------------- entry point
//
// Launched with hipLaunchCooperativeKernel so all 304 workgroups are guaranteed
// co-resident: a scheduler that waits for an absent worker would deadlock, and
// the cooperative launch removes that failure mode by construction rather than
// by hoping the occupancy works out.

extern "C" __global__ __launch_bounds__(kBlock, kBlocksPerCU)
void fleet_decode_step(RuntimeState rt, ModelDims d, Weights w,
                       Activations act, KVCache kv) {
    __shared__ Scratch s;
    __shared__ int slot_smem;
    __shared__ int ok_smem;

    const int xcd = xcd_id();
    const int worker = claim_role(rt, xcd, &slot_smem);
    if (aborted(rt)) return;          // wrong XCD distribution: everyone leaves

    if (worker < 0) {
        run_scheduler(rt, xcd);
        return;
    }

    for (int tok = 0; tok < rt.n_tokens; ++tok) {
        TokenCtx tc;
        tc.tok = tok;
        tc.epoch = rt.epoch0 + (uint32_t)tok;
        tc.pos = d.context + tok;
        tc.seq_len = tc.pos + 1;
        tc.cos = act.cos + (int64_t)tc.pos * d.qk_rope;
        tc.sin = act.sin + (int64_t)tc.pos * d.qk_rope;
        tc.dump = rt.dump_layers && tok == 0;
        tc.trace = rt.trace != nullptr && tok == rt.trace_token;

        bool stop = false;
        for (int slot = 0;; ++slot) {
            const TaskDescriptor* t = fetch_task(rt, xcd, worker, slot);
            if (t == nullptr) break;

            // cross-task prefetch: in flight across the wait, consumed after
            uint4 pf[8];
            if (rt.prefetch_next) prefetch_issue(prefetch_plan(*t, d, w, xcd, worker), pf);

            uint64_t t_start = 0, t_ready = 0;
            if (threadIdx.x == 0) {
                if (tc.trace) t_start = __builtin_amdgcn_s_memrealtime();
                bool ok = true;
                // Token boundary: the embed of token t waits for token t-1's
                // final event (the previous epoch of the graph's last event).
                if (t->kind == TASK_EMBED && tok > 0) {
                    ok = wait_event(rt, tc.epoch - 1, rt.done_event, SCOPE_GLOBAL, xcd, 1);
                }
                if (ok) {
                    ok = wait_event(rt, tc.epoch, t->wait_event, (EventScope)t->wait_scope,
                                    xcd, t->wait_count);
                }
                ok_smem = ok ? 1 : 0;
                if (tc.trace) t_ready = __builtin_amdgcn_s_memrealtime();
            }
            __syncthreads();
            if (rt.prefetch_next) {
                uint32_t sink = 0;
#pragma unroll
                for (int u = 0; u < 8; ++u) sink ^= pf[u].x;
                asm volatile("" :: "v"(sink));
            }
            if (!ok_smem) { stop = true; break; }   // abort raised somewhere

            bool do_signal = true;
            if (threadIdx.x == 0) { s.stamp = 0; s.stamp_stage = 0; }
            if (!rt.smoke) {
                do_signal = run_task(*t, rt, d, w, act, kv, s, tc, xcd, worker);
            } else if (t->flags & FLAG_CHUNK_SIGNAL) {
                // no body: stand in for the 4 waves' per-chunk arrivals
                if (threadIdx.x == 0) {
                    for (int c = 0; c < t->kv_chunk; ++c) {
                        __hip_atomic_fetch_add(
                            rt.xcd_counters + (int64_t)xcd * kMaxEvents + t->local_event + c,
                            (uint32_t)kWaves, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    }
                }
            } else if (t->flags & FLAG_SIGNAL_LAST) {
                __shared__ int last_smem2;
                if (threadIdx.x == 0) {
                    last_smem2 = arrive_last(rt, tc.epoch, t->local_event, xcd,
                                             max(1, (int)t->n_split)) ? 1 : 0;
                }
                __syncthreads();
                do_signal = last_smem2 != 0;
            }
            __syncthreads();

            if (threadIdx.x == 0 && do_signal) {
                signal_event(rt, tc.epoch, t->signal_event, (EventScope)t->signal_scope,
                             xcd, t->signal_xcd_count);
            }
            if (threadIdx.x == 0) {
                if (tc.trace) {
                    uint64_t* tr = rt.trace + (int64_t)t->index * 4;
                    tr[0] = t_start;                                  // began waiting
                    tr[1] = t_ready;                                  // wait satisfied
                    tr[2] = __builtin_amdgcn_s_memrealtime();         // signalled
                    // xcd in the low byte; prologue ticks (ready -> end of
                    // prologue, 0 if the kind has none) above it
                    const uint64_t pro = s.stamp ? (s.stamp - t_ready) : 0;
                    const uint64_t stg = s.stamp_stage ? (s.stamp_stage - t_ready) : 0;
                    tr[3] = (uint64_t)xcd | ((stg & 0xffffff) << 8) | ((pro & 0xffffff) << 32);
                }
            }
        }
        if (stop) break;
    }
}

}  // namespace fleet

// ---------------------------------------------------------------- host entry
//
// The launch lives in this translation unit so the host stub the runtime
// registers for fleet_decode_step is the one being launched; the launcher
// only ever sees these C functions.

__global__ void fleet_probe_xcc(uint32_t* hist) {
    if (threadIdx.x == 0) atomicAdd(&hist[fleet::xcd_id() & 15], 1u);
}

extern "C" hipError_t fleet_probe_xcds(uint32_t* d_hist, hipStream_t stream) {
    hipLaunchKernelGGL(fleet_probe_xcc, dim3(fleet::kGrid), dim3(fleet::kBlock), 0,
                       stream, d_hist);
    return hipGetLastError();
}

extern "C" hipError_t fleet_occupancy(int* blocks_per_cu) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks_per_cu, (const void*)fleet::fleet_decode_step, fleet::kBlock, 0);
}

extern "C" hipError_t fleet_launch_step(const fleet::RuntimeState* rt,
                                        const fleet::ModelDims* d,
                                        const fleet::Weights* w,
                                        const fleet::Activations* act,
                                        const fleet::KVCache* kv,
                                        hipStream_t stream) {
    void* args[] = {(void*)rt, (void*)d, (void*)w, (void*)act, (void*)kv};
    return hipLaunchCooperativeKernel((const void*)fleet::fleet_decode_step,
                                      dim3(fleet::kGrid), dim3(fleet::kBlock),
                                      args, 0, stream);
}
