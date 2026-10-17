// ETX_DSV2_QKV_XB=1 (docs/BATCH-2026-10-14.md section 5, stage B2, qkv only): this source replaces fleet_shim.hip ONLY
// in that graph (graph.py XB_T, build_xb_qkv). The default TU does not include it. No edits to the pinned fleet export.
//
// Two tiles per layer (graph.py build_xb_qkv):
//   fleet_xb_norm  one task per request and XCD (8 per request): fleet's TASK_QKV_FUSED prologue with that request's
//                  FleetParams (stage_residual with the same fold and residual publication, dump_layer, the input
//                  RMSNorm the build's levers selected), then the normalized vector as bf16 into this XCD's slice of
//                  the request's act.x_norm. The norm's output is bf16-valued (bf16_round(bf16_round(v * inv) *
//                  gamma)), so the bf16 copy is exact. x_norm is free here: its last readers (layer L-1's expert /
//                  dense gate_up) finished before the layer input event, its next writer (layer L's router) runs after
//                  this layer's qkv.
//   fleet_qkv_xb   the batched qkv grid (8 XCDs x 74 workers): copies the B bf16 vectors of its XCD into LDS, then
//                  ONE pass over this worker's rows of the fused [3072 + 576, 2048] weight: each 16-byte weight chunk
//                  is loaded once and multiplied with all B vectors (fleet::fma8, the same per-lane order and the
//                  same wave_sum tree as fleet's gemv_rows_pipelined), then fleet's EPI_BF16 epilogue into request
//                  j's act.q / act.kv_a. Per-request state is three pointers per request, kept in LDS.
// Row split: fleet's TASK_QKV_FUSED split with n_workers = 74 instead of 37 (q rows: this XCD's 384, kv_a: the full 576
// per XCD, or this XCD's 1/8 with FLAG_KVA_SHARED's split).
//
// LDS: no new allocation. Slots 0..B-1 of the bf16 vectors are fleet::Scratch regions (part: 2 slots, row + q_c +
// q_pe: 1 slot, vec: 1 slot); the 3 x B pointers live in s.logits (unused by fleet's qkv). The tile's LDS use is
// sizeof(fleet::Scratch), the same as every other tile (graph.py SCRATCH_BYTES).
//
// ETX_DSV2_OPROJ_XB=1 (B = 4 only, stacked on ETX_DSV2_QKV_XB=1; graph.py build_xb_oproj) adds a third tile:
//   fleet_oproj_xb the batched o_proj grid (8 XCDs x 74 workers): fleet's K-split TASK_O_PROJ for B requests at once.
//                  It copies the B fp32 o slices of its XCD (256 columns each, 1 KB) into s.vec, then ONE pass over
//                  this worker's rows of the XCD's [2048, 256] column block of W_o: each 16-byte weight chunk is loaded
//                  once and multiplied with all B slices. Same layout as fleet's gemv_rows_pipelined_half<8> (one row
//                  per half-wave, 4 rows per half-wave per group, two groups in flight), the same per-lane order and
//                  the same half-wave butterfly, so every partial is bitwise the value of fleet's per-request task.
//                  The fp32 partials go to request j's act.oproj_partial[xcd] with fleet's coherent store (EPI_NONE);
//                  the router prologue of each request folds them as today. No norm, no new LDS: the 2 x B pointers
//                  live in s.logits.
//
// ETX_DSV2_LMHEAD_XB=1 (B = 4 only, stacked on ETX_DSV2_OPROJ_XB=1; graph.py build_xb_lm) adds the token tail:
//   fleet_xb_final_norm  one task per request and XCD: fleet's TASK_LM_HEAD prologue (stage_residual with the expert
//                  fold, dump of the last layer, the final RMSNorm the build's lever selected), then the normalized
//                  vector as bf16 into this XCD's slice of the request's act.x_norm (exact, as in fleet_xb_norm).
//                  x_norm is free: its last readers (the last layer's experts) finished before the request's last
//                  E_down; its next writer (the next token's layer-0 norm grid) runs after this token's argmax.
//   fleet_lm_head_xb the batched lm_head grid (8 XCDs x 74 workers): the B vectors into the qkv tile's LDS slots,
//                  then gemv_rows_xb over this XCD's 12800 rows of W_lm (one weight stream for all B requests),
//                  fleet's EPI_BF16 with plain stores (as fleet's lm_head) into request j's act.logits.
//   fleet_xb_lm_cand fleet::lm_head_candidate unchanged, per request on fleet's 37 workers, after an XCD-local event:
//                  the same 296 candidates in act.argmax_val / argmax_idx, so fleet's argmax task is unchanged.
#include "examples/dsv2lite/fleet_shim.hip"
#include <stddef.h>

#if !FLEET_CONST_PARAMS
#error "fleet_xb.hip reads every request's FleetParams from etx_fleet_const (FLEET_CONST_PARAMS=1)"
#endif

namespace etx_xb {

constexpr int kB = ETX_DS_REQUESTS;
constexpr int kMaxB = 4;                         // graph.py XB_MAX_REQUESTS
static_assert(kB >= 2 && kB <= kMaxB, "ETX_DSV2_QKV_XB needs -DETX_DS_REQUESTS=2..4");
constexpr int kD = 4;                            // 16-byte chunks per lane per row: K <= 4 * 64 * 8 = 2048
constexpr int kRPI = 1;                          // rows per wave per group, two groups in flight: 32 VGPRs of weights
                                                 // (fleet's pipelined<2, 4> has 64; with B = 4 the megakernel needs them)
constexpr size_t kVecBytes = (size_t)fleet::kMaxHidden * sizeof(__hip_bfloat16);   // 4096 B per request
constexpr int kFold = 1, kKvs = 2;               // grid constant flags (graph.py XB_FOLD, XB_KVS)

// LDS slot j's byte offset in fleet::Scratch
constexpr size_t kPartOff = offsetof(fleet::Scratch, part);
constexpr size_t kRowOff = offsetof(fleet::Scratch, row);
static_assert(sizeof(fleet::Scratch::part) >= 2 * kVecBytes, "s.part holds two bf16 slots");
static_assert(offsetof(fleet::Scratch, q_c) == kRowOff + sizeof(fleet::Scratch::row) &&
              offsetof(fleet::Scratch, q_pe) == offsetof(fleet::Scratch, q_c) + sizeof(fleet::Scratch::q_c) &&
              sizeof(fleet::Scratch::row) + sizeof(fleet::Scratch::q_c) + sizeof(fleet::Scratch::q_pe) >= kVecBytes,
              "row + q_c + q_pe are contiguous and hold one bf16 slot");
static_assert(sizeof(fleet::Scratch::vec) >= kVecBytes, "s.vec holds the last slot");
static_assert(sizeof(fleet::Scratch::logits) >= 3 * kMaxB * sizeof(void*), "s.logits holds the 3 x B pointers");
static_assert(kPartOff % 16 == 0 && kRowOff % 16 == 0 && offsetof(fleet::Scratch, vec) % 16 == 0, "16-byte LDS reads");

// j is a compile-time index everywhere (unrolled loops): a constant offset from the LDS base
__device__ __forceinline__ uint4* slot(fleet::Scratch& s, int j) {
  char* base = reinterpret_cast<char*>(&s);
  if (j == kB - 1) return reinterpret_cast<uint4*>(s.vec);
  if (j < 2) return reinterpret_cast<uint4*>(base + kPartOff + (size_t)j * kVecBytes);
  return reinterpret_cast<uint4*>(base + kRowOff);
}

// The input norm of TASK_QKV_FUSED. Levers norm_gamma_early / norm_tree / norm_tree_all rename fleet's call site to
// their own helper (levers/norm_gamma_early.py); pick the same helper here, found by argument-dependent lookup on
// fleet::ModelDims, so this tile runs the build's norm. The levers are alternatives (patch_fleet refuses two).
struct rank0 {};
struct rank1 : rank0 {};
struct rank2 : rank1 {};
template <class D>
__device__ __forceinline__ auto input_norm(rank2, const D& d, const __hip_bfloat16* g, float* v, float* sm)
    -> decltype(rms_norm_lds_tree_early(d, g, v, sm)) { return rms_norm_lds_tree_early(d, g, v, sm); }
template <class D>
__device__ __forceinline__ auto input_norm(rank1, const D& d, const __hip_bfloat16* g, float* v, float* sm)
    -> decltype(rms_norm_lds_gamma_early(d, g, v, sm)) { return rms_norm_lds_gamma_early(d, g, v, sm); }
template <class D>
__device__ __forceinline__ void input_norm(rank0, const D& d, const __hip_bfloat16* g, float* v, float* sm) {
  fleet::rms_norm_lds(d, g, v, sm);
}

// the `coh` that fleet::run_task computes: levers coherent_loads and l1_loads set it true at compile time
__device__ __forceinline__ bool run_task_coh(const fleet::RuntimeState& rt) {
#if defined(FLEET_COH_LOADS) || defined(FLEET_L1_LOADS)
  (void)rt;
  return true;
#else
  return rt.coherent_acts != 0;
#endif
}

// 16 bytes of an activation another task of this XCD wrote, loaded the way the build loads activations (lever
// l1_loads: past L1 from the XCD's L2; coherent: agent-scope atomics; otherwise plain, after the event's acquire)
__device__ __forceinline__ uint4 ld_act16(const uint4* p, bool coh) {
#if defined(FLEET_L1_LOADS)
  (void)coh;
  return fleet::ld_l1_u4(p);
#else
  if (!coh) return *p;
  const uint64_t* q = reinterpret_cast<const uint64_t*>(p);
  const uint64_t lo = __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  const uint64_t hi = __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  return make_uint4((uint32_t)lo, (uint32_t)(lo >> 32), (uint32_t)hi, (uint32_t)(hi >> 32));
#endif
}

// ---------------------------------------------------------------- fleet_xb_norm
// grid constants (layer, request, flags: kFold); one task per XCD (domain), worker index 0 for fleet's helpers, so XCD
// 0's task publishes the residual and dumps the layer once, as fleet's (xcd 0, worker 0) qkv task does
__device__ __forceinline__ void norm(const etx_ctx* c) {
  const int L = LAYER(c), xcd = (int)c->domain;
  const bool fold = (c->cst[2] & kFold) != 0;
  const FleetParams* fp = FP(c);
  const fleet::ModelDims& d = fp->d;
  const fleet::Activations& act = fp->act;
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  fleet::TokenCtx tc;
  tc.tok = fp->tok; tc.epoch = fp->epoch; tc.pos = fp->pos; tc.seq_len = fp->pos + 1;
  tc.cos = act.cos + (int64_t)fp->pos * d.qk_rope; tc.sin = act.sin + (int64_t)fp->pos * d.qk_rope;
  tc.dump = fp->dump != 0; tc.trace = false;
  const float* src = (L == 0) ? act.x[0] : act.x[1];
  const __hip_bfloat16* parts = fold ? act.expert_out : nullptr;
  fleet::stage_residual(d, src, parts, (L == 0) ? nullptr : act.x[0], s.vec, xcd, 0, run_task_coh(fp->rt));
  fleet::dump_layer(d, act, L - 1, s.vec, xcd, 0, tc);
  input_norm(rank2{}, d, fp->w.layer_norm_in[L], s.vec, s.smem);   // ends with __syncthreads
  // the bf16 copy (exact: the values are bf16 already), 8 values = 16 bytes per thread and step
  uint4* dst = reinterpret_cast<uint4*>(act.x_norm + (int64_t)xcd * d.hidden);
  for (int i = (int)threadIdx.x; i < d.hidden / 8; i += fleet::kBlockThreads) {
    union { __hip_bfloat16 h[8]; uint4 u; } p;
#pragma unroll
    for (int k = 0; k < 8; ++k) p.h[k] = __float2bfloat16(s.vec[i * 8 + k]);
    dst[i] = p.u;
  }
}

// ---------------------------------------------------------------- fleet_qkv_xb
__device__ __forceinline__ void unpack8(const uint4& u, float* x8) {
  fleet::unpack_bf16x2(u.x, x8[0], x8[1]); fleet::unpack_bf16x2(u.y, x8[2], x8[3]);
  fleet::unpack_bf16x2(u.z, x8[4], x8[5]); fleet::unpack_bf16x2(u.w, x8[6], x8[7]);
}

// fleet::gemv_rows(w, ld, x, y, nullptr, EPI_BF16, 1.f, N, K, xcd, n_xcds, worker, n_workers) for kB vectors at
// once: rows [begin, end) of this XCD's slice, this worker's rows first + k * n_workers, every weight chunk loaded once.
// Same software pipeline as gemv_rows_pipelined (two groups in flight, unconditional loads clamped to the last row,
// masked FMAs) with kRPI rows per group. x: slot(s, j) at lane + 64 dd (fixed LDS offsets; chunks past K are read but
// masked). ys: the B output pointers, in LDS, read by the storing lane.
__device__ __forceinline__ void gemv_rows_xb(const __hip_bfloat16* __restrict__ w, int ld, fleet::Scratch& s,
                                             float* const* ys, int N, int K, int xcd, int n_xcds, int worker,
                                             int n_workers) {
  const fleet::RowSlice sl = fleet::xcd_rows(N, xcd, n_xcds);
  const int first = sl.begin + worker;
  if (first >= sl.end) return;
  const int cnt = (sl.end - first + n_workers - 1) / n_workers;
  // the thread index through an empty asm: lane-derived LDS / weight offsets are then computed here, not hoisted as
  // loop invariants out of the megakernel's task loop, where each one held a VGPR for the whole kernel (local ROCm
  // 7.2.4 compile: 4 VGPRs, the difference between 1 and 2 workgroups per CU)
  int tid = (int)threadIdx.x;
  asm volatile("" : "+v"(tid));
  const int wave = tid / fleet::kWaveLanes;
  const int lane = tid % fleet::kWaveLanes;
  const int n4 = K / 8;
  if (n4 > kD * fleet::kWaveLanes) __builtin_trap();   // a row must fit one slot (K <= 2048)
  const int stride = kRPI * fleet::kWaves;
  int k = kRPI * wave;
  if (k >= cnt) return;
  uint4 bufA[kRPI][kD], bufB[kRPI][kD];
  int idx[kD];
  bool valid[kD];
#pragma unroll
  for (int dd = 0; dd < kD; ++dd) {
    valid[dd] = lane + dd * fleet::kWaveLanes < n4;
    idx[dd] = valid[dd] ? lane + dd * fleet::kWaveLanes : 0;
  }
#define XB_ROW(KK, R) (w + (int64_t)(first + min((KK) + (R), cnt - 1) * n_workers) * ld)
#define XB_ISSUE(BUF, KK)                                                                   \
  _Pragma("unroll") for (int r = 0; r < kRPI; ++r) {                                        \
    const uint4* r4 = reinterpret_cast<const uint4*>(XB_ROW(KK, r));                        \
    _Pragma("unroll") for (int dd = 0; dd < kD; ++dd) BUF[r][dd] = fleet::load_weight(r4 + idx[dd]); \
  }
#define XB_REDUCE(BUF, KK)                                                                  \
  {                                                                                         \
    float acc[kRPI][kB];                                                                    \
    _Pragma("unroll") for (int r = 0; r < kRPI; ++r)                                        \
      _Pragma("unroll") for (int j = 0; j < kB; ++j) acc[r][j] = 0.f;                       \
    _Pragma("unroll") for (int dd = 0; dd < kD; ++dd) {                                     \
      if (valid[dd]) {                                                                      \
        _Pragma("unroll") for (int j = 0; j < kB; ++j) {                                    \
          float x8[8];                                                                      \
          unpack8(slot(s, j)[lane + dd * fleet::kWaveLanes], x8);                           \
          _Pragma("unroll") for (int r = 0; r < kRPI; ++r) fleet::fma8(BUF[r][dd], x8, acc[r][j]); \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
    _Pragma("unroll") for (int r = 0; r < kRPI; ++r)                                        \
      _Pragma("unroll") for (int j = 0; j < kB; ++j) acc[r][j] = fleet::wave_sum(acc[r][j]); \
    if (lane == 0) {                                                                        \
      _Pragma("unroll") for (int r = 0; r < kRPI; ++r) {                                    \
        if ((KK) + r < cnt) {                                                               \
          const int n = first + ((KK) + r) * n_workers;                                     \
          _Pragma("unroll") for (int j = 0; j < kB; ++j)                                    \
            fleet::epi_st(ys[j] + n, fleet::bf16_round(1.f * fleet::bf16_round(acc[r][j])), false); \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }
  XB_ISSUE(bufA, k)
  while (true) {
    const int k2 = k + stride;
    XB_ISSUE(bufB, k2)
    XB_REDUCE(bufA, k)
    if (k2 >= cnt) break;
    const int k3 = k2 + stride;
    XB_ISSUE(bufA, k3)
    XB_REDUCE(bufB, k2)
    if (k3 >= cnt) break;
    k = k3;
  }
#undef XB_ROW
#undef XB_ISSUE
#undef XB_REDUCE
}

// grid constants (layer, workers per XCD, first request, flags: kKvs). The first request is always 0 (graph.py); it is
// a grid constant, not a literal, so the request parameter loads stay inside the task.
__device__ __forceinline__ void qkv(const etx_ctx* c) {
  const int L = LAYER(c), xcd = (int)c->domain, worker = (int)c->coord[1], nw = (int)c->cst[1], r0 = (int)c->cst[2];
  const bool kvs = (c->cst[3] & kKvs) != 0;
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  const FleetParams* fq = &etx_fleet_const[r0];
  const fleet::ModelDims& d = fq->d;
  const __hip_bfloat16* wq = fq->w.layer_qkv[L];
  const int q_rows = (d.heads / fleet::kNumXCDs) * d.q_head_dim;      // 384
  const int q0 = xcd * q_rows;
  const int kv_rows = d.kv_lora + d.qk_rope;                         // 576
  const int kv0 = d.heads * d.q_head_dim;
  // per-request pointers, one thread per request, into LDS (s.logits): [0, kMaxB) act.q, [kMaxB, 2 kMaxB) act.kv_a,
  // [2 kMaxB, 3 kMaxB) this XCD's normalized vector (fleet_xb_norm)
  void** tab = reinterpret_cast<void**>(s.logits);
  if (threadIdx.x < kB) {
    const FleetParams& p = fq[threadIdx.x];
    // one weight stream serves every request only if they share it
    if (p.w.layer_qkv[L] != wq || p.d.hidden != d.hidden) __builtin_trap();
    tab[threadIdx.x] = p.act.q + q0;
    tab[kMaxB + threadIdx.x] = p.act.kv_a + (kvs ? 0 : (int64_t)xcd * kv_rows);
    tab[2 * kMaxB + threadIdx.x] = p.act.x_norm + (int64_t)xcd * d.hidden;
  }
  __syncthreads();
  const bool coh = run_task_coh(fq->rt);
  // the B vectors of this XCD into LDS: one 16-byte chunk per thread and request, all B loads in flight (a loop over
  // the chunks cost the megakernel 2 VGPRs)
  static_assert(fleet::kMaxHidden / 8 <= fleet::kBlockThreads, "one 16-byte chunk per thread and request");
  if ((int)threadIdx.x < d.hidden / 8) {
#pragma unroll
    for (int j = 0; j < kB; ++j)
      slot(s, j)[threadIdx.x] = ld_act16(reinterpret_cast<const uint4*>(tab[2 * kMaxB + j]) + threadIdx.x, coh);
  }
  __syncthreads();
  float* const* out = reinterpret_cast<float* const*>(tab);
  gemv_rows_xb(wq + (int64_t)q0 * d.hidden, d.hidden, s, out, q_rows, d.hidden, 0, 1, worker, nw);
  // FLAG_KVA_SHARED: this XCD's 1/8 of the rows into one buffer; otherwise this XCD's own full copy
  gemv_rows_xb(wq + (int64_t)kv0 * d.hidden, d.hidden, s, out + kMaxB, kv_rows, d.hidden, kvs ? xcd : 0,
               kvs ? fleet::kNumXCDs : 1, worker, nw);
}

// ---------------------------------------------------------------- fleet_oproj_xb
constexpr int kOprojK = 256;                     // K-split columns per XCD: heads * v_head / kNumXCDs (16 * 128 / 8)
constexpr int kOH = 4;                           // rows per half-wave per group: fleet's gemv_rows_pipelined_half<8>
static_assert(kOprojK / 8 == fleet::kWaveLanes / 2, "one K-split row = one 16-byte chunk per lane of a half-wave");
static_assert(sizeof(fleet::Scratch::vec) >= (size_t)kMaxB * kOprojK * sizeof(float), "s.vec holds the B fp32 o slices");
static_assert(kMaxB * kOprojK / 4 <= fleet::kBlockThreads, "one 16-byte chunk per thread for all B slices");
static_assert(sizeof(fleet::Scratch::logits) >= 2 * kMaxB * sizeof(void*), "s.logits holds the 2 x B pointers");

// fleet::gemv_rows(w, ld, x, y, nullptr, EPI_NONE, 1.f, N, kOprojK, 0, 1, worker, n_workers, coh) for kB vectors at
// once, through fleet's half-wave path: rows first + k * n_workers, row k + 2 j + half of a group on half-wave `half`,
// lane hl owns 16-byte chunk hl of the row (all 32 chunks valid at K = 256). xs: s.vec, slice b at b * kOprojK.
// ys: the B output pointers, in LDS, read by the storing lane. One request at a time inside a group (sums, then
// stores): only kOH accumulators are live, as in fleet's half-wave path.
__device__ __forceinline__ void gemv_rows_half_xb(const __hip_bfloat16* __restrict__ w, int ld, const float* xs,
                                                  float* const* ys, int N, int worker, int n_workers, bool coh) {
  const int first = worker;
  if (first >= N) return;
  const int cnt = (N - first + n_workers - 1) / n_workers;
  int tid = (int)threadIdx.x;                    // see gemv_rows_xb: keeps the lane offsets inside the task
  asm volatile("" : "+v"(tid));
  const int wave = tid / fleet::kWaveLanes;
  const int lane = tid % fleet::kWaveLanes;
  const int half = lane >> 5, hl = lane & 31;
  const int stride = 2 * kOH * fleet::kWaves;
  int k = 2 * kOH * wave;
  if (k >= cnt) return;
  uint4 bufA[kOH], bufB[kOH];
#define XBO_ROW(KK, J) (w + (int64_t)(first + min((KK) + 2 * (J) + half, cnt - 1) * n_workers) * ld)
#define XBO_ISSUE(BUF, KK)                                                                  \
  _Pragma("unroll") for (int j = 0; j < kOH; ++j)                                           \
    BUF[j] = fleet::load_weight(reinterpret_cast<const uint4*>(XBO_ROW(KK, j)) + hl);
#define XBO_REDUCE(BUF, KK)                                                                 \
  _Pragma("unroll") for (int b = 0; b < kB; ++b) {                                          \
    float acc[kOH];                                                                         \
    const float* x8 = xs + b * kOprojK + hl * 8;                                            \
    _Pragma("unroll") for (int j = 0; j < kOH; ++j) {                                       \
      acc[j] = 0.f;                                                                         \
      fleet::fma8(BUF[j], x8, acc[j]);                                                      \
      acc[j] = fleet::half_wave_sum(acc[j]);                                                \
    }                                                                                       \
    if (hl == 0) {                                                                          \
      _Pragma("unroll") for (int j = 0; j < kOH; ++j) {                                     \
        const int r = (KK) + 2 * j + half;                                                  \
        if (r < cnt) fleet::epi_st(ys[b] + first + r * n_workers, acc[j], coh);             \
      }                                                                                     \
    }                                                                                       \
  }
  XBO_ISSUE(bufA, k)
  while (k + 2 * stride < cnt) {
    const int k2 = k + stride;
    XBO_ISSUE(bufB, k2)
    XBO_REDUCE(bufA, k)
    const int k3 = k2 + stride;
    XBO_ISSUE(bufA, k3)
    XBO_REDUCE(bufB, k2)
    k = k3;
  }
  if (k + stride < cnt) {
    const int k2 = k + stride;
    XBO_ISSUE(bufB, k2)
    XBO_REDUCE(bufA, k)
    XBO_REDUCE(bufB, k2)
  } else {
    XBO_REDUCE(bufA, k)
  }
#undef XBO_ROW
#undef XBO_ISSUE
#undef XBO_REDUCE
}

// pro_trace slot of a batched o_proj task in stamped builds: fleet kind slot kXboProKind (no fleet task kind uses it),
// coordinate xcd * 128 + worker. PRO_SLOT's coord[0] * 64 + coord[1] does not fit 74 workers.
constexpr int kXboProKind = 31;
static_assert(kXboProKind > fleet::TASK_Q_ABSORB && kXboProKind < 32, "a pro_trace kind slot of its own");
#define XBO_PRO_SLOT(L, xcd, worker) ((((int64_t)(L) + 1) * 32 + kXboProKind) * 1024 + (int64_t)(xcd) * 128 + (worker))

// grid constants (layer, workers per XCD, first request). The first request is always 0 (graph.py), a grid constant
// for the same reason as in qkv().
__device__ __forceinline__ void oproj(const etx_ctx* c) {
  const int L = LAYER(c), xcd = (int)c->domain, worker = (int)c->coord[1], nw = (int)c->cst[1], r0 = (int)c->cst[2];
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  const FleetParams* fq = &etx_fleet_const[r0];
#if ETX_DS_PRO || ETX_DSV2_STAMPS2
  // stamped builds only (as pro_begin / pro_end; the host fills request 0's pro_trace with ETX_DS_PROLOGUE, the
  // ETX_TRACE runs of a stamped build): body start in s.stamp, kept in LDS, not in a register
  if (fq->pro_trace != nullptr && threadIdx.x == 0) s.stamp = __builtin_amdgcn_s_memrealtime();
#endif
  const fleet::ModelDims& d = fq->d;
  const int hv = d.heads * d.v_head;                                  // 2048: W_o's row length
  const int kc = hv / fleet::kNumXCDs;                                // 256: this XCD's columns
  const __hip_bfloat16* wo = fq->w.layer_o_proj[L];
  // per-request pointers, one thread per request, into LDS (s.logits): [0, kMaxB) this XCD's act.oproj_partial
  // slice, [kMaxB, 2 kMaxB) this XCD's act.o slice (merge_uv's output)
  void** tab = reinterpret_cast<void**>(s.logits);
  if (threadIdx.x < kB) {
    const FleetParams& p = fq[threadIdx.x];
    // one weight stream serves every request only if they share it; the half-wave layout needs K = 256
    if (p.w.layer_o_proj[L] != wo || p.d.hidden != d.hidden || kc != kOprojK) __builtin_trap();
    tab[threadIdx.x] = p.act.oproj_partial + (int64_t)xcd * d.hidden;
    tab[kMaxB + threadIdx.x] = p.act.o + (int64_t)xcd * kc;
  }
  __syncthreads();
  const bool coh = run_task_coh(fq->rt);
  // the B o slices into s.vec: thread t loads 16-byte chunk t % 64 of request t / 64 (fleet's stage_vector, with the
  // build's activation loads)
  // the thread index through an empty asm, as in gemv_rows_xb: with threadIdx.x the staging offsets were hoisted out
  // of the task loop (local ROCm 7.2.4 compile: 252 + 6 VGPRs, 1 workgroup per CU; with the asm: 248 + 6, 2)
  int t = (int)threadIdx.x;
  asm volatile("" : "+v"(t));
  if (t < kB * kOprojK / 4) {
    const int j = t / (kOprojK / 4), i = t % (kOprojK / 4);
    reinterpret_cast<uint4*>(s.vec)[t] = ld_act16(reinterpret_cast<const uint4*>(tab[kMaxB + j]) + i, coh);
  }
  __syncthreads();
#if ETX_DS_PRO || ETX_DSV2_STAMPS2
  // end of the staging: the trace splits the staging of the B o slices from the GEMV
  if (fq->pro_trace != nullptr && threadIdx.x == 0) s.stamp_stage = __builtin_amdgcn_s_memrealtime();
#endif
  gemv_rows_half_xb(wo + (int64_t)xcd * kc, hv, s.vec, reinterpret_cast<float* const*>(tab), d.hidden, worker, nw, coh);
#if ETX_DS_PRO || ETX_DSV2_STAMPS2
  // row {body start, end of staging, end of GEMV}, the layout of pro_end, in slot XBO_PRO_SLOT. Thread 0 is lane 0 of
  // wave 0, which always runs the GEMV to its end (cnt >= 27 rows per worker), so it reaches this point.
  if (fq->pro_trace != nullptr && threadIdx.x == 0) {
#if ETX_DSV2_STAMPS2
    uint64_t* r = fq->pro_trace + XBO_PRO_SLOT(L, xcd, worker) * 16;
    for (int i = 3; i < 16; ++i) r[i] = 0;
#else
    uint64_t* r = fq->pro_trace + XBO_PRO_SLOT(L, xcd, worker) * 3;
#endif
    r[0] = s.stamp; r[1] = s.stamp_stage; r[2] = __builtin_amdgcn_s_memrealtime();
  }
#endif
}

// ---------------------------------------------------------------- fleet_xb_final_norm
// grid constants (layer = -1, request, flags: kFold); one task per XCD, worker index 0 for fleet's helpers, so XCD 0's
// task dumps the last layer once, as fleet's (xcd 0, worker 0) lm_head task does. No residual publication (fleet's
// lm_head passes dst = nullptr).
__device__ __forceinline__ void final_norm(const etx_ctx* c) {
  const int xcd = (int)c->domain;
  const bool fold = (c->cst[2] & kFold) != 0;
  const FleetParams* fp = FP(c);
  const fleet::ModelDims& d = fp->d;
  const fleet::Activations& act = fp->act;
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  fleet::TokenCtx tc;
  tc.tok = fp->tok; tc.epoch = fp->epoch; tc.pos = fp->pos; tc.seq_len = fp->pos + 1;
  tc.cos = act.cos + (int64_t)fp->pos * d.qk_rope; tc.sin = act.sin + (int64_t)fp->pos * d.qk_rope;
  tc.dump = fp->dump != 0; tc.trace = false;
  fleet::stage_residual(d, act.x[1], fold ? act.expert_out : nullptr, nullptr, s.vec, xcd, 0, run_task_coh(fp->rt));
  fleet::dump_layer(d, act, d.layers - 1, s.vec, xcd, 0, tc);
  // the lever's helper, as fleet's lm_head after norm_tree_all (graph.py refuses norm_tree / norm_gamma_early, which
  // leave fleet's final norm unpatched)
  input_norm(rank2{}, d, fp->w.final_norm, s.vec, s.smem);   // ends with __syncthreads
  uint4* dst = reinterpret_cast<uint4*>(act.x_norm + (int64_t)xcd * d.hidden);
  for (int i = (int)threadIdx.x; i < d.hidden / 8; i += fleet::kBlockThreads) {
    union { __hip_bfloat16 h[8]; uint4 u; } p;
#pragma unroll
    for (int k = 0; k < 8; ++k) p.h[k] = __float2bfloat16(s.vec[i * 8 + k]);
    dst[i] = p.u;
  }
}

// ---------------------------------------------------------------- fleet_lm_head_xb
// grid constants (layer = -1, workers per XCD, first request = 0)
__device__ __forceinline__ void lm_head(const etx_ctx* c) {
  const int xcd = (int)c->domain, worker = (int)c->coord[1], nw = (int)c->cst[1], r0 = (int)c->cst[2];
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  const FleetParams* fq = &etx_fleet_const[r0];
  const fleet::ModelDims& d = fq->d;
  const __hip_bfloat16* wl = fq->w.lm_head;
  // per-request pointers into LDS (s.logits): [0, kMaxB) act.logits, [2 kMaxB, 3 kMaxB) this XCD's normalized vector
  void** tab = reinterpret_cast<void**>(s.logits);
  if (threadIdx.x < kB) {
    const FleetParams& p = fq[threadIdx.x];
    if (p.w.lm_head != wl || p.d.hidden != d.hidden || p.d.vocab != d.vocab) __builtin_trap();
    tab[threadIdx.x] = p.act.logits;
    tab[2 * kMaxB + threadIdx.x] = p.act.x_norm + (int64_t)xcd * d.hidden;
  }
  __syncthreads();
  const bool coh = run_task_coh(fq->rt);
  if ((int)threadIdx.x < d.hidden / 8) {
#pragma unroll
    for (int j = 0; j < kB; ++j)
      slot(s, j)[threadIdx.x] = ld_act16(reinterpret_cast<const uint4*>(tab[2 * kMaxB + j]) + threadIdx.x, coh);
  }
  __syncthreads();
  // fleet: gemv_rows(w.lm_head, d.hidden, s.vec, act.logits, nullptr, EPI_BF16, 1.f, d.vocab, d.hidden, xcd,
  // kNumXCDs, worker, nw), here with 74 workers
  gemv_rows_xb(wl, d.hidden, s, reinterpret_cast<float* const*>(tab), d.vocab, d.hidden, xcd, fleet::kNumXCDs, worker, nw);
}

// ---------------------------------------------------------------- fleet_xb_lm_cand
// grid constants (layer = -1, request); fleet's 37 workers of the request's slot
__device__ __forceinline__ void lm_cand(const etx_ctx* c) {
  const FleetParams* fp = FP(c);
  fleet::Activations& act = *const_cast<fleet::Activations*>(&fp->act);
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  fleet::lm_head_candidate(fp->d, act, (int)c->domain, (int)c->coord[1], fleet::kWorkersPerXCD, s.smem, s.ids_blk,
                           run_task_coh(fp->rt));
}

}  // namespace etx_xb

extern "C" __device__ void fleet_xb_norm(const etx_ctx* c) { etx_xb::norm(c); }
extern "C" __device__ void fleet_qkv_xb(const etx_ctx* c) { etx_xb::qkv(c); }
extern "C" __device__ void fleet_oproj_xb(const etx_ctx* c) { etx_xb::oproj(c); }
extern "C" __device__ void fleet_xb_final_norm(const etx_ctx* c) { etx_xb::final_norm(c); }
extern "C" __device__ void fleet_lm_head_xb(const etx_ctx* c) { etx_xb::lm_head(c); }
extern "C" __device__ void fleet_xb_lm_cand(const etx_ctx* c) { etx_xb::lm_cand(c); }
