// ETX_DSV2_QKV_XB=1 (docs/BATCH-2026-10-14.md section 5, stage B2, qkv only): this source replaces fleet_shim.hip ONLY
// in that graph (graph.py XB_T, build_xb_qkv). The default TU does not include it. No edits to the pinned fleet export.
//
// One task of the batched qkv grid (8 XCDs x 74 workers) does, for its (xcd, worker):
//   1. for every request j: fleet's TASK_QKV_FUSED prologue with request j's FleetParams (stage_residual with the
//      same fold and publication, dump_layer, the input RMSNorm the build's levers selected), then the normalized
//      vector as bf16 into LDS slot j. The norm's output is bf16-valued (bf16_round(bf16_round(v * inv) * gamma)), so
//      the bf16 copy is exact.
//   2. one pass over this worker's rows of the fused [3072 + 576, 2048] weight: each 16-byte weight chunk is loaded
//      once and multiplied with all B vectors (fleet::fma8, the same per-lane order and the same wave_sum tree as
//      fleet's gemv_rows_pipelined<2, 4>), then fleet's EPI_BF16 epilogue into request j's act.q / act.kv_a.
// Row split: fleet's TASK_QKV_FUSED split with n_workers = 74 instead of 37 (q rows: this XCD's 384, kv_a: the full 576
// per XCD, or this XCD's 1/8 with FLAG_KVA_SHARED's split). Grid constants: (layer, workers per XCD, first request,
// flags), see qkv() below.
// The first request is always 0 (graph.py); it is a grid constant, not a literal, so the request parameter loads stay
// inside the task (with literal indices they are loop-invariant __constant__ loads the compiler may hoist out of the
// megakernel's task loop).
//
// LDS: no new allocation. The 2 x B output pointers live in s.logits (unused by fleet's tasks). Slots 0..B-2 are fleet::Scratch regions that TASK_QKV_FUSED does not use (part: 2 slots,
// row + q_c + q_pe: 1 slot); slot B-1 is s.vec itself, converted in place after the last request's norm. The tile's
// LDS use is sizeof(fleet::Scratch), the same as every other tile (graph.py SCRATCH_BYTES).
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
constexpr int kRPI = 2;                          // rows per wave per group, as gemv_rows_pipelined<2, 4> (K = 2048)
constexpr size_t kVecBytes = (size_t)fleet::kMaxHidden * sizeof(__hip_bfloat16);   // 4096 B per request

// LDS slot j's byte offset in fleet::Scratch (slot kB - 1 is s.vec, see the file comment)
constexpr size_t kPartOff = offsetof(fleet::Scratch, part);
constexpr size_t kRowOff = offsetof(fleet::Scratch, row);
static_assert(sizeof(fleet::Scratch::part) >= 2 * kVecBytes, "s.part holds two bf16 slots");
static_assert(offsetof(fleet::Scratch, q_c) == kRowOff + sizeof(fleet::Scratch::row) &&
              offsetof(fleet::Scratch, q_pe) == offsetof(fleet::Scratch, q_c) + sizeof(fleet::Scratch::q_c) &&
              sizeof(fleet::Scratch::row) + sizeof(fleet::Scratch::q_c) + sizeof(fleet::Scratch::q_pe) >= kVecBytes,
              "row + q_c + q_pe are contiguous and hold one bf16 slot");
static_assert(sizeof(fleet::Scratch::vec) >= kVecBytes, "s.vec holds the last slot in place");
static_assert(sizeof(fleet::Scratch::logits) >= 2 * kMaxB * sizeof(float*), "s.logits holds the 2 x B output pointers");
static_assert(kPartOff % 16 == 0 && kRowOff % 16 == 0 && offsetof(fleet::Scratch, vec) % 16 == 0, "16-byte LDS reads");

__device__ __forceinline__ __hip_bfloat16* slot(fleet::Scratch& s, int j) {
  char* base = reinterpret_cast<char*>(&s);
  if (j == kB - 1) return reinterpret_cast<__hip_bfloat16*>(s.vec);
  if (j < 2) return reinterpret_cast<__hip_bfloat16*>(base + kPartOff + (size_t)j * kVecBytes);
  return reinterpret_cast<__hip_bfloat16*>(base + kRowOff);
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

// fleet's TASK_QKV_FUSED prologue for request j, then its normalized vector into slot j (bf16, exact). Called in a
// runtime loop over the requests (not unrolled): four inlined copies raised the megakernel's spills (local ROCm 7.14
// compile: scratch 640 B/lane unrolled, 144 as a loop).
__device__ __forceinline__ void prologue(fleet::Scratch& s, const FleetParams* fp, int j, bool fold, int L, int xcd,
                                        int worker) {
  const fleet::ModelDims& d = fp->d;
  const fleet::Activations& act = fp->act;
  fleet::TokenCtx tc;
  tc.tok = fp->tok; tc.epoch = fp->epoch; tc.pos = fp->pos; tc.seq_len = fp->pos + 1;
  tc.cos = act.cos + (int64_t)fp->pos * d.qk_rope; tc.sin = act.sin + (int64_t)fp->pos * d.qk_rope;
  tc.dump = fp->dump != 0; tc.trace = false;
  const float* src = (L == 0) ? act.x[0] : act.x[1];
  const __hip_bfloat16* parts = fold ? act.expert_out : nullptr;
  fleet::stage_residual(d, src, parts, (L == 0) ? nullptr : act.x[0], s.vec, xcd, worker, run_task_coh(fp->rt));
  fleet::dump_layer(d, act, L - 1, s.vec, xcd, worker, tc);
  input_norm(rank2{}, d, fp->w.layer_norm_in[L], s.vec, s.smem);   // ends with __syncthreads
  constexpr int kPer = fleet::kMaxHidden / fleet::kBlockThreads;     // 8
  __hip_bfloat16* dst = slot(s, j);
  // every thread reads its values before any thread writes: slot kB - 1 is s.vec itself (in place)
  float v[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int i = (int)threadIdx.x + k * fleet::kBlockThreads;
    v[k] = i < d.hidden ? s.vec[i] : 0.f;
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int i = (int)threadIdx.x + k * fleet::kBlockThreads;
    if (i < d.hidden) dst[i] = __float2bfloat16(v[k]);
  }
  __syncthreads();            // slot j complete; s.vec free for request j + 1's staging
}

__device__ __forceinline__ void unpack8(const uint4& u, float* x8) {
  fleet::unpack_bf16x2(u.x, x8[0], x8[1]); fleet::unpack_bf16x2(u.y, x8[2], x8[3]);
  fleet::unpack_bf16x2(u.z, x8[4], x8[5]); fleet::unpack_bf16x2(u.w, x8[6], x8[7]);
}

// fleet::gemv_rows(w, ld, x, y, nullptr, EPI_BF16, 1.f, N, K, xcd, n_xcds, worker, n_workers) for kB vectors at
// once: rows [begin, end) of this XCD's slice, this worker's rows first + k * n_workers, every weight chunk loaded once.
// Same software pipeline as gemv_rows_pipelined<2, 4> (unconditional loads clamped to the last row, masked FMAs).
// x: slot(s, j), LDS addresses at fixed offsets from s (ds_read immediates). ys: the B output pointers, in LDS, read by
// the storing lane.
__device__ __forceinline__ void gemv_rows_xb(const __hip_bfloat16* __restrict__ w, int ld, fleet::Scratch& s,
                                             float* const* ys, int N, int K, int xcd, int n_xcds, int worker,
                                             int n_workers) {
  const fleet::RowSlice sl = fleet::xcd_rows(N, xcd, n_xcds);
  const int first = sl.begin + worker;
  if (first >= sl.end) return;
  const int cnt = (sl.end - first + n_workers - 1) / n_workers;
  const int wave = threadIdx.x / fleet::kWaveLanes;
  const int lane = threadIdx.x % fleet::kWaveLanes;
  const int n4 = K / 8;
  if (n4 > kD * fleet::kWaveLanes) __builtin_trap();   // a row must fit one buffer (K <= 2048)
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
          unpack8(reinterpret_cast<const uint4*>(slot(s, j))[idx[dd]], x8);                      \
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

// grid constants (layer, workers per XCD, first request, flags): flags bit 0 = fold the previous layer's expert partials
// (fleet's FLAG_FOLD_PARTIALS), bit 1 = FLAG_KVA_SHARED. One body for all four cases: one inlined copy in the megakernel.
constexpr int kFold = 1, kKvs = 2;
__device__ __forceinline__ void qkv(const etx_ctx* c) {
  const int L = LAYER(c), xcd = (int)c->domain, worker = (int)c->coord[1], nw = (int)c->cst[1], r0 = (int)c->cst[2];
  const bool fold = (c->cst[3] & kFold) != 0, kvs = (c->cst[3] & kKvs) != 0;
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  const FleetParams* fq = &etx_fleet_const[r0];
  // the prologues first, with nothing else live: the megakernel's binding limit is scalar registers (SGPR spills go to
  // VGPR lanes), so the GEMV's parameters are loaded only after the loop
#pragma nounroll
  for (int j = 0; j < kB; ++j) prologue(s, fq + j, j, fold, L, xcd, worker);
  const fleet::ModelDims& d = fq->d;
  const __hip_bfloat16* wq = fq->w.layer_qkv[L];
#pragma unroll
  for (int j = 1; j < kB; ++j)               // one weight stream serves every request only if they share it
    if (fq[j].w.layer_qkv[L] != wq || fq[j].d.hidden != d.hidden) __builtin_trap();
  const int q_rows = (d.heads / fleet::kNumXCDs) * d.q_head_dim;      // 384
  const int q0 = xcd * q_rows;
  const int kv_rows = d.kv_lora + d.qk_rope;                         // 576
  const int kv0 = d.heads * d.q_head_dim;
  // the 2 x B output pointers go to LDS (s.logits, unused by every fleet task), read by the storing lane
  float** out = reinterpret_cast<float**>(s.logits);
  if (threadIdx.x < kB) {
    const int j = threadIdx.x;
    out[j] = fq[j].act.q + q0;
    out[kMaxB + j] = fq[j].act.kv_a + (kvs ? 0 : (int64_t)xcd * kv_rows);
  }
  __syncthreads();
  gemv_rows_xb(wq + (int64_t)q0 * d.hidden, d.hidden, s, out, q_rows, d.hidden, 0, 1, worker, nw);
  // FLAG_KVA_SHARED: this XCD's 1/8 of the rows into one buffer; otherwise this XCD's own full copy
  gemv_rows_xb(wq + (int64_t)kv0 * d.hidden, d.hidden, s, out + kMaxB, kv_rows, d.hidden, kvs ? xcd : 0,
               kvs ? fleet::kNumXCDs : 1, worker, nw);
}

}  // namespace etx_xb

extern "C" __device__ void fleet_qkv_xb(const etx_ctx* c) { etx_xb::qkv(c); }
