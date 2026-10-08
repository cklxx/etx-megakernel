// Small helpers for hand-written tile bodies (link mode). Nothing here touches
// events or queues; tiles only compute. Wave size comes from the lowering header.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "etx/abi.h"

#ifndef ETX_WAVE_SIZE
#define ETX_WAVE_SIZE 64
#endif

typedef __hip_bfloat16 etx_bf16;

static __device__ __forceinline__ float etx_b2f(etx_bf16 v) { return __bfloat162float(v); }
static __device__ __forceinline__ int etx_lane() { return threadIdx.x % ETX_WAVE_SIZE; }
static __device__ __forceinline__ int etx_wave() { return threadIdx.x / ETX_WAVE_SIZE; }
static __device__ __forceinline__ int etx_nwaves() { return blockDim.x / ETX_WAVE_SIZE; }

static __device__ __forceinline__ float etx_wave_sum(float v) {
  for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, ETX_WAVE_SIZE);
  return v;
}
static __device__ __forceinline__ float etx_wave_max(float v) {
  for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, ETX_WAVE_SIZE));
  return v;
}
#ifdef ETX_TOPK_KEY32
// ---- ETX_TOPK_KEY32 (default off): the softmax top-k selects on one 32-bit key per expert.
// See docs/TOPK-KEY32-2026-10-08.md. One wave of 64 lanes holds 2 logits per lane, so E <= 128
// (host.hip already refuses E > 128 for dsl_topk and for the folds).
//   key = bit 23 (valid) | order16(logit) << 7 | (127 - e);  key 0 = taken, or e >= E.
// order16 maps the bf16 logit to an unsigned value with the same order: 0x8000 +- magnitude (offset binary).
// -0 and +0 both give 0x8000, so they tie and the smaller index wins (the float compare of the old path).
// NaN gives 0, below -inf (0x0080). The unsigned max of the keys is the largest logit, and on equal logits
// the smaller e (larger 127 - e) wins: vLLM's tie rule. Softmax is monotone, so the ids need no softmax.
static_assert(ETX_WAVE_SIZE == 64, "the ETX_TOPK_KEY32 DPP reduction is written for wave64");
#define ETX_TOPK_KEY32_EMAX 128
static_assert(2 * ETX_WAVE_SIZE == ETX_TOPK_KEY32_EMAX, "2 logits per lane cover E <= 128; the low key field has 7 bits");
static __device__ __forceinline__ unsigned etx_key32_order(float x) {
  const unsigned mag = (__float_as_uint(x) >> 16) & 0x7fffu;          // a bf16 value converted to float: low 16 bits 0
  const unsigned u = x < 0.f ? 0x8000u - mag : 0x8000u + mag;          // -0 is not < 0: it maps with +0
  return x != x ? 0u : u;                                              // explicit NaN test, not the bit mapping
}
static __device__ __forceinline__ unsigned etx_key32(float x, int e) {
  return 0x800000u | (etx_key32_order(x) << 7) | (unsigned)(127 - e);
}
// the logit of a key (order16 inverse); NaN for order 0. -0 comes back as +0, which gives the same expf(x - m).
static __device__ __forceinline__ float etx_key32_logit(unsigned key) {
  const unsigned u = (key >> 7) & 0xffffu;
  if (u == 0u) return __uint_as_float(0x7fc00000u);
  return __uint_as_float((u >= 0x8000u ? u - 0x8000u : 0x8000u | (0x8000u - u)) << 16);
}
static __device__ __forceinline__ int etx_key32_expert(unsigned key) { return key ? 127 - (int)(key & 127u) : 0; }
static __device__ __forceinline__ unsigned etx_umax(unsigned a, unsigned b) { return a > b ? a : b; }
// Wave max of an unsigned value with DPP, no LDS: row_shr 1, 2, 4, 8 (bound_ctrl: an invalid source reads 0, the
// identity of the unsigned max) leaves each row's max in its lane 15; row_bcast:15 (row_mask 0xa) adds row 0 into
// row 1 and row 2 into row 3; row_bcast:31 (row_mask 0xc) adds lanes 0-31 into rows 2 and 3. Lane 63 then holds
// the max; v_readlane_b32 puts it in an SGPR (wave-uniform). The builtin lets the compiler insert the gfx9 wait
// states between a VALU write and the DPP read of that VGPR. Disabled lanes keep `old` = 0. All 64 lanes active.
static __device__ __forceinline__ unsigned etx_wave_max_u32(unsigned v) {
  v = etx_umax(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x111, 0xf, 0xf, true));   // row_shr:1
  v = etx_umax(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x112, 0xf, 0xf, true));   // row_shr:2
  v = etx_umax(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x114, 0xf, 0xf, true));   // row_shr:4
  v = etx_umax(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x118, 0xf, 0xf, true));   // row_shr:8
  v = etx_umax(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x142, 0xa, 0xf, false));  // row_bcast:15
  v = etx_umax(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x143, 0xc, 0xf, false));  // row_bcast:31
  return (unsigned)__builtin_amdgcn_readlane((int)v, 63);
}
// The softmax top-k of one wave. l[k] is the logit of expert lane + 64 k (any value when that e >= E).
// put(t, id, w) runs once in lane t, for each t < K (K <= 64): the K results leave in one store per output, and
// no lane keeps an array with a runtime index (that array costs a VGPR block and gpr-index moves in every round).
// Arithmetic of the weights = the old path: m = the max logit (here the round-0 winner), p = expf(l - m), s =
// etx_wave_sum(p) (the same butterfly order), p *= 1 / s, weight = p of the winner, sel = the weights summed in
// round order, renorm: weight / sel. The sum runs once, after round 0, outside the selection loop.
// kPerRound = false (etx_topk_key32): lane t keeps round t's id and weight; put(t, id, w) runs once in lane t < K.
// kPerRound = true (etx_topk_key32_lds, for LDS outputs): put(t, id, w) runs in lane 0 in round t, with the
// weight before renorm, and the function returns sel (wave-uniform); the caller divides by sel (the old code's way).
// The two forms give the same ids and weight bits. Both callers use the lane-t form now. The kPerRound form stays as
// the tested alternative: the form only steers the register allocation of the whole kernel (128 VGPRs, the limit),
// and which form spills less depends on the tree and the compiler (docs/TOPK-KEY32-2026-10-08.md, section 7.4).
template <bool kPerRound, class Put>
static __device__ __forceinline__ float etx_topk_key32_impl(const float (&l)[2], int E, int K, bool renorm, int lane, Put put) {
  unsigned key[2]; bool ok[2];
#pragma unroll
  for (int k = 0; k < 2; ++k) { const int e = lane + k * ETX_WAVE_SIZE; ok[k] = e < E; key[k] = ok[k] ? etx_key32(l[k], e) : 0u; }
  unsigned km = etx_wave_max_u32(etx_umax(key[0], key[1]));
#pragma unroll
  for (int k = 0; k < 2; ++k) if (key[k] == km) key[k] = 0u;           // the winner lane clears its key
  const float m = etx_key32_logit(km);
  float p[2], s = 0.f;
#pragma unroll
  for (int k = 0; k < 2; ++k) { p[k] = ok[k] ? expf(l[k] - m) : 0.f; s += p[k]; }
  s = etx_wave_sum(s);
  const float inv = 1.f / s;
#pragma unroll
  for (int k = 0; k < 2; ++k) p[k] = ok[k] ? p[k] * inv : 0.f;
  float sel = 0.f, myw = 0.f; int myid = 0;                            // kPerRound = false: lane t keeps round t
  for (int t = 0; t < K; ++t) {
    if (t > 0) {
      km = etx_wave_max_u32(etx_umax(key[0], key[1]));
#pragma unroll
      for (int k = 0; k < 2; ++k) if (key[k] == km) key[k] = 0u;
    }
    const int bi = etx_key32_expert(km);
    const float bv = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(bi >= ETX_WAVE_SIZE ? p[1] : p[0]), bi & (ETX_WAVE_SIZE - 1)));
    if constexpr (kPerRound) { if (lane == 0) put(t, bi, bv); }
    else if (lane == t) { myid = bi; myw = bv; }
    sel += bv;
  }
  if constexpr (!kPerRound) { if (lane < K) put(lane, myid, renorm ? myw / sel : myw); }
  return sel;
}
template <class Put>
static __device__ __forceinline__ void etx_topk_key32(const float (&l)[2], int E, int K, bool renorm, int lane, Put put) {
  etx_topk_key32_impl<false>(l, E, K, renorm, lane, put);
}
template <class Put>
static __device__ __forceinline__ float etx_topk_key32_lds(const float (&l)[2], int E, int K, int lane, Put put) {
  return etx_topk_key32_impl<true>(l, E, K, false, lane, put);
}
#endif
// block-wide reductions; `red` is a scratch array of >= nwaves floats in LDS; result in every thread
static __device__ __forceinline__ float etx_block_sum(float v, float* red) {
  v = etx_wave_sum(v);
  if (etx_lane() == 0) red[etx_wave()] = v;
  __syncthreads();
  float t = 0.f;
  for (int w = 0; w < etx_nwaves(); ++w) t += red[w];
  __syncthreads();
  return t;
}
static __device__ __forceinline__ float etx_block_max(float v, float* red) {
  v = etx_wave_max(v);
  if (etx_lane() == 0) red[etx_wave()] = v;
  __syncthreads();
  float t = red[0];
  for (int w = 1; w < etx_nwaves(); ++w) t = fmaxf(t, red[w]);
  __syncthreads();
  return t;
}
// wave-cooperative dot product: bf16 weight row (n elements) with an fp32 vector
static __device__ __forceinline__ float etx_wave_dot(const etx_bf16* __restrict__ w, const float* __restrict__ x, int n) {
  float acc = 0.f;
  for (int i = etx_lane(); i < n; i += ETX_WAVE_SIZE) acc += etx_b2f(w[i]) * x[i];
  return etx_wave_sum(acc);
}
// Pass 7 hook helper: touch `bytes` of memory so the lines land in L2 (l2_warm method).
// One 4-byte load per 128-byte line, spread over the workgroup; the asm keeps the loads alive.
static __device__ __forceinline__ void etx_touch(const void* p, size_t bytes) {
  const char* base = (const char*)p;
  const size_t lines = bytes / 128;
  int acc = 0;
  for (size_t l = threadIdx.x; l < lines; l += blockDim.x) acc += *(const volatile int*)(base + l * 128);
  asm volatile("" :: "v"(acc));
}
