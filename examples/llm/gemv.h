// The batch-1 GEMV of examples/llm, shared by tiles.hip and bench/gemv_bench.hip.
// Needs: etx/tile_util.h (etx_lane/etx_wave/etx_wave_sum), dot8(), and the LLM_GEMV_U / LLM_NT_WEIGHTS switches.
#pragma once
// ---------------------------------------------------------------- GEMV
// rows [r0, r1) of a row-major bf16 matrix W with K columns, times xs (fp32, LDS). The task's rows are one
// contiguous block of (r1-r0)*K bf16, so it is streamed as a flat sequence of segments of 64 lanes x EPL
// elements (EPL = 8 -> 16-byte loads when K % 512 == 0, else 4 -> 8-byte loads when K % 256 == 0); each
// segment lies inside one row. Wave w takes a contiguous quarter of the segments and keeps U loads in flight
// per lane (U x 16 B x 256 lanes = 64 KB per CU at U=16), which is what hides HBM latency on small tasks.
// A wave's running sum is reduced when its segment run crosses a row boundary; the per-wave partials meet
// in `part` (LDS, >= nwaves * (r1 - r0) floats) and are summed in wave order (deterministic); epi(row, dot)
// is then called once per row by one thread. LLM_GEMV_U overrides the depth; LLM_NT_WEIGHTS=1 uses
// non-temporal weight loads (weights are read once per token).
#ifndef LLM_GEMV_U
#define LLM_GEMV_U 16          // measured on MI300X 2026-09-25: 16 beats 32 at every task size (3.0-3.3 vs 1.2-2.7 TB/s)
#endif
typedef unsigned int llm_u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int llm_u32x2 __attribute__((ext_vector_type(2)));
template <int EPL> struct wvec;
template <> struct wvec<8> { typedef llm_u32x4 T; };
template <> struct wvec<4> { typedef llm_u32x2 T; };
template <> struct wvec<2> { typedef unsigned int T; };

// xs lives in LDS: it is read through an explicit address_space(3) pointer, otherwise the compiler emits flat
// loads for it (a generic pointer), and every flat load in the dot product waits on the vector memory counter,
// i.e. on the weight stream (measured in the assembly: flat_load_dwordx4 + s_waitcnt per product)
typedef __attribute__((address_space(3))) const float llm_lds_f;
typedef __attribute__((address_space(3))) float llm_lds_fw;
static __device__ __forceinline__ float dotn(llm_u32x4 w, const llm_lds_f* x) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { s += __uint_as_float(w[i] << 16) * x[2 * i]; s += __uint_as_float(w[i] & 0xffff0000u) * x[2 * i + 1]; }
  return s;
}
static __device__ __forceinline__ float dotn(unsigned int w, const llm_lds_f* x) {
  return __uint_as_float(w << 16) * x[0] + __uint_as_float(w & 0xffff0000u) * x[1];
}
static __device__ __forceinline__ float dotn(llm_u32x2 w, const llm_lds_f* x) {
  return __uint_as_float(w.x << 16) * x[0] + __uint_as_float(w.x & 0xffff0000u) * x[1]
       + __uint_as_float(w.y << 16) * x[2] + __uint_as_float(w.y & 0xffff0000u) * x[3];
}
template <class V> static __device__ __forceinline__ V wload(const V* p) {
#if defined(LLM_NT_WEIGHTS) && LLM_NT_WEIGHTS
  return __builtin_nontemporal_load(p);
#else
  return *p;
#endif
}

template <int EPL, class Epi>
static __device__ __forceinline__ void gemv_stream(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  typedef typename wvec<EPL>::T V;
  constexpr int U = LLM_GEMV_U, SEG = ETX_WAVE_SIZE * EPL;
  const llm_lds_f* xl = (const llm_lds_f*)xs;
  llm_lds_fw* pl = (llm_lds_fw*)part;
  const int lane = etx_lane(), wv = etx_wave(), nw = etx_nwaves(), nrt = r1 - r0;
  const int spr = K / SEG;                                   // segments per row
  const int S = nrt * spr;
  const int per = (S + nw - 1) / nw, sb = wv * per, se = min(S, sb + per);
  for (int i = lane; i < nrt; i += ETX_WAVE_SIZE) pl[wv * nrt + i] = 0.f;
  const V* base = (const V*)(W + (size_t)r0 * K) + lane;     // segment s starts at element s * SEG
  int cur = sb / (spr > 0 ? spr : 1);
  float acc = 0.f;
  auto consume = [&](int sg, V w) {
    const int row = sg / spr;
    if (row != cur) {                                        // uniform across the wave
      const float v = etx_wave_sum(acc);
      if (lane == 0) pl[wv * nrt + cur] += v;
      acc = 0.f; cur = row;
    }
    acc += dotn(w, xl + (sg - row * spr) * SEG + lane * EPL);
  };
  auto load = [&](V* w, int s) {
#pragma unroll
    for (int u = 0; u < U; ++u) w[u] = wload(base + (size_t)(s + u) * ETX_WAVE_SIZE);
  };
  auto use = [&](const V* w, int s) {
#pragma unroll
    for (int u = 0; u < U; ++u) consume(s + u, w[u]);
  };
  // one batch of U loads, then the products (a software-pipelined variant with two batches in flight
  // measured no gain at 4 waves per CU and a loss at 8, 2026-09-25; the megakernel spills there)
  int s0 = sb;
  V w[U];
  for (; s0 + U <= se; s0 += U) { load(w, s0); use(w, s0); }
  if (s0 < se) {                                             // tail of fewer than U segments
#pragma unroll
    for (int u = 0; u < U; ++u) if (s0 + u < se) w[u] = wload(base + (size_t)(s0 + u) * ETX_WAVE_SIZE);
#pragma unroll
    for (int u = 0; u < U; ++u) if (s0 + u < se) consume(s0 + u, w[u]);
  }
  if (sb < se) { const float v = etx_wave_sum(acc); if (lane == 0) pl[wv * nrt + cur] += v; }
  __syncthreads();
  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) {
    float v = 0.f;
    for (int w2 = 0; w2 < nw; ++w2) v += pl[w2 * nrt + r];
    epi(r0 + r, v);
  }
  __syncthreads();
}

template <class Epi>
static __device__ __forceinline__ void gemv_rows(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  if (K % (ETX_WAVE_SIZE * 8) == 0) { gemv_stream<8>(W, K, xs, r0, r1, part, epi); return; }
  if (K % (ETX_WAVE_SIZE * 4) == 0) { gemv_stream<4>(W, K, xs, r0, r1, part, epi); return; }
  if (K % (ETX_WAVE_SIZE * 2) == 0) { gemv_stream<2>(W, K, xs, r0, r1, part, epi); return; }
  // generic fallback (K % 8 == 0): rows to waves, lanes stride K
  const int lane = etx_lane(), nw = etx_nwaves(), nrt = r1 - r0;
  for (int r = r0 + etx_wave(); r < r1; r += nw) {
    float acc = 0.f;
    for (int k = lane * 8; k < K; k += ETX_WAVE_SIZE * 8) acc += dot8(*(const uint4*)(W + (size_t)r * K + k), xs + k);
    const float v = etx_wave_sum(acc);
    if (lane == 0) part[r - r0] = v;
  }
  __syncthreads();
  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) epi(r0 + r, part[r]);
  __syncthreads();
}

