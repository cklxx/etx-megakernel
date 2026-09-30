"""The streaming batch-1 GEMV as an ETX op: a generator that emits the tile source and the argument-block layout,
in the same form as the adapter's imported tiles (etx.importer.adapter), so model.py, the plan and the host treat it
like any other export (docs/MODEL-vllm-kernels.md section 13).

Why generated and not imported: vLLM's streaming kernels (wvSplitK, Triton fused_moe) keep 16-36 KB in flight per
CU and pay a gap per trip; the lf microbenchmark (bench/hw/hw_bench.hip) reaches 3.6-3.9 TB/s with 64 KB in flight.
The stream loop is examples/llm/gemv.h's (U loads per lane in flight, 64 KB per 16-wave workgroup, per-wave partials summed in wave order:
deterministic), with the depth per element width and the epilogue chosen here.

Forms (one export each; the host passes the same argument block: W, x, out, ids, wts, expert stride, M, K, rows per
task, tasks per expert; the grid words follow):
  dsl_gemv      out[r] = bf16(W[r, :] . x)                        rows [blk * rpt, ...) of W [M, K]
  dsl_moe_up    act[s, r] = bf16(silu(bf16(g)) * bf16(u))         expert e = ids[s], g = W_e[r, :] . x, u = W_e[M + r, :] . x
                                                                   (w13: gate rows then up rows; s = blk / tpe, r in chunk blk % tpe)
  dsl_moe_down  out[s, r] = bf16((W_e[r, :] . x_s) * wts[s])      x_s = act[s, :]
x and W are bf16, dots accumulate in fp32 (as vLLM's kernels; the summation order differs, so the outputs are compared
with a tolerance, host --check-dsl, not byte for byte).
"""
from __future__ import annotations

import json
from pathlib import Path

ARG_BYTES, GRID_OFF = 136, 120
# the "_fold" forms carry a prologue (docs/MODEL-vllm-kernels.md section 16): norm fold (mode 1): x = rms_norm(bf16(bf16(sum
# of R rows) + resid_in)) with weight ln (vLLM's moe_sum + fused_add_rms_norm numerics: the sum rounded, the add rounded
# per element, the variance over the rounded values, out = bf16(z * rsqrt * w)); task 0 stores the new residual to
# resid_out. top-k fold (mode 2): every expert task recomputes the softmax top-k from the logits (deterministic) and
# takes its slot's expert and weight; no top-k launch, no ids / wts buffers.
PARAMS = [("W", "ptr addrspace(1)", 0, 8), ("x", "ptr addrspace(1)", 8, 8), ("out", "ptr addrspace(1)", 16, 8),
          ("ids", "ptr addrspace(1)", 24, 8), ("wts", "ptr addrspace(1)", 32, 8), ("estride", "i64", 40, 8),
          ("M", "i32", 48, 4), ("K", "i32", 52, 4), ("rpt", "i32", 56, 4), ("tpe", "i32", 60, 4),
          ("rows", "ptr addrspace(1)", 64, 8), ("resid_in", "ptr addrspace(1)", 72, 8), ("resid_out", "ptr addrspace(1)", 80, 8),
          ("ln", "ptr addrspace(1)", 88, 8), ("R", "i32", 96, 4), ("eps", "float", 100, 4), ("mode", "i32", 104, 4),
          ("E", "i32", 108, 4), ("logits", "ptr addrspace(1)", 112, 8)]
FORMS = {"dsl_gemv": "plain", "dsl_moe_up": "silu_mul", "dsl_moe_down": "scale",
         "dsl_gemv_fold": "plain", "dsl_moe_up_fold": "silu_mul", "dsl_moe_down_fold": "scale"}
K_MAX, RCH, THREADS = 4096, 64, 1024                              # x (K floats) + 2 x RCH (moe_up's g) + 16 waves x RCH partials
LDS_BYTES = 4 * (K_MAX + 2 * RCH + (THREADS // 64) * RCH + 32)     # 21120 B (+ the top-k ids / weights); host-checked K, rpt


def import_entries(threads: int = 1024) -> dict:
    """imports.json entries for the three forms (what the adapter's host header and model.py read)."""
    ps = [{"ty": ty, "name": f"%{n}", "size": sz, "align": min(sz, 8), "offset": off} for n, (_, ty, off, sz) in enumerate(PARAMS)]
    return {e: {"dsl": form, "export": f"etx_tile_{e}", "kernel": e, "block": [threads, 1, 1], "threads": threads, "slots": 1,
                "slot_threads": threads, "params": ps, "grid_offset": GRID_OFF, "arg_bytes": ARG_BYTES, "lds_bytes_per_slot": LDS_BYTES,
                "lds_bytes": LDS_BYTES, "barriers": 0, "slot_barrier": False, "spec": f"ETX DSL GEMV ({form})"} for e, form in FORMS.items()}


STREAM = r'''
// ---- the stream loop (examples/llm/gemv.h): segments of 64 lanes x EPL bf16 inside one row, U loads per lane in flight
typedef unsigned int dsl_u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int dsl_u32x2 __attribute__((ext_vector_type(2)));
template <int EPL> struct dsl_wvec;
template <> struct dsl_wvec<8> { typedef dsl_u32x4 T; };
template <> struct dsl_wvec<4> { typedef dsl_u32x2 T; };
#define DSL_G __attribute__((address_space(1)))          // global loads, not flat (flat waits on the LDS counter too)
typedef __attribute__((address_space(3))) const float dsl_lds_f;
typedef __attribute__((address_space(3))) float dsl_lds_fw;
static __device__ __forceinline__ float dsl_dot(dsl_u32x4 w, const dsl_lds_f* x) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { s += __uint_as_float(w[i] << 16) * x[2 * i]; s += __uint_as_float(w[i] & 0xffff0000u) * x[2 * i + 1]; }
  return s;
}
static __device__ __forceinline__ float dsl_dot(dsl_u32x2 w, const dsl_lds_f* x) {
  return __uint_as_float(w.x << 16) * x[0] + __uint_as_float(w.x & 0xffff0000u) * x[1]
       + __uint_as_float(w.y << 16) * x[2] + __uint_as_float(w.y & 0xffff0000u) * x[3];
}
static __device__ __forceinline__ float dsl_bfr(float v) { return __bfloat162float(__float2bfloat16(v)); }
static __device__ __forceinline__ float dsl_silu(float g) { return g / (1.f + __expf(-g)); }
// rows [r0, r1) of W (row-major bf16, K columns) times xs (fp32 in LDS); part: LDS, >= nwaves * (r1 - r0) floats;
// epi(row, dot) once per row by one thread. Deterministic: per-wave partials are summed in wave order.
template <int EPL, int U, class Epi>
static __device__ __forceinline__ void dsl_gemv_rows(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  typedef typename dsl_wvec<EPL>::T V;
  constexpr int SEG = ETX_WAVE_SIZE * EPL;
  const dsl_lds_f* xl = (const dsl_lds_f*)xs;
  dsl_lds_fw* pl = (dsl_lds_fw*)part;
  const int lane = etx_lane(), wv = etx_wave(), nw = etx_nwaves(), nrt = r1 - r0;
  const int spr = K / SEG, S = nrt * spr;
  const int per = (S + nw - 1) / nw, sb = wv * per, se = min(S, sb + per);
  for (int i = (int)threadIdx.x; i < nw * nrt; i += blockDim.x) pl[i] = 0.f;
  __syncthreads();
  const DSL_G V* base = (const DSL_G V*)(W + (size_t)r0 * K) + lane;
  int cur = sb / spr;
  float acc = 0.f;
  auto consume = [&](int sg, V w) {
    const int row = sg / spr;
    if (row != cur) { const float v = etx_wave_sum(acc); if (lane == 0) pl[wv * nrt + cur] += v; acc = 0.f; cur = row; }
    acc += dsl_dot(w, xl + (sg - row * spr) * SEG + lane * EPL);
  };
  int s0 = sb;
  V w[U];
  for (; s0 + U <= se; s0 += U) {
#pragma unroll
    for (int u = 0; u < U; ++u) w[u] = base[(size_t)(s0 + u) * ETX_WAVE_SIZE];
#pragma unroll
    for (int u = 0; u < U; ++u) consume(s0 + u, w[u]);
  }
  if (s0 < se) {
#pragma unroll
    for (int u = 0; u < U; ++u) if (s0 + u < se) w[u] = base[(size_t)(s0 + u) * ETX_WAVE_SIZE];
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
// K % 512 == 0: 16 B loads, 4 in flight per lane; K % 256 == 0: 8 B loads, 8 in flight: 64 KB per workgroup of 16 waves
// (the megakernel's 1024 threads; 16 per lane as examples/llm/gemv.h's 4-wave tiles spilled 359 VGPRs at 128 per lane)
template <class Epi>
static __device__ __forceinline__ void dsl_gemv(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
#ifdef DSL_ABLATE_EPL4                                   // ablation: force the 8 B load path everywhere (moe_down's,
  dsl_gemv_rows<4, 8>(W, K, xs, r0, r1, part, epi);      // since K = 768 is not a multiple of 512) -- output stays correct
#else
  if (K % (ETX_WAVE_SIZE * 8) == 0) dsl_gemv_rows<8, 4>(W, K, xs, r0, r1, part, epi);
  else dsl_gemv_rows<4, 8>(W, K, xs, r0, r1, part, epi);
#endif
}
static __device__ __forceinline__ void dsl_stage(const __hip_bfloat16* __restrict__ x, int K, float* xs) {   // bf16 -> fp32 LDS, 16 B per thread
#ifdef DSL_ABLATE_STAGE
  __syncthreads(); return;          // ablation: the barrier without the fill (the dot then reads stale LDS: WRONG output)
#endif
#ifdef ETX_DSL_COH
  for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x) {
    // activation: two device-coherent 8 B loads (both stay in flight; the compiler places the waits)
    const unsigned long long lo = dsl_a_ld64((const unsigned long long*)(x + 8 * i));
    const unsigned long long hi = dsl_a_ld64((const unsigned long long*)(x + 8 * i) + 1);
    dsl_u32x4 v; v[0] = (unsigned)(lo & 0xffffffffull); v[1] = (unsigned)(lo >> 32); v[2] = (unsigned)(hi & 0xffffffffull); v[3] = (unsigned)(hi >> 32);
#else
  const DSL_G dsl_u32x4* xv = (const DSL_G dsl_u32x4*)x;
  for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x) {
    const dsl_u32x4 v = xv[i];
#endif
#pragma unroll
    for (int j = 0; j < 4; ++j) { xs[8 * i + 2 * j] = __uint_as_float(v[j] << 16); xs[8 * i + 2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u); }
  }
  __syncthreads();
}
// rows per pass of the stream loop: the partials take nwaves x DSL_RCH floats of LDS whatever the task's row count
// (lm_head tasks hold 500 rows; unbounded partials overran the 32 KB arena into the barrier words, review 2026-09-29)
#define DSL_RCH 64
template <class Epi>
static __device__ __forceinline__ void dsl_gemv_chunked(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  for (int r = r0; r < r1; r += DSL_RCH) dsl_gemv(W, K, xs, r, min(r1, r + DSL_RCH), part, epi);
}
struct DslArgs { const __hip_bfloat16* W; const __hip_bfloat16* x; __hip_bfloat16* out; const int* ids; const float* wts; long long estride; int M, K, rpt, tpe;
                 const __hip_bfloat16* rows; const __hip_bfloat16* resid_in; __hip_bfloat16* resid_out; const __hip_bfloat16* ln; int R; float eps; int mode, E;
                 const __hip_bfloat16* logits; };
static __device__ __forceinline__ DslArgs dsl_args(const unsigned char* a) {
  DslArgs d;
  d.W = *(const __hip_bfloat16* const*)(a + 0); d.x = *(const __hip_bfloat16* const*)(a + 8); d.out = *(__hip_bfloat16* const*)(a + 16);
  d.ids = *(const int* const*)(a + 24); d.wts = *(const float* const*)(a + 32); d.estride = *(const long long*)(a + 40);
  d.M = *(const int*)(a + 48); d.K = *(const int*)(a + 52); d.rpt = *(const int*)(a + 56); d.tpe = *(const int*)(a + 60);
  d.rows = *(const __hip_bfloat16* const*)(a + 64); d.resid_in = *(const __hip_bfloat16* const*)(a + 72); d.resid_out = *(__hip_bfloat16* const*)(a + 80);
  d.ln = *(const __hip_bfloat16* const*)(a + 88); d.R = *(const int*)(a + 96); d.eps = *(const float*)(a + 100); d.mode = *(const int*)(a + 104);
  d.E = *(const int*)(a + 108); d.logits = *(const __hip_bfloat16* const*)(a + 112);
  return d;
}
// norm fold: xs = bf16(rms_norm(z)) as floats, z = bf16(bf16(sum of R rows) + resid_in) per element; task 0 stores z
static __device__ __forceinline__ void dsl_stage_norm(const DslArgs& d, float* xs, float* red, bool store) {
  const int K = d.K;
  float ss = 0.f;
  for (int i = (int)threadIdx.x; i < K; i += blockDim.x) {
    float acc = 0.f;
    for (int s = 0; s < d.R; ++s) acc += dsl_a_ldbf(d.rows + (size_t)s * K + i);
    const float in = d.R > 1 ? dsl_bfr(acc) : acc;                                   // moe_sum rounds its fp32 sum once
    const float z = dsl_bfr(in + dsl_a_ldbf(d.resid_in + i));                        // the bf16 add of fused_add_rms_norm
    xs[i] = z; ss += z * z;
    if (store && d.resid_out != nullptr) dsl_a_stbf(d.resid_out + i, __float2bfloat16(z));
  }
  ss = etx_block_sum(ss, red);
  const float sv = rsqrtf(ss / (float)K + d.eps);
  for (int i = (int)threadIdx.x; i < K; i += blockDim.x) {
    xs[i] = dsl_bfr(xs[i] * sv * __bfloat162float(d.ln[i]));
    if (store && d.x != nullptr) dsl_a_stbf((__hip_bfloat16*)d.x + i, __float2bfloat16(xs[i]));   // the normed vector, for later consumers
  }
  __syncthreads();
}
// top-k fold: wave 0 computes the softmax top-k of the logits (etx/ops/topk.py's arithmetic) into LDS
static __device__ __forceinline__ void dsl_topk_lds(const __hip_bfloat16* logits, int E, int K, int renorm, int* ids, float* wts) {
  if (etx_wave() == 0) {
    const int lane = etx_lane();
    float p[2]; bool taken[2]; float m = -INFINITY;
#pragma unroll
    for (int k = 0; k < 2; ++k) { const int e = lane + k * ETX_WAVE_SIZE; p[k] = e < E ? dsl_a_ldbf(logits + e) : -INFINITY; taken[k] = e >= E; m = fmaxf(m, p[k]); }
    m = etx_wave_max(m);
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 2; ++k) { p[k] = taken[k] ? 0.f : expf(p[k] - m); s += p[k]; }
    s = etx_wave_sum(s);
    const float inv = 1.f / s;
#pragma unroll
    for (int k = 0; k < 2; ++k) p[k] = taken[k] ? 0.f : p[k] * inv;
    float sel = 0.f;
    for (int t = 0; t < K; ++t) {
      float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
      for (int k = 0; k < 2; ++k) { const int e = lane + k * ETX_WAVE_SIZE; if (!taken[k] && p[k] > bv) { bv = p[k]; bi = e; } }
      for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor(bv, off, ETX_WAVE_SIZE); const int oi = __shfl_xor(bi, off, ETX_WAVE_SIZE);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
#pragma unroll
      for (int k = 0; k < 2; ++k) if (lane + k * ETX_WAVE_SIZE == bi) taken[k] = true;
      if (lane == 0) { ids[t] = bi; wts[t] = bv; }
      sel += bv;
    }
    if (renorm && lane == 0) for (int t = 0; t < K; ++t) wts[t] = wts[t] / sel;
  }
  __syncthreads();
}
// ---- touch (the megakernel's touch hook, ETX_VL_TOUCH=1): while wave 0 waits for the task's event, waves 1.. load
// the first U vectors of their own segment range of the task's first DSL_RCH-row pass, exactly what dsl_gemv_rows
// loads first, so that trip hits L2 after the acquire (the weights depend on no event). 15 waves x 64 lanes x U x 16 B
// = 60 KB per workgroup: one trip, since 38 workgroups x 64 KB already fill most of an XCD's 4 MB L2.
template <int EPL, int U>
static __device__ __forceinline__ void dsl_touch_rows(const __hip_bfloat16* __restrict__ W, int K, int r0, int r1) {
  typedef typename dsl_wvec<EPL>::T V;
  constexpr int SEG = ETX_WAVE_SIZE * EPL;
  const int lane = etx_lane(), wv = etx_wave(), nw = etx_nwaves(), nrt = r1 - r0;
  const int spr = K / SEG, S = nrt * spr;
  const int per = (S + nw - 1) / nw, sb = wv * per, se = min(S, sb + per);
  const DSL_G V* base = (const DSL_G V*)(W + (size_t)r0 * K) + lane;
  unsigned acc = 0;
  V w[U];
#pragma unroll
  for (int u = 0; u < U; ++u) if (sb + u < se) w[u] = base[(size_t)(sb + u) * ETX_WAVE_SIZE];
#pragma unroll
  for (int u = 0; u < U; ++u) if (sb + u < se) acc ^= w[u][0];
  asm volatile("" :: "v"(acc));                     // the loads are kept (and complete here: these waves idle until the barrier anyway)
}
static __device__ __forceinline__ void dsl_touch_plain(const unsigned char* a, int blk) {
  const int nblk = *(const int*)(a + 120 + 12);
  if (blk >= nblk) return;
  const DslArgs d = dsl_args(a);
  const int r0 = blk * d.rpt, r1 = min(d.M, min(r0 + d.rpt, r0 + DSL_RCH));   // the first pass of dsl_gemv_chunked
  if (r0 >= r1) return;
  if (d.K % (ETX_WAVE_SIZE * 8) == 0) dsl_touch_rows<8, 4>(d.W, d.K, r0, r1); else dsl_touch_rows<4, 8>(d.W, d.K, r0, r1);
}
'''

TOUCH_FORMS = ("plain",)     # the dense GEMVs: an expert task's rows are not known before its own wait (top-k)

BODIES = {
    "plain": '''
  const int r0 = blk * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
  if (d.mode == 1) dsl_stage_norm(d, xs, xs + d.K + 16 * DSL_RCH, blk == 0); else dsl_stage(d.x, d.K, xs);
  __hip_bfloat16* out = d.out;
  dsl_gemv_chunked(d.W, d.K, xs, r0, r1, xs + d.K, [&](int r, float v) { dsl_a_stbf(out + r, __float2bfloat16(v)); });''',
    "silu_mul": '''
  const int s = blk / d.tpe, j = blk - s * d.tpe;
  const int r0 = j * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
  float* g = xs + d.K; float* part = g + 2 * d.rpt;
  int* lids = (int*)(part + 16 * DSL_RCH); float* lwts = (float*)(lids + 16);   // top-k fold: R = k, mode 3 = renormalised
#ifdef DSL_ABLATE_TOPK
  if (etx_wave() == 0 && etx_lane() == 0) { lids[s] = s; lwts[s] = 1.f; }       // ablation: slot s -> expert s (WRONG output)
  __syncthreads();
#else
  if (d.mode >= 2) dsl_topk_lds(d.logits, d.E, d.R, d.mode == 3 ? 1 : 0, lids, lwts);
#endif
  const __hip_bfloat16* We = d.W + (size_t)(d.mode >= 2 ? lids[s] : dsl_a_ldi(d.ids + s)) * d.estride;     // g: 2 x rpt floats (rpt <= DSL_RCH, host-checked)
  dsl_stage(d.x, d.K, xs);
  dsl_gemv_chunked(We, d.K, xs, r0, r1, part, [&](int r, float v) { g[r - r0] = dsl_bfr(v); });
#ifndef DSL_ABLATE_ONEPASS                                                      // ablation: gate only, no up pass (WRONG output)
  dsl_gemv_chunked(We + (size_t)d.M * d.K, d.K, xs, r0, r1, part, [&](int r, float v) { g[d.rpt + r - r0] = dsl_bfr(v); });
#endif
  __hip_bfloat16* act = d.out + (size_t)s * d.M;
#ifdef DSL_ABLATE_EPI                                                           // ablation: no silu, no multiply (WRONG output)
  for (int r = r0 + (int)threadIdx.x; r < r1; r += blockDim.x) dsl_a_stbf(act + r, __float2bfloat16(g[r - r0]));
#else
  for (int r = r0 + (int)threadIdx.x; r < r1; r += blockDim.x) dsl_a_stbf(act + r, __float2bfloat16(dsl_silu(g[r - r0]) * g[d.rpt + r - r0]));
#endif''',
    "scale": '''
  const int s = blk / d.tpe, j = blk - s * d.tpe;
  const int r0 = j * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
  int* lids = (int*)(xs + d.K + 16 * DSL_RCH); float* lwts = (float*)(lids + 16);
#ifdef DSL_ABLATE_TOPK
  if (etx_wave() == 0 && etx_lane() == 0) { lids[s] = s; lwts[s] = 1.f; }       // ablation: slot s -> expert s (WRONG output)
  __syncthreads();
#else
  if (d.mode >= 2) dsl_topk_lds(d.logits, d.E, d.R, d.mode == 3 ? 1 : 0, lids, lwts);
#endif
  const __hip_bfloat16* We = d.W + (size_t)(d.mode >= 2 ? lids[s] : dsl_a_ldi(d.ids + s)) * d.estride;
  dsl_stage(d.x + (size_t)s * d.K, d.K, xs);
  __hip_bfloat16* out = d.out + (size_t)s * d.M;
#ifdef DSL_ABLATE_EPI                                                           // ablation: no routing-weight multiply (WRONG output)
  dsl_gemv_chunked(We, d.K, xs, r0, r1, xs + d.K, [&](int r, float v) { dsl_a_stbf(out + r, __float2bfloat16(v)); });
#else
  const float ws = d.mode >= 2 ? lwts[s] : dsl_a_ldf(d.wts + s);
  dsl_gemv_chunked(We, d.K, xs, r0, r1, xs + d.K, [&](int r, float v) { dsl_a_stbf(out + r, __float2bfloat16(v * ws)); });
#endif''',
}


def emit_tiles(lds_floats: int = 8192) -> str:
    """The tile source: for every form the plain, XCD-affine (x) and pre-wait entry points the plan links
    (examples/vllm_llm/model.py), reading the argument block like an imported tile; LDS from the import arena."""
    from etx.ops import coh
    out = ["// generated by etx.ops.gemv: the DSL streaming GEMVs", "#pragma once",
           '#include "etx/abi.h"', '#include "etx/import.h"', '#include "etx/tile_util.h"', coh.HELPERS, STREAM]
    for e, form in FORMS.items():
        out.append(f"// ---- {e}: {form}")
        out.append(f"static __device__ __attribute__((always_inline)) void dsl_run_{e}(const unsigned char* a, int blk) {{")
        out.append(f"  const int nblk = *(const int*)(a + {GRID_OFF + 12});")
        out.append("  if (blk >= nblk) return;")
        out.append("  const DslArgs d = dsl_args(a);")
        out.append(f"  float* xs = (float*)etx_lds_arena;   // {lds_floats} floats: x (K), then partials")
        out.append(BODIES[form])
        out.append("}")
        out.append(f'extern "C" __device__ __attribute__((always_inline)) void etx_tile_{e}(const etx_ctx* c) {{')
        out.append(f"  dsl_run_{e}(etx_imp_argblock(c->args, c->cst[0]), c->coord[0]);\n}}")
        out.append(f'extern "C" __device__ __attribute__((always_inline)) void etx_tile_x_{e}(const etx_ctx* c) {{')
        out.append(f"  dsl_run_{e}(etx_imp_argblock(c->args, c->cst[0] + c->coord[0] * c->cst[1]), c->coord[0] * c->cst[2] + c->coord[1]);\n}}")
        out.append(f'extern "C" __device__ __attribute__((always_inline)) void etx_pre_{e}(const etx_ctx* c) {{ etx_imp_touch(etx_imp_argblock(c->args, c->cst[0]), {ARG_BYTES}); }}')
        out.append(f'extern "C" __device__ __attribute__((always_inline)) void etx_pre_x_{e}(const etx_ctx* c) {{ etx_imp_touch(etx_imp_argblock(c->args, c->cst[0] + c->coord[0] * c->cst[1]), {ARG_BYTES}); }}')
        if form in TOUCH_FORMS:
            out.append(f'extern "C" __device__ __attribute__((always_inline)) void etx_touch_{e}(const etx_ctx* c) {{ dsl_touch_plain(etx_imp_argblock(c->args, c->cst[0]), c->coord[0]); }}')
            out.append(f'extern "C" __device__ __attribute__((always_inline)) void etx_touch_x_{e}(const etx_ctx* c) {{ dsl_touch_plain(etx_imp_argblock(c->args, c->cst[0] + c->coord[0] * c->cst[1]), c->coord[0] * c->cst[2] + c->coord[1]); }}')
        out.append("")
    return "\n".join(out) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="emit the DSL GEMV tiles and add their entries to an imports.json")
    ap.add_argument("imports_json")
    ap.add_argument("--tiles", required=True)
    a = ap.parse_args()
    imp = json.loads(Path(a.imports_json).read_text())
    threads = max([i.get("threads", 1024) for i in imp.values()] + [1024])
    imp.update(import_entries(threads))
    Path(a.imports_json).write_text(json.dumps(imp, indent=1))
    Path(a.tiles).write_text(emit_tiles())
    print(f"dsl gemv: {', '.join(FORMS)} -> {a.tiles}; {a.imports_json} updated")


if __name__ == "__main__":
    main()
