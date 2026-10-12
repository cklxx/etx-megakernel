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

MoE residual, measured in-tile (2026-10-02, round 37; traced decode, layer-11 mean bodies, control run twice around
the ladder at 18.32/18.41 up and 10.82/11.39 down; every DSL_ABLATE_* build is timing-only WRONG output):
  rung                       moe_up   moe_down     reads as
  control                     18.37      11.11
  NOGEMV  (topk+stage only)    1.50       1.43     prologue 8%
  ONEPASS (gate, no up)       10.45          -     the second pass costs its own stream, consistent
  NORED   (no wave_sum/epi)   15.45       8.64     reduction + coherent output stores 16% / 22%
  NOLDS   (const x in dot)    17.81      10.68     the dot's LDS reads 3% -- the 8-float lane stride is NOT a conflict
  NOVMEM  (one weight vector) 11.55       7.64     the critical path WITHOUT the stream is 63% of the body
Exposed VMEM (control - NOVMEM) is 6.8 us = 37% for up, 3.5 = 31% for down: the stream hides under the skeleton, so
the budget's "65-70% of achievable bandwidth" framing was wrong -- the tile is skeleton-bound, not stream-bound, and
no single rung owns >= 50%. ~6.5 us/task of the skeleton is loop/flush/drain machinery no rung isolates; separating
it needs in-kernel timestamps, not ablations. Closed as spread at the structural floor of this tile shape.
x and W are bf16, dots accumulate in fp32 (as vLLM's kernels; the summation order differs, so the outputs are compared
with a tolerance, host --check-dsl, not byte for byte).
"""
from __future__ import annotations

import json
from pathlib import Path

import os as _os

# ETX_VL_BATCH: the dense GEMV serves NB activation columns from one weight read (a skinny GEMM). The columns are
# request 0..NB-1 of the same step; the host allocates each batched tensor contiguously across requests, so column b
# is at `ptr + b * stride` -- five strides (x, out, rows, resid_in, resid_out), element counts, not bytes.
#
# MEASURED 2026-10-01, correct but NOT worth enabling at NB = 2 (both requests 32/32 + CHECK-DSL):
#   B=2 batched 7.104 ms/step vs B=2 replicated 6.811 (same build generation) -- batching is 0.29 ms SLOWER.
# Two causes, both from the layer-11 traces:
#   1. the shared weight read buys little. qkv body 12.02 us at NB=1 -> 22.84 at NB=2: the second column costs 90% of
#      the first. NOT because it doubles the FMAs and the LDS reads -- that was asserted here and is wrong. Itemised
#      2026-10-01 for qkv (20 rows/task, K=2048, so 40960 MACs per column): the extra column adds 0.076 us of FMA
#      (256 FMA/clk/CU) and 0.61 us of LDS (128 B/clk/CU), 6.3% of the 10.8 us regression, 0.70% for the FMAs alone.
#      So a matrix instruction cannot pay: it attacks the 0.70%, and at NB=2 it idles 14 of 16 output columns
#      (v_mfma_f32_16x16x16_bf16) or 2 of 4 (v_mfma_f32_4x4x4_16b_bf16, which is what vLLM's own wvSplitK already
#      uses -- and dsl_gemv_rows beats it 6.29 vs 6.75 us on the identical 5120x2048 shape, docs/PHASES-2026-10-01.md).
#      Register pressure is not it either: batched spills 20 against replicated 21, LDS 50216 vs 29736, both leaving
#      one workgroup per CU. The dense PHASES in fact win -- qkv spans 20.9 us batched against 25.9 for the replicated
#      pair; it is cause 2 below that loses the step. Dense bodies over a layer: 37.88 us batched vs 40.78 replicated.
#   2. the join costs more than that. A batched phase waits for EVERY request's previous event, so it cannot start
#      until the slowest request's attention is done: o_proj wait 24.04 us batched vs 12.20 / 6.03 replicated. The
#      replicated graph overlaps request 0's compute with request 1's stalls; batching removes that slack at every
#      dense phase (9 joins per layer).
# So batching pays only if the WHOLE layer batches -- MoE included -- leaving no per-phase join. THAT WAS BUILT AND
# MEASURED TOO (round 33, 2026-10-01): topk batched (one replica set, wave w does column w) and the MoE batched
# (requests x slots tasks, S = s * nb + rq, each task one column), leaving only the o_proj join. B=2 whole-layer
# batched 7.483 ms/step against 6.901 replicated and 7.121 dense-only batched, all 32/32 -- WORSE than batching
# less. The trace says why: the batched moe_up is 384 tasks over 304 workers (48 per XCD on 38 CUs), so ten tasks
# per XCD run as a second serial wave (last-ready spread 28.2 us, phase 40.5 us against a 22.45 us body), and the
# lockstep layer removes the cross-request overlap that replication gets free (replicated r0/r1 phases interleave
# ~12 us apart, each filling the other's stalls). Joins were never the binding cost; worker oversubscription and
# lost overlap are. A union dedup cannot fix that at B=2 (it saves ~3% of expert bytes).
# Also measured and rejected 2026-10-01: restoring the replicated task count at NB=2 (ETX_VL_CU=512, 119956 tasks of
# 10 rows against 83604 of 20) is far worse -- 9.410 ms/step against 7.121 batched and 6.865 replicated, all 32/32.
# Kept behind the switch (default off): correct, gated, and the measured record of why it is off.
DSL_NB = max(1, int(_os.environ.get("ETX_VL_NB", "1")))
# ETX_VL_DSL_EXT (build.sh, from the model's manifest: PhiMoE-class models): appends two pointers AFTER the grid
# words -- `bias` (the GEMV output bias, o_proj / lm_head / the folded qkv) and `lnb` (the LayerNorm bias of the
# folded norm, mode 4). The grid offset does not move and a build without the env is byte-identical, which is the
# Qwen regression gate. The matching device code compiles under ETX_EMIT_DSL_EXT (the feature manifest); model.py
# asserts the imports.json layout and the plan's codegen flag agree, so the env and the manifest cannot drift apart.
DSL_EXT = _os.environ.get("ETX_VL_DSL_EXT", "0") == "1"
ARG_BYTES, GRID_OFF = (176, 144) if DSL_EXT else (160, 144)
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
          ("E", "i32", 108, 4), ("logits", "ptr addrspace(1)", 112, 8),
          ("nb", "i32", 120, 4), ("sx", "i32", 124, 4), ("sout", "i32", 128, 4), ("srows", "i32", 132, 4),
          ("srin", "i32", 136, 4), ("srout", "i32", 140, 4)] + (
         [("bias", "ptr addrspace(1)", 160, 8), ("lnb", "ptr addrspace(1)", 168, 8)] if DSL_EXT else [])
DSL_ROUTER = _os.environ.get("ETX_VL_ROUTER_NARROW", "0") == "1"
FORMS = {"dsl_gemv": "plain", "dsl_moe_up": "silu_mul", "dsl_moe_down": "scale",
         "dsl_gemv_fold": "plain", "dsl_moe_up_fold": "silu_mul", "dsl_moe_down_fold": "scale"}
if DSL_ROUTER:
    if DSL_NB != 1 or DSL_EXT:
        raise ValueError("ETX_VL_ROUTER_NARROW requires NB=1 and no DSL_EXT")
    FORMS["dsl_gemv_router_fold"] = "router"
K_MAX, RCH, THREADS = 4096, 64, 1024                              # x (K floats) + 2 x RCH (moe_up's g) + 16 waves x RCH partials
# NB columns scale x and the partials: B=1 21120 B, B=2 41600 B (megakernel 61928 of 65536), B=4 82560 -- over the limit.
LDS_BYTES = 4 * (DSL_NB * K_MAX + 2 * RCH + DSL_NB * (THREADS // 64) * RCH + 32)


def import_entries(threads: int = 1024) -> dict:
    """imports.json entries for the three forms (what the adapter's host header and model.py read)."""
    ps = [{"ty": ty, "name": f"%{n}", "size": sz, "align": min(sz, 8), "offset": off} for n, (_, ty, off, sz) in enumerate(PARAMS)]
    return {e: {"dsl": form, "export": f"etx_tile_{e}", "kernel": e, "block": [threads, 1, 1], "threads": threads, "slots": 1,
                "slot_threads": threads, "params": ps, "grid_offset": GRID_OFF, "arg_bytes": ARG_BYTES, "lds_bytes_per_slot": LDS_BYTES,
                "lds_bytes": LDS_BYTES, "barriers": 0, "slot_barrier": False, "spec": f"ETX DSL GEMV ({form})"} for e, form in FORMS.items()}


STREAM = r'''
@@ETX_PRIM:uniform@@
// ---- the stream loop (examples/llm/gemv.h): segments of 64 lanes x EPL bf16 inside one row, U loads per lane in flight
#ifndef DSL_U32X4_DEFINED                                 // also defined by etx/ops/coh.py, whichever is emitted first
#define DSL_U32X4_DEFINED
typedef unsigned int dsl_u32x4 __attribute__((ext_vector_type(4)));
#endif
typedef unsigned int dsl_u32x2 __attribute__((ext_vector_type(2)));
template <int EPL> struct dsl_wvec;
template <> struct dsl_wvec<8> { typedef dsl_u32x4 T; };
template <> struct dsl_wvec<4> { typedef dsl_u32x2 T; };
#ifndef DSL_G
#define DSL_G __attribute__((address_space(1)))          // global loads, not flat (flat waits on the LDS counter too)
#endif
// ETX_DSL_NT: weight loads marked non-temporal. Weights are read exactly once per token (6.08 GB), so they have no
// reuse to preserve, yet they pass through the XCD's 4 MB L2 and the 256 MB MALL and evict what does have reuse (the
// 104 MB KV cache, the activations). Measured L2 hit rate without it: 31.9 % (rocprofv3 TCC_HIT_sum / (HIT+MISS)).
template <class T> static __device__ __forceinline__ T dsl_wload(const DSL_G T* p) {
#ifdef ETX_DSL_NT
  return __builtin_nontemporal_load(p);
#else
  return *p;
#endif
}
typedef __attribute__((address_space(3))) const float dsl_lds_f;
typedef __attribute__((address_space(3))) float dsl_lds_fw;
static __device__ __forceinline__ float dsl_dot(dsl_u32x4 w, const dsl_lds_f* x) {
  float s = 0.f;
#ifdef DSL_ABLATE_NOLDS                                  // ablation: constants in place of the staged x (WRONG output) --
#pragma unroll                                           // isolates the LDS read cost of the dot (lane-strided, 8 floats)
  for (int i = 0; i < 4; ++i) { s += __uint_as_float(w[i] << 16) * 0.5f; s += __uint_as_float(w[i] & 0xffff0000u) * 0.25f; }
#else
#pragma unroll
  for (int i = 0; i < 4; ++i) { s += __uint_as_float(w[i] << 16) * x[2 * i]; s += __uint_as_float(w[i] & 0xffff0000u) * x[2 * i + 1]; }
#endif
  return s;
}
static __device__ __forceinline__ float dsl_dot(dsl_u32x2 w, const dsl_lds_f* x) {
#ifdef DSL_ABLATE_NOLDS
  return __uint_as_float(w.x << 16) * 0.5f + __uint_as_float(w.x & 0xffff0000u) * 0.25f
       + __uint_as_float(w.y << 16) * 0.5f + __uint_as_float(w.y & 0xffff0000u) * 0.25f;
#else
  return __uint_as_float(w.x << 16) * x[0] + __uint_as_float(w.x & 0xffff0000u) * x[1]
       + __uint_as_float(w.y << 16) * x[2] + __uint_as_float(w.y & 0xffff0000u) * x[3];
#endif
}
#ifdef ETX_DSL_WSUM
@@ETX_PRIM:wave_sum0@@
#endif
static __device__ __forceinline__ float dsl_bfr(float v) { return __bfloat162float(__float2bfloat16(v)); }
static __device__ __forceinline__ float dsl_silu(float g) { return g / (1.f + __expf(-g)); }
// rows [r0, r1) of W (row-major bf16, K columns) times xs (fp32 in LDS); part: LDS, >= nwaves * (r1 - r0) floats;
// epi(row, dot) once per row by one thread. Deterministic: per-wave partials are summed in wave order.
// NB columns: the weight vector is loaded ONCE and dotted with NB staged activation vectors (column b at xs + b * KS),
// giving NB partials per (wave, row). NB = 1 is the batch-1 GEMV and collapses to the original indexing.
// ETX_DSL_GU1: rows [gsplit, r1 - r0) of the pass sit ggap elements further on (moe_up's up rows behind its gate rows),
// so gate and up stream as ONE pass. The caller guarantees no wave's segment range crosses gsplit (one base per wave).
template <int EPL, int U, int NB, class Epi>
static __device__ __forceinline__ void dsl_gemv_rows_nb(const __hip_bfloat16* __restrict__ W, int K, int KS, const float* xs, int r0, int r1, float* part, Epi epi
#ifdef ETX_DSL_GU1
                                                        , int gsplit = 0, long long ggap = 0
#endif
                                                        ) {
  typedef typename dsl_wvec<EPL>::T V;
  constexpr int SEG = ETX_WAVE_SIZE * EPL;
  const dsl_lds_f* xl = (const dsl_lds_f*)xs;
  dsl_lds_fw* pl = (dsl_lds_fw*)part;
#ifdef ETX_DSL_SCALAR                                    // all of these are the same in every lane: say so to the compiler
  K = etx_uniform(K); r0 = etx_uniform(r0); r1 = etx_uniform(r1);
  const int lane = etx_lane(), wv = etx_uniform(etx_wave()), nw = etx_nwaves(), nrt = r1 - r0;
#else
  const int lane = etx_lane(), wv = etx_wave(), nw = etx_nwaves(), nrt = r1 - r0;
#endif
  const int spr = K / SEG, S = nrt * spr;
  const int per = (S + nw - 1) / nw, sb = wv * per, se = min(S, sb + per);
  for (int i = (int)threadIdx.x; i < nw * nrt * NB; i += blockDim.x) pl[i] = 0.f;
  __syncthreads();
#if defined(ETX_DSL_SCALAR) || defined(ETX_DSL_GU1)
  const DSL_G V* ubase = (const DSL_G V*)(W + (size_t)r0 * K
#ifdef ETX_DSL_GU1
                                          + ((gsplit > 0 && sb >= gsplit * spr) ? ggap : 0)
#endif
                                          );
#ifdef DSL_ABLATE_NOVMEM
  const DSL_G V* base = ubase + lane;
#endif
#define DSL_WP(sg) (ubase + (size_t)(sg) * ETX_WAVE_SIZE + lane)   // uniform part first: one scalar base per trip
#else
  const DSL_G V* base = (const DSL_G V*)(W + (size_t)r0 * K) + lane;
#define DSL_WP(sg) (base + (size_t)(sg) * ETX_WAVE_SIZE)
#endif
  int cur = sb / spr;
  float acc[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) acc[b] = 0.f;
  auto flush = [&]() {
#ifndef DSL_ABLATE_NORED                                 // ablation: no per-row wave_sum, no partial writes (WRONG output)
#pragma unroll
#ifdef ETX_DSL_WSUM
    for (int b = 0; b < NB; ++b) { const float v = dsl_wave_sum0(acc[b]); if (lane == 0) pl[(wv * nrt + cur) * NB + b] += v; acc[b] = 0.f; }
#else
    for (int b = 0; b < NB; ++b) { const float v = etx_wave_sum(acc[b]); if (lane == 0) pl[(wv * nrt + cur) * NB + b] += v; acc[b] = 0.f; }
#endif
#endif
  };
#ifdef ETX_DSL_SCALAR
  // (row, segment) advanced by one per consumed vector: segments are consumed in order, so this is sg / spr and
  // sg % spr without the division (dsl_gemv_accum's walk, minus its second reduction)
  int row = cur, seg = sb - cur * spr;
  auto consume = [&](int, V w) {
    if (row != cur) { flush(); cur = row; }
    const int off = seg * SEG + lane * EPL;
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[b] += dsl_dot(w, xl + b * KS + off);
    if (++seg == spr) { seg = 0; ++row; }
  };
#else
  auto consume = [&](int sg, V w) {
    const int row = sg / spr;
    if (row != cur) { flush(); cur = row; }
    const int off = (sg - row * spr) * SEG + lane * EPL;
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[b] += dsl_dot(w, xl + b * KS + off);   // one load, NB dots: the point of the batch
  };
#endif
  int s0 = sb;
  V w[U];
#ifdef DSL_ABLATE_NOVMEM                                 // ablation: ONE weight vector per wave, reused every trip (WRONG
  const V w00 = sb < se ? dsl_wload(base + (size_t)sb * ETX_WAVE_SIZE) : V{};   // output) -- loop, LDS reads and dots intact
#endif
  for (; s0 + U <= se; s0 += U) {
#pragma unroll
#ifdef DSL_ABLATE_NOVMEM
    for (int u = 0; u < U; ++u) w[u] = w00;
#else
    for (int u = 0; u < U; ++u) w[u] = dsl_wload(DSL_WP(s0 + u));
#endif
#pragma unroll
    for (int u = 0; u < U; ++u) consume(s0 + u, w[u]);
  }
  if (s0 < se) {
#pragma unroll
#ifdef DSL_ABLATE_NOVMEM
    for (int u = 0; u < U; ++u) if (s0 + u < se) w[u] = w00;
#else
    for (int u = 0; u < U; ++u) if (s0 + u < se) w[u] = dsl_wload(DSL_WP(s0 + u));
#endif
#pragma unroll
    for (int u = 0; u < U; ++u) if (s0 + u < se) consume(s0 + u, w[u]);
  }
  if (sb < se) flush();
#ifdef DSL_ABLATE_NORED
#pragma unroll
  for (int b = 0; b < NB; ++b) asm volatile("" :: "v"(acc[b]));   // the dots are kept; the reduction and epi are not
  __syncthreads();
  __syncthreads();
#else
  __syncthreads();
  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) {
    float v[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) v[b] = 0.f;
    for (int w2 = 0; w2 < nw; ++w2)
#pragma unroll
      for (int b = 0; b < NB; ++b) v[b] += pl[(w2 * nrt + r) * NB + b];
    epi(r0 + r, v);
  }
  __syncthreads();
#endif
#undef DSL_WP
}
#ifdef ETX_DSL_ROW_WAVE
// One wave owns whole rows: same loads, same segment order, same lane offsets, so the sum is the
// same bit pattern. What goes away is the partial matrix (zeroing, write, gather) and one barrier.
// The GU1 offset becomes a per-row condition: a wave covers segments [0, spr) of a row, so
// seg >= gsplit * spr is true exactly when the row index reaches gsplit.
template <int EPL, int NB, class Epi>
static __device__ __forceinline__ void dsl_gemv_rows_rw(const __hip_bfloat16* __restrict__ W, int K, int KS,
                                                        const float* xs, int r0, int r1, Epi epi
#ifdef ETX_DSL_GU1
                                                        , int gsplit = 0, long long ggap = 0
#endif
                                                        ) {
  typedef typename dsl_wvec<EPL>::T V;
  constexpr int SEG = ETX_WAVE_SIZE * EPL;
  const dsl_lds_f* xl = (const dsl_lds_f*)xs;
#ifdef ETX_DSL_SCALAR
  K = etx_uniform(K); r0 = etx_uniform(r0); r1 = etx_uniform(r1);
  const int lane = etx_lane(), wv = etx_uniform(etx_wave()), nw = etx_uniform(etx_nwaves()), nrt = r1 - r0;
#else
  const int lane = etx_lane(), wv = etx_wave(), nw = etx_nwaves(), nrt = r1 - r0;
#endif
  const int spr = K / SEG;
  const int per = (nrt + nw - 1) / nw, rb = wv * per, re2 = min(nrt, rb + per);
  const DSL_G V* base0 = (const DSL_G V*)(W + (size_t)r0 * K);
  for (int r = rb; r < re2; ++r) {
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[b] = 0.f;
    long long gg = 0;
#ifdef ETX_DSL_GU1
    if (gsplit > 0 && r >= gsplit) gg = ggap;
#endif
    const DSL_G V* rbase = base0 + (size_t)r * K + gg;
    for (int sg = 0; sg < spr; ++sg) {
      const V w = dsl_wload(rbase + (size_t)sg * ETX_WAVE_SIZE + lane);
      const int off = sg * SEG + lane * EPL;
#pragma unroll
      for (int b = 0; b < NB; ++b) acc[b] += dsl_dot(w, xl + b * KS + off);
    }
    float v[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
#ifdef ETX_DSL_WSUM
      v[b] = dsl_wave_sum0(acc[b]);
#else
      v[b] = etx_wave_sum(acc[b]);
#endif
    }
    if (lane == 0) epi(r0 + r, v);
  }
  __syncthreads();
}
#endif
template <int EPL, int U, class Epi>
static __device__ __forceinline__ void dsl_gemv_rows(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  dsl_gemv_rows_nb<EPL, U, 1>(W, K, 0, xs, r0, r1, part, [&](int r, const float* v) { epi(r, v[0]); });
}
// DSL_SPLIT_TAIL: the same stream loop with the row stride (ld) separated from the segment span (kcols), so a row can
// be swept in pieces. dsl_gemv_rows' flat `base[sg * 64]` indexing only works when the row stride equals spr * SEG;
// here the (row, segment) pair is tracked incrementally instead, which costs one add and one compare per load and no
// division. Accumulates into the caller's partials; the caller zeroes and reduces them.
// MEASURED 2026-10-01 and NOT worth enabling: moe_down (K = 768) on 512 wide + 256 narrow instead of three 256-wide
// segments made its body 12.01 -> 14.71 us/layer, the step 5.056 -> 5.200 ms/token (3 runs each, spread 0.006), while
// every other phase was unchanged. Round 7's ablation bounded the win at 0.021 ms/token; the per-load address tracking
// and the second per-row wave reduction cost more than the wider loads save. Correct (32/32 + CHECK-DSL) but slower.
template <int EPL, int U>
static __device__ __forceinline__ void dsl_gemv_accum(const __hip_bfloat16* __restrict__ W, int ld, int kcols, int k0,
                                                      const float* xs, int r0, int r1, float* part) {
  typedef typename dsl_wvec<EPL>::T V;
  constexpr int SEG = ETX_WAVE_SIZE * EPL;
  const dsl_lds_f* xl = (const dsl_lds_f*)xs;
  dsl_lds_fw* pl = (dsl_lds_fw*)part;
  const int lane = etx_lane(), wv = etx_wave(), nw = etx_nwaves(), nrt = r1 - r0;
  const int spr = kcols / SEG, S = nrt * spr;
  const int per = (S + nw - 1) / nw, sb = wv * per, se = min(S, sb + per);
  if (sb >= se) return;                                  // wave-uniform: spr, sb, se come from wv only
  int row = sb / spr, sw = sb - row * spr, cur = row;    // the only division, once per wave
  const __hip_bfloat16* Wr = W + (size_t)(r0 + row) * (size_t)ld + k0 + lane * EPL;
  float acc = 0.f;
  for (int s0 = sb; s0 < se; ) {
    const int n = min(U, se - s0);
    V w[U];
    int lr = row, ls = sw; const __hip_bfloat16* lw = Wr;   // walk ahead to issue n loads, then walk again to consume
#pragma unroll
    for (int u = 0; u < U; ++u) if (u < n) {
      w[u] = dsl_wload((const DSL_G V*)(lw + ls * SEG));
      if (++ls == spr) { ls = 0; ++lr; lw += ld; }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) if (u < n) {
      if (row != cur) { const float v = etx_wave_sum(acc); if (lane == 0) pl[wv * nrt + cur] += v; acc = 0.f; cur = row; }
      acc += dsl_dot(w[u], xl + k0 + sw * SEG + lane * EPL);
      if (++sw == spr) { sw = 0; ++row; Wr += ld; }
    }
    s0 += n;
  }
  const float v = etx_wave_sum(acc); if (lane == 0) pl[wv * nrt + cur] += v;
}
// K = q * 512 + 256 (moe_down: 768): the first q * 512 columns on the 16 B path, the 256-column tail on the 8 B one,
// both accumulating into the same per-(wave, row) partials. Deterministic: fixed pass order, partials summed in wave order.
template <class Epi>
static __device__ __forceinline__ void dsl_gemv_split(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  constexpr int WIDE = ETX_WAVE_SIZE * 8;
  const int nw = etx_nwaves(), nrt = r1 - r0, KW = (K / WIDE) * WIDE;
  dsl_lds_fw* pl = (dsl_lds_fw*)part;
  for (int i = (int)threadIdx.x; i < nw * nrt; i += blockDim.x) pl[i] = 0.f;
  __syncthreads();
  dsl_gemv_accum<8, 4>(W, K, KW, 0, xs, r0, r1, part);
  dsl_gemv_accum<4, 4>(W, K, K - KW, KW, xs, r0, r1, part);
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
// Half in-flight depth (8 dwords per lane) is the DEFAULT since round 34: 4.832 against 4.884 wall ms/token for
// the full depth (3 interleaved runs each, same box), VGPR/spill/occupancy identical (128/8/4) -- the win is VMEM
// queue pressure and 1 MB less code, not registers. Round 32's Little's-law headroom at full depth was 4.8x, so
// half still covers the latency 2.4x over. Occupancy itself CANNOT move here: the persistent kernel launches one
// 16-wave workgroup per CU, so 4 waves/SIMD is fixed by the grid, not by VGPRs -- forcing VGPR 64 via launch
// bounds left runtime occupancy at 4.000 and cost 2x in spill traffic (9.39 wall, 1678 spills). -DETX_DSL_U4
// restores the old depth.
#ifdef ETX_DSL_U4
#define DSL_UW 4
#define DSL_UN 8
#else
#define DSL_UW 2
#define DSL_UN 4
#endif
template <class Epi>
static __device__ __forceinline__ void dsl_gemv(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
#ifdef DSL_ABLATE_EPL4                                   // ablation: force the 8 B load path everywhere (moe_down's,
  dsl_gemv_rows<4, DSL_UN>(W, K, xs, r0, r1, part, epi); // since K = 768 is not a multiple of 512) -- output stays correct
#else
  if (K % (ETX_WAVE_SIZE * 8) == 0) dsl_gemv_rows<8, DSL_UW>(W, K, xs, r0, r1, part, epi);
#ifdef DSL_SPLIT_TAIL                                    // moe_down (K = 768): 512 columns wide, 256 narrow
  else if (K >= ETX_WAVE_SIZE * 8 && K % (ETX_WAVE_SIZE * 4) == 0) dsl_gemv_split(W, K, xs, r0, r1, part, epi);
#endif
  else dsl_gemv_rows<4, DSL_UN>(W, K, xs, r0, r1, part, epi);
#endif
}
// rows per pass of the stream loop: the partials take nwaves x DSL_RCH floats of LDS whatever the task's row count
// (lm_head tasks hold 500 rows; unbounded partials overran the 32 KB arena into the barrier words, review 2026-09-29)
#define DSL_RCH 64
// rows per batch of the norm fold's coherent loads (ETX_DSL_RBATCH): one batch for every top-k we build (2 or 8)
#define DSL_RB 8
template <int NB, class Epi>
static __device__ __forceinline__ void dsl_gemv_nb(const __hip_bfloat16* __restrict__ W, int K, int KS, const float* xs, int r0, int r1, float* part, Epi epi) {
  if (K % (ETX_WAVE_SIZE * 8) == 0) dsl_gemv_rows_nb<8, DSL_UW, NB>(W, K, KS, xs, r0, r1, part, epi);
  else dsl_gemv_rows_nb<4, DSL_UN, NB>(W, K, KS, xs, r0, r1, part, epi);
}
template <int NB, class Epi>
static __device__ __forceinline__ void dsl_gemv_chunked_nb(const __hip_bfloat16* __restrict__ W, int K, int KS, const float* xs, int r0, int r1, float* part, Epi epi) {
  for (int r = r0; r < r1; r += DSL_RCH) dsl_gemv_nb<NB>(W, K, KS, xs, r, min(r1, r + DSL_RCH), part, epi);
}
// NB activation vectors into LDS, column b at xs + b * KS (KS = the staged stride, K rounded to the LDS layout)
static __device__ __forceinline__ void dsl_stage_nb(const __hip_bfloat16* __restrict__ x, int K, int sx, int nb, int KS, float* xs);
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
static __device__ __forceinline__ void dsl_stage_nb(const __hip_bfloat16* __restrict__ x, int K, int sx, int nb, int KS, float* xs) {
  for (int b = 0; b < nb; ++b) dsl_stage(x + (size_t)b * sx, K, xs + b * KS);   // each ends in a barrier: all waves fill every column
}
template <class Epi>
static __device__ __forceinline__ void dsl_gemv_chunked(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  for (int r = r0; r < r1; r += DSL_RCH) dsl_gemv(W, K, xs, r, min(r1, r + DSL_RCH), part, epi);
}
#if ETX_EMIT_DSL_BIGK
// K > DSL_K_MAX (Mixtral moe_down: 14336): x staged in DSL_K_MAX-column chunks, every chunk's partial dots
// accumulated into the same per-(wave, row) LDS array, the epilogue once per row chunk after the last column chunk.
// LDS unchanged (the x region holds one chunk); the weight walk per chunk is today's walk for a K-chunk matrix whose
// row stride is the full K (dsl_gemv_accum separates ld from kcols), so contiguity within a segment is untouched.
// Deterministic: fixed chunk order, partials accumulated in k order per wave, final sum in wave order. 16 B loads for
// the 512-aligned span of each chunk, the 8 B path for a %512 tail (a tail exists only in the last chunk: chunks are
// DSL_K_MAX = 4096 columns, and K % 256 == 0 is host-checked). Costs one stage + one barrier per (row chunk, column
// chunk) -- cheap when rows-per-task <= DSL_RCH (one row chunk, the MoE shapes), quadratic-ish beyond: the dense
// forms keep the host's K <= DSL_K_MAX guard, only the MoE forms route here.
template <class Epi>
static __device__ __forceinline__ void dsl_gemv_bigk(const __hip_bfloat16* __restrict__ W, const __hip_bfloat16* __restrict__ x,
                                                     int K, float* xs, int r0, int r1, float* part, Epi epi) {
  constexpr int WIDE = ETX_WAVE_SIZE * 8;
  const int nw = etx_nwaves();
  dsl_lds_fw* pl = (dsl_lds_fw*)part;
  for (int r = r0; r < r1; r += DSL_RCH) {
    const int rr = min(r1, r + DSL_RCH), nrt = rr - r;
    for (int i = (int)threadIdx.x; i < nw * nrt; i += blockDim.x) pl[i] = 0.f;
    for (int k0 = 0; k0 < K; k0 += DSL_K_MAX) {
      const int kc = min(DSL_K_MAX, K - k0), KW = (kc / WIDE) * WIDE;
      dsl_stage(x + k0, kc, xs);                       // ends in __syncthreads: zeroed partials + staged chunk visible
      dsl_gemv_accum<8, 4>(W + k0, K, KW, 0, xs, r, rr, part);
      dsl_gemv_accum<4, 4>(W + k0, K, kc - KW, KW, xs, r, rr, part);
      __syncthreads();                                 // partials written, xs free for the next chunk
    }
    for (int q = (int)threadIdx.x; q < nrt; q += blockDim.x) {
      float v = 0.f;
      for (int w2 = 0; w2 < nw; ++w2) v += pl[w2 * nrt + q];
      epi(r + q, v);
    }
    __syncthreads();
  }
}
#endif
struct DslArgs { const __hip_bfloat16* W; const __hip_bfloat16* x; __hip_bfloat16* out; const int* ids; const float* wts; long long estride; int M, K, rpt, tpe;
                 const __hip_bfloat16* rows; const __hip_bfloat16* resid_in; __hip_bfloat16* resid_out; const __hip_bfloat16* ln; int R; float eps; int mode, E;
                 const __hip_bfloat16* logits; int nb, sx, sout, srows, srin, srout;
#if ETX_EMIT_DSL_EXT
                 const __hip_bfloat16* bias; const __hip_bfloat16* lnb;   // output bias; LayerNorm bias (mode 4)
#endif
};
// ETX_DSL_SCALAR: the argument block is read with scalar loads, as the imported tiles read theirs (etx_imp_argblock),
// so every size and pointer is wave-uniform for the compiler. Through the generic pointer they were flat loads into
// VGPRs, which the divergence analysis treats as per-lane values: the stream loop's row / segment arithmetic (a 17-
// instruction signed division per segment) and its branches all ran on the vector ALU, under exec masks.
// The pre-wait hook (etx_pre_*) already pulls these lines into the K$; only this form reads them from there.
#ifdef ETX_DSL_SCALAR
#define DSL_ARG_AS __attribute__((address_space(4)))
static __device__ __forceinline__ DslArgs dsl_args(const unsigned char* a_) {
  const unsigned char DSL_ARG_AS* a = (const unsigned char DSL_ARG_AS*)(uintptr_t)a_;   // a_ is wave-uniform (etx_imp_argblock)
#else
#define DSL_ARG_AS
static __device__ __forceinline__ DslArgs dsl_args(const unsigned char* a) {
#endif
  DslArgs d;
  d.W = *(const __hip_bfloat16* const DSL_ARG_AS*)(a + 0); d.x = *(const __hip_bfloat16* const DSL_ARG_AS*)(a + 8); d.out = *(__hip_bfloat16* const DSL_ARG_AS*)(a + 16);
  d.ids = *(const int* const DSL_ARG_AS*)(a + 24); d.wts = *(const float* const DSL_ARG_AS*)(a + 32); d.estride = *(const long long DSL_ARG_AS*)(a + 40);
  d.M = *(const int DSL_ARG_AS*)(a + 48); d.K = *(const int DSL_ARG_AS*)(a + 52); d.rpt = *(const int DSL_ARG_AS*)(a + 56); d.tpe = *(const int DSL_ARG_AS*)(a + 60);
  d.rows = *(const __hip_bfloat16* const DSL_ARG_AS*)(a + 64); d.resid_in = *(const __hip_bfloat16* const DSL_ARG_AS*)(a + 72); d.resid_out = *(__hip_bfloat16* const DSL_ARG_AS*)(a + 80);
  d.ln = *(const __hip_bfloat16* const DSL_ARG_AS*)(a + 88); d.R = *(const int DSL_ARG_AS*)(a + 96); d.eps = *(const float DSL_ARG_AS*)(a + 100); d.mode = *(const int DSL_ARG_AS*)(a + 104);
  d.E = *(const int DSL_ARG_AS*)(a + 108); d.logits = *(const __hip_bfloat16* const DSL_ARG_AS*)(a + 112);
  d.nb = *(const int DSL_ARG_AS*)(a + 120); d.sx = *(const int DSL_ARG_AS*)(a + 124); d.sout = *(const int DSL_ARG_AS*)(a + 128);
  d.srows = *(const int DSL_ARG_AS*)(a + 132); d.srin = *(const int DSL_ARG_AS*)(a + 136); d.srout = *(const int DSL_ARG_AS*)(a + 140);
#if ETX_EMIT_DSL_EXT
  d.bias = *(const __hip_bfloat16* const DSL_ARG_AS*)(a + 160); d.lnb = *(const __hip_bfloat16* const DSL_ARG_AS*)(a + 168);
#endif
  return d;
}
// norm fold: xs = bf16(rms_norm(z)) as floats, z = bf16(bf16(sum of R rows) + resid_in) per element; task 0 stores z
template <int FixedK = 0, int FixedR = -1>
static __device__ __forceinline__ void dsl_stage_norm_at(const DslArgs& d, const __hip_bfloat16* rows, const __hip_bfloat16* rin,
                                                        __hip_bfloat16* rout, __hip_bfloat16* xout, float* xs, float* red, bool store) {
  const int K = FixedK ? FixedK : d.K;
  const int R = FixedR >= 0 ? FixedR : d.R;
  float ss = 0.f;
#ifdef ETX_DSL_VECNORM
  // 8 elements per thread per trip: the per-element accessors issue one coherent request each (K=2048 gives 4096 of
  // them against o_proj's 1024 wide ones, round 23). The R reduction keeps its per-element order; the sum of squares
  // is accumulated over a different partition of the elements, which is a fp32 rounding difference, not a reordering
  // of a dependence. Needs K % 8 == 0, true for every shape we build and asserted by the host.
  for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x) {
    float z[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) z[j] = 0.f;
#ifdef ETX_DSL_RBATCH
    // ETX_DSL_RBATCH: the residual and up to DSL_RB rows issued before the first is used. The runtime-R loop below
    // kept one row's pair of coherent loads in flight and waited for it (s_waitcnt vmcnt(0)) before issuing the
    // next: R + 1 serial round trips past L2 per trip, 9 for qkv's R = 8. Same additions, same order.
    const dsl_u32x4 ri = dsl_a_ld128(rin + 8 * i);
    for (int s0 = 0; s0 < R; s0 += DSL_RB) {
      dsl_u32x4 rv[DSL_RB];
#pragma unroll
      for (int u = 0; u < DSL_RB; ++u) if (s0 + u < R) rv[u] = dsl_a_ld128(rows + (size_t)(s0 + u) * K + 8 * i);
#pragma unroll
      for (int u = 0; u < DSL_RB; ++u) if (s0 + u < R) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          z[2 * j] += __uint_as_float(rv[u][j] << 16); z[2 * j + 1] += __uint_as_float(rv[u][j] & 0xffff0000u);
        }
      }
    }
#else
    for (int s = 0; s < R; ++s) {
      const dsl_u32x4 r = dsl_a_ld128(rows + (size_t)s * K + 8 * i);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        z[2 * j] += __uint_as_float(r[j] << 16); z[2 * j + 1] += __uint_as_float(r[j] & 0xffff0000u);
      }
    }
    const dsl_u32x4 ri = dsl_a_ld128(rin + 8 * i);
#endif
    dsl_u32x4 ro;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a0 = R > 1 ? dsl_bfr(z[2 * j]) : z[2 * j], a1 = R > 1 ? dsl_bfr(z[2 * j + 1]) : z[2 * j + 1];
      z[2 * j] = dsl_bfr(a0 + __uint_as_float(ri[j] << 16));
      z[2 * j + 1] = dsl_bfr(a1 + __uint_as_float(ri[j] & 0xffff0000u));
      ro[j] = (__float_as_uint(z[2 * j]) >> 16) | (__float_as_uint(z[2 * j + 1]) & 0xffff0000u);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) { xs[8 * i + j] = z[j]; ss += z[j] * z[j]; }
    if (store && rout != nullptr) dsl_a_st128(rout + 8 * i, ro);
  }
#ifdef ETX_DSL_NORM_HOIST
  // Immutable norm weights: one 16-byte vector per active thread, live across
  // the block sum's two joins. A compiler memory fence prevents sinking;
  // it emits no device fence and preserves the original cached load policy.
  // Keep the original load for later trips when K > 8 * blockDim.x.
  dsl_u32x4 norm_w = {};
  if ((int)threadIdx.x < K / 8)
    norm_w = *(const DSL_G dsl_u32x4*)(d.ln + 8 * (int)threadIdx.x);
  asm volatile("" ::: "memory");
#endif
#if ETX_EMIT_DSL_EXT
  // mode 4: LayerNorm (PhiMoE): mean-centred, biased -- out = (z - mean) * rsqrt(var + eps) * w + b, fp32 inside,
  // one bf16 rounding, as torch's nn.LayerNorm on a bf16 tensor. mode 1 keeps the RMS path bit-for-bit.
  float mean = 0.f;
  if (d.mode == 4) {
    __syncthreads();                    // mode 4 reads the WHOLE staged xs[] (mean), written by every thread --
                                        // mode 1 never needed this: each thread reduces only its own ss during
                                        // staging. Without it the mean reads unwritten LDS (NaN at full occupancy).
    float s1 = 0.f;
    for (int i = (int)threadIdx.x; i < K; i += blockDim.x) s1 += xs[i];
    mean = etx_block_sum(s1, red) / (float)K;
    float s2 = 0.f;
    for (int i = (int)threadIdx.x; i < K; i += blockDim.x) { const float zc = xs[i] - mean; s2 += zc * zc; }
    ss = etx_block_sum(s2, red);
  } else
#endif
  ss = etx_block_sum(ss, red);
  const float sv = rsqrtf(ss / (float)K + d.eps);
  for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x) {
#ifdef ETX_DSL_NORM_HOIST
    const dsl_u32x4 w = i == (int)threadIdx.x ? norm_w : *(const DSL_G dsl_u32x4*)(d.ln + 8 * i);
#else
    const dsl_u32x4 w = *(const DSL_G dsl_u32x4*)(d.ln + 8 * i);          // the norm weight is read-only: stays cached
#endif
    dsl_u32x4 o;
#if ETX_EMIT_DSL_EXT
    if (d.mode == 4) {
      const dsl_u32x4 bq = *(const DSL_G dsl_u32x4*)(d.lnb + 8 * i);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float v0 = dsl_bfr((xs[8 * i + 2 * j] - mean) * sv * __uint_as_float(w[j] << 16) + __uint_as_float(bq[j] << 16));
        const float v1 = dsl_bfr((xs[8 * i + 2 * j + 1] - mean) * sv * __uint_as_float(w[j] & 0xffff0000u) + __uint_as_float(bq[j] & 0xffff0000u));
        xs[8 * i + 2 * j] = v0; xs[8 * i + 2 * j + 1] = v1;
        o[j] = (__float_as_uint(v0) >> 16) | (__float_as_uint(v1) & 0xffff0000u);
      }
      if (store && xout != nullptr) dsl_a_st128(xout + 8 * i, o);
      continue;
    }
#endif
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dsl_bfr(xs[8 * i + 2 * j] * sv * __uint_as_float(w[j] << 16));
      const float v1 = dsl_bfr(xs[8 * i + 2 * j + 1] * sv * __uint_as_float(w[j] & 0xffff0000u));
      xs[8 * i + 2 * j] = v0; xs[8 * i + 2 * j + 1] = v1;
      o[j] = (__float_as_uint(v0) >> 16) | (__float_as_uint(v1) & 0xffff0000u);
    }
    if (store && xout != nullptr) dsl_a_st128(xout + 8 * i, o);
  }
#else
  for (int i = (int)threadIdx.x; i < K; i += blockDim.x) {
    float acc = 0.f;
    for (int s = 0; s < R; ++s) acc += dsl_a_ldbf(rows + (size_t)s * K + i);
    const float in = R > 1 ? dsl_bfr(acc) : acc;                                   // moe_sum rounds its fp32 sum once
    const float z = dsl_bfr(in + dsl_a_ldbf(rin + i));                        // the bf16 add of fused_add_rms_norm
    xs[i] = z; ss += z * z;
    if (store && rout != nullptr) dsl_a_stbf(rout + i, __float2bfloat16(z));
  }
#if ETX_EMIT_DSL_EXT
  float mean = 0.f;
  if (d.mode == 4) {
    __syncthreads();                    // mode 4 reads the WHOLE staged xs[] (mean), written by every thread --
                                        // mode 1 never needed this: each thread reduces only its own ss during
                                        // staging. Without it the mean reads unwritten LDS (NaN at full occupancy).
    float s1 = 0.f;
    for (int i = (int)threadIdx.x; i < K; i += blockDim.x) s1 += xs[i];
    mean = etx_block_sum(s1, red) / (float)K;
    float s2 = 0.f;
    for (int i = (int)threadIdx.x; i < K; i += blockDim.x) { const float zc = xs[i] - mean; s2 += zc * zc; }
    ss = etx_block_sum(s2, red);
  } else
#endif
  ss = etx_block_sum(ss, red);
  const float sv = rsqrtf(ss / (float)K + d.eps);
  for (int i = (int)threadIdx.x; i < K; i += blockDim.x) {
#if ETX_EMIT_DSL_EXT
    xs[i] = d.mode == 4 ? dsl_bfr((xs[i] - mean) * sv * __bfloat162float(d.ln[i]) + __bfloat162float(d.lnb[i]))
                        : dsl_bfr(xs[i] * sv * __bfloat162float(d.ln[i]));
#else
    xs[i] = dsl_bfr(xs[i] * sv * __bfloat162float(d.ln[i]));
#endif
    if (store && xout != nullptr) dsl_a_stbf(xout + i, __float2bfloat16(xs[i]));   // the normed vector, for later consumers
  }
#endif
  __syncthreads();
}
static __device__ __forceinline__ void dsl_stage_norm(const DslArgs& d, float* xs, float* red, bool store) {
  dsl_stage_norm_at(d, d.rows, d.resid_in, d.resid_out, (__hip_bfloat16*)d.x, xs, red, store);
}
// NB columns of the norm fold: column b takes request b's rows / residual and writes its own normed vector
static __device__ __forceinline__ void dsl_stage_norm_nb(const DslArgs& d, int nb, int KS, float* xs, float* red, bool store) {
  for (int b = 0; b < nb; ++b)
    dsl_stage_norm_at(d, d.rows + (size_t)b * d.srows, d.resid_in + (size_t)b * d.srin,
                      d.resid_out ? d.resid_out + (size_t)b * d.srout : nullptr,
                      d.x ? (__hip_bfloat16*)d.x + (size_t)b * d.sx : nullptr, xs + b * KS, red, store);
}
// top-k fold: wave 0 computes the softmax top-k of the logits (etx/ops/topk.py's arithmetic) into LDS
static __device__ __forceinline__ void dsl_topk_lds(const __hip_bfloat16* logits, int E, int K, int renorm, int* ids, float* wts) {
  if (etx_wave() == 0) {
    const int lane = etx_lane();
    float p[2]; bool taken[2]; float m = -INFINITY;
#pragma unroll
    for (int k = 0; k < 2; ++k) { const int e = lane + k * ETX_WAVE_SIZE; p[k] = e < E ? dsl_a_ldbf(logits + e) : -INFINITY; taken[k] = e >= E; m = fmaxf(m, p[k]); }
#ifdef ETX_TOPK_KEY32                                    // one 32-bit key per expert, DPP wave max (etx/tile_util.h)
    // lane t stores result t (the dsl_run_dsl_topk form). etx_topk_key32_lds (lane 0 stores each round) gives the same
    // bits, but on this tree it adds 11 spilled VGPRs to the fold build (docs/TOPK-KEY32-2026-10-08.md, section 7.4).
    etx_topk_key32(p, E, K, renorm != 0, lane, [&](int t, int bi, float v) { ids[t] = bi; wts[t] = v; });
#else
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
#endif
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
  const int nblk = *(const int*)(a + DSL_GRID_OFF + 12);
  if (blk >= nblk) return;
  const DslArgs d = dsl_args(a);
  const int r0 = blk * d.rpt, r1 = min(d.M, min(r0 + d.rpt, r0 + DSL_RCH));   // the first pass of dsl_gemv_chunked
  if (r0 >= r1) return;
  if (d.K % (ETX_WAVE_SIZE * 8) == 0) dsl_touch_rows<8, DSL_UW>(d.W, d.K, r0, r1); else dsl_touch_rows<4, DSL_UN>(d.W, d.K, r0, r1);
}
'''

TOUCH_FORMS = ("plain", "router")     # the dense GEMVs: an expert task's rows are not known before its own wait (top-k)


ROUTER_BODY = r"""
  // Isolated Qwen3-30B-A3B router instantiation: one output row, four
  // 512-element segments, R=1 norm. No generic chunk/tail/batch dispatch.
  // Host and plan reject other shapes before launch; all waves take joins.
  if (blk >= 128) return;
  float* part = xs + DSL_K_MAX;
  dsl_stage_norm_at<2048, 1>(d, d.rows, d.resid_in, d.resid_out,
                            (__hip_bfloat16*)d.x, xs, part + 16 * DSL_RCH, blk == 0);
  const int lane = etx_lane(), wv = etx_wave();
  if (wv < 4) {
    const int off = wv * 512 + lane * 8;
    const dsl_u32x4 w = dsl_wload((const DSL_G dsl_u32x4*)(d.W + (size_t)blk * 2048 + off));
    float acc = 0.f;
    acc += dsl_dot(w, (const dsl_lds_f*)xs + off);
#ifdef ETX_DSL_WSUM
    const float v = dsl_wave_sum0(acc);
#else
    const float v = etx_wave_sum(acc);
#endif
    if (lane == 0) part[wv] = 0.f + v;   // original clear + deposit arithmetic
  }
  __syncthreads();                      // every producer's partial must be visible
  if (threadIdx.x == 0) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) v += w < 4 ? part[w] : 0.f;
    dsl_a_stbf(d.out + blk, __float2bfloat16(v));
  }
  __syncthreads();                      // protect the arena until all consumers finish
"""

BODIES = {
    "router": ROUTER_BODY,
    "plain": '''
  const int r0 = blk * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
  const int KS = DSL_K_MAX;                                    // columns are DSL_K_MAX floats apart in LDS
  float* part = xs + DSL_NB * KS;
  if (d.mode == 1
#if ETX_EMIT_DSL_EXT
      || d.mode == 4
#endif
     ) dsl_stage_norm_nb(d, DSL_NB, KS, xs, part + DSL_NB * 16 * DSL_RCH, blk == 0);
  else dsl_stage_nb(d.x, d.K, d.sx, DSL_NB, KS, xs);
  __hip_bfloat16* out = d.out; const int sout = d.sout;
#if ETX_EMIT_DSL_EXT
  const __hip_bfloat16* bias = d.bias;                 // output bias (o_proj / lm_head / the folded qkv); null = none
  dsl_gemv_chunked_nb<DSL_NB>(d.W, d.K, KS, xs, r0, r1, part, [&](int r, const float* v) {
    const float bv = bias ? __bfloat162float(bias[r]) : 0.f;
#pragma unroll
    for (int b = 0; b < DSL_NB; ++b) dsl_a_stbf(out + (size_t)b * sout + r, __float2bfloat16(v[b] + bv));
  });
#else
  dsl_gemv_chunked_nb<DSL_NB>(d.W, d.K, KS, xs, r0, r1, part, [&](int r, const float* v) {
#pragma unroll
    for (int b = 0; b < DSL_NB; ++b) dsl_a_stbf(out + (size_t)b * sout + r, __float2bfloat16(v[b]));
  });
#endif''',
    "silu_mul": '''
  // ETX_VL_BATCH (nb > 1): the slot index runs over requests x slots as S = s * nb + rq, so slot s of every request
  // lands on the same XCD (the xw->x affinity is unchanged) and the per-request tensors are reached through the
  // request strides the host passes (sx: x, sout: out, srows: ids, srin: wts -- the slabs are one allocation).
  const int S0 = blk / d.tpe, j = blk - S0 * d.tpe;
  int s = S0, rq = 0;
  if (DSL_NB > 1 && d.nb > 1) { s = S0 / d.nb; rq = S0 - s * d.nb; }
  const int r0 = j * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
#if ETX_EMIT_DSL_BIGK
  float* g = xs + min(d.K, DSL_K_MAX); float* part = g + 2 * d.rpt;   // min: K > DSL_K_MAX takes the chunked path below, whose x region is one chunk
#else
  float* g = xs + d.K; float* part = g + 2 * d.rpt;
#endif
  int* lids = (int*)(part + 16 * DSL_RCH); float* lwts = (float*)(lids + 16);   // top-k fold: R = k, mode 3 = renormalised
#ifdef DSL_ABLATE_TOPK
  if (etx_wave() == 0 && etx_lane() == 0) { lids[s] = s; lwts[s] = 1.f; }       // ablation: slot s -> expert s (WRONG output)
  __syncthreads();
#else
  if (d.mode >= 2) dsl_topk_lds(d.logits, d.E, d.R, d.mode == 3 ? 1 : 0, lids, lwts);
#endif
  const int dsl_eid = d.mode >= 2 ? lids[s] : dsl_a_ldi(d.ids + (size_t)rq * d.srows + s);
  const __hip_bfloat16* We = d.W + (size_t)dsl_eid * d.estride;     // g: 2 x rpt floats (rpt <= DSL_RCH, host-checked)
#if ETX_EMIT_DSL_BIGK
  // readfirstlane: d.K is one value per task, but without ETX_DSL_SCALAR it is a flat load into a VGPR, so the compiler
  // took this branch as divergent and structurised both paths together. In the one-function megakernel that pushed
  // Mixtral (the only BIGK model) from 2 to 60 spilled VGPRs, all of them in the inlined attention (mixtral_spill).
  if (etx_uniform(d.K) > DSL_K_MAX) {   // K > the staged window (host allows it for the MoE forms only):
    float* gb = xs + DSL_K_MAX; float* pb = gb + 2 * d.rpt;      // fixed LDS bases -- xs + d.K would overflow the block
    const __hip_bfloat16* xr = d.x + (size_t)rq * d.sx;
    dsl_gemv_bigk(We, xr, d.K, xs, r0, r1, pb, [&](int r, float v) { gb[r - r0] = dsl_bfr(v); });
    dsl_gemv_bigk(We + (size_t)d.M * d.K, xr, d.K, xs, r0, r1, pb, [&](int r, float v) { gb[d.rpt + r - r0] = dsl_bfr(v); });
    __hip_bfloat16* actb = d.out + (size_t)rq * d.sout + (size_t)s * d.M;
    for (int r = r0 + (int)threadIdx.x; r < r1; r += blockDim.x) dsl_a_stbf(actb + r, __float2bfloat16(dsl_silu(gb[r - r0]) * gb[d.rpt + r - r0]));
    return;
  }
#endif
  dsl_stage(d.x + (size_t)rq * d.sx, d.K, xs);
#ifdef DSL_ABLATE_NOGEMV                                 // ablation: prologue only -- topk + stage, no passes, no stores (WRONG output)
  asm volatile("" :: "v"((unsigned)(size_t)We)); return;
#endif
#ifdef ETX_DSL_GU1
  // gate and up rows as ONE stream pass, taken only where it cannot change a bit: every row lies inside one wave's
  // segment range both as two passes of n rows and as one pass of 2n (so each row is one wave's in-order sum either
  // way, plus exact zeros from the other waves), and no wave straddles gate and up. Saves the second pass's partial
  // zeroing, cross-wave reduction and two of its barriers. Qwen3-30B-A3B: n = 32, K = 2048: 8 then 16 segments per
  // wave, 4 per row. Otherwise the two passes below run unchanged.
  const int gn = r1 - r0, gspr = d.K / (ETX_WAVE_SIZE * 8), gnw = etx_nwaves();
  const int gp1 = (gn * gspr + gnw - 1) / gnw, gp2 = (2 * gn * gspr + gnw - 1) / gnw;
  if (d.K % (ETX_WAVE_SIZE * 8) == 0 && 2 * gn <= DSL_RCH && gp1 % gspr == 0 && gp2 % gspr == 0 && (gn * gspr) % gp2 == 0) {
#ifdef ETX_DSL_ROW_WAVE
    dsl_gemv_rows_rw<8, 1>(We, d.K, 0, xs, r0, r0 + 2 * gn, [&](int r, const float* v) {
      const int q = r - r0; g[q < gn ? q : d.rpt + q - gn] = dsl_bfr(v[0]); }, gn, (long long)(d.M - gn) * d.K);
#else
    dsl_gemv_rows_nb<8, DSL_UW, 1>(We, d.K, 0, xs, r0, r0 + 2 * gn, part, [&](int r, const float* v) {
      const int q = r - r0; g[q < gn ? q : d.rpt + q - gn] = dsl_bfr(v[0]); }, gn, (long long)(d.M - gn) * d.K);
#endif
  } else
#endif
  {
  dsl_gemv_chunked(We, d.K, xs, r0, r1, part, [&](int r, float v) { g[r - r0] = dsl_bfr(v); });
#ifndef DSL_ABLATE_ONEPASS                                                      // ablation: gate only, no up pass (WRONG output)
  dsl_gemv_chunked(We + (size_t)d.M * d.K, d.K, xs, r0, r1, part, [&](int r, float v) { g[d.rpt + r - r0] = dsl_bfr(v); });
#endif
  }
  __hip_bfloat16* act = d.out + (size_t)rq * d.sout + (size_t)s * d.M;
#ifdef DSL_ABLATE_EPI                                                           // ablation: no silu, no multiply (WRONG output)
  for (int r = r0 + (int)threadIdx.x; r < r1; r += blockDim.x) dsl_a_stbf(act + r, __float2bfloat16(g[r - r0]));
#else
  for (int r = r0 + (int)threadIdx.x; r < r1; r += blockDim.x) dsl_a_stbf(act + r, __float2bfloat16(dsl_silu(g[r - r0]) * g[d.rpt + r - r0]));
#endif''',
    "scale": '''
  const int S0 = blk / d.tpe, j = blk - S0 * d.tpe;    // batched: S = s * nb + rq, as silu_mul
  int s = S0, rq = 0;
  if (DSL_NB > 1 && d.nb > 1) { s = S0 / d.nb; rq = S0 - s * d.nb; }
  const int r0 = j * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
#if ETX_EMIT_DSL_BIGK
  int* lids = (int*)(xs + min(d.K, DSL_K_MAX) + 16 * DSL_RCH); float* lwts = (float*)(lids + 16);   // min: as silu_mul
#else
  int* lids = (int*)(xs + d.K + 16 * DSL_RCH); float* lwts = (float*)(lids + 16);
#endif
#ifdef DSL_ABLATE_TOPK
  if (etx_wave() == 0 && etx_lane() == 0) { lids[s] = s; lwts[s] = 1.f; }       // ablation: slot s -> expert s (WRONG output)
  __syncthreads();
#else
  if (d.mode >= 2) dsl_topk_lds(d.logits, d.E, d.R, d.mode == 3 ? 1 : 0, lids, lwts);
#endif
  const int dsl_eid = d.mode >= 2 ? lids[s] : dsl_a_ldi(d.ids + (size_t)rq * d.srows + s);
  const __hip_bfloat16* We = d.W + (size_t)dsl_eid * d.estride;
#if ETX_EMIT_DSL_BIGK
  if (etx_uniform(d.K) > DSL_K_MAX) {   // chunked staging, uniform branch; fixed LDS bases (see silu_mul)
    float* pb = xs + DSL_K_MAX;
    const float wsb = d.mode >= 2 ? lwts[s] : dsl_a_ldf(d.wts + (size_t)rq * d.srin + s);
    __hip_bfloat16* outb = d.out + (size_t)rq * d.sout + (size_t)s * d.M;
    dsl_gemv_bigk(We, d.x + (size_t)rq * d.sx + (size_t)s * d.K, d.K, xs, r0, r1, pb,
                  [&](int r, float v) { dsl_a_stbf(outb + r, __float2bfloat16(v * wsb)); });
    return;
  }
#endif
  dsl_stage(d.x + (size_t)rq * d.sx + (size_t)s * d.K, d.K, xs);
#ifdef DSL_ABLATE_NOGEMV
  asm volatile("" :: "v"((unsigned)(size_t)We)); return;
#endif
  __hip_bfloat16* out = d.out + (size_t)rq * d.sout + (size_t)s * d.M;
#if defined(DSL_ABLATE_EPI) || defined(DSL_ABLATE_WTS)   // ablation: no routing-weight multiply (WRONG output). WTS is
  // the down-only form: EPI also drops moe_up's silu, so it cannot price what moe_down does that vLLM's kernel does not
  // (vLLM applies the routing weight in its separate moe_sum, which our fold deletes).
  dsl_gemv_chunked(We, d.K, xs, r0, r1, xs + d.K, [&](int r, float v) { dsl_a_stbf(out + r, __float2bfloat16(v)); });
#else
  const float ws = d.mode >= 2 ? lwts[s] : dsl_a_ldf(d.wts + (size_t)rq * d.srin + s);
  dsl_gemv_chunked(We, d.K, xs, r0, r1, xs + d.K, [&](int r, float v) { dsl_a_stbf(out + r, __float2bfloat16(v * ws)); });
#endif''',
}


def emit_tiles(lds_floats: int = 8192, machine=None) -> str:
    """The tile source: for every form the plain, XCD-affine (x) and pre-wait entry points the plan links
    (examples/vllm_llm/model.py), reading the argument block like an imported tile; LDS from the import arena.
    machine: the machine model (or its name) whose tile primitives fill the source (etx/ops/prims.py); default gfx942."""
    from etx.ops import coh, prims
    out = ["// generated by etx.ops.gemv: the DSL streaming GEMVs", "#pragma once",
           '#include "etx/abi.h"', '#include "etx/import.h"', '#include "etx/tile_util.h"',
           f"#define DSL_NB {DSL_NB}            // activation columns per weight read (ETX_VL_NB)",
           f"#define DSL_K_MAX {K_MAX}          // the staged-column stride in LDS floats",
           f"#define DSL_GRID_OFF {GRID_OFF}",
           "#ifdef ETX_IMPORT_LDS   // the staged columns and partials must fit the shared import arena",
           f"_Static_assert({LDS_BYTES} <= ETX_IMPORT_LDS, \"etx.ops.gemv: LDS_BYTES exceeds ETX_IMPORT_LDS"
           " (raise the arena or lower ETX_VL_NB)\");", "#endif", coh.HELPERS, prims.expand(STREAM, machine)]
    for e, form in FORMS.items():
        out.append(f"// ---- {e}: {form}")
        out.append(f"static __device__ __attribute__((always_inline)) void dsl_run_{e}(const unsigned char* a, int blk) {{")
        out.append("#ifdef ETX_DSL_SCALAR")
        out.append("  blk = etx_uniform(blk);")
        out.append(f"  const int nblk = *(const int DSL_ARG_AS*)(uintptr_t)(a + {GRID_OFF + 12});")
        out.append("#else")
        out.append(f"  const int nblk = *(const int*)(a + {GRID_OFF + 12});")
        out.append("#endif")
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
    source = "\n".join(out) + "\n"
    if _os.environ.get("ETX_INBODY_EMIT") == "1":
        from etx.codegen.load_stamps import instrument
        source = instrument(source)
    return source


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
