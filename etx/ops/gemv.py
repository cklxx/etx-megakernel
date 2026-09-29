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

ARG_BYTES, GRID_OFF = 80, 64
PARAMS = [("W", "ptr addrspace(1)", 0, 8), ("x", "ptr addrspace(1)", 8, 8), ("out", "ptr addrspace(1)", 16, 8),
          ("ids", "ptr addrspace(1)", 24, 8), ("wts", "ptr addrspace(1)", 32, 8), ("estride", "i64", 40, 8),
          ("M", "i32", 48, 4), ("K", "i32", 52, 4), ("rpt", "i32", 56, 4), ("tpe", "i32", 60, 4)]
FORMS = {"dsl_gemv": "plain", "dsl_moe_up": "silu_mul", "dsl_moe_down": "scale"}
K_MAX, RCH, THREADS = 4096, 64, 1024                              # x (K floats) + 2 x RCH (moe_up's g) + 16 waves x RCH partials
LDS_BYTES = 4 * (K_MAX + 2 * RCH + (THREADS // 64) * RCH)          # 20992 B; the host refuses K > K_MAX and rpt > RCH


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
  if (K % (ETX_WAVE_SIZE * 8) == 0) dsl_gemv_rows<8, 4>(W, K, xs, r0, r1, part, epi);
  else dsl_gemv_rows<4, 8>(W, K, xs, r0, r1, part, epi);
}
static __device__ __forceinline__ void dsl_stage(const __hip_bfloat16* __restrict__ x, int K, float* xs) {   // bf16 -> fp32 LDS, 16 B per thread
  const DSL_G dsl_u32x4* xv = (const DSL_G dsl_u32x4*)x;
  for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x) {
    const dsl_u32x4 v = xv[i];
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
struct DslArgs { const __hip_bfloat16* W; const __hip_bfloat16* x; __hip_bfloat16* out; const int* ids; const float* wts; long long estride; int M, K, rpt, tpe; };
static __device__ __forceinline__ DslArgs dsl_args(const unsigned char* a) {
  DslArgs d;
  d.W = *(const __hip_bfloat16* const*)(a + 0); d.x = *(const __hip_bfloat16* const*)(a + 8); d.out = *(__hip_bfloat16* const*)(a + 16);
  d.ids = *(const int* const*)(a + 24); d.wts = *(const float* const*)(a + 32); d.estride = *(const long long*)(a + 40);
  d.M = *(const int*)(a + 48); d.K = *(const int*)(a + 52); d.rpt = *(const int*)(a + 56); d.tpe = *(const int*)(a + 60);
  return d;
}
'''

BODIES = {
    "plain": '''
  const int r0 = blk * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
  dsl_stage(d.x, d.K, xs);
  __hip_bfloat16* out = d.out;
  dsl_gemv_chunked(d.W, d.K, xs, r0, r1, xs + d.K, [&](int r, float v) { out[r] = __float2bfloat16(v); });''',
    "silu_mul": '''
  const int s = blk / d.tpe, j = blk - s * d.tpe;
  const int r0 = j * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
  const __hip_bfloat16* We = d.W + (size_t)d.ids[s] * d.estride;
  float* g = xs + d.K; float* part = g + 2 * d.rpt;                  // g: 2 x rpt floats (rpt <= DSL_RCH, host-checked)
  dsl_stage(d.x, d.K, xs);
  dsl_gemv_chunked(We, d.K, xs, r0, r1, part, [&](int r, float v) { g[r - r0] = dsl_bfr(v); });
  dsl_gemv_chunked(We + (size_t)d.M * d.K, d.K, xs, r0, r1, part, [&](int r, float v) { g[d.rpt + r - r0] = dsl_bfr(v); });
  __hip_bfloat16* act = d.out + (size_t)s * d.M;
  for (int r = r0 + (int)threadIdx.x; r < r1; r += blockDim.x) act[r] = __float2bfloat16(dsl_silu(g[r - r0]) * g[d.rpt + r - r0]);''',
    "scale": '''
  const int s = blk / d.tpe, j = blk - s * d.tpe;
  const int r0 = j * d.rpt, r1 = min(d.M, r0 + d.rpt);
  if (r0 >= r1) return;
  const __hip_bfloat16* We = d.W + (size_t)d.ids[s] * d.estride;
  dsl_stage(d.x + (size_t)s * d.K, d.K, xs);
  __hip_bfloat16* out = d.out + (size_t)s * d.M;
  const float ws = d.wts[s];
  dsl_gemv_chunked(We, d.K, xs, r0, r1, xs + d.K, [&](int r, float v) { out[r] = __float2bfloat16(v * ws); });''',
}


def emit_tiles(lds_floats: int = 8192) -> str:
    """The tile source: for every form the plain, XCD-affine (x) and pre-wait entry points the plan links
    (examples/vllm_llm/model.py), reading the argument block like an imported tile; LDS from the import arena."""
    out = ["// generated by etx.ops.gemv: the DSL streaming GEMVs", "#pragma once",
           '#include "etx/abi.h"', '#include "etx/import.h"', '#include "etx/tile_util.h"', STREAM]
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
