"""Use the kept GEMV bodies and the exact vector fold text.

The source anchors fail closed. This module never patches the shared emitter.
The probe has 1024 threads and only the vector RMS path. The scalar, batch,
LayerNorm and local-fold variants are outside this experiment.
"""
from etx.ops import gemv


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("expert merge source anchor changed: " + old[:90])
    return text.replace(old, new, 1)


def parts():
    stream = gemv.STREAM
    start = stream.index("template <int FixedK = 0, int FixedR = -1>")
    end = stream.index("static __device__ __forceinline__ void dsl_stage_norm(", start)
    original = stream[start:end]
    begin = original.index("  for (int i = (int)threadIdx.x; i < K / 8;")
    endfold = original.index("#ifdef ETX_DSL_NORM_HOIST", begin)
    return original, original[begin:endfold]


def emit(candidate: bool, diagnostic: bool) -> str:
    if gemv.DSL_NB != 1 or gemv.DSL_EXT:
        raise ValueError("expert merge requires NB=1 and no DSL_EXT")
    original, fold = parts()
    # Keep only the existing vector branch. This avoids a second implementation
    # of the norm arithmetic, including both bf16 boundaries and the wave tree.
    vector = original[:original.index("#else\n  for (int i = (int)threadIdx.x; i < K;")]
    vector += "#endif\n  __syncthreads();\n}\n"
    vector = once(vector, "dsl_stage_norm_at(", "eem_stage_norm(")
    vector = once(vector, "const DslArgs& d,", "const DslArgs& d, const etx_ctx* c,")
    if candidate:
        stage = '''  for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < 8; ++j) xs[8*i+j] = dsl_a_ldbf((const __hip_bfloat16*)c->args[1] + 8*i+j);
  }
'''
        vector = once(vector, fold, stage)
        vector = once(vector, "  ss = etx_block_sum(ss, red);\n  const float sv = rsqrtf(ss / (float)K + d.eps);",
                      "  const float sv = dsl_a_ldf((const float*)c->args[3]);")
    else:
        # Diagnostics observe the original values. Only blk 0 owns these copies.
        observe = '''
#if EEM_DIAGNOSTIC
  // The copy's scalar index reads values folded by other waves. Join all
  // writers before any diagnostic reader; store is block-uniform.
  __syncthreads();
  if (store) {
    if (threadIdx.x < 256) dsl_a_stf((float*)c->args[2] + threadIdx.x, ss);
    for (int i = (int)threadIdx.x; i < K; i += blockDim.x)
      dsl_a_stbf((__hip_bfloat16*)c->args[1] + i, __float2bfloat16(xs[i]));
  }
#endif
'''
        anchor = "#ifdef ETX_DSL_NORM_HOIST\n  // Immutable norm weights:"
        vector = once(vector, anchor, observe + anchor)
        vector = once(vector, "  const float sv = rsqrtf(ss / (float)K + d.eps);",
                      "  const float sv = rsqrtf(ss / (float)K + d.eps);\n"
                      "#if EEM_DIAGNOSTIC\n  if (store && threadIdx.x == 0) dsl_a_stf((float*)c->args[3], sv);\n#endif")
    body = once(gemv.BODIES["plain"],
                "dsl_stage_norm_nb(d, DSL_NB, KS, xs, part + DSL_NB * 16 * DSL_RCH, blk == 0);",
                "eem_stage_norm(d, c, d.rows, d.resid_in, d.resid_out, (__hip_bfloat16*)d.x, xs, part + DSL_NB * 16 * DSL_RCH, blk == 0);")
    body = once(body, "  __hip_bfloat16* out = d.out;", '''
#if EEM_DIAGNOSTIC
  for (int i = (int)threadIdx.x; i < 2048; i += blockDim.x)
    dsl_a_stbf((__hip_bfloat16*)c->args[5] + (size_t)blk*2048+i, __float2bfloat16(xs[i]));
#endif
  __hip_bfloat16* out = d.out;''')
    # Each row owner uses eight original vector-thread accumulators. A group
    # scalar is not an exact substitute for these 256 original statistics.
    rowfold = once(fold, "for (int i = (int)threadIdx.x; i < K / 8; i += blockDim.x)",
                   "for (int i = group*8 + (int)threadIdx.x; i < group*8+8; i += blockDim.x)")
    rowfold = once(rowfold, "    if (store && rout != nullptr)",
                   "    dsl_a_stf((float*)c->args[2] + i, ss);\n"
                   "    dsl_a_st128((__hip_bfloat16*)c->args[1] + 8*i, ro);\n"
                   "    if (store && rout != nullptr)")
    appended = '''
static __device__ __forceinline__ void eem_append(const etx_ctx* c, int group) {
  const DslArgs d = dsl_args(etx_imp_argblock(c->args, 320));
  const int K = 2048, R = 8;
  const __hip_bfloat16 *rows = d.rows, *rin = d.resid_in;
  __hip_bfloat16* rout = d.resid_out;
  const bool store = true;
  float* xs = (float*)etx_lds_arena;
  float ss = 0.f;
  // All producing waves drain, then the block joins BEFORE the leader RMW.
  // Coherent stores/loads carry every cross-XCD payload. The atomics use the
  // original DEVICE event scope/order. They are not a cached-data acquire.
  ETX_RELEASE_DOMAIN();
  __syncthreads();
  int* elected = (int*)(xs + 4096);
  if (threadIdx.x == 0) *elected = ETX_ARRIVE_DEVICE((int*)c->args[4] + group) == 1;
  __syncthreads();
  if (!*elected) return;  // block-uniform, no task waits for another task
#if EEM_DIAGNOSTIC
  if (threadIdx.x == 0) dsl_a_sti((int*)c->args[7] + group, c->coord[0]);
#endif
''' + rowfold + '''
  ETX_RELEASE_DOMAIN();
  __syncthreads();
  if (threadIdx.x == 0) *elected = ETX_ARRIVE_DEVICE((int*)c->args[4] + 32) == 1;
  __syncthreads();
  if (!*elected) return;
  // Same 1024 virtual threads, same 64-lane butterfly, then 16 wave
  // results added in order. The 768 unused lanes contribute positive zero.
  const float v = threadIdx.x < 256 ? dsl_a_ldf((const float*)c->args[2] + threadIdx.x) : 0.f;
  const float total = etx_block_sum(v, xs + 4096 + 16);
  if (threadIdx.x == 0) dsl_a_stf((float*)c->args[3], rsqrtf(total / 2048.f + d.eps));
  // Keep runtime all-wave drain and the full down completion after return.
}
'''
    source = gemv.emit_tiles()
    prefix = f'''// Isolated expert merge: no full-model switch exists.
#include "probe_config.h"
static_assert(EEM_CANDIDATE == {int(candidate)} && EEM_DIAGNOSTIC == {int(diagnostic)}, "probe source/config identity mismatch");
#define ETX_IMPORT_LDS {gemv.LDS_BYTES}
#if !defined(ETX_DSL_VECNORM) || !defined(ETX_DSL_NORM_HOIST) || !defined(ETX_DSL_COH)
#error "expert merge requires the kept vector norm, hoist and coherent path"
#endif
#if defined(ETX_NO_ALLWAVE_DRAIN) || defined(ETX_ACQUIRE_DOMAIN_RELAXED) || defined(ETX_DSL_COH_L1)
#error "expert merge requires system-coherent payloads and the original all-wave drain"
#endif
'''
    wrappers = '''
extern "C" __device__ __attribute__((always_inline)) void etx_eem_up(const etx_ctx* c) {
  int j = c->coord[1];
#if EEM_DIAGNOSTIC
  j = (j + dsl_a_ldi((const int*)c->args[6])) % 24;
#endif
  dsl_run_dsl_moe_up_fold(etx_imp_argblock(c->args, 0), c->coord[0]*24+j);
}
extern "C" __device__ __attribute__((always_inline)) void etx_eem_down(const etx_ctx* c) {
  int j = c->coord[1];
#if EEM_DIAGNOSTIC
  j = (j + dsl_a_ldi((const int*)c->args[6])) % 32;
#endif
  dsl_run_dsl_moe_down_fold(etx_imp_argblock(c->args, 160), c->coord[0]*32+j);
#if EEM_CANDIDATE
  eem_append(c, j);
#endif
}
extern "C" __device__ __attribute__((always_inline)) void etx_eem_consumer(const etx_ctx* c) {
  const unsigned char* a = etx_imp_argblock(c->args, 320);
  const int blk = etx_uniform(c->coord[0]);
  const DslArgs d = dsl_args(a);
  float* xs = (float*)etx_lds_arena;
''' + body + "\n}\n"
    return prefix + source + vector + (appended if candidate else "") + wrappers
