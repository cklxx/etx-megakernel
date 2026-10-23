"""The DSL ops' activation accessors. With -DETX_DSL_COH (the ETX_VL_COH=1 builds) they are device-coherent
(relaxed system-scope atomics: sc0 sc1, past the XCD's L1 and L2), so the megakernel emits no acquire
`buffer_inv sc1` for these tiles and the weights (and the touch hook's lines) survive the phase boundary in L2
(docs/qwen/RESULTS-2026-09-30.md: the whole-L2 invalidate per task is why every phase streamed from cold caches).
Without the flag they reduce to the plain accesses the acquire-based builds always used, so a build without
ETX_VL_COH is unchanged. Weights are never accessed through these: they are read-only for the whole run and
must stay cached.

Every op module prepends HELPERS to its emitted source; the include guard makes double emission harmless when
etx.ops.emit concatenates several ops into one file.
"""
from __future__ import annotations

HELPERS = r'''
#ifndef ETX_DSL_COH_HELPERS
#define ETX_DSL_COH_HELPERS
#define ETX_DSL_COH_L1_IMPL 1      // this source implements ETX_DSL_COH_L1 (the megakernel refuses one that does not)
#ifndef DSL_U32X4_DEFINED                                 // also defined by etx/ops/gemv.py, whichever is emitted first
#define DSL_U32X4_DEFINED
typedef unsigned int dsl_u32x4 __attribute__((ext_vector_type(4)));
#endif
#ifndef DSL_G
#define DSL_G __attribute__((address_space(1)))
#endif
#if defined(ETX_DSL_COH) || defined(ETX_DSL_COH_L1)
#define DSL_COH_SCOPE __HIP_MEMORY_SCOPE_SYSTEM
// ETX_DSL_COH_L1 (coherence mode "l1", published by the plan): loads at AGENT scope (sc1: past this CU's L1, never a
// stale L2 line -- sync_bench loadscope xl1 / cl1, 0 stale words); stores stay system scope (write-through), so a tile
// still declared "coherent" keeps every guarantee it had
#ifdef ETX_DSL_COH_L1
#define DSL_COH_LD_SCOPE __HIP_MEMORY_SCOPE_AGENT
#else
#define DSL_COH_LD_SCOPE __HIP_MEMORY_SCOPE_SYSTEM
#endif
// the atomics go through address_space(1) pointers so they lower to global_ ops with sc0 sc1, not flat_ ones
// (a flat access also waits on the LDS counter: the DSL_G note in etx/ops/gemv.py)
#define DSL_COH_G __attribute__((address_space(1)))
static __device__ __forceinline__ float dsl_a_ldbf(const __hip_bfloat16* p) {
  const unsigned short u = __hip_atomic_load((const DSL_COH_G unsigned short*)p, __ATOMIC_RELAXED, DSL_COH_LD_SCOPE);
  __hip_bfloat16 h; __builtin_memcpy(&h, &u, 2); return __bfloat162float(h);
}
static __device__ __forceinline__ void dsl_a_stbf(__hip_bfloat16* p, __hip_bfloat16 v) {
  unsigned short u; __builtin_memcpy(&u, &v, 2); __hip_atomic_store((DSL_COH_G unsigned short*)p, u, __ATOMIC_RELAXED, DSL_COH_SCOPE);
}
static __device__ __forceinline__ int dsl_a_ldi(const int* p) { return __hip_atomic_load((const DSL_COH_G int*)p, __ATOMIC_RELAXED, DSL_COH_LD_SCOPE); }
static __device__ __forceinline__ void dsl_a_sti(int* p, int v) { __hip_atomic_store((DSL_COH_G int*)p, v, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
static __device__ __forceinline__ float dsl_a_ldf(const float* p) { return __hip_atomic_load((const DSL_COH_G float*)p, __ATOMIC_RELAXED, DSL_COH_LD_SCOPE); }
static __device__ __forceinline__ void dsl_a_stf(float* p, float v) { __hip_atomic_store((DSL_COH_G float*)p, v, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
static __device__ __forceinline__ long long dsl_a_ldll(const long long* p) { return __hip_atomic_load((const DSL_COH_G long long*)p, __ATOMIC_RELAXED, DSL_COH_LD_SCOPE); }
static __device__ __forceinline__ unsigned long long dsl_a_ld64(const unsigned long long* p) { return __hip_atomic_load((const DSL_COH_G unsigned long long*)p, __ATOMIC_RELAXED, DSL_COH_LD_SCOPE); }
static __device__ __forceinline__ void dsl_a_st64(unsigned long long* p, unsigned long long v) { __hip_atomic_store((DSL_COH_G unsigned long long*)p, v, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
// 16 B of activation in one pair of requests. A coherent access bypasses L1/L2, so a per-element 2 B accessor
// issues one memory request per lane per element: dsl_stage_norm's K=2048 fold made 4096 of them per task against
// o_proj's 1024 eight-byte ones, and o_proj sits at its achievable rate while qkv does not (round 23). Built from
// dsl_a_ld64 because that lowering (address_space(1) -> global_ with sc0 sc1, not flat_) is the one already proven.
static __device__ __forceinline__ dsl_u32x4 dsl_a_ld128(const __hip_bfloat16* p) {
  const unsigned long long lo = dsl_a_ld64((const unsigned long long*)p);
  const unsigned long long hi = dsl_a_ld64((const unsigned long long*)p + 1);
  dsl_u32x4 v; v[0] = (unsigned)(lo & 0xffffffffull); v[1] = (unsigned)(lo >> 32);
  v[2] = (unsigned)(hi & 0xffffffffull); v[3] = (unsigned)(hi >> 32); return v;
}
static __device__ __forceinline__ void dsl_a_st128(__hip_bfloat16* p, dsl_u32x4 v) {
  dsl_a_st64((unsigned long long*)p, ((unsigned long long)v[1] << 32) | v[0]);
  dsl_a_st64((unsigned long long*)p + 1, ((unsigned long long)v[3] << 32) | v[2]);
}

// A scalar every lane needs (positions[0], slot_mapping[0]). A coherent load bypasses L1 and L2, so the usual
// "lane 0 misses, the rest hit" does not happen: each lane's load is its own memory request to the same address.
// -DETX_DSL_UNIFORM loads it once per wave and broadcasts. Without the flag it is the plain per-lane coherent load.
#ifdef ETX_DSL_UNIFORM
static __device__ __forceinline__ long long dsl_a_ldll_u(const long long* p) {
  long long v = 0;
  if (etx_lane() == 0) v = dsl_a_ldll(p);
  const int lo = __shfl((int)((unsigned long long)v & 0xffffffffull), 0, ETX_WAVE_SIZE);
  const int hi = __shfl((int)((unsigned long long)v >> 32), 0, ETX_WAVE_SIZE);
  return (long long)(((unsigned long long)(unsigned int)hi << 32) | (unsigned int)lo);
}
#else
static __device__ __forceinline__ long long dsl_a_ldll_u(const long long* p) { return dsl_a_ldll(p); }
#endif
#else
static __device__ __forceinline__ float dsl_a_ldbf(const __hip_bfloat16* p) { return __bfloat162float(*p); }
static __device__ __forceinline__ void dsl_a_stbf(__hip_bfloat16* p, __hip_bfloat16 v) { *p = v; }
static __device__ __forceinline__ int dsl_a_ldi(const int* p) { return *p; }
static __device__ __forceinline__ void dsl_a_sti(int* p, int v) { *p = v; }
static __device__ __forceinline__ float dsl_a_ldf(const float* p) { return *p; }
static __device__ __forceinline__ void dsl_a_stf(float* p, float v) { *p = v; }
static __device__ __forceinline__ long long dsl_a_ldll(const long long* p) { return *p; }
static __device__ __forceinline__ unsigned long long dsl_a_ld64(const unsigned long long* p) { return *p; }
static __device__ __forceinline__ void dsl_a_st64(unsigned long long* p, unsigned long long v) { *p = v; }
static __device__ __forceinline__ dsl_u32x4 dsl_a_ld128(const __hip_bfloat16* p) { return *(const DSL_G dsl_u32x4*)p; }
static __device__ __forceinline__ void dsl_a_st128(__hip_bfloat16* p, dsl_u32x4 v) { *(DSL_G dsl_u32x4*)p = v; }
static __device__ __forceinline__ long long dsl_a_ldll_u(const long long* p) { return *p; }
#endif
#endif
'''


def emit_tiles(machine="gfx942") -> str:
    """Emit accessors using the machine's payload-coherence contract."""
    from etx.ops import prims
    return prims.emit_source(HELPERS, machine)
