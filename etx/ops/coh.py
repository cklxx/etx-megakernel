"""The DSL ops' activation accessors. With -DETX_DSL_COH (the ETX_VL_COH=1 builds) they are device-coherent
(relaxed system-scope atomics: sc0 sc1, past the XCD's L1 and L2), so the megakernel emits no acquire
`buffer_inv sc1` for these tiles and the weights (and the touch hook's lines) survive the phase boundary in L2
(docs/RESULTS-2026-09-30.md: the whole-L2 invalidate per task is why every phase streamed from cold caches).
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
#ifdef ETX_DSL_COH
#define DSL_COH_SCOPE __HIP_MEMORY_SCOPE_SYSTEM
// the atomics go through address_space(1) pointers so they lower to global_ ops with sc0 sc1, not flat_ ones
// (a flat access also waits on the LDS counter: the DSL_G note in etx/ops/gemv.py)
#define DSL_COH_G __attribute__((address_space(1)))
static __device__ __forceinline__ float dsl_a_ldbf(const __hip_bfloat16* p) {
  const unsigned short u = __hip_atomic_load((const DSL_COH_G unsigned short*)p, __ATOMIC_RELAXED, DSL_COH_SCOPE);
  __hip_bfloat16 h; __builtin_memcpy(&h, &u, 2); return __bfloat162float(h);
}
static __device__ __forceinline__ void dsl_a_stbf(__hip_bfloat16* p, __hip_bfloat16 v) {
  unsigned short u; __builtin_memcpy(&u, &v, 2); __hip_atomic_store((DSL_COH_G unsigned short*)p, u, __ATOMIC_RELAXED, DSL_COH_SCOPE);
}
static __device__ __forceinline__ int dsl_a_ldi(const int* p) { return __hip_atomic_load((const DSL_COH_G int*)p, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
static __device__ __forceinline__ void dsl_a_sti(int* p, int v) { __hip_atomic_store((DSL_COH_G int*)p, v, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
static __device__ __forceinline__ float dsl_a_ldf(const float* p) { return __hip_atomic_load((const DSL_COH_G float*)p, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
static __device__ __forceinline__ void dsl_a_stf(float* p, float v) { __hip_atomic_store((DSL_COH_G float*)p, v, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
static __device__ __forceinline__ long long dsl_a_ldll(const long long* p) { return __hip_atomic_load((const DSL_COH_G long long*)p, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
static __device__ __forceinline__ unsigned long long dsl_a_ld64(const unsigned long long* p) { return __hip_atomic_load((const DSL_COH_G unsigned long long*)p, __ATOMIC_RELAXED, DSL_COH_SCOPE); }
#else
static __device__ __forceinline__ float dsl_a_ldbf(const __hip_bfloat16* p) { return __bfloat162float(*p); }
static __device__ __forceinline__ void dsl_a_stbf(__hip_bfloat16* p, __hip_bfloat16 v) { *p = v; }
static __device__ __forceinline__ int dsl_a_ldi(const int* p) { return *p; }
static __device__ __forceinline__ void dsl_a_sti(int* p, int v) { *p = v; }
static __device__ __forceinline__ float dsl_a_ldf(const float* p) { return *p; }
static __device__ __forceinline__ void dsl_a_stf(float* p, float v) { *p = v; }
static __device__ __forceinline__ long long dsl_a_ldll(const long long* p) { return *p; }
static __device__ __forceinline__ unsigned long long dsl_a_ld64(const unsigned long long* p) { return *p; }
#endif
#endif
'''
