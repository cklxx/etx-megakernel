// AMD-specific helpers referenced by the machine YAML lowering entries.
// UNTESTED ON HARDWARE: verify the HW_ID encodings on the first bring-up
// (bench/calib/discover_domain.hip prints what every workgroup sees).
#pragma once
#include <hip/hip_runtime.h>

// XCC_ID hardware register (gfx940+): s_getreg_b32 HW_REG_XCC_ID, bits [3:0].
// s_getreg immediate = (size-1) << 11 | offset << 6 | hwRegId ; XCC_ID = 20.
static __device__ __forceinline__ uint32_t etx_amdgcn_xcc_id() {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
  return __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20) & 0xF;
#else
  return 0u;
#endif
}

// Counter poll through L2: drop the per-CU L1 (buffer_inv sc0), then a plain
// load. Measured on MI300X 2026-09-22 (bench/calib/atomic_pingpong): observes
// agent-scope RMWs from any XCD, one-way 642 ns same-XCD / 742 ns cross-XCD vs
// 873 / 892 ns for an agent-scope atomic load. A plain or sc0 load without the
// invalidate never observes the arrive (stale L1).
static __device__ __forceinline__ int32_t etx_amdgcn_poll_l2(const int32_t* p) {
  asm volatile("buffer_inv sc0" ::: "memory");
  return *(volatile const int32_t*)p;
}

// Sentinel signalling (capability sentinel_signal): poll a data word with an
// agent-scope load instead of a counter. Kog measured 0.8 us vs 7.6 us for a
// whole-GPU phase switch on MI300X. Use only when the producer writes the
// payload with a release and the sentinel is a value the payload cannot take.
static __device__ __forceinline__ int32_t etx_amdgcn_load_agent(const int32_t* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Relaxed device-scope store (relay mirror writes).
static __device__ __forceinline__ void etx_store_relaxed_device(int32_t* p, int32_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
