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

// Sentinel signalling (capability sentinel_signal): poll a data word with an
// agent-scope load instead of a counter. Kog measured 0.8 us vs 7.6 us for a
// whole-GPU phase switch on MI300X. Use only when the producer writes the
// payload with a release and the sentinel is a value the payload cannot take.
static __device__ __forceinline__ int32_t etx_amdgcn_load_agent(const int32_t* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
