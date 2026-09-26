// Support for kernels imported at the IR level (etx/importer/ir_import.py).
//
// etx_lds_arena   the LDS every imported tile uses instead of its own static __shared__ variables; imported
//                 tiles run one after another on a workgroup, so they share it. The megakernel TU defines
//                 it once (ETX_IMPORT_LDS bytes; the importer reports what each import needs).
// etx_imp_slot_barrier(slot, nwaves)
//                 the kernel's __syncthreads when k of its blocks share one megakernel workgroup: a barrier
//                 over the nwaves waves of `slot` only, with LDS counters (generation counting, so it can be
//                 reused without a reset). Release/acquire at workgroup scope like s_barrier's fences.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef ETX_IMPORT_LDS
#define ETX_IMPORT_LDS 16
#endif
#ifndef ETX_IMPORT_SLOTS
#define ETX_IMPORT_SLOTS 16    // the most slots a 1024-thread workgroup can hold (one wave each)
#endif

extern "C" {   // a linkage block, not `extern "C" <decl>`: this must be the definition
__shared__ __attribute__((aligned(16))) unsigned char etx_lds_arena[ETX_IMPORT_LDS];
}
__shared__ uint32_t etx_imp_bar_count[ETX_IMPORT_SLOTS], etx_imp_bar_gen[ETX_IMPORT_SLOTS];

extern "C" __device__ __attribute__((always_inline)) void etx_imp_slot_barrier(int slot, int nwaves) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  if (__builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)) == 0) {   // lane 0 of the wave
    const uint32_t g = __hip_atomic_load(&etx_imp_bar_gen[slot], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    if (__hip_atomic_fetch_add(&etx_imp_bar_count[slot], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_WORKGROUP) == (uint32_t)nwaves - 1) {
      __hip_atomic_store(&etx_imp_bar_count[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
      __hip_atomic_fetch_add(&etx_imp_bar_gen[slot], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
    } else {
      while (__hip_atomic_load(&etx_imp_bar_gen[slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) == g) __builtin_amdgcn_s_sleep(1);
    }
  }
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// zero the barrier words once per workgroup before the first imported task (the megakernel prologue)
static __device__ __forceinline__ void etx_import_init() {
  if (threadIdx.x < ETX_IMPORT_SLOTS) { etx_imp_bar_count[threadIdx.x] = 0; etx_imp_bar_gen[threadIdx.x] = 0; }
}
