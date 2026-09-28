// Support for kernels imported at the IR level (etx/importer/ir_import.py).
//
// etx_lds_arena   the LDS every imported tile uses instead of its own static __shared__ variables; imported
//                 tiles run one after another on a workgroup, so they share it. The megakernel TU defines
//                 it once (ETX_IMPORT_LDS bytes; the importer reports what each import needs).
// etx_imp_slot_barrier(slot, nwaves)
//                 the kernel's __syncthreads when its block is smaller than the megakernel workgroup: the
//                 workgroup's s_barrier with idle waves taking part in rounds (default), or, with
//                 -DETX_IMP_SW_BARRIER, a barrier over the slot's waves only with LDS counters.
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

#ifdef ETX_IMP_SW_BARRIER
// The software slot barrier (the imported kernel's __syncthreads over its own slot's waves): LDS counter + generation
// word, polling. Kept for A/B; measured 0.2-0.26 us per barrier against 77 ns for s_barrier (docs/MODEL-vllm-kernels.md).
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
#define ETX_IMP_ROUNDS 0
#else
// Hardware barrier in rounds: the imported kernel's __syncthreads is the workgroup's s_barrier, and every wave of the
// workgroup takes part in every one. A wave with no block, or done with its block, repeats s_barrier (etx_imp_rounds_end)
// until every wave has finished. Blocks of one kernel reach the same barriers in the same order, so a round is each
// unfinished wave's next barrier; a block's barrier is also a barrier over the other slots, which only orders more.
// Every counted barrier bumps the wave's own round number (etx_imp_nbar[wave], never reset): all waves hold the same
// number at every barrier, so the exit is decided by round number, not by re-reading a count (no read/increment race).
#define ETX_IMP_ROUNDS 1
#endif
__shared__ uint32_t etx_imp_done, etx_imp_exit, etx_imp_nbar[16];
static __device__ __forceinline__ bool etx_imp_lane0() { return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)) == 0; }
#if ETX_IMP_ROUNDS
static __device__ __forceinline__ void etx_imp_counted_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  if (etx_imp_lane0())                                             // wave-private: only this wave's lane 0 writes it
    __hip_atomic_fetch_add(&etx_imp_nbar[threadIdx.x / 64], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}
extern "C" __device__ __attribute__((always_inline)) void etx_imp_slot_barrier(int slot, int nwaves) {
  (void)slot; (void)nwaves;
  etx_imp_counted_barrier();
}
#endif
// before the blocks of one launch (kept for symmetry: the round protocol needs no reset here)
static __device__ __forceinline__ void etx_imp_rounds_begin() {}
// after this wave's block (or none). The wave whose increment completes the count (the last to finish) resets the count
// and sets the exit round to its next round, all before its next barrier; every wave leaves right after that barrier.
// A stale exit value from an earlier use is below the current round, so it never matches.
static __device__ __forceinline__ void etx_imp_rounds_end() {
#if ETX_IMP_ROUNDS
  const uint32_t w = threadIdx.x / 64, nw = blockDim.x / 64;
  // lane 0's own view of its own counter (same thread as every increment), broadcast to the wave
  uint32_t r = (uint32_t)__builtin_amdgcn_readfirstlane((int)__hip_atomic_load(&etx_imp_nbar[w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP));
  if (etx_imp_lane0()) {
    if (__hip_atomic_fetch_add(&etx_imp_done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) == nw - 1) {
      __hip_atomic_store(&etx_imp_done, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
      __hip_atomic_store(&etx_imp_exit, r + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
  }
  for (;;) {
    etx_imp_counted_barrier();
    ++r;
    if ((uint32_t)__builtin_amdgcn_readfirstlane((int)__hip_atomic_load(&etx_imp_exit, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)) == r) break;
  }
#else
  __syncthreads();   // the software-barrier build: the per-iteration barrier etx_all_blocks always had
#endif
}

// The argument block of an imported tile: args[0] + cst[0], made wave-uniform so the kernel's parameter reads are
// scalar loads (through the K$, which the vector-cache invalidates of ETX's acquires leave alone).
static __device__ __forceinline__ uint64_t etx_imp_sgpr64(uint64_t v) {
  // readfirstlane returns int: take both halves as uint32 before widening (a sign-extended low half corrupts the pointer)
  const uint32_t lo = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)v);
  const uint32_t hi = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)(v >> 32));
  return ((uint64_t)hi << 32) | (uint64_t)lo;
}
static __device__ __forceinline__ const unsigned char* etx_imp_argblock(const void* const* args, int32_t off) {
#ifdef ETX_IMP_VECTOR_ARGS   // A/B: the previous form (the parameters became vector loads through the invalidated L1/L2)
  return (const unsigned char*)args[0] + off;
#endif
  const void* const* a = (const void* const*)etx_imp_sgpr64((uint64_t)args);
  const __attribute__((address_space(4))) uint64_t* a4 = (const __attribute__((address_space(4))) uint64_t*)(uintptr_t)a;
  return (const unsigned char*)etx_imp_sgpr64(a4[0]) + (int32_t)__builtin_amdgcn_readfirstlane(off);
}
// pull `bytes` of an argument block into the scalar cache (one scalar load per 64 B line) and wait for them: run by the
// waiting wave before its event wait, so the kernel's own parameter loads after the acquire hit the K$
static __device__ __forceinline__ void etx_imp_touch(const unsigned char* a, int bytes) {
  const __attribute__((address_space(4))) uint32_t* a4 = (const __attribute__((address_space(4))) uint32_t*)(uintptr_t)a;
  uint32_t acc = a4[(bytes - 4) / 4];                  // the last dword too: the block need not start on a 64 B line
  for (int o = 0; o < bytes; o += 64) acc ^= a4[o / 4];
  asm volatile("" :: "s"(acc));
}

// zero the barrier words once per workgroup before the first imported task (the megakernel prologue)
static __device__ __forceinline__ void etx_import_init() {
  if (threadIdx.x < ETX_IMPORT_SLOTS) { etx_imp_bar_count[threadIdx.x] = 0; etx_imp_bar_gen[threadIdx.x] = 0; }
  if (threadIdx.x < 16) etx_imp_nbar[threadIdx.x] = 0;
  if (threadIdx.x == 0) { etx_imp_done = 0; etx_imp_exit = 0; }
}
