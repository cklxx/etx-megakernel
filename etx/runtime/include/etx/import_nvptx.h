#pragma once
// Reserve CTA named barriers 1..15 for ETX imports. Pass slot-local tid to
// imported functions, and keep all threads of a slot on the same task.
#ifndef ETX_IMPORT_LDS
#define ETX_IMPORT_LDS 16
#endif
extern "C" {
__device__ __shared__ __attribute__((aligned(16))) unsigned char etx_lds_arena[ETX_IMPORT_LDS];
}
static __device__ __forceinline__ void etx_imp_slot_barrier(int slot, int threads) {
  asm volatile("bar.sync %0, %1;" :: "r"(1 + slot), "r"(threads) : "memory");
}
