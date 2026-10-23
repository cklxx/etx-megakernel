// NVIDIA-specific helpers referenced by the machine YAML lowering entries.
#pragma once
#include <cuda_runtime.h>

#ifdef ETX_SLOT_BY_CU
#error "ETX_SLOT_BY_CU is supported only on AMD"
#endif

// CUDA's unsuffixed atomicSub is relaxed and device-scoped.
static __device__ __forceinline__ int32_t etx_atomic_sub_device(int32_t* p, int32_t n) {
  return atomicSub(p, n);
}

static __device__ __forceinline__ int32_t etx_ld_acquire_gpu(const int32_t* p) {
  int32_t v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
static __device__ __forceinline__ int32_t etx_ld_acquire_sys(const int32_t* p) {
  int32_t v;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
static __device__ __forceinline__ int32_t etx_atomic_sub_sys(int32_t* p) {
  return atomicSub_system(p, 1);
}

// Relaxed device-scope store (relay mirror writes).
static __device__ __forceinline__ void etx_store_relaxed_device(int32_t* p, int32_t v) {
  asm volatile("st.relaxed.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
