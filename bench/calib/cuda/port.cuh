#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Match HIP wall_clock64's 10 ns units, using the GPU-wide PTX nanosecond timer.
__device__ __forceinline__ unsigned long long wall_clock64() {
  unsigned long long ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  return ns / 10;
}
__device__ __forceinline__ uint32_t sm_id() {
  uint32_t id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}
// NVIDIA has one device-wide L2 domain; an SM is not an AMD XCD.
__device__ __forceinline__ uint32_t cuda_domain() { return 0; }
__device__ __forceinline__ int port_load(const int* p) {
  return atomicAdd(const_cast<int*>(p), 0);
}
__device__ __forceinline__ void port_store(int* p, int v) { atomicExch(p, v); }
__device__ __forceinline__ void port_sleep(unsigned n) { __nanosleep(n * 32); }
#define ETX_DOMAIN_ID() cuda_domain()
#define ETX_TIMER() wall_clock64()
#define ETX_POLL_DEVICE(p) port_load(p)
#define ETX_RELEASE_DEVICE() __threadfence()
#define ETX_RELEASE_DOMAIN() __threadfence()
#define ETX_BACKOFF() port_sleep(1)
struct etx_task { int32_t type; int32_t coord[4]; };
struct etx_queue { int32_t* slots; int32_t capacity; int32_t* head; int32_t* tail; };
__device__ __forceinline__ int32_t etx_try_pop(const etx_queue* q, int32_t* ticket) {
  if (!q->capacity) return -1;
  if (*ticket < 0) {
    if (port_load(q->head) >= port_load(q->tail)) return -1;
    *ticket = atomicAdd(q->head, 1);
  }
  int v = port_load(q->slots + (*ticket % q->capacity));
  if (v < 0 || (v >> 24) != (*ticket / q->capacity)) return -1;
  *ticket = -1;
  return v & 0x00ffffff;
}
inline bool port_help(int argc, char** argv, const char* usage) {
  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    printf("%s\nCUDA syntax port: compiled, never run. No GPU validation.\n", usage);
    return true;
  }
  return false;
}
#define PORT_LAUNCH(kernel, grid, block, shared, stream, ...) \
  kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)

#define PORT_CHECK(expr) do { \
  cudaError_t error_ = (expr); \
  if (error_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(error_)); \
    exit(1); \
  } \
} while (0)
