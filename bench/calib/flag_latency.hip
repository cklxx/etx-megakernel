// One-way flag latency with the exact release/acquire sequence the lowering
// table prescribes for DEVICE scope, plus the fence cost in isolation.
#include <hip/hip_runtime.h>
#include <cstdio>
#include "etx_lowering.h"
#include "etx/primitives.h"

__global__ void oneway(int32_t* flag, int32_t* payload, int producer, int consumer, int iters, long long* out) {
  if (threadIdx.x != 0) return;
  const int me = blockIdx.x;
  if (me == producer) {
    for (int i = 1; i <= iters; ++i) {
      payload[0] = i;
      ETX_RELEASE_DEVICE();
      __hip_atomic_store(flag, i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      while (ETX_POLL_DEVICE(flag + 1) != i) { ETX_BACKOFF(); }   // ack
    }
  } else if (me == consumer) {
    long long acc = 0;
    for (int i = 1; i <= iters; ++i) {
      long long t0 = ETX_TIMER();
      while (ETX_POLL_DEVICE(flag) != i) { }
      ETX_ACQUIRE_DEVICE();
      if (payload[0] != i) out[1]++;                                  // stale payload = broken visibility
      acc += ETX_TIMER() - t0;
      __hip_atomic_store(flag + 1, i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    out[0] = acc;
  }
}

__global__ void fence_cost(int iters, long long* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long t0 = ETX_TIMER();
  for (int i = 0; i < iters; ++i) { ETX_RELEASE_DEVICE(); }
  long long t1 = ETX_TIMER();
  for (int i = 0; i < iters; ++i) { ETX_ACQUIRE_DEVICE(); }
  long long t2 = ETX_TIMER();
  out[0] = t1 - t0; out[1] = t2 - t1;
}

int main() {
  hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
  int32_t *flag, *payload; hipMalloc(&flag, 8); hipMalloc(&payload, 4); hipMemset(flag, 0, 8);
  long long* out; hipMallocManaged(&out, 16); out[0] = out[1] = 0;
  const int iters = 10000;
  oneway<<<prop.multiProcessorCount, 64>>>(flag, payload, 0, prop.multiProcessorCount - 1, iters, out); hipDeviceSynchronize();
  printf("one-way flag (wg0 -> last wg): %.1f ns, stale payloads: %lld\n", (double)out[0] * 10.0 / iters, out[1]);
  out[0] = out[1] = 0;
  fence_cost<<<1, 64>>>(iters, out); hipDeviceSynchronize();
  printf("release fence %.1f ns, acquire fence %.1f ns\n", (double)out[0] * 10.0 / iters, (double)out[1] * 10.0 / iters);
  return 0;
}
