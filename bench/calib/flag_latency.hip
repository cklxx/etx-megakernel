// One-way flag latency measured with the global 100 MHz s_memrealtime counter:
// the producer stores its timestamp as the payload; the consumer computes
// (now - payload) after its acquire. Also checks payload visibility (stale
// reads = broken release/acquire) and the isolated fence costs.
#include <hip/hip_runtime.h>
#include <cstdio>
#include "etx_lowering.h"
#include "etx/primitives.h"

__global__ void oneway(int32_t* flag, long long* payload, int producer, int consumer, int iters, long long* out) {
  if (threadIdx.x != 0) return;
  const int me = blockIdx.x;
  if (me == producer) {
    for (int i = 1; i <= iters; ++i) {
      payload[0] = ETX_TIMER();
      ETX_RELEASE_DEVICE();
      __hip_atomic_store(flag, i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      while (ETX_POLL_DEVICE(flag + 1) != i) { ETX_BACKOFF(); }   // ack, keeps the pair in lockstep
    }
  } else if (me == consumer) {
    long long acc = 0, stale = 0;
    for (int i = 1; i <= iters; ++i) {
      while (ETX_POLL_DEVICE(flag) != i) { }
      ETX_ACQUIRE_DEVICE();
      long long ts = payload[0];
      long long now = ETX_TIMER();
      if (ts == 0 || now < ts) stale++; else acc += now - ts;
      __hip_atomic_store(flag + 1, i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    out[0] = acc; out[1] = stale;
  }
}

__global__ void fence_cost(int iters, long long* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long t0 = ETX_TIMER();
  for (int i = 0; i < iters; ++i) { ETX_RELEASE_DEVICE(); }
  long long t1 = ETX_TIMER();
  for (int i = 0; i < iters; ++i) { ETX_ACQUIRE_DEVICE(); }
  long long t2 = ETX_TIMER();
  for (int i = 0; i < iters; ++i) { ETX_RELEASE_DOMAIN(); ETX_ACQUIRE_DOMAIN(); }
  long long t3 = ETX_TIMER();
  out[0] = t1 - t0; out[1] = t2 - t1; out[2] = t3 - t2;
}

int main() {
  hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
  int32_t* flag; long long* payload; hipMalloc(&flag, 8); hipMalloc(&payload, 8); hipMemset(flag, 0, 8); hipMemset(payload, 0, 8);
  long long* out; hipMallocManaged(&out, 32); out[0] = out[1] = out[2] = 0;
  const int iters = 10000;
  const int consumers[2] = {8, prop.multiProcessorCount - 1};     // wg 8 usually shares wg 0's domain; the last one usually does not
  for (int k = 0; k < 2; ++k) {
    out[0] = out[1] = 0; hipMemset(flag, 0, 8);
    oneway<<<prop.multiProcessorCount, 64>>>(flag, payload, 0, consumers[k], iters, out); hipDeviceSynchronize();
    printf("one-way flag wg0 -> wg%d: %.1f ns, stale payloads: %lld\n", consumers[k], (double)out[0] * 10.0 / iters, out[1]);
  }
  fence_cost<<<1, 64>>>(iters, out); hipDeviceSynchronize();
  printf("DEVICE release %.1f ns, DEVICE acquire %.1f ns, DOMAIN release+acquire %.1f ns\n",
         (double)out[0] * 10.0 / iters, (double)out[1] * 10.0 / iters, (double)out[2] * 10.0 / iters);
  return 0;
}
