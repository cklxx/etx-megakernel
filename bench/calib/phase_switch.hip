// Whole-GPU phase switch with every resident workgroup participating:
//   counter : each WG does one atomic RMW on a per-phase counter, then polls it
//   flags   : each WG stores its own flag word (no RMW); WG 0 polls all flags
//             with inv-L1 loads and releases the phase with one "go" word
// Reports microseconds per phase switch for both. Corresponds to the
// t_phase_switch_* entries of the cost table (Kog's counter-vs-sentinel data).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "etx_lowering.h"
#include "etx/primitives.h"

__global__ void counter_barrier(int32_t* counters, int phases, int n_wg, long long* out) {
  if (threadIdx.x != 0) return;
  long long t0 = ETX_TIMER();
  for (int p = 0; p < phases; ++p) {
    ETX_RELEASE_DEVICE();
    atomicAdd(counters + p, 1);
    while (ETX_POLL_DEVICE(counters + p) < n_wg) { ETX_BACKOFF(); }
    ETX_ACQUIRE_DEVICE();
  }
  long long t1 = ETX_TIMER();
  if (blockIdx.x == 0) out[0] = t1 - t0;
}

__global__ void flag_barrier(int32_t* flags, int32_t* go, int phases, int n_wg, long long* out) {
  if (threadIdx.x != 0) return;
  const int me = blockIdx.x;
  long long t0 = ETX_TIMER();
  for (int p = 1; p <= phases; ++p) {
    ETX_RELEASE_DEVICE();
    __hip_atomic_store(flags + me, p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (me == 0) {
      for (int w = 0; w < n_wg; ++w) while (ETX_POLL_DEVICE(flags + w) < p) { }
      __hip_atomic_store(go, p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      while (ETX_POLL_DEVICE(go) < p) { ETX_BACKOFF(); }
    }
    ETX_ACQUIRE_DEVICE();
  }
  long long t1 = ETX_TIMER();
  if (me == 0) out[0] = t1 - t0;
}

int main() {
  hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
  const int phases = 2000;
  for (int mult = 1; mult <= 2; ++mult) {
    const int n_wg = prop.multiProcessorCount * mult;
    int32_t *counters, *flags, *go; long long* out;
    hipMalloc(&counters, phases * 4); hipMemset(counters, 0, phases * 4);
    hipMalloc(&flags, n_wg * 4); hipMemset(flags, 0, n_wg * 4);
    hipMalloc(&go, 4); hipMemset(go, 0, 4);
    hipMallocManaged(&out, 8);
    void* a1[] = {&counters, (void*)&phases, (void*)&n_wg, &out};
    hipLaunchCooperativeKernel((const void*)counter_barrier, dim3(n_wg), dim3(64), a1, 0, 0); hipDeviceSynchronize();
    double c_us = (double)out[0] * 10.0 / phases / 1000.0;
    void* a2[] = {&flags, &go, (void*)&phases, (void*)&n_wg, &out};
    hipLaunchCooperativeKernel((const void*)flag_barrier, dim3(n_wg), dim3(64), a2, 0, 0); hipDeviceSynchronize();
    double f_us = (double)out[0] * 10.0 / phases / 1000.0;
    printf("%d workgroups: counter barrier %.2f us/phase, flag(leader) barrier %.2f us/phase\n", n_wg, c_us, f_us);
  }
  return 0;
}
