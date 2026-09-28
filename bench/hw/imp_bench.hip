// The imported kernels' barrier protocols, measured with the header the megakernel uses (etx/runtime/include/etx/import.h),
// in the megakernel's workgroup (1024 threads, one per CU, every CU busy). Built twice:
//   hipcc -O3 --offload-arch=gfx942 -Ietx/runtime/include bench/hw/imp_bench.hip -o imp_bench_hw        (s_barrier in rounds)
//   ... -DETX_IMP_SW_BARRIER -o imp_bench_sw                                                              (software slot barrier)
// A "block" is what the adapter runs per task (examples: fused_moe, one 256-thread slot and 12 idle waves; topk, four
// 256-thread slots; wvSplitK, one 1024-thread slot): etx_imp_rounds_begin(); if (slot < A) { R barriers }; etx_imp_rounds_end().
// Between barriers each active thread writes and reads one LDS word (a barrier that orders something).
// Reports per block: the time, and (R > 0) the time per barrier after subtracting the R = 0 block (the exit protocol alone).
// Parameter of the kernel model it replaces: t_swbar (0.26 us, estimated from the instruction count) and, for the round
// protocol, the per-barrier cost the model took as a bare s_barrier (77 ns).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include "etx/import.h"

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(1); } } while (0)

__global__ void __launch_bounds__(1024) k_bar(int S, int A, int R, int reps, uint64_t* t, float* sink) {
  __shared__ float lds[1024];
  etx_import_init();
  __syncthreads();
  const int slot = threadIdx.x / S, nw = S / 64;
  float a = threadIdx.x;
  const uint64_t t0 = wall_clock64();
  for (int rep = 0; rep < reps; ++rep) {
    etx_imp_rounds_begin();
    if (slot < A) {
      for (int r = 0; r < R; ++r) {
        lds[threadIdx.x] = a + r;
        etx_imp_slot_barrier(slot, nw);
        a += lds[slot * S + ((threadIdx.x - slot * S + 64) % S)];
      }
    }
    etx_imp_rounds_end();
  }
  const uint64_t t1 = wall_clock64();
  if (threadIdx.x == 0) t[blockIdx.x] = t1 - t0;
  if (a == 1.2345f) sink[blockIdx.x] = a;
}

// the plain workgroup barrier, same pattern (the floor)
__global__ void __launch_bounds__(1024) k_sync(int R, int reps, uint64_t* t, float* sink) {
  __shared__ float lds[1024];
  float a = threadIdx.x;
  const uint64_t t0 = wall_clock64();
  for (int rep = 0; rep < reps; ++rep)
    for (int r = 0; r < R; ++r) { lds[threadIdx.x] = a + r; __syncthreads(); a += lds[(threadIdx.x + 64) % 1024]; __syncthreads(); }
  const uint64_t t1 = wall_clock64();
  if (threadIdx.x == 0) t[blockIdx.x] = t1 - t0;
  if (a == 1.2345f) sink[blockIdx.x] = a;
}

static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, 0));
  const int ncu = prop.multiProcessorCount;
#ifdef ETX_IMP_SW_BARRIER
  const char* proto = "sw    ";
#else
  const char* proto = "rounds";
#endif
  printf("device %s, %d CUs, protocol %s\n", prop.gcnArchName, ncu, proto);
  uint64_t* t; CHECK(hipMalloc(&t, ncu * 8)); float* sink; CHECK(hipMalloc(&sink, ncu * 4));
  const size_t lds = 40 * 1024;                            // one workgroup per CU
  const int reps = 200;
  auto run = [&](auto launch) {
    launch(); CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> h(ncu); CHECK(hipMemcpy(h.data(), t, ncu * 8, hipMemcpyDeviceToHost));
    std::vector<double> us; for (int i = 0; i < ncu; ++i) us.push_back(h[i] * 0.01 / reps);
    return med(us);
  };
  {
    const double b = run([&] { hipLaunchKernelGGL(k_sync, dim3(ncu), dim3(1024), lds, 0, 32, reps, t, sink); });
    printf("RESULT impbar %s __syncthreads (1024 threads): %6.1f ns per barrier\n", proto, b / 64 * 1e3);
  }
  struct C { int S, A; const char* what; };
  for (const C& c : {C{256, 1, "fused_moe (1 x 256, 12 idle waves)"}, C{256, 4, "topk (4 x 256)"}, C{1024, 1, "wvSplitK (1 x 1024)"}}) {
    double r0 = 0;
    for (int R : {0, 1, 8, 32}) {
      const double us = run([&] { hipLaunchKernelGGL(k_bar, dim3(ncu), dim3(1024), lds, 0, c.S, c.A, R, reps, t, sink); });
      if (R == 0) r0 = us;
      if (R == 0) printf("RESULT impbar %s %-36s R %2d: block %7.3f us (the exit protocol alone)\n", proto, c.what, R, us);
      else printf("RESULT impbar %s %-36s R %2d: block %7.3f us, per barrier %6.1f ns\n", proto, c.what, R, us, (us - r0) / R * 1e3);
    }
  }
  return 0;
}
