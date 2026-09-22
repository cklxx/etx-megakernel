// Atomic ping-pong between two workgroups. Pin them to domains by launching a
// full persistent grid, reading the domain id of every workgroup, and letting
// only the chosen pair play. Reports ns per round trip for (same, cross) domain.
// Build: hipcc -O2 --offload-arch=gfx942 -I etx/runtime/include -I build/calib atomic_pingpong.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "etx_lowering.h"
#include "etx/primitives.h"

__global__ void discover(uint32_t* dom) { if (threadIdx.x == 0) dom[blockIdx.x] = ETX_DOMAIN_ID(); }

__global__ void pingpong(int32_t* flag, int a, int b, int iters, long long* out) {
  if (threadIdx.x != 0) return;
  const int me = blockIdx.x;
  if (me != a && me != b) return;
  const int32_t mine = (me == a) ? 0 : 1;
  long long t0 = ETX_TIMER();
  for (int i = 0; i < iters; ++i) {
    while ((ETX_POLL_DEVICE(flag) & 1) != mine) { ETX_BACKOFF(); }
    ETX_RELEASE_DEVICE();
    atomicAdd(flag, 1);
  }
  long long t1 = ETX_TIMER();
  if (me == a) out[0] = t1 - t0;
}

int main() {
  int dev_count = 0; hipGetDeviceCount(&dev_count);
  hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
  const int grid = prop.multiProcessorCount;
  uint32_t* dom; hipMalloc(&dom, grid * sizeof(uint32_t));
  discover<<<grid, 64>>>(dom); hipDeviceSynchronize();
  std::vector<uint32_t> h(grid); hipMemcpy(h.data(), dom, grid * 4, hipMemcpyDeviceToHost);
  int same_b = -1, cross_b = -1;
  for (int i = 1; i < grid; ++i) { if (h[i] == h[0] && same_b < 0) same_b = i; if (h[i] != h[0] && cross_b < 0) cross_b = i; }
  int32_t* flag; hipMalloc(&flag, 4); hipMemset(flag, 0, 4);
  long long* out; hipMallocManaged(&out, 8);
  const int iters = 20000;
  for (int pair = 0; pair < 2; ++pair) {
    int b = pair == 0 ? same_b : cross_b;
    if (b < 0) continue;
    hipMemset(flag, 0, 4);
    pingpong<<<grid, 64>>>(flag, 0, b, iters, out); hipDeviceSynchronize();
    // ETX_TIMER on AMD is s_memrealtime at 100 MHz -> 10 ns ticks
    printf("%s-domain round trip: %.1f ns (wg 0 on domain %u, wg %d on domain %u)\n",
           pair == 0 ? "same" : "cross", (double)out[0] * 10.0 / iters / 2.0, h[0], b, h[b]);
  }
  return 0;
}
