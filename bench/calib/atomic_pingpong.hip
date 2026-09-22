// Atomic ping-pong between two workgroups under three polling protocols, for a
// same-domain pair and a cross-domain pair. Every arrive is an agent-scope
// atomic RMW. Protocols:
//   DEVICE : release = buffer_wbl2 sc1 ; poll = agent-scope atomic load
//   INVL1  : release = s_waitcnt only  ; poll = buffer_inv sc0 (drop L1) + plain load (L2 hit if the RMW landed in L2)
//   SC0    : release = s_waitcnt only  ; poll = global_load ... sc0
// A protocol that cannot observe the other party's arrive is reported as
// "NO PROGRESS" instead of hanging (bounded spin). Reports ns per one-way hop.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "etx_lowering.h"
#include "etx/primitives.h"

__global__ void discover(uint32_t* dom) { if (threadIdx.x == 0) dom[blockIdx.x] = ETX_DOMAIN_ID(); }

static __device__ __forceinline__ int32_t load_sc0(const int32_t* p) {
#if defined(__gfx942__) || defined(__gfx950__)
  int32_t v;
  asm volatile("global_load_dword %0, %1, off sc0\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  return v;
#else
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}
static __device__ __forceinline__ int32_t load_invl1(const int32_t* p) {
  asm volatile("buffer_inv sc0" ::: "memory");
  return *(volatile const int32_t*)p;
}

template <int PROTO>
__global__ void pingpong(int32_t* flag, int a, int b, int iters, long long* out) {
  if (threadIdx.x != 0) return;
  const int me = blockIdx.x;
  if (me != a && me != b) return;
  const int32_t mine = (me == a) ? 0 : 1;
  long long t0 = ETX_TIMER();
  long long spins = 0;
  for (int i = 0; i < iters; ++i) {
    for (;;) {
      int32_t v = (PROTO == 0) ? ETX_POLL_DEVICE(flag) : (PROTO == 1) ? load_invl1(flag) : load_sc0(flag);
      if ((v & 1) == mine) break;
      if (++spins > 200000000LL) { if (me == a) { out[0] = -1; } return; }
    }
    if (PROTO == 0) ETX_RELEASE_DEVICE(); else ETX_RELEASE_DOMAIN();
    __hip_atomic_fetch_add(flag, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  long long t1 = ETX_TIMER();
  if (me == a) out[0] = t1 - t0;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
  const int grid = prop.multiProcessorCount;
  uint32_t* dom; hipMalloc(&dom, grid * sizeof(uint32_t));
  discover<<<grid, 64>>>(dom); hipDeviceSynchronize();
  std::vector<uint32_t> h(grid); hipMemcpy(h.data(), dom, grid * 4, hipMemcpyDeviceToHost);
  printf("workgroup -> domain (first 16): ");
  for (int i = 0; i < 16 && i < grid; ++i) printf("%u ", h[i]);
  printf("\n");
  int same_b = -1, cross_b = -1;
  for (int i = 1; i < grid; ++i) { if (h[i] == h[0] && same_b < 0) same_b = i; if (h[i] != h[0] && cross_b < 0) cross_b = i; }
  int32_t* flag; hipMalloc(&flag, 4);
  long long* out; hipMallocManaged(&out, 8);
  const int iters = 20000;
  const char* proto_name[3] = {"DEVICE wbl2+agent-poll", "INVL1  inv-sc0+load   ", "SC0    load-sc0       "};
  for (int proto = 0; proto < 3; ++proto) {
    for (int pair = 0; pair < 2; ++pair) {
      int b = pair == 0 ? same_b : cross_b;
      if (b < 0) continue;
      hipMemset(flag, 0, 4); out[0] = 0;
      if (proto == 0) pingpong<0><<<grid, 64>>>(flag, 0, b, iters, out);
      else if (proto == 1) pingpong<1><<<grid, 64>>>(flag, 0, b, iters, out);
      else pingpong<2><<<grid, 64>>>(flag, 0, b, iters, out);
      hipDeviceSynchronize();
      if (out[0] < 0) printf("%s %s-domain: NO PROGRESS (poll never observes the arrive)\n", proto_name[proto], pair == 0 ? "same " : "cross");
      else printf("%s %s-domain one-way: %.1f ns (wg 0 on domain %u, wg %d on domain %u)\n",
                  proto_name[proto], pair == 0 ? "same " : "cross", (double)out[0] * 10.0 / iters / 2.0, h[0], b, h[b]);
    }
  }
  return 0;
}
