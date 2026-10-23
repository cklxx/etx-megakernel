// CUDA syntax port of bench/calib/atomic_pingpong.hip. Status: compiled, never run.
// Source comments below describe the HIP experiment; see README.md for CUDA mappings and exclusions.
// Atomic ping-pong between two workgroups under three polling protocols, for a
// same-domain pair and a cross-domain pair. Every arrive is an agent-scope
// atomic RMW. Protocols:
//   DEVICE : release = buffer_wbl2 sc1 ; poll = agent-scope atomic load
//   INVL1  : release = s_waitcnt only  ; poll = buffer_inv sc0 (drop L1) + plain load (L2 hit if the RMW landed in L2)
//   SC0    : release = s_waitcnt only  ; poll = global_load ... sc0
// A protocol that cannot observe the other party's arrive is reported as
// "NO PROGRESS" instead of hanging (bounded spin). Reports ns per one-way hop.
#include "port.cuh"
#include <cstdio>
#include <vector>



__global__ void discover(uint32_t* dom) { if (threadIdx.x == 0) dom[blockIdx.x] = ETX_DOMAIN_ID(); }

static __device__ __forceinline__ int32_t load_sc0(const int32_t* p) {
  int32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
static __device__ __forceinline__ int32_t load_invl1(const int32_t* p) {
  __threadfence();
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
    atomicAdd(flag, 1);
  }
  long long t1 = ETX_TIMER();
  if (me == a) out[0] = t1 - t0;
}

int main(int argc, char** argv) {
  if (port_help(argc, argv, "atomic_pingpong (DEVICE, same device-wide L2 domain)")) return 0;
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaDeviceProp prop; PORT_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount;
  uint32_t* dom; PORT_CHECK(cudaMalloc(&dom, grid * sizeof(uint32_t)));
  discover<<<grid, 64>>>(dom); cudaDeviceSynchronize();
  std::vector<uint32_t> h(grid); PORT_CHECK(cudaMemcpy(h.data(), dom, grid * 4, cudaMemcpyDeviceToHost));
  printf("workgroup -> domain (first 16): ");
  for (int i = 0; i < 16 && i < grid; ++i) printf("%u ", h[i]);
  printf("\n");
  int same_b = -1, cross_b = -1;
  for (int i = 1; i < grid; ++i) { if (h[i] == h[0] && same_b < 0) same_b = i; if (h[i] != h[0] && cross_b < 0) cross_b = i; }
  int32_t* flag; PORT_CHECK(cudaMalloc(&flag, 4));
  long long* out; PORT_CHECK(cudaMallocManaged(&out, 8));
  if (same_b < 0) { fprintf(stderr, "No pair of workgroups found\n"); return 2; }
  const int iters = 20000;
  const char* proto_name[3] = {"DEVICE wbl2+agent-poll", "INVL1  inv-sc0+load   ", "SC0    load-sc0       "};
  for (int proto = 0; proto < 1; ++proto) {
    for (int pair = 0; pair < 2; ++pair) {
      int b = pair == 0 ? same_b : cross_b;
      if (b < 0) continue;
      PORT_CHECK(cudaMemset(flag, 0, 4)); out[0] = 0;
      if (proto == 0) pingpong<0><<<grid, 64>>>(flag, 0, b, iters, out);
      else if (proto == 1) pingpong<1><<<grid, 64>>>(flag, 0, b, iters, out);
      else pingpong<2><<<grid, 64>>>(flag, 0, b, iters, out);
      cudaDeviceSynchronize();
      if (out[0] < 0) printf("%s %s-domain: NO PROGRESS (poll never observes the arrive)\n", proto_name[proto], pair == 0 ? "same " : "cross");
      else printf("%s %s-domain one-way: %.1f ns (wg 0 on domain %u, wg %d on domain %u)\n",
                  proto_name[proto], pair == 0 ? "same " : "cross", (double)out[0] * 10.0 / iters / 2.0, h[0], b, h[b]);
    }
  }
  return 0;
}
