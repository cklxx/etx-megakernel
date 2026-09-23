// Cross-device synchronisation for MoE decode (expert-parallel dispatch):
// one-way flag latency between two GPUs as a function of where the flag lives
// (device 0 / device 1 fine-grained memory / host pinned), with and without a
// 4 KB payload made visible through the release/acquire sequence, idle and
// under a streaming load on both GPUs. Each device measures its own round trip
// (arrive -> see the reply) with its own s_memrealtime, so no cross-device
// clock comparison is needed; one-way = round trip / 2, reported per initiator.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include "etx_lowering.h"
#include "etx/primitives.h"

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

__global__ void pingpong(int32_t* flag, float* payload, int32_t mine, int iters, int payload_words, long long* out) {
  if (blockIdx.x != 0) return;
  long long acc_t = 0;
  for (int i = 0; i < iters; ++i) {
    if (threadIdx.x == 0) { while ((ETX_POLL_SYSTEM(flag) & 1) != mine) { } }
    __syncthreads();
    ETX_ACQUIRE_SYSTEM();
    float acc = 0.f;
    for (int w = threadIdx.x; w < payload_words; w += blockDim.x) acc += payload[w];
    for (int w = threadIdx.x; w < payload_words; w += blockDim.x) payload[w] = acc + i;
    __syncthreads();
    if (threadIdx.x == 0) {
      const long long t0 = ETX_TIMER();
      ETX_RELEASE_SYSTEM();
      __hip_atomic_fetch_add(flag, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      // round trip: from our arrive until the peer's reply is visible
      while ((ETX_POLL_SYSTEM(flag) & 1) != mine) { }
      acc_t += ETX_TIMER() - t0;
      // hand the token back so the loop structure stays symmetric: the peer's own arrive flips it again
    }
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_fetch_add(flag, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __syncthreads();
  }
  if (threadIdx.x == 0) out[mine] = acc_t;
}

__global__ void stream_load(const float4* src, float4* dst, size_t n, int reps) {
  float4 acc = make_float4(0, 0, 0, 0);
  for (int r = 0; r < reps; ++r)
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = src[i]; acc.x += v.x; }
  if (acc.x == 12345.f) dst[0] = acc;
}

static void* alloc_flag(int where) {   // 0: device 0 fine-grained, 1: device 1 fine-grained, 2: host pinned coherent
  void* p = nullptr;
  if (where == 2) { CK(hipHostMalloc(&p, 8192, hipHostMallocCoherent)); for (int i = 0; i < 2048; ++i) ((int32_t*)p)[i] = 0; }
  else { CK(hipSetDevice(where)); CK(hipExtMallocWithFlags(&p, 8192, hipDeviceMallocFinegrained)); CK(hipMemset(p, 0, 8192)); }
  return p;
}

int main() {
  int n = 0; CK(hipGetDeviceCount(&n));
  if (n < 2) { printf("need 2 devices\n"); return 0; }
  for (int d = 0; d < 2; ++d) { CK(hipSetDevice(d)); hipError_t e = hipDeviceEnablePeerAccess(1 - d, 0); if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) { printf("peer access %d->%d: %s\n", d, 1 - d, hipGetErrorString(e)); } }
  long long* out; CK(hipHostMalloc((void**)&out, 64, hipHostMallocCoherent));
  const int iters = 2000;
  const char* where_name[3] = {"flag on device 0", "flag on device 1", "flag on host    "};
  float4 *src0, *dst0, *src1, *dst1; const size_t nload = (1u << 30) / 16;
  CK(hipSetDevice(0)); CK(hipMalloc(&src0, 1u << 30)); CK(hipMalloc(&dst0, 64));
  CK(hipSetDevice(1)); CK(hipMalloc(&src1, 1u << 30)); CK(hipMalloc(&dst1, 64));
  hipStream_t ls0, ls1;
  CK(hipSetDevice(0)); CK(hipStreamCreateWithFlags(&ls0, hipStreamNonBlocking));
  CK(hipSetDevice(1)); CK(hipStreamCreateWithFlags(&ls1, hipStreamNonBlocking));
  printf("%-28s %-18s %8s %14s %14s\n", "condition", "flag location", "payload", "dev0 one-way", "dev1 one-way");
  for (int load = 0; load < 2; ++load) {
    for (int where = 0; where < 3; ++where) {
      for (int pw : {0, 1024}) {
        int32_t* flag = (int32_t*)alloc_flag(where);
        float* payload = (float*)(flag + 64);
        out[0] = out[1] = -1;
        if (load) {
          CK(hipSetDevice(0)); stream_load<<<1024, 256, 0, ls0>>>(src0, dst0, nload, 400); CK(hipGetLastError());
          CK(hipSetDevice(1)); stream_load<<<1024, 256, 0, ls1>>>(src1, dst1, nload, 400); CK(hipGetLastError());
        }
        std::thread t1([&] { CK(hipSetDevice(1)); pingpong<<<1, 256>>>(flag, payload, 1, iters, pw, out); CK(hipGetLastError()); CK(hipDeviceSynchronize()); });
        CK(hipSetDevice(0)); pingpong<<<1, 256>>>(flag, payload, 0, iters, pw, out); CK(hipGetLastError()); CK(hipDeviceSynchronize());
        t1.join();
        CK(hipSetDevice(0)); CK(hipStreamSynchronize(ls0)); CK(hipSetDevice(1)); CK(hipStreamSynchronize(ls1));
        printf("%-28s %-18s %6d B %11.2f us %11.2f us\n", load ? "under 2x1GB streaming load" : "idle", where_name[where], pw * 4,
               (double)out[0] * 10.0 / iters / 2.0 / 1000.0, (double)out[1] * 10.0 / iters / 2.0 / 1000.0);
        if (where == 2) CK(hipHostFree(flag)); else { CK(hipSetDevice(where)); CK(hipFree(flag)); }
      }
    }
  }
  return 0;
}
