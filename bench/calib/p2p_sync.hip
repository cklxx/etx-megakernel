// Cross-device synchronisation for MoE decode (expert-parallel dispatch):
// one-way flag latency between two GPUs as a function of where the flag
// lives (device 0 / device 1 / host pinned), with and without a 4 KB payload
// made visible through the release/acquire sequence, and under a streaming
// load on both GPUs. Latency is measured with s_memrealtime stamps (100 MHz,
// consistent across the two devices on one board is NOT guaranteed, so the
// round trip is measured on one device and halved).
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include "etx_lowering.h"
#include "etx/primitives.h"

__global__ void pingpong(int32_t* flag, float* payload, int32_t mine, int iters, int payload_words, long long* out) {
  if (blockIdx.x != 0) return;
  long long t0 = ETX_TIMER();
  for (int i = 0; i < iters; ++i) {
    if (threadIdx.x == 0) { while ((ETX_POLL_SYSTEM(flag) & 1) != mine) { } }
    __syncthreads();
    ETX_ACQUIRE_SYSTEM();
    float acc = 0.f;
    for (int w = threadIdx.x; w < payload_words; w += blockDim.x) acc += payload[w];        // consume the payload
    for (int w = threadIdx.x; w < payload_words; w += blockDim.x) payload[w] = acc + i;     // produce the next one
    __syncthreads();
    if (threadIdx.x == 0) { ETX_RELEASE_SYSTEM(); __hip_atomic_fetch_add(flag, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
  }
  long long t1 = ETX_TIMER();
  if (mine == 0 && threadIdx.x == 0) out[0] = t1 - t0;
}

__global__ void stream_load(const float4* src, float4* dst, size_t n, int reps) {
  float4 acc = make_float4(0, 0, 0, 0);
  for (int r = 0; r < reps; ++r)
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = src[i]; acc.x += v.x; }
  if (acc.x == 12345.f) dst[0] = acc;
}

static void* alloc_flag(int where) {   // 0: device 0 fine-grained, 1: device 1 fine-grained, 2: host pinned
  void* p;
  if (where == 2) { hipHostMalloc(&p, 4096, hipHostMallocCoherent); }
  else { hipSetDevice(where); hipExtMallocWithFlags(&p, 4096, hipDeviceMallocFinegrained); }
  hipMemset(p, 0, 4096);
  return p;
}

int main() {
  int n = 0; hipGetDeviceCount(&n);
  if (n < 2) { printf("need 2 devices\n"); return 0; }
  for (int d = 0; d < 2; ++d) { hipSetDevice(d); hipDeviceEnablePeerAccess(1 - d, 0); }
  long long* out; hipSetDevice(0); hipMallocManaged(&out, 8);
  const int iters = 2000;
  const char* where_name[3] = {"flag on device 0", "flag on device 1", "flag on host   "};
  // streaming load buffers (1 GB each device)
  float4 *src0, *dst0, *src1, *dst1; const size_t nload = (1u << 30) / 16;
  hipSetDevice(0); hipMalloc(&src0, 1u << 30); hipMalloc(&dst0, 64);
  hipSetDevice(1); hipMalloc(&src1, 1u << 30); hipMalloc(&dst1, 64);
  for (int load = 0; load < 2; ++load) {
    for (int where = 0; where < 3; ++where) {
      for (int pw : {0, 1024}) {                       // 0 words, or 4 KB payload
        int32_t* flag = (int32_t*)alloc_flag(where);
        float* payload = (float*)(flag + 16);
        hipStream_t ls0, ls1;
        hipSetDevice(0); hipStreamCreateWithFlags(&ls0, hipStreamNonBlocking);
        hipSetDevice(1); hipStreamCreateWithFlags(&ls1, hipStreamNonBlocking);
        if (load) { hipSetDevice(0); stream_load<<<1024, 256, 0, ls0>>>(src0, dst0, nload, 200); hipSetDevice(1); stream_load<<<1024, 256, 0, ls1>>>(src1, dst1, nload, 200); }
        std::thread t1([&] { hipSetDevice(1); pingpong<<<1, 256>>>(flag, payload, 1, iters, pw, out); hipDeviceSynchronize(); });
        hipSetDevice(0); pingpong<<<1, 256>>>(flag, payload, 0, iters, pw, out); hipDeviceSynchronize();
        t1.join();
        hipSetDevice(0); hipStreamSynchronize(ls0); hipSetDevice(1); hipStreamSynchronize(ls1);
        printf("%s | %s | payload %4d B : one-way %.2f us\n", load ? "under 2x1GB streaming load" : "idle                      ", where_name[where], pw * 4, (double)out[0] * 10.0 / iters / 2.0 / 1000.0);
        if (where == 2) hipHostFree(flag); else hipFree(flag);
      }
    }
  }
  return 0;
}
