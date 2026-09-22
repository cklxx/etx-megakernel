// Two devices: system-scope flag ping-pong through fine-grained memory owned by
// device 0 and mapped into device 1 (peer access), plus peer store bandwidth.
// Feeds t_dev_ns and the cross-device path of Pass 4 / the GEMM+RS example.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <thread>
#include "etx_lowering.h"
#include "etx/primitives.h"

__global__ void pingpong(int32_t* flag, int32_t mine, int iters, long long* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long t0 = ETX_TIMER();
  for (int i = 0; i < iters; ++i) {
    while ((ETX_POLL_SYSTEM(flag) & 1) != mine) { }
    ETX_RELEASE_SYSTEM();
    __hip_atomic_fetch_add(flag, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  long long t1 = ETX_TIMER();
  if (mine == 0) out[0] = t1 - t0;
}

__global__ void peer_store(float4* dst, int n, long long* out) {
  long long t0 = ETX_TIMER();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = make_float4(1.f, 2.f, 3.f, 4.f);
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = ETX_TIMER() - t0;
}

int main() {
  int n = 0; hipGetDeviceCount(&n);
  if (n < 2) { printf("need 2 devices, have %d\n", n); return 0; }
  int can = 0; hipDeviceCanAccessPeer(&can, 1, 0);
  printf("device 1 can access device 0: %d\n", can);
  hipSetDevice(0);
  int32_t* flag; hipExtMallocWithFlags((void**)&flag, 64, hipDeviceMallocFinegrained); hipMemset(flag, 0, 64);
  long long* out0; hipMallocManaged(&out0, 8);
  hipSetDevice(1); hipDeviceEnablePeerAccess(0, 0);
  long long* out1; hipMallocManaged(&out1, 8);
  const int iters = 5000;
  std::thread t1([&] { hipSetDevice(1); pingpong<<<1, 64>>>(flag, 1, iters, out1); hipDeviceSynchronize(); });
  hipSetDevice(0); pingpong<<<1, 64>>>(flag, 0, iters, out0); hipDeviceSynchronize();
  t1.join();
  printf("cross-device system-scope flag one-way: %.1f ns (fine-grained memory on device 0)\n", (double)out0[0] * 10.0 / iters / 2.0);
  // peer store bandwidth: device 1 writes 256 MB into device 0 memory
  hipSetDevice(0); float4* buf; hipMalloc(&buf, 256u << 20);
  hipSetDevice(1);
  const int n4 = (256 << 20) / 16;
  peer_store<<<1024, 256>>>(buf, n4, out1); hipDeviceSynchronize();      // warm
  peer_store<<<1024, 256>>>(buf, n4, out1); hipDeviceSynchronize();
  printf("peer store bandwidth (device 1 -> device 0 HBM): %.1f GB/s\n", 256.0 / ((double)out1[0] * 10.0 / 1e9) / 1024.0);
  hipSetDevice(0); float4* local; hipMalloc(&local, 256u << 20);
  peer_store<<<1024, 256>>>(local, n4, out0); hipDeviceSynchronize();
  peer_store<<<1024, 256>>>(local, n4, out0); hipDeviceSynchronize();
  printf("local store bandwidth (device 0 -> own HBM, same kernel): %.1f GB/s\n", 256.0 / ((double)out0[0] * 10.0 / 1e9) / 1024.0);
  return 0;
}
