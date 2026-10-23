// CUDA port of bench/calib/p2p_sync.hip. Status: compiled, never run.
// Each initiator times a round trip on its own clock; one-way = RTT / 2.
#include "port.cuh"
#include <cuda/atomic>
#include <initializer_list>

using system_flag = cuda::atomic_ref<int, cuda::thread_scope_system>;
constexpr unsigned long long deadline_ticks = 500000000ULL; // five seconds

__device__ bool await_flag(int* flag, int value, unsigned long long start) {
  while (system_flag(*flag).load(cuda::memory_order_acquire) != value) {
    if (ETX_TIMER() - start > deadline_ticks) return false;
  }
  return true;
}

// Both request and reply, plus the optional payload, live on the consumer.
__global__ void exchange(int* request, int* reply, float* payload, int words,
                         int iters, bool initiator, long long* elapsed) {
  const auto start = ETX_TIMER();
  long long total = 0;
  for (int i = 1; i <= iters; ++i) {
    if (initiator) {
      for (int w = 0; w < words; ++w) payload[w] = float(i);
      const auto t0 = ETX_TIMER();
      system_flag(*request).store(i, cuda::memory_order_release);
      if (!await_flag(reply, i, start)) { *elapsed = -1; return; }
      total += ETX_TIMER() - t0;
    } else {
      if (!await_flag(request, i, start)) { *elapsed = -1; return; }
      for (int w = 0; w < words; ++w) {
        if (payload[w] != float(i)) { *elapsed = -2; return; }
      }
      system_flag(*reply).store(i, cuda::memory_order_release);
    }
  }
  *elapsed = total;
}

// Finite streaming load; bounded even if the signalling kernels cannot progress.
__global__ void stream_load(const float4* src, float4* dst, size_t n) {
  const auto start = ETX_TIMER();
  float acc = 0;
  do {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x) acc += src[i].x;
  } while (ETX_TIMER() - start < deadline_ticks);
  dst[blockIdx.x * blockDim.x + threadIdx.x] = make_float4(acc, 0, 0, 0);
}

int main(int argc, char** argv) {
  if (port_help(argc, argv, "p2p_sync: two CUDA peers; consumer-memory system-scope flag, idle/loaded")) return 0;
  if (argc != 1) { fprintf(stderr, "Use --help for usage\n"); return 2; }
  setvbuf(stdout, nullptr, _IONBF, 0);
  int count;
  PORT_CHECK(cudaGetDeviceCount(&count));
  if (count < 2) { fprintf(stderr, "need 2 devices\n"); return 2; }
  cudaStream_t signal[2], load[2];
  float4 *src[2], *dst[2];
  long long* result[2];
  int blocks[2];
  constexpr size_t bytes = size_t(1) << 30;
  for (int d = 0; d < 2; ++d) {
    PORT_CHECK(cudaSetDevice(d));
    int access, native;
    PORT_CHECK(cudaDeviceCanAccessPeer(&access, d, 1 - d));
    PORT_CHECK(cudaDeviceGetP2PAttribute(&native, cudaDevP2PAttrNativeAtomicSupported, d, 1 - d));
    if (!access || !native) { fprintf(stderr, "peer %d->%d requires access and native system atomics\n", d, 1-d); return 2; }
    cudaError_t e = cudaDeviceEnablePeerAccess(1-d, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled) PORT_CHECK(e);
    else (void)cudaGetLastError(); // clear the accepted already-enabled status
    cudaDeviceProp prop;
    PORT_CHECK(cudaGetDeviceProperties(&prop, d));
    blocks[d] = prop.multiProcessorCount;
    PORT_CHECK(cudaStreamCreateWithFlags(&signal[d], cudaStreamNonBlocking));
    PORT_CHECK(cudaStreamCreateWithFlags(&load[d], cudaStreamNonBlocking));
    PORT_CHECK(cudaMalloc(&src[d], bytes));
    PORT_CHECK(cudaMemset(src[d], 0, bytes));
    PORT_CHECK(cudaMalloc(&dst[d], blocks[d] * 128 * sizeof(float4)));
    PORT_CHECK(cudaMalloc(&result[d], sizeof(long long)));
    PORT_CHECK(cudaDeviceSynchronize());
  }
  constexpr int iters = 2000;
  printf("%-28s %-18s %8s %14s %14s\n", "condition", "flag location", "payload", "dev0 one-way", "dev1 one-way");
  for (int loaded = 0; loaded < 2; ++loaded) {
    for (int words : {0, 1024}) {
      double us[2];
      for (int producer = 0; producer < 2; ++producer) {
        const int consumer = 1-producer;
        PORT_CHECK(cudaSetDevice(consumer));
        int* flags;
        PORT_CHECK(cudaMalloc(&flags, 8192));
        PORT_CHECK(cudaMemset(flags, 0, 8192));
        PORT_CHECK(cudaDeviceSynchronize());
        if (loaded) for (int d = 0; d < 2; ++d) {
          PORT_CHECK(cudaSetDevice(d));
          stream_load<<<blocks[d], 128, 0, load[d]>>>(src[d], dst[d], bytes / sizeof(float4));
          PORT_CHECK(cudaGetLastError());
        }
        for (int d = 0; d < 2; ++d) {
          PORT_CHECK(cudaSetDevice(d));
          exchange<<<1, 1, 0, signal[d]>>>(flags, flags+1, reinterpret_cast<float*>(flags+64),
                                         words, iters, d == producer, result[d]);
          PORT_CHECK(cudaGetLastError());
        }
        long long elapsed[2];
        for (int d = 0; d < 2; ++d) {
          PORT_CHECK(cudaSetDevice(d));
          PORT_CHECK(cudaStreamSynchronize(signal[d]));
          PORT_CHECK(cudaMemcpy(&elapsed[d], result[d], sizeof(long long), cudaMemcpyDeviceToHost));
          PORT_CHECK(cudaStreamSynchronize(load[d]));
        }
        if (elapsed[0] < 0 || elapsed[1] < 0) {
          fprintf(stderr, "NO PROGRESS or payload mismatch: %lld %lld\n", elapsed[0], elapsed[1]); return 1;
        }
        us[producer] = double(elapsed[producer]) * 10.0 / iters / 2.0 / 1000.0;
        PORT_CHECK(cudaSetDevice(consumer));
        PORT_CHECK(cudaFree(flags));
      }
      printf("%-28s %-18s %6d B %11.2f us %11.2f us\n", loaded ? "under 2x1GB streaming load" : "idle",
             "consumer device", words * 4, us[0], us[1]);
    }
  }
  for (int d = 0; d < 2; ++d) {
    PORT_CHECK(cudaSetDevice(d));
    PORT_CHECK(cudaFree(src[d])); PORT_CHECK(cudaFree(dst[d])); PORT_CHECK(cudaFree(result[d]));
    PORT_CHECK(cudaStreamDestroy(signal[d])); PORT_CHECK(cudaStreamDestroy(load[d]));
  }
}
