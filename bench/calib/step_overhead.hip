// Baselines for the per-step fixed cost of a persistent megakernel step.
//   A  cooperative launch of an empty kernel (608 x 256)            -> launch floor
//   B0 ETX-shaped worker loop, N no-op tasks, workers poll ctrl_done  -> the v0.3 loop
//   B1 same, workers exit when their static queue is drained          -> no idle polling
//   B2 same as B1 but no per-task ctrl_done atomic                    -> pure loop + descriptor fetch
//   C  ordinary (non-cooperative) empty kernel launch                 -> non-cooperative floor
//   D  unfused split-K: two ordinary kernels per step (n=64, n=1024)  -> the kernel-per-op baseline
// All timings are host wall-clock over many iterations on one stream.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include "etx_lowering.h"
#include "etx/primitives.h"

#define WORKERS 608
#define THREADS 256

__global__ void empty_kernel() {}

struct loop_params {
  const etx_task* descs; const int32_t* sq; const int32_t* sbegin; const int32_t* send;
  int32_t* ctrl_done; int32_t* domain_slots; int32_t n_tasks; int mode;   // 0 = poll done, 1 = exit when drained, 2 = drained + no done atomic
};

__global__ void __launch_bounds__(THREADS) loop_kernel(loop_params p) {
  __shared__ uint32_t s_worker; __shared__ int32_t s_tid;
  if (threadIdx.x == 0) {
    const uint32_t d = ETX_DOMAIN_ID();
    const int32_t slot = atomicAdd(p.domain_slots + d, 1);
    s_worker = d * (WORKERS / 8) + slot;
  }
  __syncthreads();
  const uint32_t worker = s_worker;
  int32_t cursor = worker < WORKERS ? p.sbegin[worker] : 0, cend = worker < WORKERS ? p.send[worker] : 0;
  for (;;) {
    if (threadIdx.x == 0) s_tid = (cursor < cend) ? p.sq[cursor++] : -1;
    __syncthreads();
    const int32_t tid = s_tid;
    if (tid < 0) {
      if (p.mode != 0) break;
      if (atomicAdd(p.ctrl_done, 0) >= p.n_tasks) break;
      ETX_BACKOFF(); __syncthreads(); continue;
    }
    const etx_task t = p.descs[tid];
    asm volatile("" :: "v"(t.coord[0]));          // pretend to use the descriptor
    if (p.mode != 2 && threadIdx.x == 0) atomicAdd(p.ctrl_done, 1);
    __syncthreads();
  }
}

// unfused split-K reference kernels (same math as examples/tiles/splitk.hip)
__global__ void partial_sum(const __hip_bfloat16* A, float* B, int rows) {
  const int i = blockIdx.x / 4, j = blockIdx.x % 4;
  for (int r = threadIdx.x; r < 32; r += blockDim.x) {
    const int row = i * 32 + r; if (row >= rows) break;
    float acc = 0.f; for (int k = j * 32; k < j * 32 + 32; ++k) acc += __bfloat162float(A[row * 128 + k]);
    B[row * 4 + j] = acc;
  }
}
__global__ void final_sum(const float* B, float* C) {
  const int i = blockIdx.x;
  for (int r = threadIdx.x; r < 32; r += blockDim.x) { const int row = i * 32 + r; C[row] = B[row * 4] + B[row * 4 + 1] + B[row * 4 + 2] + B[row * 4 + 3]; }
}

template <class F> static double time_us(F f, int iters) {
  f(); hipDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) f();
  hipDeviceSynchronize();
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int main() {
  hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  const int iters = 200;
  // A / C
  void* noargs[] = {};
  double tA = time_us([&] { hipLaunchCooperativeKernel((const void*)empty_kernel, dim3(WORKERS), dim3(THREADS), noargs, 0, s); }, iters);
  double tC = time_us([&] { empty_kernel<<<WORKERS, THREADS, 0, s>>>(); }, iters);
  printf("A  cooperative launch, empty kernel, %d x %d:   %.1f us\n", WORKERS, THREADS, tA);
  printf("C  ordinary launch,    empty kernel, %d x %d:   %.1f us\n", WORKERS, THREADS, tC);
  // B: worker loop with N no-op tasks
  int32_t *ctrl, *slots; hipExtMallocWithFlags((void**)&ctrl, 64, hipDeviceMallocFinegrained); hipMalloc(&slots, 8 * 4);
  for (int n_tasks : {0, 608, 5120, 20480}) {
    std::vector<etx_task> descs(n_tasks > 0 ? n_tasks : 1);
    for (int i = 0; i < n_tasks; ++i) descs[i] = etx_task{0, {i, 0, 0, 0}};
    std::vector<int32_t> sq(n_tasks > 0 ? n_tasks : 1), sb(WORKERS), se(WORKERS);
    int k = 0;
    for (int w = 0; w < WORKERS; ++w) { sb[w] = k; for (int i = w; i < n_tasks; i += WORKERS) sq[k++] = i; se[w] = k; }
    etx_task* d_descs; int32_t *d_sq, *d_sb, *d_se;
    hipMalloc(&d_descs, descs.size() * sizeof(etx_task)); hipMemcpy(d_descs, descs.data(), descs.size() * sizeof(etx_task), hipMemcpyHostToDevice);
    hipMalloc(&d_sq, sq.size() * 4); hipMemcpy(d_sq, sq.data(), sq.size() * 4, hipMemcpyHostToDevice);
    hipMalloc(&d_sb, WORKERS * 4); hipMemcpy(d_sb, sb.data(), WORKERS * 4, hipMemcpyHostToDevice);
    hipMalloc(&d_se, WORKERS * 4); hipMemcpy(d_se, se.data(), WORKERS * 4, hipMemcpyHostToDevice);
    for (int mode = 0; mode < 3; ++mode) {
      loop_params p{d_descs, d_sq, d_sb, d_se, ctrl, slots, n_tasks, mode};
      void* args[] = {&p};
      double t = time_us([&] {
        hipMemsetAsync(ctrl, 0, 64, s); hipMemsetAsync(slots, 0, 32, s);
        hipLaunchCooperativeKernel((const void*)loop_kernel, dim3(WORKERS), dim3(THREADS), args, 0, s);
      }, iters);
      printf("B%d loop, %5d no-op tasks, %s: %.1f us (%.2f us per task-slot beyond launch)\n", mode, n_tasks,
             mode == 0 ? "poll ctrl_done      " : mode == 1 ? "exit when drained   " : "drained, no atomic  ", t,
             n_tasks ? (t - tA) / ((double)n_tasks / WORKERS) : 0.0);
    }
  }
  // D: unfused split-K
  for (int n : {64, 1024}) {
    const int rows = n * 32;
    __hip_bfloat16* A; float *B, *C;
    hipMalloc(&A, (size_t)rows * 128 * 2); hipMalloc(&B, (size_t)rows * 4 * 4); hipMalloc(&C, (size_t)rows * 4);
    double t = time_us([&] { partial_sum<<<n * 4, 256, 0, s>>>(A, B, rows); final_sum<<<n, 256, 0, s>>>(B, C); }, iters);
    printf("D  unfused split-K n=%d: two ordinary kernels per step: %.1f us\n", n, t);
  }
  return 0;
}
