// Batch-1 GEMV microbenchmark for the examples/llm tile: one launch per (rows per task, K) point, one
// workgroup per task, xs in LDS, y written once per row. Prints effective TB/s of the weight stream.
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 -I etx/runtime/include -I . [-DLLM_NT_WEIGHTS=1 -DLLM_GEMV_U=32 -DBENCH_WG_PER_CU=1] bench/gemv_bench.hip -o build/gemv_bench
//   build/gemv_bench [rows_per_task,...] [K,...]      e.g. build/gemv_bench 8,16,32,64,128 512,1536,2048,4096,12288
#include "etx/tile_util.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#ifndef BENCH_WG_PER_CU
#define BENCH_WG_PER_CU 1
#endif
static __device__ __forceinline__ float dot8(uint4 w, const float* x) {
  const uint32_t u[4] = {w.x, w.y, w.z, w.w}; float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { s += __uint_as_float(u[i] << 16) * x[2 * i]; s += __uint_as_float(u[i] & 0xffff0000u) * x[2 * i + 1]; }
  return s;
}
#include "examples/llm/gemv.h"

__global__ void __launch_bounds__(256, BENCH_WG_PER_CU) gemv_kernel(const __hip_bfloat16* W, int K, int rows, int rpt, const float* x, float* y) {
  extern __shared__ float lds[];
  float* xs = lds; float* part = lds + K;
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const int r0 = blockIdx.x * rpt, r1 = min(rows, r0 + rpt);
  if (r0 >= r1) return;
  gemv_rows(W, K, xs, r0, r1, part, [&](int r, float v) { y[r] = v; });
}
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

static std::vector<int> parse(const char* s, std::vector<int> d) {
  if (!s) return d; std::vector<int> v; std::string t; for (const char* p = s;; ++p) { if (*p == ',' || !*p) { v.push_back(atoi(t.c_str())); t.clear(); if (!*p) break; } else t += *p; } return v;
}

int main(int argc, char** argv) {
  const std::vector<int> RPT = parse(argc > 1 ? argv[1] : nullptr, {8, 16, 32, 64, 128});
  const std::vector<int> KS = parse(argc > 2 ? argv[2] : nullptr, {512, 1536, 2048, 4096, 12288});
  const size_t bytes = 1ull << 30;                      // 1 GB weight stream per point (beyond any cache)
  __hip_bfloat16* W; CK(hipMalloc(&W, bytes)); CK(hipMemset(W, 0x3c, bytes));
  float* x; CK(hipMalloc(&x, 16384 * 4)); CK(hipMemset(x, 0, 16384 * 4));
  float* y; CK(hipMalloc(&y, (bytes / 1024) * 4));
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  printf("%-6s %-7s %-7s %-9s %-8s  (U=%d NT=%d wg/CU=%d)\n", "K", "rows/t", "tasks", "task_KB", "TB/s", LLM_GEMV_U,
#if defined(LLM_NT_WEIGHTS) && LLM_NT_WEIGHTS
         1,
#else
         0,
#endif
         BENCH_WG_PER_CU);
  for (int K : KS) for (int rpt : RPT) {
    const int rows = (int)(bytes / ((size_t)K * 2)); const int tasks = (rows + rpt - 1) / rpt;
    const size_t lds = (size_t)K * 4 + (size_t)4 * rpt * 4 + 1024;
    if (lds > 65536) { printf("%-6d %-7d skipped: LDS %zu\n", K, rpt, lds); continue; }
    for (int i = 0; i < 2; ++i) gemv_kernel<<<tasks, 256, lds>>>(W, K, rows, rpt, x, y);   // warm-up
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    const int reps = 5;
    for (int i = 0; i < reps; ++i) gemv_kernel<<<tasks, 256, lds>>>(W, K, rows, rpt, x, y);
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms = 0; CK(hipEventElapsedTime(&ms, a, b));
    printf("%-6d %-7d %-7d %-9.0f %-8.2f\n", K, rpt, tasks, rpt * K * 2 / 1024.0, (double)bytes * reps / (ms * 1e-3) / 1e12);
  }
  return 0;
}
