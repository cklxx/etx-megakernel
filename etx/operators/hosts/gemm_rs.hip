// Two-device host for examples/gemm_rs.py: one kernel instance per GPU, a
// shared system-scope event buffer in fine-grained memory on device 0, C
// buffers peer-mapped, CPU reference for the reduce-scattered rows.
// Build (after `python -m etx compile examples/gemm_rs.py --arch gfx942 --out build/gemm_rs`):
//   hipcc -O2 --offload-arch=gfx942 -fgpu-rdc -I etx/runtime/include -I build/gemm_rs \
//     build/gemm_rs/megakernel_d0.hip build/gemm_rs/megakernel_d1.hip examples/tiles/gemm_rs.hip \
//     examples/hosts/gemm_rs.hip -o build/gemm_rs/run -pthread
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>
#include "plan_data.h"
#include "etx/host.h"

extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_megakernel_d1(etx_params);

typedef __hip_bfloat16 bf16;

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 10;
  int ndev = 0; ETX_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) { printf("need 2 devices\n"); return 0; }
  const int M = etx_shape[ETX_SYM_M], K = etx_shape[ETX_SYM_K], N = etx_shape[ETX_SYM_N];
  const int WORLD = 2, ML = M / WORLD;
  for (int d = 0; d < WORLD; ++d) { ETX_CHECK(hipSetDevice(d)); for (int o = 0; o < WORLD; ++o) if (o != d) hipDeviceEnablePeerAccess(o, 0); }

  // allocation policy: a tensor written on device d lives on d; inputs are replicated per device
  std::vector<std::vector<void*>> args(WORLD, std::vector<void*>(ETX_N_ARGS, nullptr));
  for (int a = 0; a < ETX_N_ARGS; ++a) {
    const size_t bytes = etx_arg_bytes[a] ? etx_arg_bytes[a] : 16;
    if (etx_arg_writer_device[a] >= 0) {
      ETX_CHECK(hipSetDevice(etx_arg_writer_device[a]));
      void* p; ETX_CHECK(hipMalloc(&p, bytes)); ETX_CHECK(hipMemset(p, 0, bytes));
      for (int d = 0; d < WORLD; ++d) args[d][a] = p;                    // same VA usable by peers
    } else {
      for (int d = 0; d < WORLD; ++d) { ETX_CHECK(hipSetDevice(d)); void* p; ETX_CHECK(hipMalloc(&p, bytes)); args[d][a] = p; }
    }
  }
  // inputs
  std::mt19937 rng(3); std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<std::vector<bf16>> A(WORLD, std::vector<bf16>((size_t)M * K)); std::vector<bf16> Bw((size_t)K * N);
  std::vector<std::vector<float>> Af(WORLD, std::vector<float>((size_t)M * K)); std::vector<float> Bf((size_t)K * N);
  for (int r = 0; r < WORLD; ++r) for (size_t i = 0; i < A[r].size(); ++i) { bf16 b = __float2bfloat16(U(rng)); A[r][i] = b; Af[r][i] = __bfloat162float(b); }
  for (size_t i = 0; i < Bw.size(); ++i) { bf16 b = __float2bfloat16(U(rng)); Bw[i] = b; Bf[i] = __bfloat162float(b); }
  for (int r = 0; r < WORLD; ++r) {
    ETX_CHECK(hipSetDevice(r));
    char name[8]; snprintf(name, sizeof name, "A%d", r);
    ETX_CHECK(hipMemcpy(args[r][etx_host::arg_index(name)], A[r].data(), A[r].size() * 2, hipMemcpyHostToDevice));
    ETX_CHECK(hipMemcpy(args[r][etx_host::arg_index("Bw")], Bw.data(), Bw.size() * 2, hipMemcpyHostToDevice));
  }
  // CPU reference: D_r = rows [r*ML, (r+1)*ML) of sum_r A_r Bw
  std::vector<std::vector<double>> D(WORLD, std::vector<double>((size_t)ML * N));
  for (int r = 0; r < WORLD; ++r)
    for (int i = 0; i < ML; ++i)
      for (int n = 0; n < N; ++n) {
        double s = 0;
        for (int rr = 0; rr < WORLD; ++rr) for (int k = 0; k < K; ++k) s += (double)Af[rr][(size_t)(r * ML + i) * K + k] * Bf[(size_t)k * N + n];
        D[r][(size_t)i * N + n] = s;
      }

  // hosts: shared events on device 0
  int32_t* events = etx_host::alloc_events(0);
  std::vector<etx_host> hs; hs.reserve(WORLD);
  for (int r = 0; r < WORLD; ++r) { hs.emplace_back(r); hs[r].kernel = r == 0 ? (const void*)etx_megakernel_d0 : (const void*)etx_megakernel_d1; hs[r].args = args[r]; hs[r].setup(events); }
  hs[0].owns_events = true; hs[1].owns_events = false;
  printf("gemm_rs: M=%d K=%d N=%d world=%d tasks=%d (dev0 %d, dev1 %d) events=%d fine_grained=%d\n", M, K, N, WORLD, ETX_N_TASKS, etx_n_tasks_dev[0], etx_n_tasks_dev[1], ETX_EVENT_WORDS, ETX_EVENTS_FINE_GRAINED);

  auto step = [&](double timeout) -> bool {
    for (int r = 0; r < WORLD; ++r) hs[r].reset_step();
    for (int r = 0; r < WORLD; ++r) if (!hs[r].check_residency()) return false;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < WORLD; ++r) hs[r].launch();
    bool ok = true;
    for (int r = 0; r < WORLD; ++r) ok = hs[r].wait(timeout, t0) && ok;
    return ok;
  };
  if (!step(10.0)) { printf("FAILED: step did not complete\n"); return 2; }
  // device-local check first: C_r = A_r Bw on each device (separates tile bugs from cross-device visibility)
  for (int r = 0; r < WORLD; ++r) {
    std::vector<float> C((size_t)M * N);
    char name[8]; snprintf(name, sizeof name, "C%d", r);
    ETX_CHECK(hipSetDevice(r));
    ETX_CHECK(hipMemcpy(C.data(), args[r][etx_host::arg_index(name)], C.size() * 4, hipMemcpyDeviceToHost));
    int cbad = 0; double cmax = 0;
    for (int i = 0; i < M; i += 37) for (int n = 0; n < N; n += 13) {
      double s = 0; for (int k = 0; k < K; ++k) s += (double)Af[r][(size_t)i * K + k] * Bf[(size_t)k * N + n];
      double e = fabs(C[(size_t)i * N + n] - s) / (1e-2 + fabs(s)); cmax = std::max(cmax, e); cbad += e > 1e-3;
    }
    printf("  device %d local C check: max rel err %.2e, %d sampled mismatches\n", r, cmax, cbad);
  }
  int bad = 0; double maxrel = 0;
  for (int r = 0; r < WORLD; ++r) {
    std::vector<float> got((size_t)ML * N);
    char name[8]; snprintf(name, sizeof name, "D%d", r);
    ETX_CHECK(hipSetDevice(r));
    ETX_CHECK(hipMemcpy(got.data(), args[r][etx_host::arg_index(name)], got.size() * 4, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
      double e = fabs(got[i] - D[r][i]) / (1e-2 + fabs(D[r][i])); maxrel = std::max(maxrel, e);
      if (e > 1e-3) { if (bad < 5) fprintf(stderr, "  D%d[%zu]: got %f want %f\n", r, i, got[i], D[r][i]); bad++; }
    }
  }
  printf("first step: %.3f ms | max rel err %.2e | mismatches %d of %d\n", std::max(hs[0].last_ms, hs[1].last_ms), maxrel, bad, WORLD * ML * N);
  if (bad) { printf("FAILED\n"); return 1; }
  double best = 1e9, sum = 0;
  for (int i = 0; i < reps; ++i) { if (!step(10.0)) return 2; double ms = std::max(hs[0].last_ms, hs[1].last_ms); best = std::min(best, ms); sum += ms; }
  printf("PASS  %d reps: best %.3f ms, mean %.3f ms (2 devices, system-scope events over xGMI)\n", reps, best, sum / reps);
  return 0;
}
