// First real execution target: split-K row sum (examples/splitk_sum.py).
// Build (after `python -m etx compile examples/splitk_sum.py --arch gfx942 --out build/splitk_sum`):
//   hipcc -O2 --offload-arch=gfx942 -fgpu-rdc -I etx/runtime/include -I build/splitk_sum \
//         build/splitk_sum/megakernel_d0.hip examples/tiles/splitk.hip examples/hosts/splitk.hip -o build/splitk_sum/run
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "plan_data.h"
#include "etx/host.h"

extern "C" __global__ void etx_megakernel_d0(etx_params);

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 20;
  etx_host h(0);
  h.kernel = (const void*)etx_megakernel_d0;
  h.setup();
  const int n = etx_shape[0];                 // symbol n
  const int rows = n * 32, K = 128;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<__hip_bfloat16> A(rows * K);
  std::vector<float> ref(rows, 0.f);
  for (int r = 0; r < rows; ++r) {
    for (int k = 0; k < K; ++k) { float v = U(rng); A[r * K + k] = __float2bfloat16(v); ref[r] += __bfloat162float(A[r * K + k]); }
  }
  ETX_CHECK(hipMemcpy(h.args[h.arg_index("A")], A.data(), A.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
  printf("splitk_sum: n=%d rows=%d tasks=%d workers=%d (%d domains x %d) threads=%d\n", n, rows, ETX_N_TASKS, ETX_N_WORKERS, ETX_N_DOMAINS, ETX_WORKERS_PER_DOMAIN, ETX_THREADS);
  if (!h.run(5.0)) { fprintf(stderr, "FAILED: kernel did not complete\n"); return 2; }
  std::vector<float> C(rows);
  ETX_CHECK(hipMemcpy(C.data(), h.args[h.arg_index("C")], rows * sizeof(float), hipMemcpyDeviceToHost));
  int bad = 0; double maxerr = 0;
  for (int r = 0; r < rows; ++r) { double e = fabs(C[r] - ref[r]); maxerr = fmax(maxerr, e); if (e > 1e-3 * (1 + fabs(ref[r]))) { if (bad < 5) fprintf(stderr, "  row %d: got %f want %f\n", r, C[r], ref[r]); bad++; } }
  printf("first run: %.3f ms, max abs err %.3g, mismatches %d of %d\n", h.last_ms, maxerr, bad, rows);
  if (bad) { printf("FAILED\n"); return 1; }
  double best = 1e9, sum = 0;
  for (int i = 0; i < reps; ++i) { if (!h.run(5.0)) return 2; best = fmin(best, h.last_ms); sum += h.last_ms; }
  printf("PASS  %d reps: best %.3f ms, mean %.3f ms (%d tasks, %d events)\n", reps, best, sum / reps, ETX_N_TASKS, ETX_EVENT_WORDS);
  return 0;
}
