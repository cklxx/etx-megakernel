// Host for examples/moe_layer.py: generates inputs, runs the megakernel, checks
// against a CPU reference (fp32 with double accumulation), reports timing.
// Build (after `python -m etx compile examples/moe_layer.py --arch gfx942 --out build/moe`):
//   hipcc -O2 --offload-arch=gfx942 -fgpu-rdc -I etx/runtime/include -I build/moe \
//         build/moe/megakernel_d0.hip examples/tiles/moe.hip examples/hosts/moe.hip -o build/moe/run
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "plan_data.h"
#include "etx/host.h"

#ifndef D
#define D 1024
#endif
#ifndef H
#define H 8
#endif
#define HD 128
#ifndef S
#define S 512
#endif
#define NE 8
#define K 2
#ifndef FF
#define FF 512
#endif
#define TPT 4

typedef __hip_bfloat16 bf16;
static float b2f(bf16 v) { return __bfloat162float(v); }

extern "C" __global__ void etx_megakernel_d0(etx_params);

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 20;
  const int B = etx_shape[ETX_SYM_B], n_gg = etx_shape[ETX_SYM_n_gg];
  etx_host hst(0);
  hst.kernel = (const void*)etx_megakernel_d0;
  hst.setup();

  // ---- inputs: x from the same LCG as the Python example (bf16-exact values)
  std::vector<bf16> x(B * D);
  std::vector<float> xf(B * D);
  uint32_t s = 12345u;
  for (int i = 0; i < B * D; ++i) { s = s * 1664525u + 1013904223u; float v = (float)((int)((s >> 8) % 257) - 128) / 128.f; x[i] = __float2bfloat16(v); xf[i] = v; }
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> Uw(-0.05f, 0.05f), Uc(-1.f, 1.f);
  auto gen = [&](std::vector<bf16>& v, std::vector<float>& f, size_t n, std::uniform_real_distribution<float>& U) {
    v.resize(n); f.resize(n);
    for (size_t i = 0; i < n; ++i) { bf16 b = __float2bfloat16(U(rng)); v[i] = b; f[i] = b2f(b); }
  };
  std::vector<bf16> Wqkv, Wo, kc, vc, Wup, Wdn; std::vector<float> Wqkvf, Wof, kcf, vcf, Wupf, Wdnf;
  gen(Wqkv, Wqkvf, (size_t)3 * D * D, Uw); gen(Wo, Wof, (size_t)D * D, Uw);
  gen(kc, kcf, (size_t)S * D, Uc); gen(vc, vcf, (size_t)S * D, Uc);
  gen(Wup, Wupf, (size_t)NE * FF * D, Uw); gen(Wdn, Wdnf, (size_t)NE * D * FF, Uw);
  auto up = [&](const char* name, const void* p, size_t bytes) { ETX_CHECK(hipMemcpy(hst.args[hst.arg_index(name)], p, bytes, hipMemcpyHostToDevice)); };
  up("x", x.data(), x.size() * 2); up("W_qkv", Wqkv.data(), Wqkv.size() * 2); up("W_o", Wo.data(), Wo.size() * 2);
  up("k_cache", kc.data(), kc.size() * 2); up("v_cache", vc.data(), vc.size() * 2);
  up("W_up", Wup.data(), Wup.size() * 2); up("W_down", Wdn.data(), Wdn.size() * 2);

  // ---- CPU reference
  std::vector<double> xn(B * D), qkv(B * 3 * D), attn(B * D), h(B * D), y(B * D);
  for (int b = 0; b < B; ++b) {
    double ss = 0; for (int i = 0; i < D; ++i) ss += (double)xf[b * D + i] * xf[b * D + i];
    double r = 1.0 / sqrt(ss / D + 1e-6);
    for (int i = 0; i < D; ++i) xn[b * D + i] = xf[b * D + i] * r;
    for (int o = 0; o < 3 * D; ++o) { double a = 0; for (int i = 0; i < D; ++i) a += (double)Wqkvf[(size_t)o * D + i] * xn[b * D + i]; qkv[b * 3 * D + o] = a; }
    for (int hh = 0; hh < H; ++hh) {
      std::vector<double> sc(S); double m = -1e300;
      for (int t = 0; t < S; ++t) { double a = 0; for (int d = 0; d < HD; ++d) a += qkv[b * 3 * D + hh * HD + d] * kcf[(size_t)t * D + hh * HD + d]; sc[t] = a / sqrt((double)HD); m = std::max(m, sc[t]); }
      double sum = 0; for (int t = 0; t < S; ++t) { sc[t] = exp(sc[t] - m); sum += sc[t]; }
      for (int d = 0; d < HD; ++d) { double a = 0; for (int t = 0; t < S; ++t) a += sc[t] * vcf[(size_t)t * D + hh * HD + d]; attn[b * D + hh * HD + d] = a / sum; }
    }
    for (int o = 0; o < D; ++o) { double a = 0; for (int i = 0; i < D; ++i) a += (double)Wof[(size_t)o * D + i] * attn[b * D + i]; h[b * D + o] = xf[b * D + o] + a; }
  }
  // routing on x[b][0..NE): top-2, ties -> lower id, stored ascending
  std::vector<int> topk(B * K); std::vector<double> tw(B * K);
  for (int b = 0; b < B; ++b) {
    int e0 = 0; for (int e = 1; e < NE; ++e) if (xf[b * D + e] > xf[b * D + e0]) e0 = e;
    int e1 = e0 == 0 ? 1 : 0; for (int e = 0; e < NE; ++e) if (e != e0 && xf[b * D + e] > xf[b * D + e1]) e1 = e;
    int lo = std::min(e0, e1), hi = std::max(e0, e1);
    double m = std::max(xf[b * D + lo], xf[b * D + hi]); double p0 = exp(xf[b * D + lo] - m), p1 = exp(xf[b * D + hi] - m);
    topk[b * K] = lo; topk[b * K + 1] = hi; tw[b * K] = p0 / (p0 + p1); tw[b * K + 1] = p1 / (p0 + p1);
  }
  for (int b = 0; b < B; ++b) {
    for (int o = 0; o < D; ++o) y[b * D + o] = h[b * D + o];
    for (int k = 0; k < K; ++k) {
      const int e = topk[b * K + k];
      std::vector<double> act(FF);
      for (int f = 0; f < FF; ++f) { double a = 0; for (int i = 0; i < D; ++i) a += (double)Wupf[((size_t)e * FF + f) * D + i] * h[b * D + i]; act[f] = a / (1.0 + exp(-a)); }
      for (int o = 0; o < D; ++o) { double a = 0; for (int f = 0; f < FF; ++f) a += (double)Wdnf[((size_t)e * D + o) * FF + f] * act[f]; y[b * D + o] += tw[b * K + k] * a; }
    }
  }

  printf("moe_layer: B=%d n_gg=%d tasks=%d events=%d workers=%d (%d domains x %d) threads=%d\n", B, n_gg, ETX_N_TASKS, ETX_EVENT_WORDS, ETX_N_WORKERS, ETX_N_DOMAINS, ETX_WORKERS_PER_DOMAIN, ETX_THREADS);
  if (!hst.run(10.0)) { fprintf(stderr, "FAILED: kernel did not complete\n"); return 2; }

  std::vector<int32_t> dtopk(B * K);
  ETX_CHECK(hipMemcpy(dtopk.data(), hst.args[hst.arg_index("topk")], B * K * 4, hipMemcpyDeviceToHost));
  int route_bad = 0; for (int i = 0; i < B * K; ++i) route_bad += dtopk[i] != topk[i];
  std::vector<float> dy(B * D);
  ETX_CHECK(hipMemcpy(dy.data(), hst.args[hst.arg_index("y")], B * D * 4, hipMemcpyDeviceToHost));
  int bad = 0; double maxrel = 0;
  for (int i = 0; i < B * D; ++i) {
    double e = fabs(dy[i] - y[i]) / (1e-3 + fabs(y[i]));
    maxrel = std::max(maxrel, e);
    if (e > 2e-3) { if (bad < 5) fprintf(stderr, "  y[%d][%d]: got %f want %f\n", i / D, i % D, dy[i], y[i]); bad++; }
  }
  printf("first run: %.3f ms | routing mismatches %d | max rel err %.2e | y mismatches %d of %d\n", hst.last_ms, route_bad, maxrel, bad, B * D);
  if (route_bad || bad) { printf("FAILED\n"); hst.dump_trace(); return 1; }
  double best = 1e9, sum = 0;
  for (int i = 0; i < reps; ++i) { if (!hst.run(10.0)) return 2; best = std::min(best, hst.last_ms); sum += hst.last_ms; }
  printf("PASS  %d reps: best %.3f ms, mean %.3f ms (%d tasks, %d events, %d eliminated)\n", reps, best, sum / reps, ETX_N_TASKS, ETX_EVENT_WORDS, (int)(sizeof(etx_ev_offset) / sizeof(int32_t)) >= 0 ? 0 : 0);
  return 0;
}
