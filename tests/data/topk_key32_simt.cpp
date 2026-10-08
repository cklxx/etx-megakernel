// Host SIMT harness for the ETX_TOPK_KEY32 block of etx/runtime/include/etx/tile_util.h (tests/test_topk_key32.py).
// The test extracts the block (between "#ifdef ETX_TOPK_KEY32" and its "#endif") into key32_block.inc and compiles
// this file with a host C++20 compiler. 64 host threads run the block as the 64 lanes of one wave. Each cross-lane builtin
// (update_dpp, readlane, __shfl_xor) publishes the lane values, meets at a barrier, and reads them.
// stdin: E K renorm, then E bf16 bit patterns (hex). stdout: K lines "id weight_bits_hex" (result t).
// argv[1] == "lds": run etx_topk_key32_lds (lane 0 stores each round; lane 0 divides by sel, as dsl_topk_lds does).
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define __device__
#define __forceinline__ inline
#define ETX_WAVE_SIZE 64
static inline unsigned __float_as_uint(float x) { unsigned u; std::memcpy(&u, &x, 4); return u; }
static inline float __uint_as_float(unsigned u) { float x; std::memcpy(&x, &u, 4); return x; }
static inline int __float_as_int(float x) { int u; std::memcpy(&u, &x, 4); return u; }
static inline float __int_as_float(int u) { float x; std::memcpy(&x, &u, 4); return x; }
static inline float expf_(float x) { return std::exp(x); }
#define expf expf_

static std::barrier<>* g_bar;
static int g_ival[64];
static float g_fval[64];
static thread_local int t_lane;

static int xchg_i(int v, int src) { g_ival[t_lane] = v; g_bar->arrive_and_wait(); int r = src >= 0 ? g_ival[src] : 0; g_bar->arrive_and_wait(); return r; }
static float __shfl_xor(float v, int off, int) { g_fval[t_lane] = v; g_bar->arrive_and_wait(); float r = g_fval[t_lane ^ off]; g_bar->arrive_and_wait(); return r; }
static float etx_wave_sum(float v) { for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64); return v; }
// gfx9 DPP: row_shr:n = 0x110 + n (source lane - n inside the 16-lane row), row_bcast:15 = 0x142 (lane 15 of the
// previous row), row_bcast:31 = 0x143 (lane 31). A lane outside row_mask/bank_mask keeps old. An invalid source
// gives 0 with bound_ctrl, else old.
static int __builtin_amdgcn_update_dpp(int old, int src, int ctrl, int row_mask, int bank_mask, bool bound_ctrl) {
  const int i = t_lane, row = i / 16, pos = i % 16;
  int s = -1;
  if (ctrl >= 0x111 && ctrl <= 0x11f) { const int n = ctrl - 0x110; s = pos >= n ? i - n : -1; }
  else if (ctrl == 0x142) s = row > 0 ? row * 16 - 1 : -1;
  else if (ctrl == 0x143) s = row > 1 ? 31 : -1;
  else { std::fprintf(stderr, "dpp ctrl %#x not modelled\n", ctrl); std::abort(); }
  const int v = xchg_i(src, s < 0 ? i : s);
  if (!((row_mask >> row) & 1) || !((bank_mask >> (pos / 4)) & 1)) return old;
  if (s < 0) return bound_ctrl ? 0 : old;
  return v;
}
static int __builtin_amdgcn_readlane(int v, int lane) {
  g_ival[t_lane] = v; g_bar->arrive_and_wait(); int r = g_ival[lane & 63]; g_bar->arrive_and_wait(); return r;
}

#include "key32_block.inc"

int main(int argc, char** argv) {
  const bool lds = argc > 1 && std::strcmp(argv[1], "lds") == 0;
  int E, K, renorm;
  while (std::scanf("%d %d %d", &E, &K, &renorm) == 3) {
    std::vector<unsigned> bits(E);
    for (int e = 0; e < E; ++e) std::scanf("%x", &bits[e]);
    std::vector<int> id(K, -1); std::vector<float> w(K, 0.f);
    std::barrier<> bar(64); g_bar = &bar;
    std::vector<std::thread> th;
    for (int lane = 0; lane < 64; ++lane)
      th.emplace_back([&, lane] {
        t_lane = lane;
        float l[2];
        for (int k = 0; k < 2; ++k) { const int e = lane + 64 * k; l[k] = e < E ? __uint_as_float(bits[e] << 16) : -INFINITY; }
        if (!lds) { etx_topk_key32(l, E, K, renorm != 0, lane, [&](int t, int bi, float v) { id[t] = bi; w[t] = v; }); return; }
        const float sel = etx_topk_key32_lds(l, E, K, lane, [&](int t, int bi, float v) { id[t] = bi; w[t] = v; });
        if (renorm && lane == 0) for (int t = 0; t < K; ++t) w[t] = w[t] / sel;
      });
    for (auto& t : th) t.join();
    for (int t = 0; t < K; ++t) std::printf("%d %08x\n", id[t], __float_as_uint(w[t]));
    std::fflush(stdout);
  }
  return 0;
}
