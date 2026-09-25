// Generic batch-1 decode tiles for examples/llm/model.py (link mode, one set for every model).
//
// Every body reads the model through the __constant__ LlmParams (llm_params.h); the layer index
// is the grid's compile-time constant (ctx->cst[0]); the task's rows come from its coordinate and
// the partition in LlmParams. Numerics follow Hugging Face's bf16 forward: values are kept in fp32
// but rounded to bf16 at every point where HF materialises a bf16 tensor (linear outputs, residual
// adds, RMSNorm before and after the weight, each RoPE product, SiLU, the MoE combine in ascending
// expert order), so the argmax agrees with HF greedy except at near-ties.
//
// The GEMV streams each task's contiguous weight block with U loads in flight per lane (gemv_stream).
#include "etx/tile_util.h"
#include "examples/llm/llm_params.h"

__constant__ LlmParams P;

extern "C" void llm_set_params(const void* host_params, hipStream_t s) {
  (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(P), host_params, sizeof(LlmParams), 0, hipMemcpyHostToDevice, s);
  (void)hipStreamSynchronize(s);
}
extern "C" size_t llm_params_bytes() { return sizeof(LlmParams); }

// ---------------------------------------------------------------- helpers
static __device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16(v)); }
static __device__ __forceinline__ float silu(float g) { return g / (1.f + __expf(-g)); }

// LDS layout of every body: [0, 1 KB) reduction scratch and small arrays, then the fp32 input vector.
#define LLM_RED(c)  ((float*)(c)->lds)
#define LLM_VEC(c)  ((float*)(c)->lds + 256)

static __device__ __forceinline__ float dot8(uint4 w, const float* x) {
  const uint32_t u[4] = {w.x, w.y, w.z, w.w};
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s += __uint_as_float(u[i] << 16) * x[2 * i];
    s += __uint_as_float(u[i] & 0xffff0000u) * x[2 * i + 1];
  }
  return s;
}

// ---------------------------------------------------------------- GEMV
// rows [r0, r1) of a row-major bf16 matrix W with K columns, times xs (fp32, LDS). The task's rows are one
// contiguous block of (r1-r0)*K bf16, so it is streamed as a flat sequence of segments of 64 lanes x EPL
// elements (EPL = 8 -> 16-byte loads when K % 512 == 0, else 4 -> 8-byte loads when K % 256 == 0); each
// segment lies inside one row. Wave w takes a contiguous quarter of the segments and keeps U loads in flight
// per lane (U x 16 B x 256 lanes = 64 KB per CU at U=16), which is what hides HBM latency on small tasks.
// A wave's running sum is reduced when its segment run crosses a row boundary; the per-wave partials meet
// in `part` (LDS, >= nwaves * (r1 - r0) floats) and are summed in wave order (deterministic); epi(row, dot)
// is then called once per row by one thread. LLM_GEMV_U overrides the depth; LLM_NT_WEIGHTS=1 uses
// non-temporal weight loads (weights are read once per token).
#ifndef LLM_GEMV_U
#define LLM_GEMV_U 16
#endif
typedef unsigned int llm_u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int llm_u32x2 __attribute__((ext_vector_type(2)));
template <int EPL> struct wvec;
template <> struct wvec<8> { typedef llm_u32x4 T; };
template <> struct wvec<4> { typedef llm_u32x2 T; };
template <> struct wvec<2> { typedef unsigned int T; };

static __device__ __forceinline__ float dotn(llm_u32x4 w, const float* x) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { s += __uint_as_float(w[i] << 16) * x[2 * i]; s += __uint_as_float(w[i] & 0xffff0000u) * x[2 * i + 1]; }
  return s;
}
static __device__ __forceinline__ float dotn(unsigned int w, const float* x) {
  return __uint_as_float(w << 16) * x[0] + __uint_as_float(w & 0xffff0000u) * x[1];
}
static __device__ __forceinline__ float dotn(llm_u32x2 w, const float* x) {
  return __uint_as_float(w.x << 16) * x[0] + __uint_as_float(w.x & 0xffff0000u) * x[1]
       + __uint_as_float(w.y << 16) * x[2] + __uint_as_float(w.y & 0xffff0000u) * x[3];
}
template <class V> static __device__ __forceinline__ V wload(const V* p) {
#if defined(LLM_NT_WEIGHTS) && LLM_NT_WEIGHTS
  return __builtin_nontemporal_load(p);
#else
  return *p;
#endif
}

template <int EPL, class Epi>
static __device__ __forceinline__ void gemv_stream(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  typedef typename wvec<EPL>::T V;
  constexpr int U = LLM_GEMV_U, SEG = ETX_WAVE_SIZE * EPL;
  const int lane = etx_lane(), wv = etx_wave(), nw = etx_nwaves(), nrt = r1 - r0;
  const int spr = K / SEG;                                   // segments per row
  const int S = nrt * spr;
  const int per = (S + nw - 1) / nw, sb = wv * per, se = min(S, sb + per);
  for (int i = lane; i < nrt; i += ETX_WAVE_SIZE) part[wv * nrt + i] = 0.f;
  const V* base = (const V*)(W + (size_t)r0 * K) + lane;     // segment s starts at element s * SEG
  int cur = sb / (spr > 0 ? spr : 1);
  float acc = 0.f;
  for (int s0 = sb; s0 < se; s0 += U) {
    V w[U];
#pragma unroll
    for (int u = 0; u < U; ++u) if (s0 + u < se) w[u] = wload(base + (size_t)(s0 + u) * ETX_WAVE_SIZE);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int sg = s0 + u;
      if (sg < se) {
        const int row = sg / spr;
        if (row != cur) {                                    // uniform across the wave
          const float v = etx_wave_sum(acc);
          if (lane == 0) part[wv * nrt + cur] += v;
          acc = 0.f; cur = row;
        }
        acc += dotn(w[u], xs + (sg - row * spr) * SEG + lane * EPL);
      }
    }
  }
  if (sb < se) { const float v = etx_wave_sum(acc); if (lane == 0) part[wv * nrt + cur] += v; }
  __syncthreads();
  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) {
    float v = 0.f;
    for (int w2 = 0; w2 < nw; ++w2) v += part[w2 * nrt + r];
    epi(r0 + r, v);
  }
  __syncthreads();
}

template <class Epi>
static __device__ __forceinline__ void gemv_rows(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, float* part, Epi epi) {
  if (K % (ETX_WAVE_SIZE * 8) == 0) { gemv_stream<8>(W, K, xs, r0, r1, part, epi); return; }
  if (K % (ETX_WAVE_SIZE * 4) == 0) { gemv_stream<4>(W, K, xs, r0, r1, part, epi); return; }
  if (K % (ETX_WAVE_SIZE * 2) == 0) { gemv_stream<2>(W, K, xs, r0, r1, part, epi); return; }
  // generic fallback (K % 8 == 0): rows to waves, lanes stride K
  const int lane = etx_lane(), nw = etx_nwaves(), nrt = r1 - r0;
  for (int r = r0 + etx_wave(); r < r1; r += nw) {
    float acc = 0.f;
    for (int k = lane * 8; k < K; k += ETX_WAVE_SIZE * 8) acc += dot8(*(const uint4*)(W + (size_t)r * K + k), xs + k);
    const float v = etx_wave_sum(acc);
    if (lane == 0) part[r - r0] = v;
  }
  __syncthreads();
  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) epi(r0 + r, part[r]);
  __syncthreads();
}

// HF Qwen2RMSNorm in bf16: weight * bf16(x * rsqrt(mean(x^2) + eps)), rounded again after the weight.
static __device__ __forceinline__ void rmsnorm_to(const float* x, const __hip_bfloat16* w, int n, float* out, float* red) {
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = x[i]; ss += v * v; }
  ss = etx_block_sum(ss, red);
  const float r = rsqrtf(ss / (float)n + P.eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = bfr(etx_b2f(w[i]) * bfr(x[i] * r));
  __syncthreads();
}

static __device__ __forceinline__ void copy_to(const float* src, int n, float* dst) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
  __syncthreads();
}

static __device__ __forceinline__ int task_rows(int i, int rpt, int rows, int* r1) {
  const int r0 = i * rpt;
  *r1 = min(r0 + rpt, rows);
  return r0;
}

// ---------------------------------------------------------------- embed / head
extern "C" __device__ void llm_embed(const etx_ctx* c) {
  if (threadIdx.x == 0) P.token_time[2 * P.tok] = __builtin_amdgcn_s_memrealtime();
  const int t = P.token_in[P.tok];
  const __hip_bfloat16* e = P.embed + (size_t)t * P.H;
  for (int i = threadIdx.x; i < P.H; i += blockDim.x) P.x[i] = etx_b2f(e[i]);
}

extern "C" __device__ void llm_lmhead(const etx_ctx* c) {
  float* red = LLM_RED(c); float* xs = LLM_VEC(c); float* lv = xs + P.H;
  rmsnorm_to(P.x, P.final_norm, P.H, xs, red);
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_lm, P.V, &r1);
  const __hip_bfloat16* W = P.tie ? P.embed : P.lm_head;
  gemv_rows(W, P.H, xs, r0, r1, lv + P.rpt_lm, [&](int r, float v) { lv[r - r0] = bfr(v); });
  __syncthreads();
  if (etx_wave() == 0) {                 // argmax over this task's rows, ties -> lowest row (torch.argmax)
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int k = etx_lane(); k < r1 - r0; k += ETX_WAVE_SIZE)
      if (lv[k] > bv) { bv = lv[k]; bi = r0 + k; }
    for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off, ETX_WAVE_SIZE); const int oi = __shfl_xor(bi, off, ETX_WAVE_SIZE);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (etx_lane() == 0) { P.lm_val[c->coord[0]] = bv; P.lm_idx[c->coord[0]] = bi; }
  }
}

extern "C" __device__ void llm_argmax(const etx_ctx* c) {
  if (etx_wave() != 0) return;
  const int n = (P.V + P.rpt_lm - 1) / P.rpt_lm;
  float bv = -INFINITY; int bi = 0x7fffffff;
  for (int k = etx_lane(); k < n; k += ETX_WAVE_SIZE) {
    const float v = P.lm_val[k]; const int i = P.lm_idx[k];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) {
    const float ov = __shfl_xor(bv, off, ETX_WAVE_SIZE); const int oi = __shfl_xor(bi, off, ETX_WAVE_SIZE);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (etx_lane() == 0) {
    P.token_out[P.tok] = bi;
    if (P.write_next) P.token_in[P.tok + 1] = bi;
    __threadfence();
    P.token_time[2 * P.tok + 1] = __builtin_amdgcn_s_memrealtime();
  }
}

// ---------------------------------------------------------------- attention block
extern "C" __device__ void llm_qkv(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  rmsnorm_to(P.x, Ly.ln1, P.H, xs, LLM_RED(c));
  const int nqkv = (P.NH + 2 * P.NKV) * P.HD;
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_qkv, nqkv, &r1);
  const __hip_bfloat16* b = Ly.bqkv;
  gemv_rows(Ly.wqkv, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { P.qkv[r] = bfr(v + (b ? etx_b2f(b[r]) : 0.f)); });
}

// q/k post-processing of one head: optional per-head RMSNorm, then RoPE (rotate_half), each product in bf16
static __device__ __forceinline__ void head_post(const float* src, const __hip_bfloat16* nw, float* dst_f, __hip_bfloat16* dst_b, float* red) {
  const int HD = P.HD, half = HD / 2, d = threadIdx.x;
  float* v = red + 64;                   // HD floats (HD <= 192 fits the 1 KB scratch)
  float x = d < HD ? src[d] : 0.f;
  if (nw) {
    const float ss = etx_block_sum(x * x, red);
    const float r = rsqrtf(ss / (float)HD + P.eps);
    if (d < HD) x = bfr(etx_b2f(nw[d]) * bfr(x * r));
  }
  if (d < HD) v[d] = x;
  __syncthreads();
  if (d < HD) {
    const int i = d < half ? d : d - half;
    const float cs = P.rope_cos[(size_t)P.pos * half + i], sn = P.rope_sin[(size_t)P.pos * half + i];
    const float rot = d < half ? -v[d + half] : v[d - half];
    const float out = bfr(bfr(v[d] * cs) + bfr(rot * sn));
    if (dst_f) dst_f[d] = out; else dst_b[d] = __float2bfloat16(out);
  }
}

extern "C" __device__ void llm_post_q(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  const int qh = c->coord[0];
  head_post(P.qkv + (size_t)qh * P.HD, P.qk_norm ? Ly.qn : nullptr, P.q + (size_t)qh * P.HD, nullptr, LLM_RED(c));
}

extern "C" __device__ void llm_post_kv(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  const int kvh = c->coord[0], which = c->coord[1];
  const size_t row = ((size_t)P.pos * P.NKV + kvh) * P.HD;
  if (which == 0) {
    head_post(P.qkv + (size_t)(P.NH + kvh) * P.HD, P.qk_norm ? Ly.kn : nullptr, nullptr, Ly.kc + row, LLM_RED(c));
  } else {
    const float* v = P.qkv + (size_t)(P.NH + P.NKV + kvh) * P.HD;
    for (int d = threadIdx.x; d < P.HD; d += blockDim.x) Ly.vc[row + d] = __float2bfloat16(v[d]);
  }
}

// one KV head group x one chunk of positions: scores, local softmax, unnormalised P.V partial per q head
extern "C" __device__ void llm_attn(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  const int kvh = c->coord[0], ch = c->coord[1], G = P.G, HD = P.HD, NC = P.NC;
  const int n = P.pos + 1, t0 = ch * P.CHUNK, t1 = min(t0 + P.CHUNK, n), len = t1 - t0;
  const int stride = HD + 2;
  if (len <= 0) {
    for (int j = 0; j < G; ++j) {
      float* part = P.attn_part + ((size_t)(kvh * G + j) * NC + ch) * stride;
      for (int d = threadIdx.x; d < HD; d += blockDim.x) part[d] = 0.f;
      if (threadIdx.x == 0) { part[HD] = -INFINITY; part[HD + 1] = 0.f; }
    }
    return;
  }
  float* red = LLM_RED(c);
  float* qs = LLM_VEC(c);                 // [G][HD]
  float* sc = qs + G * HD;                // [G][CHUNK]
  for (int i = threadIdx.x; i < G * HD; i += blockDim.x) qs[i] = P.q[(size_t)kvh * G * HD + i];
  __syncthreads();
  const float scale = rsqrtf((float)HD);
  for (int idx = threadIdx.x; idx < G * len; idx += blockDim.x) {
    const int j = idx / len, t = idx % len;
    const __hip_bfloat16* k = Ly.kc + ((size_t)(t0 + t) * P.NKV + kvh) * HD;
    const float* q = qs + j * HD;
    float s = 0.f;
    for (int d = 0; d < HD; d += 8) s += dot8(*(const uint4*)(k + d), q + d);
    sc[j * P.CHUNK + t] = s * scale;
  }
  __syncthreads();
  float* mrow = red + 64; float* lrow = mrow + 16;   // per head max / sum (G <= 16)
  for (int j = etx_wave(); j < G; j += etx_nwaves()) {
    float m = -INFINITY;
    for (int t = etx_lane(); t < len; t += ETX_WAVE_SIZE) m = fmaxf(m, sc[j * P.CHUNK + t]);
    m = etx_wave_max(m);
    float l = 0.f;
    for (int t = etx_lane(); t < len; t += ETX_WAVE_SIZE) { const float p = __expf(sc[j * P.CHUNK + t] - m); sc[j * P.CHUNK + t] = p; l += p; }
    l = etx_wave_sum(l);
    if (etx_lane() == 0) { mrow[j] = m; lrow[j] = l; }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < G * HD; idx += blockDim.x) {
    const int j = idx / HD, d = idx % HD;
    float acc = 0.f;
    for (int t = 0; t < len; ++t) acc += sc[j * P.CHUNK + t] * etx_b2f(Ly.vc[((size_t)(t0 + t) * P.NKV + kvh) * HD + d]);
    float* part = P.attn_part + ((size_t)(kvh * G + j) * NC + ch) * stride;
    part[d] = acc;
    if (d == 0) { part[HD] = mrow[j]; part[HD + 1] = lrow[j]; }
  }
}

extern "C" __device__ void llm_merge(const etx_ctx* c) {
  // chunk weights once into LDS (w_c = exp(m_c - M), empty chunks 0), then every thread sums its d over
  // the chunks with independent loads (the old loop re-read m per chunk per thread: 31 us per layer at NC=64)
  const int qh = c->coord[0], HD = P.HD, NC = P.NC, stride = HD + 2;
  const float* part = P.attn_part + (size_t)qh * NC * stride;
  float* red = LLM_RED(c);
  float* wch = LLM_VEC(c);                // NC weights
  float m = -INFINITY;
  for (int ch = threadIdx.x; ch < NC; ch += blockDim.x) m = fmaxf(m, part[ch * stride + HD]);
  const float M = etx_block_max(m, red);
  float l = 0.f;
  for (int ch = threadIdx.x; ch < NC; ch += blockDim.x) {
    const float mc = part[ch * stride + HD];
    const float w = mc > -INFINITY ? __expf(mc - M) : 0.f;
    wch[ch] = w; l += w * part[ch * stride + HD + 1];
  }
  const float Lsum = etx_block_sum(l, red);   // its __syncthreads also publishes wch
  for (int d = threadIdx.x; d < HD; d += blockDim.x) {
    float acc = 0.f;
#pragma unroll 8
    for (int ch = 0; ch < NC; ++ch) acc += wch[ch] * part[ch * stride + d];
    P.o[(size_t)qh * HD + d] = bfr(acc / Lsum);
  }
}

extern "C" __device__ void llm_oproj(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  const int K = P.NH * P.HD;
  copy_to(P.o, K, xs);
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_o, P.H, &r1);
  gemv_rows(Ly.wo, K, xs, r0, r1, xs + K, [&](int r, float v) { P.x[r] = bfr(P.x[r] + bfr(v)); });
}

// ---------------------------------------------------------------- dense MLP
extern "C" __device__ void llm_gateup(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c);
  float* gtmp = red + 64;                 // up to 192 rows per task
  rmsnorm_to(P.x, Ly.ln2, P.H, xs, red);
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_gu, P.INTER, &r1);
  gemv_rows(Ly.wg, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { gtmp[r - r0] = bfr(v); });
  gemv_rows(Ly.wu, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { P.h[r] = bfr(bfr(silu(gtmp[r - r0])) * bfr(v)); });
}

extern "C" __device__ void llm_down(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  copy_to(P.h, P.INTER, xs);
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_dn, P.H, &r1);
  gemv_rows(Ly.wd, P.INTER, xs, r0, r1, xs + P.INTER, [&](int r, float v) { P.x[r] = bfr(P.x[r] + bfr(v)); });
}

// ---------------------------------------------------------------- MoE
extern "C" __device__ void llm_router(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  rmsnorm_to(P.x, Ly.ln2, P.H, xs, LLM_RED(c));
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_r, P.E, &r1);
  gemv_rows(Ly.wr, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { P.rlogits[r] = bfr(v); });
}

// Every expert task recomputes the token's routing from the E logits (one wave; E <= 256):
// fp32 softmax over all experts, top-k (ties -> lower id), optional renormalisation, bf16 weights.
// ids / weights land in red[64..80) / red[80..96).
static __device__ __forceinline__ void topk_route(float* red) {
  int* ids = (int*)(red + 64); float* wts = red + 80;
  if (etx_wave() == 0) {
    const int lane = etx_lane(), E = P.E;
    float p[4]; bool taken[4];
    float m = -INFINITY;
#pragma unroll
    for (int k = 0; k < 4; ++k) { const int e = lane + k * ETX_WAVE_SIZE; p[k] = e < E ? P.rlogits[e] : -INFINITY; taken[k] = e >= E; m = fmaxf(m, p[k]); }
    m = etx_wave_max(m);
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) { p[k] = taken[k] ? 0.f : __expf(p[k] - m); s += p[k]; }
    s = etx_wave_sum(s);
    float sel = 0.f;
    for (int t = 0; t < P.TOPK; ++t) {
      float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
      for (int k = 0; k < 4; ++k) { const int e = lane + k * ETX_WAVE_SIZE; if (!taken[k] && p[k] / s > bv) { bv = p[k] / s; bi = e; } }
      for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor(bv, off, ETX_WAVE_SIZE); const int oi = __shfl_xor(bi, off, ETX_WAVE_SIZE);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) if (lane + k * ETX_WAVE_SIZE == bi) taken[k] = true;
      if (lane == 0) { ids[t] = bi; wts[t] = bv; }
      sel += bv;
    }
    if (lane == 0) for (int t = 0; t < P.TOPK; ++t) wts[t] = bfr(P.norm_topk ? wts[t] / sel : wts[t]);
  }
  __syncthreads();
}

extern "C" __device__ void llm_egu(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c);
  topk_route(red);
  const int s = c->coord[0], e = ((int*)(red + 64))[s];
  float* gtmp = red + 96;                 // up to 160 rows per task
  rmsnorm_to(P.x, Ly.ln2, P.H, xs, red);
  int r1; const int r0 = task_rows(c->coord[1], P.rpt_egu, P.MI, &r1);
  const size_t eoff = (size_t)e * P.MI * P.H;
  gemv_rows(Ly.eg + eoff, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { gtmp[r - r0] = bfr(v); });
  gemv_rows(Ly.eu + eoff, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { P.hexp[(size_t)s * P.MI + r] = bfr(bfr(silu(gtmp[r - r0])) * bfr(v)); });
}

// rows of x: x[r] += sum over the token's experts in ascending id of bf16(bf16(down_e . h_s) * w_s)
extern "C" __device__ void llm_edn(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c);
  topk_route(red);
  const int* ids = (const int*)(red + 64); const float* wts = red + 80;
  int* order = (int*)(red + 96);
  float* ys = red + 112;                  // [TOPK][rpt]  (TOPK * rpt <= 144)
  copy_to(P.hexp, P.TOPK * P.MI, xs);
  if (threadIdx.x == 0) {                 // slots sorted by expert id (HF's index_add_ order)
    for (int t = 0; t < P.TOPK; ++t) order[t] = t;
    for (int a = 1; a < P.TOPK; ++a) for (int b = a; b > 0 && ids[order[b]] < ids[order[b - 1]]; --b) { const int tmp = order[b]; order[b] = order[b - 1]; order[b - 1] = tmp; }
  }
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_edn, P.H, &r1);
  const int nr = r1 - r0;
  for (int s = 0; s < P.TOPK; ++s)
    gemv_rows(Ly.ed + (size_t)ids[s] * P.H * P.MI, P.MI, xs + (size_t)s * P.MI, r0, r1, xs + (size_t)P.TOPK * P.MI,
              [&](int r, float v) { ys[s * nr + (r - r0)] = bfr(bfr(v) * wts[s]); });
  for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) {
    float acc = 0.f;
    for (int t = 0; t < P.TOPK; ++t) acc = bfr(acc + ys[order[t] * nr + (r - r0)]);
    P.x[r] = bfr(P.x[r] + acc);
  }
}

// ================================================================ sliced layer (examples/llm/model_sliced.py)
// One slice per XCD: coordinate (x, w) = XCD and worker; P.qh0/nqh/kv0/nkv/i0/ni/h0/nh describe XCD x's
// slice. Partial sums are folded ONCE per XCD by a fold task (the only device-scope consumer) into an
// XCD-local vector; every other task waits on a local event and reads that vector with an L1-only acquire.

static __device__ __forceinline__ int rows_of(int n, int w, int W, int* r1) {   // worker w's rows of n
  const int rpt = (n + W - 1) / W, r0 = min(n, w * rpt);
  *r1 = min(n, r0 + rpt);
  return r0;
}

// residual entering layer L: xb for L == 0, else bf16(xa + bf16(sum of the down partials)); MoE sums the
// slots in ascending expert id in bf16 (HF's index_add_). Written to `out` (global, H floats).
// The partials were written by other XCDs, so every load is a remote miss (~2 us). Issue the loads of
// eight output elements at once (up to 8 x TOPK values in registers) instead of one element at a time:
// measured 36 us -> a few us for H = 4096.
template <int NB>
static __device__ __forceinline__ void fold_sum_rows(const float* base, int n, int H, const float* addend, float* out, bool bf16_chain,
                                                     const int* order) {
  for (int i0 = threadIdx.x; i0 < H; i0 += blockDim.x * NB) {
    float v[NB][16]; float a[NB];
#pragma unroll
    for (int m = 0; m < NB; ++m) {
      const int i = i0 + m * blockDim.x;
      a[m] = i < H ? addend[i] : 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) v[m][k] = (k < n && i < H) ? base[(size_t)(order ? order[k] : k) * H + i] : 0.f;
    }
#pragma unroll
    for (int m = 0; m < NB; ++m) {
      const int i = i0 + m * blockDim.x;
      if (i >= H) continue;
      float acc = 0.f;
      if (bf16_chain) { for (int k = 0; k < n; ++k) acc = bfr(acc + v[m][k]); out[i] = bfr(a[m] + acc); }
      else { for (int k = 0; k < n; ++k) acc += v[m][k]; out[i] = bfr(a[m] + bfr(acc)); }
    }
  }
}

static __device__ __forceinline__ void fold_down_to(int L, float* out, int* order_scratch) {
  const int H = P.H;
  if (L <= 0) { for (int i = threadIdx.x; i < H; i += blockDim.x) out[i] = P.x[i]; return; }
  if (P.moe) {
    if (threadIdx.x == 0) {                 // slots sorted by expert id (HF's index_add_ order); TOPK <= 16
      for (int t = 0; t < P.TOPK; ++t) order_scratch[t] = t;
      for (int a = 1; a < P.TOPK; ++a) for (int b = a; b > 0 && P.eid[order_scratch[b]] < P.eid[order_scratch[b - 1]]; --b) { const int tmp = order_scratch[b]; order_scratch[b] = order_scratch[b - 1]; order_scratch[b - 1] = tmp; }
    }
    __syncthreads();
    fold_sum_rows<8>(P.d_part, P.TOPK, H, P.xa, out, true, order_scratch);
  } else {
    fold_sum_rows<8>(P.d_part, P.X, H, P.xa, out, false, nullptr);
  }
}

// fold task of XCD x before layer L: xin_x[x] = folded residual; also stores its slice of xb (= P.x)
extern "C" __device__ void llm_s_fold_d(const etx_ctx* c) {
  const int L = c->cst[0], x = c->coord[0];
  float* out = P.xin_x + (size_t)x * P.H;
  fold_down_to(L, out, (int*)LLM_RED(c));
  __syncthreads();
  if (L > 0) for (int i = P.h0[x] + threadIdx.x; i < P.h0[x] + P.nh[x]; i += blockDim.x) P.x[i] = out[i];
}

// fold task of XCD x after o_proj: xa_x[x] = bf16(xb + bf16(sum of the o_proj partials)); stores its slice of xa
extern "C" __device__ void llm_s_fold_o(const etx_ctx* c) {
  const int x = c->coord[0], H = P.H;
  float* out = P.xa_x + (size_t)x * H;
  fold_sum_rows<8>(P.o_part, P.X, H, P.x, out, false, nullptr);
  __syncthreads();
  for (int i = P.h0[x] + threadIdx.x; i < P.h0[x] + P.nh[x]; i += blockDim.x) P.xa[i] = out[i];
}

extern "C" __device__ void llm_s_lmfold(const etx_ctx* c) {
  fold_down_to(c->cst[0], P.xfin, (int*)LLM_RED(c));
}

// local qkv row r of XCD x -> global row in P.qkv (q rows, then [k; v] per needed KV head)
static __device__ __forceinline__ int qkv_row(int x, int r) {
  const int HD = P.HD, nq = P.nqh[x] * HD;
  if (r < nq) return P.qh0[x] * HD + r;
  const int rr = r - nq, j = rr / (2 * HD), o = rr % (2 * HD);
  return o < HD ? (P.NH + P.kv0[x] + j) * HD + o : (P.NH + P.NKV + P.kv0[x] + j) * HD + (o - HD);
}

extern "C" __device__ void llm_s_qkv(const etx_ctx* c) {
  const int L = c->cst[0], x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[L];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c);
  rmsnorm_to(P.xin_x + (size_t)x * P.H, Ly.ln1, P.H, xs, red);
  const int nrows = (P.nqh[x] + 2 * P.nkv[x]) * P.HD;
  int r1; const int r0 = rows_of(nrows, w, P.W, &r1);
  if (r0 >= r1) return;
  const __hip_bfloat16* b = Ly.bqkv_x[x];
  gemv_rows(Ly.wqkv_x[x], P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { P.qkv[qkv_row(x, r)] = bfr(v + (b ? etx_b2f(b[r]) : 0.f)); });
}

extern "C" __device__ void llm_s_post_q(const etx_ctx* c) {
  const int x = c->coord[0], j = c->coord[1];
  if (j >= P.nqh[x]) return;
  const LlmLayer& Ly = P.layer[c->cst[0]];
  const int qh = P.qh0[x] + j;
  head_post(P.qkv + (size_t)qh * P.HD, P.qk_norm ? Ly.qn : nullptr, P.q + (size_t)qh * P.HD, nullptr, LLM_RED(c));
}

extern "C" __device__ void llm_s_post_kv(const etx_ctx* c) {
  const int x = c->coord[0], j = c->coord[1], which = c->coord[2];
  if (j >= P.nkv[x]) return;
  const LlmLayer& Ly = P.layer[c->cst[0]];
  const int kvh = P.kv0[x] + j;
  const size_t row = ((size_t)P.pos * P.NKV + kvh) * P.HD;
  if (which == 0) head_post(P.qkv + (size_t)(P.NH + kvh) * P.HD, P.qk_norm ? Ly.kn : nullptr, nullptr, Ly.kc + row, LLM_RED(c));
  else { const float* v = P.qkv + (size_t)(P.NH + P.NKV + kvh) * P.HD; for (int d = threadIdx.x; d < P.HD; d += blockDim.x) Ly.vc[row + d] = __float2bfloat16(v[d]); }
}

// one chunk of positions for ALL of XCD x's heads: heads that share a KV head share its K/V loads
static __device__ __forceinline__ void attn_heads(const LlmLayer& Ly, int qh_begin, int nheads, int kvh, int ch, float* red, float* qs) {
  const int HD = P.HD, NC = P.NC, stride = HD + 2;
  const int n = P.pos + 1, t0 = ch * P.CHUNK, t1 = min(t0 + P.CHUNK, n), len = t1 - t0;
  if (len <= 0) {
    for (int j = 0; j < nheads; ++j) {
      float* part = P.attn_part + ((size_t)(qh_begin + j) * NC + ch) * stride;
      for (int d = threadIdx.x; d < HD; d += blockDim.x) part[d] = 0.f;
      if (threadIdx.x == 0) { part[HD] = -INFINITY; part[HD + 1] = 0.f; }
    }
    __syncthreads();
    return;
  }
  float* sc = qs + nheads * HD;                            // [nheads][CHUNK]
  for (int i = threadIdx.x; i < nheads * HD; i += blockDim.x) qs[i] = P.q[(size_t)qh_begin * HD + i];
  __syncthreads();
  const float scale = rsqrtf((float)HD);
  for (int idx = threadIdx.x; idx < nheads * len; idx += blockDim.x) {
    const int j = idx / len, t = idx % len;
    const __hip_bfloat16* k = Ly.kc + ((size_t)(t0 + t) * P.NKV + kvh) * HD;
    const float* q = qs + j * HD;
    float sacc = 0.f;
    for (int d = 0; d < HD; d += 8) sacc += dot8(*(const uint4*)(k + d), q + d);
    sc[j * P.CHUNK + t] = sacc * scale;
  }
  __syncthreads();
  float* mrow = red + 64; float* lrow = mrow + 16;
  for (int j = etx_wave(); j < nheads; j += etx_nwaves()) {
    float m = -INFINITY;
    for (int t = etx_lane(); t < len; t += ETX_WAVE_SIZE) m = fmaxf(m, sc[j * P.CHUNK + t]);
    m = etx_wave_max(m);
    float l = 0.f;
    for (int t = etx_lane(); t < len; t += ETX_WAVE_SIZE) { const float pv = __expf(sc[j * P.CHUNK + t] - m); sc[j * P.CHUNK + t] = pv; l += pv; }
    l = etx_wave_sum(l);
    if (etx_lane() == 0) { mrow[j] = m; lrow[j] = l; }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < nheads * HD; idx += blockDim.x) {
    const int j = idx / HD, d = idx % HD;
    float acc = 0.f;
    for (int t = 0; t < len; ++t) acc += sc[j * P.CHUNK + t] * etx_b2f(Ly.vc[((size_t)(t0 + t) * P.NKV + kvh) * HD + d]);
    float* part = P.attn_part + ((size_t)(qh_begin + j) * NC + ch) * stride;
    part[d] = acc;
    if (d == 0) { part[HD] = mrow[j]; part[HD + 1] = lrow[j]; }
  }
  __syncthreads();
}

extern "C" __device__ void llm_s_attn(const etx_ctx* c) {
  const int x = c->coord[0], ch = c->coord[1];
  if (P.nqh[x] == 0) return;
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* qs = LLM_VEC(c);
  const int qend = P.qh0[x] + P.nqh[x];
  for (int qh = P.qh0[x]; qh < qend;) {                     // one call per KV head the XCD's heads use
    const int kvh = qh / P.G, last = min(qend, (kvh + 1) * P.G);
    attn_heads(Ly, qh, last - qh, kvh, ch, red, qs);
    qh = last;
  }
}

extern "C" __device__ void llm_s_merge(const etx_ctx* c) {
  const int x = c->coord[0], j = c->coord[1];
  if (j >= P.nqh[x]) return;
  etx_ctx cc = *c; cc.coord[0] = P.qh0[x] + j;
  llm_merge(&cc);
}

// o_part[x] = Wo[:, columns of XCD x's heads] . o[those heads]; zero when the XCD owns no head
extern "C" __device__ void llm_s_oproj(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  const int K = P.nqh[x] * P.HD;
  int r1; const int r0 = rows_of(P.H, w, P.W, &r1);
  float* out = P.o_part + (size_t)x * P.H;
  if (K == 0) { for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) out[r] = 0.f; return; }
  copy_to(P.o + (size_t)P.qh0[x] * P.HD, K, xs);
  if (r0 >= r1) return;
  gemv_rows(Ly.wo_x[x], K, xs, r0, r1, xs + K, [&](int r, float v) { out[r] = v; });
}

extern "C" __device__ void llm_s_gateup(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c); float* gtmp = red + 64;
  rmsnorm_to(P.xa_x + (size_t)x * P.H, Ly.ln2, P.H, xs, red);
  int r1; const int r0 = rows_of(P.ni[x], w, P.W, &r1);
  if (r0 >= r1) return;
  const int base = P.i0[x];
  gemv_rows(Ly.wg + (size_t)base * P.H, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { gtmp[r - r0] = bfr(v); });
  gemv_rows(Ly.wu + (size_t)base * P.H, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { P.h[base + r] = bfr(bfr(silu(gtmp[r - r0])) * bfr(v)); });
}

extern "C" __device__ void llm_s_down(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  const int K = P.ni[x];
  copy_to(P.h + P.i0[x], K, xs);
  int r1; const int r0 = rows_of(P.H, w, P.W, &r1);
  if (r0 >= r1) return;
  float* out = P.d_part + (size_t)x * P.H;
  gemv_rows(Ly.wd_x[x], K, xs, r0, r1, xs + K, [&](int r, float v) { out[r] = v; });
}

// MoE: every XCD computes all E logits into its own copy (no barrier), then handles slots x, x+X, ...
extern "C" __device__ void llm_s_router(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c);
  rmsnorm_to(P.xa_x + (size_t)x * P.H, Ly.ln2, P.H, xs, red);
  int r1; const int r0 = rows_of(P.E, w, P.W, &r1);
  if (r0 >= r1) return;
  float* out = P.rlog_x + (size_t)x * P.E;
  gemv_rows(Ly.wr, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { out[r] = bfr(v); });
}

static __device__ __forceinline__ void topk_route_x(int x, float* red) {   // topk_route on XCD x's logits
  int* ids = (int*)(red + 64); float* wts = red + 80;
  if (etx_wave() == 0) {
    const int lane = etx_lane(), E = P.E;
    const float* lg = P.rlog_x + (size_t)x * E;
    float p[4]; bool taken[4]; float m = -INFINITY;
#pragma unroll
    for (int k = 0; k < 4; ++k) { const int e = lane + k * ETX_WAVE_SIZE; p[k] = e < E ? lg[e] : -INFINITY; taken[k] = e >= E; m = fmaxf(m, p[k]); }
    m = etx_wave_max(m);
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) { p[k] = taken[k] ? 0.f : __expf(p[k] - m); s += p[k]; }
    s = etx_wave_sum(s);
    float sel = 0.f;
    for (int t = 0; t < P.TOPK; ++t) {
      float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
      for (int k = 0; k < 4; ++k) { const int e = lane + k * ETX_WAVE_SIZE; if (!taken[k] && p[k] / s > bv) { bv = p[k] / s; bi = e; } }
      for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor(bv, off, ETX_WAVE_SIZE); const int oi = __shfl_xor(bi, off, ETX_WAVE_SIZE);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) if (lane + k * ETX_WAVE_SIZE == bi) taken[k] = true;
      if (lane == 0) { ids[t] = bi; wts[t] = bv; }
      sel += bv;
    }
    if (lane == 0) for (int t = 0; t < P.TOPK; ++t) wts[t] = bfr(P.norm_topk ? wts[t] / sel : wts[t]);
  }
  __syncthreads();
}

extern "C" __device__ void llm_s_egu(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c); float* gtmp = red + 96;
  topk_route_x(x, red);
  const int* ids = (const int*)(red + 64);
  rmsnorm_to(P.xa_x + (size_t)x * P.H, Ly.ln2, P.H, xs, red);
  int r1; const int r0 = rows_of(P.MI, w, P.W, &r1);
  for (int s = x; s < P.TOPK; s += P.X) {
    const int e = ids[s];
    if (w == 0 && threadIdx.x == 0) P.eid[s] = e;
    if (r0 >= r1) continue;
    const size_t eoff = (size_t)e * P.MI * P.H;
    gemv_rows(Ly.eg + eoff, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { gtmp[r - r0] = bfr(v); });
    gemv_rows(Ly.eu + eoff, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { P.hexp[(size_t)s * P.MI + r] = bfr(bfr(silu(gtmp[r - r0])) * bfr(v)); });
  }
}

extern "C" __device__ void llm_s_edn(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c);
  topk_route_x(x, red);
  const int* ids = (const int*)(red + 64); const float* wts = red + 80;
  int r1; const int r0 = rows_of(P.H, w, P.W, &r1);
  for (int s = x; s < P.TOPK; s += P.X) {
    copy_to(P.hexp + (size_t)s * P.MI, P.MI, xs);
    if (r0 >= r1) continue;
    float* out = P.d_part + (size_t)s * P.H;
    const float ws = wts[s];
    gemv_rows(Ly.ed + (size_t)ids[s] * P.H * P.MI, P.MI, xs, r0, r1, xs + P.MI, [&](int r, float v) { out[r] = bfr(bfr(v) * ws); });
  }
}

extern "C" __device__ void llm_s_lmhead(const etx_ctx* c) {
  float* red = LLM_RED(c); float* xs = LLM_VEC(c); float* lv = xs + P.H;
  rmsnorm_to(P.xfin, P.final_norm, P.H, xs, red);
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_lm, P.V, &r1);
  const __hip_bfloat16* W = P.tie ? P.embed : P.lm_head;
  gemv_rows(W, P.H, xs, r0, r1, lv + P.rpt_lm, [&](int r, float v) { lv[r - r0] = bfr(v); });
  __syncthreads();
  if (etx_wave() == 0) {
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int k = etx_lane(); k < r1 - r0; k += ETX_WAVE_SIZE) if (lv[k] > bv) { bv = lv[k]; bi = r0 + k; }
    for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off, ETX_WAVE_SIZE); const int oi = __shfl_xor(bi, off, ETX_WAVE_SIZE);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (etx_lane() == 0) { P.lm_val[c->coord[0]] = bv; P.lm_idx[c->coord[0]] = bi; }
  }
}
