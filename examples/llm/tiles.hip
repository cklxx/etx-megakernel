// Generic batch-1 decode tiles for examples/llm/model.py (link mode, one set for every model).
//
// Every body reads the model through the __constant__ LlmParams (llm_params.h); the layer index
// is the grid's compile-time constant (ctx->cst[0]); the task's rows come from its coordinate and
// the partition in LlmParams. Numerics follow Hugging Face's bf16 forward: values are kept in fp32
// but rounded to bf16 at every point where HF materialises a bf16 tensor (linear outputs, residual
// adds, RMSNorm before and after the weight, each RoPE product, SiLU, the MoE combine in ascending
// expert order), so the argmax agrees with HF greedy except at near-ties.
//
// The GEMV is simple and general, not tuned: each wave streams 4 weight rows at a time with 16-byte
// loads against the input vector held in LDS as fp32.
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

// rows [r0, r1) of a row-major bf16 matrix W with K columns, times xs (fp32, LDS). Each wave takes
// groups of 4 rows; lane 0 of the wave calls epi(row, dot) for each finished row.
template <class Epi>
static __device__ __forceinline__ void gemv_rows(const __hip_bfloat16* __restrict__ W, int K, const float* xs, int r0, int r1, Epi epi) {
  const int lane = etx_lane(), nw = etx_nwaves();
  for (int base = r0 + etx_wave() * 4; base < r1; base += nw * 4) {
    const int nr = min(4, r1 - base);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const __hip_bfloat16* w0 = W + (size_t)base * K;
#pragma unroll 2
    for (int k = lane * 8; k < K; k += ETX_WAVE_SIZE * 8) {
      float xv[8];
      const float4 a = *(const float4*)(xs + k), b = *(const float4*)(xs + k + 4);
      xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w; xv[4] = b.x; xv[5] = b.y; xv[6] = b.z; xv[7] = b.w;
      uint4 wv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) if (j < nr) wv[j] = *(const uint4*)(w0 + (size_t)j * K + k);
#pragma unroll
      for (int j = 0; j < 4; ++j) if (j < nr) acc[j] += dot8(wv[j], xv);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v = etx_wave_sum(acc[j]);
      if (j < nr && lane == 0) epi(base + j, v);
    }
  }
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
  gemv_rows(W, P.H, xs, r0, r1, [&](int r, float v) { lv[r - r0] = bfr(v); });
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
  gemv_rows(Ly.wqkv, P.H, xs, r0, r1, [&](int r, float v) { P.qkv[r] = bfr(v + (b ? etx_b2f(b[r]) : 0.f)); });
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
  const int qh = c->coord[0], HD = P.HD, NC = P.NC, stride = HD + 2;
  const float* part = P.attn_part + (size_t)qh * NC * stride;
  float M = -INFINITY;
  for (int ch = 0; ch < NC; ++ch) M = fmaxf(M, part[ch * stride + HD]);
  float Lsum = 0.f;
  for (int ch = 0; ch < NC; ++ch) { const float m = part[ch * stride + HD]; if (m > -INFINITY) Lsum += __expf(m - M) * part[ch * stride + HD + 1]; }
  for (int d = threadIdx.x; d < HD; d += blockDim.x) {
    float acc = 0.f;
    for (int ch = 0; ch < NC; ++ch) { const float m = part[ch * stride + HD]; if (m > -INFINITY) acc += __expf(m - M) * part[ch * stride + d]; }
    P.o[(size_t)qh * HD + d] = bfr(acc / Lsum);
  }
}

extern "C" __device__ void llm_oproj(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  const int K = P.NH * P.HD;
  copy_to(P.o, K, xs);
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_o, P.H, &r1);
  gemv_rows(Ly.wo, K, xs, r0, r1, [&](int r, float v) { P.x[r] = bfr(P.x[r] + bfr(v)); });
}

// ---------------------------------------------------------------- dense MLP
extern "C" __device__ void llm_gateup(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c);
  float* gtmp = red + 64;                 // up to 192 rows per task
  rmsnorm_to(P.x, Ly.ln2, P.H, xs, red);
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_gu, P.INTER, &r1);
  gemv_rows(Ly.wg, P.H, xs, r0, r1, [&](int r, float v) { gtmp[r - r0] = bfr(v); });
  __syncthreads();
  gemv_rows(Ly.wu, P.H, xs, r0, r1, [&](int r, float v) { P.h[r] = bfr(bfr(silu(gtmp[r - r0])) * bfr(v)); });
}

extern "C" __device__ void llm_down(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  copy_to(P.h, P.INTER, xs);
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_dn, P.H, &r1);
  gemv_rows(Ly.wd, P.INTER, xs, r0, r1, [&](int r, float v) { P.x[r] = bfr(P.x[r] + bfr(v)); });
}

// ---------------------------------------------------------------- MoE
extern "C" __device__ void llm_router(const etx_ctx* c) {
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  rmsnorm_to(P.x, Ly.ln2, P.H, xs, LLM_RED(c));
  int r1; const int r0 = task_rows(c->coord[0], P.rpt_r, P.E, &r1);
  gemv_rows(Ly.wr, P.H, xs, r0, r1, [&](int r, float v) { P.rlogits[r] = bfr(v); });
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
  gemv_rows(Ly.eg + eoff, P.H, xs, r0, r1, [&](int r, float v) { gtmp[r - r0] = bfr(v); });
  __syncthreads();
  gemv_rows(Ly.eu + eoff, P.H, xs, r0, r1, [&](int r, float v) { P.hexp[(size_t)s * P.MI + r] = bfr(bfr(silu(gtmp[r - r0])) * bfr(v)); });
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
    gemv_rows(Ly.ed + (size_t)ids[s] * P.H * P.MI, P.MI, xs + (size_t)s * P.MI, r0, r1,
              [&](int r, float v) { ys[s * nr + (r - r0)] = bfr(bfr(v) * wts[s]); });
  __syncthreads();
  for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) {
    float acc = 0.f;
    for (int t = 0; t < P.TOPK; ++t) acc = bfr(acc + ys[order[t] * nr + (r - r0)]);
    P.x[r] = bfr(P.x[r] + acc);
  }
}
