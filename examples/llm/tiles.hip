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

typedef __attribute__((address_space(3))) const float lds_cf;
typedef __attribute__((address_space(3))) float lds_f;
static __device__ __forceinline__ float dot8(uint4 w, lds_cf* x) {
  const uint32_t u[4] = {w.x, w.y, w.z, w.w};
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s += __uint_as_float(u[i] << 16) * x[2 * i];
    s += __uint_as_float(u[i] & 0xffff0000u) * x[2 * i + 1];
  }
  return s;
}
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

#include "examples/llm/gemv.h"

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
  __syncthreads();                       // the previous call's threads may still be reading v[]; this call rewrites it
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
  gemv_rows(Ly.wgu, P.H, xs, 2 * r0, 2 * r1, xs + P.H, [&](int r, float v) { gtmp[r - 2 * r0] = bfr(v); });   // rows: g0 u0 g1 u1 ...
  for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) P.h[r] = bfr(bfr(silu(gtmp[2 * (r - r0)])) * gtmp[2 * (r - r0) + 1]);
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
  const size_t eoff = (size_t)e * 2 * P.MI * P.H;
  gemv_rows(Ly.egu + eoff, P.H, xs, 2 * r0, 2 * r1, xs + P.H, [&](int r, float v) { gtmp[r - 2 * r0] = bfr(v); });
  for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) P.hexp[(size_t)s * P.MI + r] = bfr(bfr(silu(gtmp[2 * (r - r0)])) * gtmp[2 * (r - r0) + 1]);
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
// n == 8 partials (8 XCDs, or top-8): four outputs per thread with all 32 loads issued before any add
// (v3 with a 16-wide register block spilled to scratch and doubled the time); other n: plain loop.
static __device__ __forceinline__ void fold_sum_rows(const float* base, int n, int H, const float* addend, float* out, bool bf16_chain,
                                                     const int* order, int i_begin, int i_end) {
  if (n == 8) {
    for (int i0 = i_begin + (int)threadIdx.x; i0 < i_end; i0 += blockDim.x * 4) {
      float v[4][8], a[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int i = min(i0 + m * (int)blockDim.x, i_end - 1);
        a[m] = addend[i];
#pragma unroll
        for (int k = 0; k < 8; ++k) v[m][k] = base[(size_t)(order ? order[k] : k) * H + i];
      }
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int i = i0 + m * (int)blockDim.x;
        if (i < i_end) {
          float acc = 0.f;
          if (bf16_chain) { for (int k = 0; k < 8; ++k) acc = bfr(acc + v[m][k]); out[i] = bfr(a[m] + acc); }
          else { for (int k = 0; k < 8; ++k) acc += v[m][k]; out[i] = bfr(a[m] + bfr(acc)); }
        }
      }
    }
    return;
  }
  for (int i = i_begin + (int)threadIdx.x; i < i_end; i += blockDim.x) {
    float acc = 0.f;
    if (bf16_chain) { for (int k = 0; k < n; ++k) acc = bfr(acc + base[(size_t)(order ? order[k] : k) * H + i]); out[i] = bfr(addend[i] + acc); }
    else { for (int k = 0; k < n; ++k) acc += base[(size_t)k * H + i]; out[i] = bfr(addend[i] + bfr(acc)); }
  }
}

// elements [i_begin, i_end) of the residual entering layer L: xb for L == 0, else bf16(xa + bf16(sum of the
// down partials)); MoE sums the slots in ascending expert id in bf16 (HF's index_add_)
static __device__ __forceinline__ void fold_down_to(int L, float* out, int* order_scratch, int i_begin, int i_end) {
  const int H = P.H;
  if (L <= 0) { for (int i = i_begin + (int)threadIdx.x; i < i_end; i += blockDim.x) out[i] = P.x[i]; return; }
  if (P.moe) {
    if (threadIdx.x == 0) {                 // slots sorted by expert id (HF's index_add_ order); TOPK <= 16
      for (int t = 0; t < P.TOPK; ++t) order_scratch[t] = t;
      for (int a = 1; a < P.TOPK; ++a) for (int b = a; b > 0 && P.eid[order_scratch[b]] < P.eid[order_scratch[b - 1]]; --b) { const int tmp = order_scratch[b]; order_scratch[b] = order_scratch[b - 1]; order_scratch[b - 1] = tmp; }
    }
    __syncthreads();
    fold_sum_rows(P.d_part, P.TOPK, H, P.xa, out, true, order_scratch, i_begin, i_end);
  } else {
    fold_sum_rows(P.d_part, P.X, H, P.xa, out, false, nullptr, i_begin, i_end);
  }
}

// sum of squares of out[i_begin, i_end) -> ssp (one float), for the consumers' RMSNorm scale
static __device__ __forceinline__ void ss_partial(const float* out, int i_begin, int i_end, float* ssp, float* red) {
  float ss = 0.f;
  for (int i = i_begin + (int)threadIdx.x; i < i_end; i += blockDim.x) { const float v = out[i]; ss += v * v; }
  ss = etx_block_sum(ss, red);
  if (threadIdx.x == 0) *ssp = ss;
}

// RMSNorm applied while copying: xs[i] = bf16(w[i] * bf16(xin[i] * r)), r from the F partial sums of squares
static __device__ __forceinline__ void norm_copy_to(const float* xin, const float* ssp, int nparts, const __hip_bfloat16* w, int n, float* xs) {
  float ss = 0.f;
  for (int f = 0; f < nparts; ++f) ss += ssp[f];
  const float r = rsqrtf(ss / (float)n + P.eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) xs[i] = bfr(etx_b2f(w[i]) * bfr(xin[i] * r));
  __syncthreads();
}

static __device__ __forceinline__ int fold_range(int f, int* i_end) {   // fold task f's elements of the vector
  const int per = (P.H + P.F - 1) / P.F, b = min(P.H, f * per);
  *i_end = min(P.H, b + per);
  return b;
}

// fold task f of XCD x before layer L: folds elements [b, e) of the residual into xin_x[x], stores the
// XCD's slice of xb that lies in [b, e), and its partial sum of squares (the consumers apply the norm)
extern "C" __device__ void llm_s_fold_d(const etx_ctx* c) {
  const int L = c->cst[0], x = c->coord[0], f = c->coord[1];
  float* red = LLM_RED(c);
  float* out = P.xin_x + (size_t)x * P.H;
  int e; const int b = fold_range(f, &e);
  fold_down_to(L, out, (int*)(red + 64), b, e);
  __syncthreads();
  if (L > 0) for (int i = max(b, P.h0[x]) + threadIdx.x; i < min(e, P.h0[x] + P.nh[x]); i += blockDim.x) P.x[i] = out[i];
  if (P.layer_dump && P.dump_step == P.tok && x == 0) for (int i = b + threadIdx.x; i < e; i += blockDim.x) P.layer_dump[(size_t)L * P.H + i] = out[i];
  ss_partial(out, b, e, P.ss_x + ((size_t)x * 2 + 0) * P.F + f, red);
}

// fold task f of XCD x after o_proj: xa_x[x][b, e) = bf16(xb + bf16(sum o_part)); stores the slice of xa; partial sum of squares
extern "C" __device__ void llm_s_fold_o(const etx_ctx* c) {
  const int x = c->coord[0], f = c->coord[1], H = P.H;
  float* red = LLM_RED(c);
  float* out = P.xa_x + (size_t)x * H;
  int e; const int b = fold_range(f, &e);
  fold_sum_rows(P.o_part, P.X, H, P.x, out, false, nullptr, b, e);
  __syncthreads();
  for (int i = max(b, P.h0[x]) + threadIdx.x; i < min(e, P.h0[x] + P.nh[x]); i += blockDim.x) P.xa[i] = out[i];
  ss_partial(out, b, e, P.ss_x + ((size_t)x * 2 + 1) * P.F + f, red);
}

extern "C" __device__ void llm_s_lmfold(const etx_ctx* c) {
  const int f = c->coord[0];
  float* red = LLM_RED(c);
  int e; const int b = fold_range(f, &e);
  fold_down_to(c->cst[0], P.xfin, (int*)(red + 64), b, e);
  __syncthreads();
  if (P.layer_dump && P.dump_step == P.tok) for (int i = b + threadIdx.x; i < e; i += blockDim.x) P.layer_dump[(size_t)c->cst[0] * P.H + i] = P.xfin[i];
  ss_partial(P.xfin, b, e, P.ss_f + f, red);
}

// Last-arriver pattern inside one XCD: every wave drains its stores, thread 0 bumps the XCD's counter, and the
// task that completes the group invalidates its L1 and returns true. Counters are monotonic; the target scales
// with the step number. Unfused launches spread the group over XCDs, so they use device-scope fences.
static __device__ __forceinline__ bool last_arriver(int32_t* ctr, int target, float* red) {
  int* flag = (int*)(red + 255);
  if (P.unfused) ETX_RELEASE_DEVICE(); else ETX_RELEASE_DOMAIN();
  __syncthreads();
  if (threadIdx.x == 0) { const int old = __hip_atomic_fetch_add(ctr, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); *flag = (old + 1 == target); }
  __syncthreads();
  const bool last = *flag != 0;
  if (last) { if (P.unfused) ETX_ACQUIRE_DEVICE(); else ETX_ACQUIRE_DOMAIN(); }
  return last;
}

// local qkv row r of XCD x -> global row in P.qkv (q rows, then [k; v] per needed KV head)
static __device__ __forceinline__ int qkv_row(int x, int r) {
  const int HD = P.HD, nq = P.nqh[x] * HD;
  if (r < nq) return P.qh0[x] * HD + r;
  const int rr = r - nq, j = rr / (2 * HD), o = rr % (2 * HD);
  return o < HD ? (P.NH + P.kv0[x] + j) * HD + o : (P.NH + P.NKV + P.kv0[x] + j) * HD + (o - HD);
}

// raw q/k/v rows of the XCD's slice; the post-processing (norm, RoPE, KV write) is done by the attention tasks
extern "C" __device__ void llm_s_qkv(const etx_ctx* c) {
  const int L = c->cst[0], x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[L];
  float* xs = LLM_VEC(c);
  norm_copy_to(P.xin_x + (size_t)x * P.H, P.ss_x + ((size_t)x * 2 + 0) * P.F, P.F, Ly.ln1, P.H, xs);
  const int nrows = (P.nqh[x] + 2 * P.nkv[x]) * P.HD;
  int r1; const int r0 = rows_of(nrows, w, P.W, &r1);
  if (r0 >= r1) return;
  const __hip_bfloat16* b = Ly.bqkv_x[x];
  gemv_rows(Ly.wqkv_x[x], P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { P.qkv[qkv_row(x, r)] = bfr(v + (b ? etx_b2f(b[r]) : 0.f)); });
}

// one chunk of positions for ALL of XCD x's heads. Every chunk task norms and rotates the q of the XCD's heads
// into LDS (cheap, redundant); the chunk task holding the newest position also writes that K/V cache row.
// K rows, then V rows, are staged through LDS with 16-byte coalesced loads. Same arithmetic order as before.
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
  uint4* kk = (uint4*)(sc + nheads * P.CHUNK);             // [CHUNK][HD] bf16 K
  const int per_row = HD / 8;
  uint4* vv = kk + P.CHUNK * per_row;                      // [CHUNK][HD] bf16 V
  float* kt = (float*)(vv + P.CHUNK * per_row);            // HD floats: the new K row before it is stored
  const bool mine = t0 <= P.pos && P.pos < t1;
  // K and V rows of the chunk into LDS first, both in flight at once, before the q post (the row of the new
  // position, if it is this chunk's, is computed below and written to the cache and the staging together)
  for (int i = threadIdx.x; i < len * per_row; i += blockDim.x) {
    const int t = i / per_row, u = i % per_row;
    if (mine && t0 + t == P.pos) continue;
    const size_t off = ((size_t)(t0 + t) * P.NKV + kvh) * HD + u * 8;
    kk[i] = *(const uint4*)(Ly.kc + off);
    vv[i] = *(const uint4*)(Ly.vc + off);
  }
  for (int j = 0; j < nheads; ++j)
    head_post(P.qkv + (size_t)(qh_begin + j) * HD, P.qk_norm ? Ly.qn : nullptr, qs + j * HD, nullptr, red);
  if (mine) {
    const size_t row = ((size_t)P.pos * P.NKV + kvh) * HD;
    head_post(P.qkv + (size_t)(P.NH + kvh) * HD, P.qk_norm ? Ly.kn : nullptr, kt, nullptr, red);
    __syncthreads();
    const float* v = P.qkv + (size_t)(P.NH + P.NKV + kvh) * HD;
    __hip_bfloat16* kb = (__hip_bfloat16*)kk; __hip_bfloat16* vb = (__hip_bfloat16*)vv;
    const int tl = P.pos - t0;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
      const __hip_bfloat16 kd = __float2bfloat16(kt[d]), vd = __float2bfloat16(v[d]);
      Ly.kc[row + d] = kd; Ly.vc[row + d] = vd; kb[tl * HD + d] = kd; vb[tl * HD + d] = vd;
    }
  }
  __syncthreads();
  const float scale = rsqrtf((float)HD);
  // LDS-typed views: through the generic pointers the compiler emits flat loads, which are slower and wait
  // on the vector memory counter as well
  typedef __attribute__((address_space(3))) const uint4 lds_u4;
  typedef __attribute__((address_space(3))) const unsigned short lds_u16;
  lds_u4* kl = (lds_u4*)kk; lds_cf* ql = (lds_cf*)qs; lds_f* scl = (lds_f*)sc;
  for (int idx = threadIdx.x; idx < nheads * len; idx += blockDim.x) {
    const int j = idx / len, t = idx % len;
    lds_u4* k = kl + t * per_row;
    lds_cf* q = ql + j * HD;
    float sacc = 0.f;
    for (int u = 0; u < per_row; ++u) sacc += dot8(k[u], q + u * 8);
    scl[j * P.CHUNK + t] = sacc * scale;
  }
  __syncthreads();
  float* mrow = red + 64; float* lrow = mrow + 16;
  for (int j = etx_wave(); j < nheads; j += etx_nwaves()) {
    float m = -INFINITY;
    for (int t = etx_lane(); t < len; t += ETX_WAVE_SIZE) m = fmaxf(m, scl[j * P.CHUNK + t]);
    m = etx_wave_max(m);
    float l = 0.f;
    for (int t = etx_lane(); t < len; t += ETX_WAVE_SIZE) { const float pv = __expf(scl[j * P.CHUNK + t] - m); scl[j * P.CHUNK + t] = pv; l += pv; }
    l = etx_wave_sum(l);
    if (etx_lane() == 0) { mrow[j] = m; lrow[j] = l; }
  }
  __syncthreads();
  lds_u16* vl = (lds_u16*)vv;
  for (int idx = threadIdx.x; idx < nheads * HD; idx += blockDim.x) {
    const int j = idx / HD, d = idx % HD;
    float acc = 0.f;
    for (int t = 0; t < len; ++t) acc += scl[j * P.CHUNK + t] * __uint_as_float((unsigned)vl[t * HD + d] << 16);
    float* part = P.attn_part + ((size_t)(qh_begin + j) * NC + ch) * stride;
    part[d] = acc;
    if (d == 0) { part[HD] = mrow[j]; part[HD + 1] = lrow[j]; }
  }
  __syncthreads();
}

extern "C" __device__ void llm_s_attn(const etx_ctx* c) {
  const int L = c->cst[0], x = c->coord[0], ch = c->coord[1];
  if (P.nqh[x] == 0) return;
  const LlmLayer& Ly = P.layer[L];
  float* red = LLM_RED(c); float* qs = LLM_VEC(c);
  const int qend = P.qh0[x] + P.nqh[x];
  for (int qh = P.qh0[x]; qh < qend;) {
    const int kvh = qh / P.G, last = min(qend, (kvh + 1) * P.G);
    attn_heads(Ly, qh, last - qh, kvh, ch, red, qs);
    qh = last;
  }
}

// merge of one head's chunks into `out`: chunk weights once into wch (NC floats), then per d
static __device__ __forceinline__ void merge_head(int qh, float* out, float* wch, float* red, float* half_sum) {
  const int HD = P.HD, NC = P.NC, stride = HD + 2;
  const float* part = P.attn_part + (size_t)qh * NC * stride;
  float m = -INFINITY;
  for (int ch = threadIdx.x; ch < NC; ch += blockDim.x) m = fmaxf(m, part[ch * stride + HD]);
  const float M = etx_block_max(m, red);
  float l = 0.f;
  for (int ch = threadIdx.x; ch < NC; ch += blockDim.x) {
    const float mc = part[ch * stride + HD];
    const float w = mc > -INFINITY ? __expf(mc - M) : 0.f;
    wch[ch] = w; l += w * part[ch * stride + HD + 1];
  }
  const float Lsum = etx_block_sum(l, red);
  // threads = HD (128) columns x ng chunk groups: each thread sums NC/ng chunks with all loads in flight
  const int ng = max(1, (int)blockDim.x / HD);
  lds_cf* wl = (lds_cf*)wch; lds_f* hs = (lds_f*)half_sum;      // LDS-typed: no flat loads in the loop
  {
    const int d = threadIdx.x % HD, g = threadIdx.x / HD;
    float acc = 0.f;
    if (g < ng)
#pragma unroll 16
      for (int ch = g; ch < NC; ch += ng) acc += wl[ch] * part[ch * stride + d];
    if (g < ng) hs[g * HD + d] = acc;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < HD; d += blockDim.x) {
    float v = 0.f;
    for (int g = 0; g < ng; ++g) v += hs[g * HD + d];
    out[d] = bfr(v / Lsum);
  }
  __syncthreads();
}

// o_part[x] = Wo[:, columns of XCD x's heads] . o[those heads]; every task merges the XCD's heads in its
// prologue (4 x 16 KB of partials from the XCD's L2, in parallel, instead of one task doing it serially)
extern "C" __device__ void llm_s_oproj(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c);
  const int K = P.nqh[x] * P.HD;
  int r1; const int r0 = rows_of(P.H, w, P.W, &r1);
  float* out = P.o_part + (size_t)x * P.H;
  if (K == 0) { for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) out[r] = 0.f; return; }
  for (int j = 0; j < P.nqh[x]; ++j) merge_head(P.qh0[x] + j, xs + j * P.HD, red + 64, red, xs + K);   // wch at red+64 (NC <= 128), ng x HD scratch after the o vector
  if (r0 >= r1) return;
  gemv_rows(Ly.wo_x[x], K, xs, r0, r1, xs + K, [&](int r, float v) { out[r] = v; });
}

// dense MLP, one interleaved gate/up stream per task; the normed input comes from the XCD's fold task
extern "C" __device__ void llm_s_gateup(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c); float* gu = red + 64;   // 2 * rows per task <= 190
  norm_copy_to(P.xa_x + (size_t)x * P.H, P.ss_x + ((size_t)x * 2 + 1) * P.F, P.F, Ly.ln2, P.H, xs);
  int r1; const int r0 = rows_of(P.ni[x], w, P.W, &r1);
  if (r0 >= r1) return;
  const int base = P.i0[x];
  gemv_rows(Ly.wgu + (size_t)2 * base * P.H, P.H, xs, 2 * r0, 2 * r1, xs + P.H, [&](int r, float v) { gu[r - 2 * r0] = bfr(v); });
  for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) P.h[base + r] = bfr(bfr(silu(gu[2 * (r - r0)])) * gu[2 * (r - r0) + 1]);
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

// MoE router: every XCD computes all E logits into its own copy; the XCD's last router task selects the
// top-k once and publishes ids / weights for the expert tasks
// top-k of softmax(lg[0..E)), ids at red+64.., bf16 weights at red+80.. Every thread ranks one logit against
// all the others with LDS broadcast reads, so there is no serial shuffle chain (the wave-reduction version cost
// about 6 us per task); ties -> lower id, like torch.topk on a sorted softmax. lg may be global or LDS;
// scratch: 2 * E floats of LDS. E % 4 == 0.
static __device__ __forceinline__ void topk_local(const float* lg, float* scratch, float* red) {
  int* ids = (int*)(red + 64); float* wts = red + 80; float* pl = scratch; float* pp = scratch + P.E;
  const int E = P.E, E4 = E / 4, t = threadIdx.x;
  for (int e = t; e < E; e += blockDim.x) pl[e] = lg[e];
  __syncthreads();
  const float4* pl4 = (const float4*)pl; const float4* pp4 = (const float4*)pp;
  float m = -INFINITY;
#pragma unroll 8
  for (int j = 0; j < E4; ++j) { const float4 v = pl4[j]; m = fmaxf(fmaxf(m, v.x), fmaxf(v.y, fmaxf(v.z, v.w))); }
  for (int e = t; e < E; e += blockDim.x) pp[e] = __expf(pl[e] - m);
  __syncthreads();
  float s = 0.f;
#pragma unroll 8
  for (int j = 0; j < E4; ++j) { const float4 v = pp4[j]; s += v.x + v.y + v.z + v.w; }
  for (int e = t; e < E; e += blockDim.x) {
    const float pe = pp[e]; int rank = 0;
#pragma unroll 8
    for (int j = 0; j < E4; ++j) {
      const float4 v = pp4[j]; const int j0 = 4 * j;
      rank += (v.x > pe || (v.x == pe && j0 < e)) + (v.y > pe || (v.y == pe && j0 + 1 < e)) + (v.z > pe || (v.z == pe && j0 + 2 < e)) + (v.w > pe || (v.w == pe && j0 + 3 < e));
    }
    if (rank < P.TOPK) { ids[rank] = e; wts[rank] = pe / s; }
  }
  __syncthreads();
  if (t == 0) {
    float sel = 0.f;
    for (int k = 0; k < P.TOPK; ++k) sel += wts[k];
    for (int k = 0; k < P.TOPK; ++k) wts[k] = bfr(P.norm_topk ? wts[k] / sel : wts[k]);
  }
  __syncthreads();
}

extern "C" __device__ void llm_s_router(const etx_ctx* c) {
  const int L = c->cst[0], x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[L];
  float* xs = LLM_VEC(c);
  norm_copy_to(P.xa_x + (size_t)x * P.H, P.ss_x + ((size_t)x * 2 + 1) * P.F, P.F, Ly.ln2, P.H, xs);
  int r1; const int r0 = rows_of(P.E, w, P.W, &r1);
  float* out = P.rlog_x + (size_t)x * P.E;
  if (r0 < r1) gemv_rows(Ly.wr, P.H, xs, r0, r1, xs + P.H, [&](int r, float v) { out[r] = bfr(v); });
}

extern "C" __device__ void llm_s_egu(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* red = LLM_RED(c); float* xs = LLM_VEC(c); float* gu = red + 96;   // ids/weights live at red+64..96
  // (computing the router inside every expert task instead of a router phase was tried: 38 tasks per XCD
  // touching the same 512 KB at once cost ~28 us per layer, the L2 does not merge those misses)
  topk_local(P.rlog_x + (size_t)x * P.E, xs, red);                  // xs is free until the copy below
  const int* ids = (const int*)(red + 64); const float* wts = red + 80;
  if (w == 0 && threadIdx.x < P.TOPK) { P.rids_x[x * 16 + threadIdx.x] = ids[threadIdx.x]; P.rw_x[x * 16 + threadIdx.x] = wts[threadIdx.x]; }   // for the down tasks
  norm_copy_to(P.xa_x + (size_t)x * P.H, P.ss_x + ((size_t)x * 2 + 1) * P.F, P.F, Ly.ln2, P.H, xs);
  int r1; const int r0 = rows_of(P.MI, w, P.W, &r1);
  for (int s = x; s < P.TOPK; s += P.X) {
    const int e = ids[s];
    if (w == 0 && threadIdx.x == 0) P.eid[s] = e;
    if (r0 >= r1) continue;
    gemv_rows(Ly.egu + (size_t)e * 2 * P.MI * P.H, P.H, xs, 2 * r0, 2 * r1, xs + P.H, [&](int r, float v) { gu[r - 2 * r0] = bfr(v); });
    for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) P.hexp[(size_t)s * P.MI + r] = bfr(bfr(silu(gu[2 * (r - r0)])) * gu[2 * (r - r0) + 1]);
    __syncthreads();
  }
}

extern "C" __device__ void llm_s_edn(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1];
  const LlmLayer& Ly = P.layer[c->cst[0]];
  float* xs = LLM_VEC(c);
  const int* ids = P.rids_x + x * 16; const float* wts = P.rw_x + x * 16;   // published by gate_up task 0 of this XCD
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
  float* xs = LLM_VEC(c); float* lv = xs + P.H;
  norm_copy_to(P.xfin, P.ss_f, P.F, P.final_norm, P.H, xs);
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

// ---------------------------------------------------------------- cross-barrier weight prefetch (Pass 7 hook)
// Runs before the task's wait, in every wave but wave 0 (thread 0 polls the event meanwhile). Touches the
// first LLM_PF_KB kilobytes of THIS task's weight block so the lines sit in the XCD's L2 when the body
// streams them: 38 tasks x 64 KB = 2.4 MB per XCD, inside the 4 MB L2. Weights never depend on events.
#ifndef LLM_PF_KB
#define LLM_PF_KB 64
#endif
static __device__ __forceinline__ void pf_touch(const void* p, size_t bytes) {
  if (etx_wave() == 0) return;
  const char* base = (const char*)p;
  const size_t lines = min(bytes, (size_t)LLM_PF_KB * 1024) / 128;
  int acc = 0;
  for (size_t l = threadIdx.x - ETX_WAVE_SIZE; l < lines; l += blockDim.x - ETX_WAVE_SIZE) acc += *(const volatile int*)(base + l * 128);
  asm volatile("" :: "v"(acc));
}
extern "C" __device__ void llm_s_qkv_pf(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1]; const LlmLayer& Ly = P.layer[c->cst[0]];
  int r1; const int r0 = rows_of((P.nqh[x] + 2 * P.nkv[x]) * P.HD, w, P.W, &r1);
  if (r0 < r1) pf_touch(Ly.wqkv_x[x] + (size_t)r0 * P.H, (size_t)(r1 - r0) * P.H * 2);
}
extern "C" __device__ void llm_s_oproj_pf(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1]; const LlmLayer& Ly = P.layer[c->cst[0]];
  const int K = P.nqh[x] * P.HD; int r1; const int r0 = rows_of(P.H, w, P.W, &r1);
  if (K && r0 < r1) pf_touch(Ly.wo_x[x] + (size_t)r0 * K, (size_t)(r1 - r0) * K * 2);
}
extern "C" __device__ void llm_s_gateup_pf(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1]; const LlmLayer& Ly = P.layer[c->cst[0]];
  int r1; const int r0 = rows_of(P.ni[x], w, P.W, &r1);
  if (r0 < r1) pf_touch(Ly.wgu + (size_t)2 * (P.i0[x] + r0) * P.H, (size_t)2 * (r1 - r0) * P.H * 2);
}
extern "C" __device__ void llm_s_down_pf(const etx_ctx* c) {
  const int x = c->coord[0], w = c->coord[1]; const LlmLayer& Ly = P.layer[c->cst[0]];
  const int K = P.ni[x]; int r1; const int r0 = rows_of(P.H, w, P.W, &r1);
  if (K && r0 < r1) pf_touch(Ly.wd_x[x] + (size_t)r0 * K, (size_t)(r1 - r0) * K * 2);
}
extern "C" __device__ void llm_s_egu_pf(const etx_ctx* c) {   // the expert is only known after the router: warm the router-independent part, nothing
  (void)c;
}
