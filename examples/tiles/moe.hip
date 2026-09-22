// Tile bodies for examples/moe_layer.py (link mode). Reference implementations:
// correct, simple, wave-per-output GEMVs; weights stream once per task. Not
// tuned for bandwidth (a GEMM over the token batch would be); the point of this
// file is exercising every mechanism of the megakernel on real data.
// Dimensions must match examples/moe_layer.py.
#include "etx/tile_util.h"

#define D 1024
#define H 8
#define HD 128
#define S 512
#define NE 8
#define K 2
#define FF 512
#define TPT 4

// ---------------------------------------------------------------- norm (pure; inlined by Pass 3)
extern "C" __device__ void etx_tile_rmsnorm(const etx_ctx* c) {
  const int b = c->coord[0];
  const etx_bf16* x = (const etx_bf16*)c->args[0] + (size_t)b * D;
  float* xn = (float*)c->args[1] + (size_t)b * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) { float v = etx_b2f(x[i]); ss += v * v; }
  ss = etx_block_sum(ss, (float*)c->lds);
  const float r = rsqrtf(ss / D + 1e-6f);
  for (int i = threadIdx.x; i < D; i += blockDim.x) xn[i] = etx_b2f(x[i]) * r;
}

// ---------------------------------------------------------------- qkv projection (b, j)
extern "C" __device__ void etx_tile_gemv_qkv(const etx_ctx* c) {
  const int b = c->coord[0], j = c->coord[1];
  const float* xn = (const float*)c->args[0] + (size_t)b * D;
  const etx_bf16* W = (const etx_bf16*)c->args[1] + (size_t)j * D * D;     // rows [j*D, j*D+D) of (3D, D)
  float* out = (float*)c->args[2] + (size_t)b * 3 * D + (size_t)j * D;
  for (int o = etx_wave(); o < D; o += etx_nwaves()) {
    const float v = etx_wave_dot(W + (size_t)o * D, xn, D);
    if (etx_lane() == 0) out[o] = v;
  }
}
extern "C" __device__ void etx_prefetch_gemv_qkv(const etx_ctx* c) {
  etx_touch((const etx_bf16*)c->args[1] + (size_t)c->coord[1] * D * D, (size_t)D * D * 2);
}

// ---------------------------------------------------------------- attention (b, h) over the cache
extern "C" __device__ void etx_tile_attention(const etx_ctx* c) {
  const int b = c->coord[0], h = c->coord[1];
  const float* q = (const float*)c->args[0] + (size_t)b * 3 * D + (size_t)h * HD;
  const etx_bf16* kc = (const etx_bf16*)c->args[1];
  const etx_bf16* vc = (const etx_bf16*)c->args[2];
  float* attn = (float*)c->args[3] + (size_t)b * D + (size_t)h * HD;
  float* sc = (float*)c->lds;              // S scores
  float* red = sc + S;                     // reduction scratch
  const float scale = rsqrtf((float)HD);
  float local_max = -1e30f;
  for (int t = threadIdx.x; t < S; t += blockDim.x) {
    const etx_bf16* k = kc + (size_t)t * D + (size_t)h * HD;
    float s = 0.f;
    for (int d = 0; d < HD; ++d) s += q[d] * etx_b2f(k[d]);
    s *= scale;
    sc[t] = s;
    local_max = fmaxf(local_max, s);
  }
  __syncthreads();
  const float m = etx_block_max(local_max, red);
  float local_sum = 0.f;
  for (int t = threadIdx.x; t < S; t += blockDim.x) { const float p = __expf(sc[t] - m); sc[t] = p; local_sum += p; }
  __syncthreads();
  const float sum = etx_block_sum(local_sum, red);
  const float inv = 1.f / sum;
  for (int d = threadIdx.x; d < HD; d += blockDim.x) {
    float acc = 0.f;
    for (int t = 0; t < S; ++t) acc += sc[t] * etx_b2f(vc[(size_t)t * D + (size_t)h * HD + d]);
    attn[d] = acc * inv;
  }
}
extern "C" __device__ void etx_prefetch_attention(const etx_ctx* c) {
  // this head's slice of K and V: S rows x HD bf16 (strided by D)
  const int h = c->coord[1];
  const etx_bf16* kc = (const etx_bf16*)c->args[1];
  const etx_bf16* vc = (const etx_bf16*)c->args[2];
  int acc = 0;
  for (int t = threadIdx.x; t < S; t += blockDim.x) {
    acc += *(const volatile int*)(kc + (size_t)t * D + (size_t)h * HD);
    acc += *(const volatile int*)(vc + (size_t)t * D + (size_t)h * HD);
  }
  asm volatile("" :: "v"(acc));
}

// ---------------------------------------------------------------- o_proj + residual (b)
extern "C" __device__ void etx_tile_gemv_o(const etx_ctx* c) {
  const int b = c->coord[0];
  const float* a = (const float*)c->args[0] + (size_t)b * D;
  const etx_bf16* x = (const etx_bf16*)c->args[1] + (size_t)b * D;
  const etx_bf16* W = (const etx_bf16*)c->args[2];
  float* hh = (float*)c->args[3] + (size_t)b * D;
  for (int o = etx_wave(); o < D; o += etx_nwaves()) {
    const float v = etx_wave_dot(W + (size_t)o * D, a, D);
    if (etx_lane() == 0) hh[o] = etx_b2f(x[o]) + v;
  }
}
extern "C" __device__ void etx_prefetch_gemv_o(const etx_ctx* c) { etx_touch(c->args[2], (size_t)D * D * 2); }

// ---------------------------------------------------------------- router (b): top-2 of x[b][0..NE)
extern "C" __device__ void etx_tile_router(const etx_ctx* c) {
  if (threadIdx.x != 0) return;
  const int b = c->coord[0];
  const etx_bf16* x = (const etx_bf16*)c->args[0] + (size_t)b * D;
  int32_t* topk = (int32_t*)c->args[1] + b * K;
  float* w = (float*)c->args[2] + b * K;
  float v[NE];
  for (int e = 0; e < NE; ++e) v[e] = etx_b2f(x[e]);
  int e0 = 0;
  for (int e = 1; e < NE; ++e) if (v[e] > v[e0]) e0 = e;              // ties -> lower id (strict >)
  int e1 = (e0 == 0) ? 1 : 0;
  for (int e = 0; e < NE; ++e) if (e != e0 && v[e] > v[e1]) e1 = e;
  const int lo = e0 < e1 ? e0 : e1, hi = e0 < e1 ? e1 : e0;
  const float m = fmaxf(v[lo], v[hi]);
  const float p0 = __expf(v[lo] - m), p1 = __expf(v[hi] - m);
  topk[0] = lo; topk[1] = hi;
  w[0] = p0 / (p0 + p1); w[1] = p1 / (p0 + p1);
}

// ---------------------------------------------------------------- grouping (single task): CSR + runtime counts
extern "C" __device__ void etx_tile_grouping(const etx_ctx* c) {
  if (threadIdx.x != 0) return;
  const int B = c->shape[0];
  const int32_t* topk = (const int32_t*)c->args[0];
  int32_t* counts = (int32_t*)c->args[1];
  int32_t* tiles = (int32_t*)c->args[2];
  int32_t* tok_indptr = (int32_t*)c->args[3];
  int32_t* indptr = (int32_t*)c->args[4];
  int32_t* tok_slot = (int32_t*)c->args[5];
  int32_t* tile_expert = (int32_t*)c->args[6];
  for (int e = 0; e < NE; ++e) counts[e] = 0;
  for (int i = 0; i < B * K; ++i) counts[topk[i]]++;
  tok_indptr[0] = 0; indptr[0] = 0;
  for (int e = 0; e < NE; ++e) {
    tiles[e] = (counts[e] + TPT - 1) / TPT;
    tok_indptr[e + 1] = tok_indptr[e] + counts[e];
    indptr[e + 1] = indptr[e] + tiles[e];
  }
  int fill[NE];
  for (int e = 0; e < NE; ++e) fill[e] = 0;
  for (int i = 0; i < B * K; ++i) { const int e = topk[i]; tok_slot[i] = tok_indptr[e] + fill[e]++; }
  for (int e = 0; e < NE; ++e) for (int lt = 0; lt < tiles[e]; ++lt) tile_expert[indptr[e] + lt] = e;
  // E_expert / E_down counters are written by the generated epilogue from counts / tiles
}

// ---------------------------------------------------------------- gather (b): h[b] -> grouped rows
extern "C" __device__ void etx_tile_gather(const etx_ctx* c) {
  const int b = c->coord[0];
  const float* hh = (const float*)c->args[0] + (size_t)b * D;
  const int32_t* slot = (const int32_t*)c->args[1] + b * K;
  float* grouped = (float*)c->args[2];
  for (int k = 0; k < K; ++k)
    for (int i = threadIdx.x; i < D; i += blockDim.x) grouped[(size_t)slot[k] * D + i] = hh[i];
}

// ---------------------------------------------------------------- group GEMM up + SiLU (tile t)
static __device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

extern "C" __device__ void etx_tile_group_gemm_up(const etx_ctx* c) {
  const int t = c->coord[0];
  const float* grouped = (const float*)c->args[0];
  const etx_bf16* W_up = (const etx_bf16*)c->args[1];
  float* act = (float*)c->args[2];
  const int32_t* tile_expert = (const int32_t*)c->args[3];
  const int32_t* indptr = (const int32_t*)c->args[4];
  const int32_t* tok_indptr = (const int32_t*)c->args[5];
  const int e = tile_expert[t];
  const int lt = t - indptr[e];
  const int r0 = tok_indptr[e] + lt * TPT;
  const int r1 = min(tok_indptr[e + 1], r0 + TPT);
  const int nr = r1 - r0;
  float* rows = (float*)c->lds;                      // nr x D
  for (int r = 0; r < nr; ++r)
    for (int i = threadIdx.x; i < D; i += blockDim.x) rows[r * D + i] = grouped[(size_t)(r0 + r) * D + i];
  __syncthreads();
  const int lane = etx_lane();
  for (int f = etx_wave(); f < FF; f += etx_nwaves()) {
    const etx_bf16* w = W_up + ((size_t)e * FF + f) * D;
    float wv[D / ETX_WAVE_SIZE];
    for (int i = 0; i < D / ETX_WAVE_SIZE; ++i) wv[i] = etx_b2f(w[lane + i * ETX_WAVE_SIZE]);
    for (int r = 0; r < nr; ++r) {
      float acc = 0.f;
      for (int i = 0; i < D / ETX_WAVE_SIZE; ++i) acc += wv[i] * rows[r * D + lane + i * ETX_WAVE_SIZE];
      acc = etx_wave_sum(acc);
      if (lane == 0) act[(size_t)(r0 + r) * FF + f] = silu(acc);
    }
  }
}
extern "C" __device__ void etx_prefetch_group_gemm_up(const etx_ctx* c) {
  const int e = ((const int32_t*)c->args[3])[c->coord[0]];
  etx_touch((const etx_bf16*)c->args[1] + (size_t)e * FF * D, (size_t)FF * D * 2);
}

// ---------------------------------------------------------------- group GEMM down (tile t)
extern "C" __device__ void etx_tile_group_gemm_down(const etx_ctx* c) {
  const int t = c->coord[0];
  const float* act = (const float*)c->args[0];
  const etx_bf16* W_down = (const etx_bf16*)c->args[1];
  float* partial = (float*)c->args[2];
  const int32_t* tile_expert = (const int32_t*)c->args[3];
  const int32_t* indptr = (const int32_t*)c->args[4];
  const int32_t* tok_indptr = (const int32_t*)c->args[5];
  const int e = tile_expert[t];
  const int lt = t - indptr[e];
  const int r0 = tok_indptr[e] + lt * TPT;
  const int r1 = min(tok_indptr[e + 1], r0 + TPT);
  const int nr = r1 - r0;
  float* rows = (float*)c->lds;                      // nr x FF
  for (int r = 0; r < nr; ++r)
    for (int i = threadIdx.x; i < FF; i += blockDim.x) rows[r * FF + i] = act[(size_t)(r0 + r) * FF + i];
  __syncthreads();
  const int lane = etx_lane();
  for (int o = etx_wave(); o < D; o += etx_nwaves()) {
    const etx_bf16* w = W_down + ((size_t)e * D + o) * FF;
    float wv[FF / ETX_WAVE_SIZE];
    for (int i = 0; i < FF / ETX_WAVE_SIZE; ++i) wv[i] = etx_b2f(w[lane + i * ETX_WAVE_SIZE]);
    for (int r = 0; r < nr; ++r) {
      float acc = 0.f;
      for (int i = 0; i < FF / ETX_WAVE_SIZE; ++i) acc += wv[i] * rows[r * FF + lane + i * ETX_WAVE_SIZE];
      acc = etx_wave_sum(acc);
      if (lane == 0) partial[(size_t)(r0 + r) * D + o] = acc;
    }
  }
}
extern "C" __device__ void etx_prefetch_group_gemm_down(const etx_ctx* c) {
  const int e = ((const int32_t*)c->args[3])[c->coord[0]];
  etx_touch((const etx_bf16*)c->args[1] + (size_t)e * D * FF, (size_t)D * FF * 2);
}

// ---------------------------------------------------------------- combine (b): residual + weighted experts
extern "C" __device__ void etx_tile_combine(const etx_ctx* c) {
  const int b = c->coord[0];
  const float* hh = (const float*)c->args[0] + (size_t)b * D;
  const float* partial = (const float*)c->args[1];
  const float* w = (const float*)c->args[2] + b * K;
  const int32_t* slot = (const int32_t*)c->args[3] + b * K;
  float* y = (float*)c->args[4] + (size_t)b * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float v = hh[i];
    for (int k = 0; k < K; ++k) v += w[k] * partial[(size_t)slot[k] * D + i];
    y[i] = v;
  }
}
