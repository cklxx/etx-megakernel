"""The attention preparation of one kv head as an ETX op (export dsl_attnprep): the q/k RMS norm + neox rope of vLLM's
fusedQKNormRopeKernel (float throughout, one bf16 rounding; x *= rms_rcp * w; rope pairs (d, d + 64) with cos/sin[d % 64])
for the head's G q heads and its k head, q and k written back in place into qkv, then the k / v rows of the head written
into the paged KV cache exactly as vllm::reshape_and_cache_kernel lays them out (key [blocks][NKV][HD/x][BS][x], value
[blocks][NKV][HD][BS], slot = slot_mapping[0]). One task per kv head (grid (NKV,)), placed on XCD h by model.py
(ETX_VL_DSL_ATTN=head) so that the head's attention partitions and its reduce wait on XCD-local events.
One wave per head (G + 1 waves); 8 elements per lane on the vectorised path, 2 on the scalar fallback.
"""
from __future__ import annotations

ARG_BYTES, GRID_OFF = 112, 96
PARAMS = [("qkv", "ptr addrspace(1)", 0, 8), ("qw", "ptr addrspace(1)", 8, 8), ("kw", "ptr addrspace(1)", 16, 8),
          ("cos_sin", "ptr addrspace(1)", 24, 8), ("positions", "ptr addrspace(1)", 32, 8), ("kc", "ptr addrspace(1)", 40, 8),
          ("vc", "ptr addrspace(1)", 48, 8), ("slot_mapping", "ptr addrspace(1)", 56, 8),
          ("NH", "i32", 64, 4), ("NKV", "i32", 68, 4), ("HD", "i32", 72, 4), ("eps", "float", 76, 4), ("BS", "i32", 80, 4), ("X", "i32", 84, 4)]


def import_entries(threads: int = 1024) -> dict:
    ps = [{"ty": ty, "name": f"%{n}", "size": sz, "align": min(sz, 8), "offset": off} for n, (_, ty, off, sz) in enumerate(PARAMS)]
    return {"dsl_attnprep": {"dsl": "attnprep", "export": "etx_tile_dsl_attnprep", "kernel": "dsl_attnprep", "block": [threads, 1, 1],
                             "threads": threads, "slots": 1, "slot_threads": threads, "params": ps, "grid_offset": GRID_OFF,
                             "arg_bytes": ARG_BYTES, "lds_bytes_per_slot": 0, "lds_bytes": 0, "barriers": 0, "slot_barrier": False,
                             "spec": "ETX DSL attention prep (q/k norm + rope + cache write of one kv head)"}}


TILE = r'''
// ---- dsl_attnprep: one kv head's q/k norm + rope (in place) and its k / v rows into the paged cache
//
// Round 22 established where this tile's ~5.7 us body goes: the coherent accessors are 2-byte atomic loads and
// stores issued PER LANE, and a coherent access bypasses L1 and L2, so a 256-byte head row is fetched as 128
// separate uncached requests. Counting the whole task at G = 8: 9 head rows x (128 loads + 128 stores) + 128 k
// cache stores + 256 for v, plus a per-lane coherent load of positions / slot_mapping in every wave -- about
// 3400 requests. It is a request-rate cost, not the dependent chain the body was first read as; hoisting the
// chain's late loads was measured neutral (round 22: +0.003 ms, body unmoved, flag since removed).
//
// Round 23 fixed exactly this disease in the GEMV's folded norm by moving to 16-byte coherent accesses (-0.105 ms),
// and its request-count table predicted correctly which phases would and would not move. This is the same fix for
// the same disease, with one extra constraint: the neox rope pairs d with d ^ 64, and the old mapping (lane L holds
// d = 2L, 2L+1) put the partner on lane L ^ 32, reached by __shfl_xor.
//
// The mapping below keeps the partner in the SAME lane, so the rope needs no shuffle at all:
//
//     lane L < 16 holds d = 4L .. 4L+3   (low half)  and  d = 4L+64 .. 4L+67   (high half)
//
// Each half is 4 contiguous bf16 = 8 B = one dsl_a_ld64, so a row costs 16 lanes x 2 = 32 requests instead of 128,
// and the partner of 4L+i is 4L+64+i, already in this lane's registers. cos/sin are indexed by d % 64, which is
// 4L+i for both halves, so one pair of cached reads serves both. The K cache layout [block][h][d/X][off][d%X] with
// X a multiple of 4 keeps each quad inside one X group and contiguous, so the cache write is one st64 per quad.
// Lanes 16..63 of a head wave are idle and contribute zero to the RMS wave sum; the phase is request-rate bound,
// not ALU bound, so idle lanes cost nothing the loads were not already paying.
//
// V is the exception: its layout [block][h][d][off] strides by BS per element, so the v stores stay scalar. Its
// load vectorises (16 lanes x 8 contiguous elements), which is where its saving comes from.
//
// New count: 9 x (32 + 32) + 32 + (32 + 128) = 768 requests, 3.5x fewer, and the two scalars are read once per
// wave instead of once per lane (round 22 measured that form alone as neutral, so it adds no win of its own --
// it is here because 64 per-lane requests for one address would otherwise dominate the 32 the row now costs).
//
// Falls back to the scalar path for any shape that does not fit (HD != 128, wave != 64, X not a multiple of 4).

static __device__ __forceinline__ __hip_bfloat16 dsl_ap_bf(unsigned long long u, int i) {
  const unsigned short s = (unsigned short)((u >> (16 * i)) & 0xffffull);
  __hip_bfloat16 h; __builtin_memcpy(&h, &s, 2); return h;
}
static __device__ __forceinline__ float dsl_ap_f(unsigned long long u, int i) { return __bfloat162float(dsl_ap_bf(u, i)); }
static __device__ __forceinline__ unsigned long long dsl_ap_pk(const float* v) {
  unsigned long long u = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __hip_bfloat16 h = __float2bfloat16(v[i]);
    unsigned short s; __builtin_memcpy(&s, &h, 2);
    u |= (unsigned long long)s << (16 * i);
  }
  return u;
}
// one coherent request per wave for a scalar every lane needs (a coherent load bypasses L1/L2, so the per-lane
// form really is 64 requests to one address)
static __device__ __forceinline__ long long dsl_ap_scalar(const long long* p) {
  long long v = 0;
  if (etx_lane() == 0) v = dsl_a_ldll(p);
  const int lo = __shfl((int)((unsigned long long)v & 0xffffffffull), 0, ETX_WAVE_SIZE);
  const int hi = __shfl((int)((unsigned long long)v >> 32), 0, ETX_WAVE_SIZE);
  return (long long)(((unsigned long long)(unsigned int)hi << 32) | (unsigned int)lo);
}

static __device__ __attribute__((always_inline)) void dsl_run_dsl_attnprep(const unsigned char* a, int blk) {
  const int nblk = *(const int*)(a + 96 + 12);
  if (blk >= nblk) return;
  __hip_bfloat16* qkv = *(__hip_bfloat16* const*)(a + 0);
  const __hip_bfloat16* qw = *(const __hip_bfloat16* const*)(a + 8);
  const __hip_bfloat16* kw = *(const __hip_bfloat16* const*)(a + 16);
  const __hip_bfloat16* cs = *(const __hip_bfloat16* const*)(a + 24);
  const long long* positions = *(const long long* const*)(a + 32);
  __hip_bfloat16* kc = *(__hip_bfloat16* const*)(a + 40);
  __hip_bfloat16* vc = *(__hip_bfloat16* const*)(a + 48);
  const long long* slot_mapping = *(const long long* const*)(a + 56);
  const int NH = *(const int*)(a + 64), NKV = *(const int*)(a + 68), HD = *(const int*)(a + 72);
  const float eps = *(const float*)(a + 76);
  const int BS = *(const int*)(a + 80), X = *(const int*)(a + 84);
  const int G = NH / NKV, h = blk;                       // this task's kv head; q heads G*h .. G*h+G-1, then its k head
  const int wave = etx_wave(), lane = etx_lane();
  const int HALF = HD / 2;
  const bool vec = (HD == 128 && ETX_WAVE_SIZE == 64 && X >= 4 && (X % 4) == 0);

  if (vec) {
    if (wave <= G) {
      const bool isq = wave < G;
      __hip_bfloat16* row = qkv + (isq ? (size_t)(G * h + wave) * HD : (size_t)NH * HD + (size_t)h * HD);
      const __hip_bfloat16* w = isq ? qw : kw;
      const bool act = lane < 16;
      const int d0 = act ? 4 * lane : 0;                  // inactive lanes read element 0: in bounds, result discarded
      unsigned long long ulo = 0, uhi = 0;
      if (act) {
        ulo = dsl_a_ld64((const unsigned long long*)(row + d0));
        uhi = dsl_a_ld64((const unsigned long long*)(row + d0 + HALF));
      }
      float lo[4], hi[4], ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        lo[i] = act ? dsl_ap_f(ulo, i) : 0.f;
        hi[i] = act ? dsl_ap_f(uhi, i) : 0.f;
        ss += lo[i] * lo[i] + hi[i] * hi[i];
      }
      ss = etx_wave_sum(ss);                              // all 64 lanes participate; 16..63 contribute zero
      const float rms_rcp = rsqrtf(ss / (float)HD + eps);
      const long long pos = dsl_ap_scalar(positions);     // uniform, and outside the divergent branch
      const __hip_bfloat16* cosp = cs + pos * HD;         // [cos (HD/2) | sin (HD/2)]
      const __hip_bfloat16* sinp = cosp + HALF;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        lo[i] *= rms_rcp * __bfloat162float(w[d0 + i]);
        hi[i] *= rms_rcp * __bfloat162float(w[d0 + HALF + i]);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {                       // partner of d0+i is d0+HALF+i, in this lane's registers
        const float c = __bfloat162float(cosp[d0 + i]), s = __bfloat162float(sinp[d0 + i]);
        const float x0 = lo[i], x1 = hi[i];
        lo[i] = x0 * c - x1 * s;                          // d < HALF: sign -
        hi[i] = x1 * c + x0 * s;                          // d >= HALF: sign +
      }
      const long long slot = isq ? 0 : dsl_ap_scalar(slot_mapping);
      if (act) {
        const unsigned long long plo = dsl_ap_pk(lo), phi = dsl_ap_pk(hi);
        dsl_a_st64((unsigned long long*)(row + d0), plo);
        dsl_a_st64((unsigned long long*)(row + d0 + HALF), phi);
        if (!isq) {                                       // the k head into the cache: [block][h][d / X][off][d % X]
          const long long block = slot / BS, off = slot % BS;
          __hip_bfloat16* kb = kc + (((block * NKV + h) * (HD / X)) * BS) * X + off * X;
          const int dh = d0 + HALF;                       // each quad sits inside one X group, so it is contiguous
          dsl_a_st64((unsigned long long*)(kb + (size_t)(d0 / X) * BS * X + (d0 % X)), plo);
          dsl_a_st64((unsigned long long*)(kb + (size_t)(dh / X) * BS * X + (dh % X)), phi);
        }
      }
    } else if (wave == G + 1) {                           // the v head into the cache: [block][h][d][off], strided
      const long long slot = dsl_ap_scalar(slot_mapping);
      if (lane < 16) {
        const long long block = slot / BS, off = slot % BS;
        const __hip_bfloat16* vrow = qkv + (size_t)(NH + NKV) * HD + (size_t)h * HD;
        __hip_bfloat16* vdst = vc + ((block * NKV + h) * HD) * BS + off;
        const int d0 = 8 * lane;
        const unsigned long long u0 = dsl_a_ld64((const unsigned long long*)(vrow + d0));
        const unsigned long long u1 = dsl_a_ld64((const unsigned long long*)(vrow + d0 + 4));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          dsl_a_stbf(vdst + (size_t)(d0 + i) * BS, dsl_ap_bf(u0, i));
          dsl_a_stbf(vdst + (size_t)(d0 + 4 + i) * BS, dsl_ap_bf(u1, i));
        }
      }
    }
    return;
  }

  // scalar fallback: 2 elements per lane, the rope partner d ^ 64 on lane ^ 32 (any shape the vector path rejects)
  if (wave <= G && HD == 2 * ETX_WAVE_SIZE) {
    const bool isq = wave < G;
    __hip_bfloat16* row = qkv + (isq ? (size_t)(G * h + wave) * HD : (size_t)NH * HD + (size_t)h * HD);
    const __hip_bfloat16* w = isq ? qw : kw;
    float e0 = dsl_a_ldbf(row + 2 * lane), e1 = dsl_a_ldbf(row + 2 * lane + 1);
    float ss = e0 * e0 + e1 * e1;
    ss = etx_wave_sum(ss);
    const float rms_rcp = rsqrtf(ss / (float)HD + eps);
    e0 *= rms_rcp * __bfloat162float(w[2 * lane]);
    e1 *= rms_rcp * __bfloat162float(w[2 * lane + 1]);
    const long long pos = dsl_a_ldll_u(positions);
    const __hip_bfloat16* cosp = cs + pos * HD;
    const __hip_bfloat16* sinp = cosp + HD / 2;
    const float p0 = __shfl_xor(e0, ETX_WAVE_SIZE / 2, ETX_WAVE_SIZE), p1 = __shfl_xor(e1, ETX_WAVE_SIZE / 2, ETX_WAVE_SIZE);
    const float sgn = lane < ETX_WAVE_SIZE / 2 ? -1.f : 1.f;
    const int d0 = (2 * lane) % (HD / 2), d1 = (2 * lane + 1) % (HD / 2);
    const float c0 = __bfloat162float(cosp[d0]), s0 = __bfloat162float(sinp[d0]);
    const float c1 = __bfloat162float(cosp[d1]), s1 = __bfloat162float(sinp[d1]);
    e0 = e0 * c0 + sgn * p0 * s0;
    e1 = e1 * c1 + sgn * p1 * s1;
    const __hip_bfloat16 o0 = __float2bfloat16(e0), o1 = __float2bfloat16(e1);
    dsl_a_stbf(row + 2 * lane, o0); dsl_a_stbf(row + 2 * lane + 1, o1);
    if (!isq) {
      const long long slot = dsl_a_ldll_u(slot_mapping);
      const long long block = slot / BS, off = slot % BS;
      __hip_bfloat16* kdst = kc + (((block * NKV + h) * (HD / X)) * BS) * X + off * X;
      dsl_a_stbf(kdst + (2 * lane) / X * BS * X + (2 * lane) % X, o0);
      dsl_a_stbf(kdst + (2 * lane + 1) / X * BS * X + (2 * lane + 1) % X, o1);
    }
  } else if (wave == G + 1 && HD == 2 * ETX_WAVE_SIZE) {
    const long long slot = dsl_a_ldll_u(slot_mapping);
    const long long block = slot / BS, off = slot % BS;
    const __hip_bfloat16* vrow = qkv + (size_t)(NH + NKV) * HD + (size_t)h * HD;
    __hip_bfloat16* vdst = vc + ((block * NKV + h) * HD) * BS + off;
    dsl_a_stbf(vdst + (2 * lane) * BS, __float2bfloat16(dsl_a_ldbf(vrow + 2 * lane)));
    dsl_a_stbf(vdst + (2 * lane + 1) * BS, __float2bfloat16(dsl_a_ldbf(vrow + 2 * lane + 1)));
  }
}
extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_attnprep(const etx_ctx* c) {
  dsl_run_dsl_attnprep(etx_imp_argblock(c->args, c->cst[0]), c->coord[0]);
}
extern "C" __device__ __attribute__((always_inline)) void etx_pre_dsl_attnprep(const etx_ctx* c) { etx_imp_touch(etx_imp_argblock(c->args, c->cst[0]), 112); }
'''


def emit_tiles() -> str:
    from etx.ops import coh
    return coh.HELPERS + TILE
