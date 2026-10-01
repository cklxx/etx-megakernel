"""The attention preparation of one kv head as an ETX op (export dsl_attnprep): the q/k RMS norm + neox rope of vLLM's
fusedQKNormRopeKernel (float throughout, one bf16 rounding; x *= rms_rcp * w; rope pairs (d, d + 64) with cos/sin[d % 64])
for the head's G q heads and its k head, q and k written back in place into qkv, then the k / v rows of the head written
into the paged KV cache exactly as vllm::reshape_and_cache_kernel lays them out (key [blocks][NKV][HD/x][BS][x], value
[blocks][NKV][HD][BS], slot = slot_mapping[0]). One task per kv head (grid (NKV,)), placed on XCD h by model.py
(ETX_VL_DSL_ATTN=head) so that the head's attention partitions and its reduce wait on XCD-local events.
One wave per head (G + 1 waves), 2 elements per lane (HD = 128).
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
  // Round 22, on where prep's ~5.7 us goes: the coherent accessors are 2-byte atomic loads/stores issued PER LANE,
  // so a head row (128 bf16 = 256 B = 2 cache lines) is fetched as 128 separate 2-byte uncached requests, and a
  // task issues roughly 2300 of them across its G+2 waves. That is a request-rate cost, not the dependent chain
  // the body was first read as: hoisting the chain's late loads changes nothing (above). Vectorising the coherent
  // path (16 B per request, 8 elements per lane) is the lever, but it moves the lane -> element mapping, which the
  // rope partner shuffle (lane ^ 32) and the 2-elements-per-lane cache writes both depend on.
#ifdef ETX_DSL_PREP_EARLY
  // positions[0] and slot_mapping[0] depend only on the argument block, but in program order they sit after the
  // wave reduction and after the row stores, so each adds a full round trip (~690 ns loaded) to a chain measured
  // at 8.5 hops. Issue them with the row load instead. Round 4 hoisted `positions` alone and lost 0.044 ms, read
  // as register pressure at spill 13-15; this pairs the hoist with the wave-uniform load, so each scalar costs one
  // request per wave rather than one per lane (a coherent load bypasses L1/L2, so the per-lane form really is 64
  // requests to one address).
  // MEASURED ROUND 22: neutral, and prep's body does not move -- 5.70 us control vs 5.68 with the hoist, step
  // +0.003 ms over three interleaved rounds (inside the 0.017 spread), VGPR/spill/LDS identical. The compiler was
  // already scheduling these loads as early as the dependences allow, so source order was never the constraint.
  // Left off by default. The body is NOT a chain of late-issued independent loads; see the request-rate note below.
  const long long dsl_pos_e = dsl_a_ldll_u(positions), dsl_slot_e = dsl_a_ldll_u(slot_mapping);
#define DSL_PREP_POS  dsl_pos_e
#define DSL_PREP_SLOT dsl_slot_e
#else
#define DSL_PREP_POS  dsl_a_ldll_u(positions)
#define DSL_PREP_SLOT dsl_a_ldll_u(slot_mapping)
#endif
  if (wave <= G && HD == 2 * ETX_WAVE_SIZE) {
    // head row: q head G*h + wave, or (wave == G) the k head h
    const bool isq = wave < G;
    __hip_bfloat16* row = qkv + (isq ? (size_t)(G * h + wave) * HD : (size_t)NH * HD + (size_t)h * HD);
    const __hip_bfloat16* w = isq ? qw : kw;
    // 2 elements per lane: d0 = 2 * lane, d1 = 2 * lane + 1 (vLLM: 4 per lane of 32; the sums differ in order only)
    float e0 = dsl_a_ldbf(row + 2 * lane), e1 = dsl_a_ldbf(row + 2 * lane + 1);
    float ss = e0 * e0 + e1 * e1;
    ss = etx_wave_sum(ss);
    const float rms_rcp = rsqrtf(ss / (float)HD + eps);
    e0 *= rms_rcp * __bfloat162float(w[2 * lane]);
    e1 *= rms_rcp * __bfloat162float(w[2 * lane + 1]);
    // neox rope over the whole head: partner element d ^ 64 lives on lane ^ 32; sign - for d < 64
    const long long pos = DSL_PREP_POS;   // hoisting this alone above the reduction measured +0.044 ms (round 4)
    const __hip_bfloat16* cosp = cs + pos * HD;           // [cos (HD/2) | sin (HD/2)]
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
    if (!isq) {                                           // the k head into the cache: [block][h][d / X][off][d % X]
      const long long slot = DSL_PREP_SLOT;
      const long long block = slot / BS, off = slot % BS;
      __hip_bfloat16* kdst = kc + (((block * NKV + h) * (HD / X)) * BS) * X + off * X;
      dsl_a_stbf(kdst + (2 * lane) / X * BS * X + (2 * lane) % X, o0);
      dsl_a_stbf(kdst + (2 * lane + 1) / X * BS * X + (2 * lane + 1) % X, o1);
    }
  } else if (wave == G + 1 && HD == 2 * ETX_WAVE_SIZE) {   // the v head into the cache: [block][h][d][off]
    const long long slot = DSL_PREP_SLOT;
    const long long block = slot / BS, off = slot % BS;
    const __hip_bfloat16* vrow = qkv + (size_t)(NH + NKV) * HD + (size_t)h * HD;
    __hip_bfloat16* vdst = vc + ((block * NKV + h) * HD) * BS + off;
    dsl_a_stbf(vdst + (2 * lane) * BS, __float2bfloat16(dsl_a_ldbf(vrow + 2 * lane)));
    dsl_a_stbf(vdst + (2 * lane + 1) * BS, __float2bfloat16(dsl_a_ldbf(vrow + 2 * lane + 1)));
  }
#undef DSL_PREP_POS
#undef DSL_PREP_SLOT
}
extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_attnprep(const etx_ctx* c) {
  dsl_run_dsl_attnprep(etx_imp_argblock(c->args, c->cst[0]), c->coord[0]);
}
extern "C" __device__ __attribute__((always_inline)) void etx_pre_dsl_attnprep(const etx_ctx* c) { etx_imp_touch(etx_imp_argblock(c->args, c->cst[0]), 112); }
'''


def emit_tiles() -> str:
    from etx.ops import coh
    return coh.HELPERS + TILE
