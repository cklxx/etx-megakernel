"""The MoE router's softmax + top-k as an ETX op (one export, dsl_topk), replacing vLLM's topkGating for one token:
softmax over E logits (expf(v - max) * (1 / sum), as topk_softmax_kernels.cu), k rounds of argmax with the lower
expert index winning ties, weights renormalised over the selected k when asked; outputs as vLLM's: topk_weights
(float), topk_ids (int), token_expert_indices (k index for one token). One wave does it (E <= 2 x 64 lanes).

mode (the former renorm word): 0 plain softmax-topk, 1 renormalised, 2 sparsemixer (PhiMoE's routing,
vllm/model_executor/models/phimoe.py sparsemixer, inference path): per round, over the working row (previous
winners at -inf), m = max, winner = argmax (lowest index on ties); elements with ((m - s) / max(|s|, m)) > 0.02
(s = the ORIGINAL logit) are masked out of that round's softmax; the round's weight is the masked softmax at the
winner = 1 / sum(exp(work - m)) over unmasked available elements. Ids equal plain top-k; weights do not, and a
softmax-topk build would pass every structural gate with wrong weights (round 46). Cross-checked against vLLM's
python on 2000 random rows (E = 8 and 16): identical ids, weights to 1e-6 (results_2026-10-01/round47/).
"""
from __future__ import annotations

ARG_BYTES, GRID_OFF = 64, 48
# nbsv (ETX_VL_BATCH): packed nb | (sv << 8) in the word that was padding, so the layout (and the B=1 plan) is
# unchanged. nb = activation columns this task serves (requests); sv = the request stride of wts/ids/tei in
# elements (NR * TOPK: the slabs are one allocation, host.hip ETX_VL_SLICE). The logits request stride is E
# itself (n_rlog), so it needs no field. nb <= 1 keeps the old single-wave behaviour exactly.
PARAMS = [("logits", "ptr addrspace(1)", 0, 8), ("wts", "ptr addrspace(1)", 8, 8), ("ids", "ptr addrspace(1)", 16, 8),
          ("tei", "ptr addrspace(1)", 24, 8), ("E", "i32", 32, 4), ("K", "i32", 36, 4), ("mode", "i32", 40, 4),
          ("nbsv", "i32", 44, 4)]


def import_entries(threads: int = 1024) -> dict:
    ps = [{"ty": ty, "name": f"%{n}", "size": sz, "align": min(sz, 8), "offset": off} for n, (_, ty, off, sz) in enumerate(PARAMS)]
    return {"dsl_topk": {"dsl": "topk", "export": "etx_tile_dsl_topk", "kernel": "dsl_topk", "block": [threads, 1, 1], "threads": threads,
                         "slots": 1, "slot_threads": threads, "params": ps, "grid_offset": GRID_OFF, "arg_bytes": ARG_BYTES,
                         "lds_bytes_per_slot": 0, "lds_bytes": 0, "barriers": 0, "slot_barrier": False, "spec": "ETX DSL softmax top-k"}}


TILE = r'''
// ---- dsl_topk: softmax + top-k of one token's router logits (wave 0)
static __device__ __attribute__((always_inline)) void dsl_run_dsl_topk(const unsigned char* a) {
  // ETX_VL_BATCH: wave w < nb does column (request) w -- the columns run concurrently, so the phase still costs
  // one body. nb <= 1 (every non-batched build) is wave 0 only, as always.
  const unsigned nbsv = *(const unsigned*)(a + 44);
  const int nb = nbsv & 0xff ? (int)(nbsv & 0xff) : 1, sv = (int)(nbsv >> 8);
  const int w = etx_wave();
  if (w >= nb) return;
  const int E = *(const int*)(a + 32), K = *(const int*)(a + 36), mode = *(const int*)(a + 40);
  const __hip_bfloat16* logits = *(const __hip_bfloat16* const*)(a + 0) + (size_t)w * E;
  float* wts = *(float* const*)(a + 8) + (size_t)w * sv;
  int* ids = *(int* const*)(a + 16) + (size_t)w * sv; int* tei = *(int* const*)(a + 24) + (size_t)w * sv;
  const int lane = etx_lane();
#ifdef ETX_ABL_TOPK_SKIP     // timing-only: no load, no scan; fixed ids and weights (WRONG OUTPUT)
  if (lane == 0) for (int t = 0; t < K; ++t) { dsl_a_sti(ids + t, t); dsl_a_sti(tei + t, t); dsl_a_stf(wts + t, 1.f / (float)K); }
  return;
#endif
  float p[2]; bool taken[2];
  float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < 2; ++k) { const int e = lane + k * ETX_WAVE_SIZE; p[k] = e < E ?
#ifdef ETX_ABL_TOPK_NOLOAD   // timing-only: synthetic logits, the two coherent loads removed (WRONG OUTPUT)
    (float)((e * 37) & 127) * 0.01f
#else
    dsl_a_ldbf(logits + e)
#endif
    : -INFINITY; taken[k] = e >= E; m = fmaxf(m, p[k]); }
#ifndef ETX_TOPK_KEY32       // the key path takes m from its round-0 winner; sparsemixer never reads this m
  m = etx_wave_max(m);
#endif
  if (mode == 2) {                                    // sparsemixer: p[] holds the RAW logits throughout
    float raw[2]; raw[0] = p[0]; raw[1] = p[1];
    for (int t = 0; t < K; ++t) {
      // the round's max and winner over the working row (taken -> excluded), lowest index on ties
      float mt = -INFINITY;
#pragma unroll
      for (int k = 0; k < 2; ++k) if (!taken[k]) mt = fmaxf(mt, raw[k]);
      mt = etx_wave_max(mt);
      float bv = -INFINITY; int bi = 0x7fffffff;
#pragma unroll
      for (int k = 0; k < 2; ++k) { const int e = lane + k * ETX_WAVE_SIZE; if (!taken[k] && raw[k] > bv) { bv = raw[k]; bi = e; } }
      for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor(bv, off, ETX_WAVE_SIZE); const int oi = __shfl_xor(bi, off, ETX_WAVE_SIZE);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (bi < 0 || bi >= E) bi = 0;      // all-NaN logits: every comparison is false and bi keeps its init;
                                          // never let a sentinel become an expert index (a wild weight pointer)
      // masked softmax of the working row: exclude taken and jitter-masked elements (mask from the ORIGINAL logits)
      float c = 0.f;
#pragma unroll
      for (int k = 0; k < 2; ++k) if (!taken[k] && !(((mt - raw[k]) / fmaxf(fabsf(raw[k]), mt)) > 0.02f)) c += expf(raw[k] - mt);
      const float S = etx_wave_sum(c);
#pragma unroll
      for (int k = 0; k < 2; ++k) if (lane + k * ETX_WAVE_SIZE == bi) taken[k] = true;
      if (lane == 0) {
        dsl_a_sti(ids + t, bi); dsl_a_sti(tei + t, t);
        dsl_a_stf(wts + t, 1.f / S);                  // = exp(mt - mt) / S, the masked softmax at the winner
      }
    }
    return;
  }
#ifdef ETX_TOPK_KEY32        // select on one 32-bit key per expert, DPP wave max (etx/runtime/include/etx/tile_util.h)
#ifdef ETX_ABL_TOPK_NOSTORE
#error "ETX_ABL_TOPK_NOSTORE has no effect with ETX_TOPK_KEY32: the key path stores its outputs itself"
#endif
  etx_topk_key32(p, E, K, mode == 1, lane, [&](int t, int bi, float v) {
    dsl_a_sti(ids + t, bi);
#ifndef ETX_ABL_TOPK_NOTEI
    dsl_a_sti(tei + t, t);                            // kept: the same outputs as the old path
#endif
    dsl_a_stf(wts + t, v); });
  return;
#endif
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 2; ++k) { p[k] = taken[k] ? 0.f : expf(p[k] - m); s += p[k]; }
  s = etx_wave_sum(s);
  const float inv = 1.f / s;
#pragma unroll
  for (int k = 0; k < 2; ++k) p[k] = taken[k] ? 0.f : p[k] * inv;
  float sel = 0.f, lw[16];                       // K <= 16 (qwen: 8); selected weights stay in registers until the one coherent store
  for (int t = 0; t < K; ++t) {
    float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
    for (int k = 0; k < 2; ++k) { const int e = lane + k * ETX_WAVE_SIZE; if (!taken[k] && p[k] > bv) { bv = p[k]; bi = e; } }
    for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off, ETX_WAVE_SIZE); const int oi = __shfl_xor(bi, off, ETX_WAVE_SIZE);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) if (lane + k * ETX_WAVE_SIZE == bi) taken[k] = true;
    // `tei` (vLLM's token_expert_indices) is written for signature parity; nothing in the megakernel reads it
    // (host.hip:241 voids it, the MoE tiles take only ids/weights). ETX_ABL_TOPK_NOTEI sizes those 8 stores.
    if (lane == 0) {
#ifndef ETX_ABL_TOPK_NOSTORE
      dsl_a_sti(ids + t, bi);
#ifndef ETX_ABL_TOPK_NOTEI
      dsl_a_sti(tei + t, t);
#endif
#else
      asm volatile("" :: "v"(bi) : "memory");
#endif
      if (t < 16) lw[t] = bv; }
    sel += bv;
  }
#ifdef ETX_ABL_TOPK_NOSTORE  // timing-only: the scan runs, the 24 coherent stores do not (WRONG OUTPUT)
  if (lane == 0) for (int t = 0; t < K && t < 16; ++t) { const float v = mode == 1 ? lw[t] / sel : lw[t]; asm volatile("" :: "v"(v)); }
#else
  if (lane == 0) for (int t = 0; t < K && t < 16; ++t) dsl_a_stf(wts + t, mode == 1 ? lw[t] / sel : lw[t]);
#endif
}
extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_topk(const etx_ctx* c) {
  dsl_run_dsl_topk(etx_imp_argblock(c->args, c->cst[0]));
}
// XCD replica r: this XCD's own copy of the outputs (cst = {block 0, stride, -1})
extern "C" __device__ __attribute__((always_inline)) void etx_tile_rep_dsl_topk(const etx_ctx* c) {
  dsl_run_dsl_topk(etx_imp_argblock(c->args, c->cst[0] + c->coord[0] * c->cst[1]));
}
extern "C" __device__ __attribute__((always_inline)) void etx_pre_dsl_topk(const etx_ctx* c) { etx_imp_touch(etx_imp_argblock(c->args, c->cst[0]), 64); }
extern "C" __device__ __attribute__((always_inline)) void etx_pre_x_dsl_topk(const etx_ctx* c) { etx_imp_touch(etx_imp_argblock(c->args, c->cst[0] + c->coord[0] * c->cst[1]), 64); }
'''


def emit_tiles() -> str:
    from etx.ops import coh
    return coh.HELPERS + TILE
