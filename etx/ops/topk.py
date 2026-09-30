"""The MoE router's softmax + top-k as an ETX op (one export, dsl_topk), replacing vLLM's topkGating for one token:
softmax over E logits (expf(v - max) * (1 / sum), as topk_softmax_kernels.cu), k rounds of argmax with the lower
expert index winning ties, weights renormalised over the selected k when asked; outputs as vLLM's: topk_weights
(float), topk_ids (int), token_expert_indices (k index for one token). One wave does it (E <= 2 x 64 lanes).
"""
from __future__ import annotations

ARG_BYTES, GRID_OFF = 64, 48
PARAMS = [("logits", "ptr addrspace(1)", 0, 8), ("wts", "ptr addrspace(1)", 8, 8), ("ids", "ptr addrspace(1)", 16, 8),
          ("tei", "ptr addrspace(1)", 24, 8), ("E", "i32", 32, 4), ("K", "i32", 36, 4), ("renorm", "i32", 40, 4)]


def import_entries(threads: int = 1024) -> dict:
    ps = [{"ty": ty, "name": f"%{n}", "size": sz, "align": min(sz, 8), "offset": off} for n, (_, ty, off, sz) in enumerate(PARAMS)]
    return {"dsl_topk": {"dsl": "topk", "export": "etx_tile_dsl_topk", "kernel": "dsl_topk", "block": [threads, 1, 1], "threads": threads,
                         "slots": 1, "slot_threads": threads, "params": ps, "grid_offset": GRID_OFF, "arg_bytes": ARG_BYTES,
                         "lds_bytes_per_slot": 0, "lds_bytes": 0, "barriers": 0, "slot_barrier": False, "spec": "ETX DSL softmax top-k"}}


TILE = r'''
// ---- dsl_topk: softmax + top-k of one token's router logits (wave 0)
static __device__ __attribute__((always_inline)) void dsl_run_dsl_topk(const unsigned char* a) {
  if (etx_wave() != 0) return;
  const __hip_bfloat16* logits = *(const __hip_bfloat16* const*)(a + 0);
  float* wts = *(float* const*)(a + 8); int* ids = *(int* const*)(a + 16); int* tei = *(int* const*)(a + 24);
  const int E = *(const int*)(a + 32), K = *(const int*)(a + 36), renorm = *(const int*)(a + 40);
  const int lane = etx_lane();
  float p[2]; bool taken[2];
  float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < 2; ++k) { const int e = lane + k * ETX_WAVE_SIZE; p[k] = e < E ? dsl_a_ldbf(logits + e) : -INFINITY; taken[k] = e >= E; m = fmaxf(m, p[k]); }
  m = etx_wave_max(m);
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
    if (lane == 0) { dsl_a_sti(ids + t, bi); dsl_a_sti(tei + t, t); if (t < 16) lw[t] = bv; }
    sel += bv;
  }
  if (lane == 0) for (int t = 0; t < K && t < 16; ++t) dsl_a_stf(wts + t, renorm ? lw[t] / sel : lw[t]);
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
