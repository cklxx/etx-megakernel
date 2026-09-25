// Shared by examples/llm/tiles.hip (device) and examples/llm/host.hip (host): the model's dimensions,
// the task partition and every pointer the tile bodies read. The host fills one LlmParams per step
// and copies it into __constant__ memory, so the bodies read it with scalar loads.
#pragma once
#include <stdint.h>
#include <hip/hip_bf16.h>

#define LLM_MAX_LAYERS 96

struct LlmLayer {
  const __hip_bfloat16 *ln1, *ln2;          // input / post-attention RMSNorm weights [H]
  const __hip_bfloat16 *wqkv, *bqkv;        // [NQKV][H], [NQKV] (bias may be null)
  const __hip_bfloat16 *qn, *kn;            // per-head q / k RMSNorm [HD] (Qwen3; may be null)
  const __hip_bfloat16 *wo;                 // [H][NH*HD]
  const __hip_bfloat16 *wg, *wu, *wd;       // dense MLP: [INTER][H], [INTER][H], [H][INTER]
  const __hip_bfloat16 *wgu;                // dense MLP, layout 2: gate/up rows interleaved [2*INTER][H] (one stream per task)
  const __hip_bfloat16 *wr;                 // MoE router [E][H]
  const __hip_bfloat16 *eg, *eu, *ed;       // MoE experts: [E][MI][H], [E][MI][H], [E][H][MI]
  const __hip_bfloat16 *egu;                // MoE experts, layout 2: gate/up rows interleaved [E][2*MI][H]
  __hip_bfloat16 *kc, *vc;                  // KV cache [CTX][NKV][HD]
  // sliced layout (examples/llm/model_sliced.py), one per XCD; null when not packed
  const __hip_bfloat16 *wqkv_x[8], *bqkv_x[8];   // [nrows_x][H]: q rows of the XCD's heads, then [k rows; v rows] per needed KV head
  const __hip_bfloat16 *wo_x[8];                 // [H][nqh_x * HD]: o_proj columns of the XCD's heads
  const __hip_bfloat16 *wd_x[8];                 // dense: [H][ninter_x]: down columns of the XCD's intermediate slice
};

struct LlmParams {
  // dimensions
  int32_t H, NH, NKV, HD, G, INTER, L, V, E, TOPK, MI, CTX, CHUNK, NC;
  int32_t qk_norm, has_bias, moe, norm_topk, tie;
  float eps;
  // rows per task (examples/llm/model.py partition())
  int32_t rpt_qkv, rpt_o, rpt_gu, rpt_dn, rpt_r, rpt_egu, rpt_edn, rpt_lm;
  // weights
  const __hip_bfloat16 *embed, *lm_head, *final_norm;
  LlmLayer layer[LLM_MAX_LAYERS];
  const float *rope_cos, *rope_sin;         // [CTX][HD/2], bf16-rounded like HF's cos/sin
  // activations (fp32 storage of bf16-rounded values, matching HF's bf16 forward)
  float *x;                                 // residual stream [H]
  float *qkv;                               // [NQKV]
  float *q;                                 // rotated q [NH*HD]
  float *attn_part;                         // [NH][NC][HD+2]: unnormalised o, running max, sum
  float *o;                                 // [NH*HD]
  float *h;                                 // dense MLP activation [INTER]
  float *rlogits;                           // MoE router logits [E]
  float *hexp;                              // MoE expert activations [TOPK][MI]
  float *lm_val; int32_t *lm_idx;           // per lm_head task: best logit and its row
  int32_t *token_in, *token_out;            // [steps + 1], [steps]
  uint64_t *token_time;                     // [steps][2]: embed start, argmax end (s_memrealtime, 100 MHz)
  // sliced layout tables (model_sliced.slices): per XCD
  int32_t X, W;                             // XCDs, workers per XCD
  int32_t qh0[8], nqh[8], kv0[8], nkv[8], i0[8], ni[8], h0[8], nh[8];
  float *xa;                                // residual after attention [H] (xb is `x`)
  float *xin_x, *xa_x;                      // [8][H] XCD-local copies of the folded residual (written by the XCD's fold task)
  float *xs1_x, *xs2_x;                     // [8][H] XCD-local normed inputs (ln1 for qkv, ln2 for the MLP / router), by the fold tasks
  float *xsf;                               // [H] final-normed residual for lm_head (by lmfold)
  int32_t *rids_x; float *rw_x;             // [8][16] the token's top-k expert ids / weights, by the XCD's last router task
  float *xfin;                              // [H] folded final residual for lm_head
  float *o_part;                            // [8][H] per-XCD o_proj partial sums
  float *d_part;                            // [max(8, TOPK)][H] per-XCD (dense) or per-slot (MoE) down partial sums
  float *rlog_x;                            // [8][E] router logits, one copy per XCD
  int32_t *eid;                             // [TOPK] expert id per slot (written by the slot's XCD)
  int32_t *ctr_qkv, *ctr_attn;              // [L][8] last-arriver counters per layer and XCD (monotonic; step_seq scales the target)
  int32_t step_seq;                         // steps launched so far in this process (never reset)
  int32_t unfused;                          // 1: one launch per grid, tasks land on any XCD -> last-arriver uses device-scope fences
  float *layer_dump;                        // [L+1][H]: residual entering each layer and the final one, when dump_step == tok
  int32_t dump_step;
  // per step
  int32_t tok, pos, write_next;
};
