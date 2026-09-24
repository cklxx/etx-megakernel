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
  const __hip_bfloat16 *wr;                 // MoE router [E][H]
  const __hip_bfloat16 *eg, *eu, *ed;       // MoE experts: [E][MI][H], [E][MI][H], [E][H][MI]
  __hip_bfloat16 *kc, *vc;                  // KV cache [CTX][NKV][HD]
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
  // per step
  int32_t tok, pos, write_next;
};
