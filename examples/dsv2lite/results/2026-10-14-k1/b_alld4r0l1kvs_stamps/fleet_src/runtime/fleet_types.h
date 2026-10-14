// Types shared by the kernel and the host launcher.
//
// These were originally defined inside fleet_kernel.hip, which meant
// fleet_launch.hip had to forward-declare them — and a forward declaration
// cannot be passed by value, so the launch signature would not compile. Both
// sides now include this header, so there is exactly one definition and the
// kernel's parameter layout cannot drift from what the host packs.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>   // __hip_bfloat16 is not part of hip_runtime.h
#include <stdint.h>

namespace fleet {

// Compile-time bounds for the LDS buffers the task bodies use. The host
// launcher refuses to run a model whose dims exceed them (fleet_launch.hip),
// so the kernel never has to check.
constexpr int kMaxHidden = 2048;   // x / x_norm / o staged in LDS
constexpr int kExpertKChunk = 2048; // rows of h per gate_up→down tile (taskgraph EXPERT_K_CHUNK);
                                    // >= moe_inter = one chunk: tiling measured no gain (STATUS.md)
constexpr int kMaxKvLora = 512;    // q_c, cache row, merge scratch
constexpr int kMaxQkRope = 64;     // q_pe, k_pe
constexpr int kMaxMoeInter = 1408; // expert h staged in LDS for down
constexpr int kMaxRouted = 64;     // router logits
constexpr int kMaxTopK = 8;
constexpr int kPartialStride = 2 + kMaxKvLora;   // attention partial: m, l, acc[512]
constexpr int kMaxKvChunks = 16;   // split-KV factor the merge unrolls to (taskgraph --kv-chunks)
constexpr int kQAbsSlices = 2;     // publishers per head for q_c (taskgraph QABS_SLICES)
constexpr int kNumXCDs = 8;        // per-XCD buffers below

// Static model shape, read from build/weights.manifest by the host.
struct ModelDims {
    int hidden, heads, q_head_dim, qk_nope, qk_rope, v_head;
    int kv_lora, moe_inter, dense_inter, n_routed, top_k, vocab;
    int layers, first_k_dense;
    int context;           // prefill length: token t of a launch sits at position context + t
    float rms_eps, softmax_scale, routed_scaling;
};

// Weight pointers into the flat blob produced by src/host/pack_weights.py.
// Experts are addressed as base + id * expert_stride so the task queue stays
// immutable while routing is data-dependent (docs/design.md §3).
struct Weights {
    const __hip_bfloat16* const* layer_qkv;       // [3072 + 576, hidden] fused
    const __hip_bfloat16* const* layer_kv_b;
    const __hip_bfloat16* const* layer_o_proj;
    const __hip_bfloat16* const* layer_norm_in;
    const __hip_bfloat16* const* layer_norm_post;
    const __hip_bfloat16* const* layer_kv_a_norm;
    const __hip_bfloat16* const* layer_router;
    const __hip_bfloat16* const* layer_experts;   // base of the 64-expert block
    const __hip_bfloat16* const* layer_shared;
    const __hip_bfloat16* const* layer_dense;
    const __hip_bfloat16* embed;
    const __hip_bfloat16* lm_head;
    const __hip_bfloat16* final_norm;
    int64_t expert_stride;                        // elements between experts
};

// Per-token activations, allocated once (docs/design.md §5).
//
// The residual stream is two buffers: x[0] is the layer input, x[1] the
// post-attention value. Every consumer folds the 8 partials it needs into
// its own LDS copy (o_proj partials in the router prologue, expert partials
// in the next q/kv_a prologue), and worker (0,0) publishes that copy to the
// other buffer, so no reader and writer ever share one. The one-writer fold
// by the last arriver was measured slower (STATUS.md). Anything an XCD produces for its own consumption
// (kv_a, the normed vector, the routing choice) is per XCD, so those
// handoffs are XCD-local events and identical values are never written to
// the same line from two chiplets.
struct Activations {
    float* __restrict__ x[2];          // [hidden] each, bf16-valued: layer input, post-attention
    float* __restrict__ x_norm;        // [kNumXCDs][hidden] post-attention norm
    float* __restrict__ q_absorbed;    // [heads][kQAbsSlices][kv_lora] row-split partials of
                                       // W_UK^T q_nope, published once per head instead of
                                       // every KV chunk streaming all 131 KB of W_UK
    float* __restrict__ q;             // [heads * q_head_dim]; XCD k writes its heads'
    float* __restrict__ kv_a;          // [kNumXCDs][kv_lora + qk_rope] raw, pre-norm
    float* __restrict__ attn_partial;  // [heads][chunks][2 + kv_lora] m, l, acc
    float* __restrict__ o;             // [heads * v_head]
    float* __restrict__ expert_h;      // [kNumXCDs][moe_inter], or [dense_inter]
    __hip_bfloat16* __restrict__ expert_out;  // [kNumXCDs][hidden] expert partials. The
                                       // values are bf16 already (EPI_BF16), so half the
                                       // bytes of the fold are pure waste; folded by the next q/kv_a
    float* __restrict__ oproj_partial; // [kNumXCDs][hidden] o_proj K-split partials, folded by the router
    float* __restrict__ logits;        // [vocab]
    float* __restrict__ argmax_val;    // [kNumXCDs * workers] per-worker best logit
    int32_t* __restrict__ argmax_idx;  // [kNumXCDs * workers] ...and its index
    int32_t* __restrict__ topk_ids;    // [kNumXCDs][kMaxTopK] published once per XCD by
    float* __restrict__ topk_w;        // the router's last arriver (TOPK_PUBLISH)
    float* __restrict__ router_logits; // [kNumXCDs][n_routed], fp32; the expert
                                       // tasks derive top-k from it themselves
    const float* __restrict__ cos;     // [max_pos][qk_rope] YaRN tables; the kernel
    const float* __restrict__ sin;     // indexes them by position
    float* __restrict__ layer_dump;    // [layers][hidden] each layer's output x on
                                       // the launch's first token, if dump_layers
    // In-kernel decode loop (design.md §4, launch model v2):
    int32_t* __restrict__ token_in;    // [n_tokens + 1]: input token of step t; the
                                       // argmax task writes token_in[t+1] unless
                                       // teacher-forced (then the host filled it)
    int32_t* __restrict__ token_out;   // [n_tokens]: argmax of step t
    uint64_t* __restrict__ token_time; // [n_tokens][2]: s_memrealtime at embed, argmax
};

// Absorbed-MLA cache: one compressed row per position, shared by all heads.
struct KVCache {
    __hip_bfloat16* __restrict__ data;  // [layers][max_pos][kv_lora + qk_rope]
    int max_pos;
    int row;
};

}  // namespace fleet
