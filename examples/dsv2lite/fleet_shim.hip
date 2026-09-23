// fleet-mi300x task bodies as ETX tiles (link mode), design §15.1 milestone M2.
//
// Build with -I <fleet-mi300x>/src -I <etx>/etx/runtime/include. This TU includes
// fleet's kernel source so that fleet::run_task and every task body compile
// unchanged; the ETX-generated megakernel replaces fleet's worker loop,
// scheduler and events. Each wrapper rebuilds the fleet TaskDescriptor the body
// expects from the ETX coordinate and the grid's own constants, and hands the
// body the same ModelDims / Weights / Activations / KVCache the fleet host set
// up (one FleetParams struct, argument 0 of every grid).
//
// In-body synchronisation in fleet's default configuration:
//   FLAG_PUB_WAIT   attention waits for the published q_c inside its body   -> ETX waits on
//                   E_qabs before the body; the in-body wait reads a saturated dummy counter
//   FLAG_CHUNK_SIGNAL / FLAG_CHUNK_WAIT with one K chunk: gate_up bumps a dummy counter per
//                   wave, down performs no in-body wait (chunk 0 is the descriptor's event)
//   FLAG_ROUTING_CACHED: down reuses the top-k its worker's gate_up left in LDS; only valid
//                   when ETX placed both on the same worker back to back (fleet_expert_down_cached)
#include "etx/abi.h"
#include "kernels/fleet_kernel.hip"

struct FleetParams {
  fleet::RuntimeState rt;
  fleet::ModelDims d;
  fleet::Weights w;
  fleet::Activations act;
  fleet::KVCache kv;
  int32_t tok;          // index of this token in the run (token_in / token_out / token_time slot)
  uint32_t epoch;
  int32_t pos;          // absolute position of this token in the cache
  int32_t dump;         // 1: copy every layer's x to act.layer_dump (first token)
};

static __device__ __forceinline__ void fleet_call(const etx_ctx* c, uint16_t kind, int16_t flags,
                                                  int head, int kv_chunk, int n_split, int expert_slot,
                                                  int worker) {
  const FleetParams* fp = (const FleetParams*)c->args[0];
  const int32_t L = *(const int32_t*)c->args[1];
  fleet::TaskDescriptor t{};
  t.kind = kind; t.layer = (int16_t)L; t.xcd = (int16_t)c->domain; t.worker = (int16_t)worker;
  t.wait_event = -1; t.signal_event = -1; t.signal_scope = 0; t.n_split = (int16_t)n_split;
  t.head = (int16_t)head; t.kv_chunk = (int16_t)kv_chunk; t.expert_slot = (int16_t)expert_slot;
  t.wait_scope = 0; t.wait_count = 0; t.local_event = 0; t.flags = flags; t.signal_xcd_count = 0;
  t.index = 0;
  fleet::TokenCtx tc;
  tc.tok = fp->tok; tc.epoch = fp->epoch; tc.pos = fp->pos; tc.seq_len = fp->pos + 1;
  tc.cos = fp->act.cos + (int64_t)fp->pos * fp->d.qk_rope;
  tc.sin = fp->act.sin + (int64_t)fp->pos * fp->d.qk_rope;
  tc.dump = fp->dump != 0;
  tc.trace = false;
  // run_task takes Activations / KVCache by non-const reference but only writes through their pointers
  fleet::Activations& act = *const_cast<fleet::Activations*>(&fp->act);
  fleet::KVCache& kv = *const_cast<fleet::KVCache*>(&fp->kv);
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  fleet::run_task(t, fp->rt, fp->d, fp->w, act, kv, s, tc, (int)c->domain, worker);
}

#define CHIPLET_WORKER (c->coord[1])
#define SLOT_OF_XCD(fp_topk) ((int)c->coord[0] < (fp_topk) ? (int)c->coord[0] : -((int)c->coord[0] - (fp_topk) + 1))

extern "C" __device__ void fleet_embed(const etx_ctx* c) { fleet_call(c, fleet::TASK_EMBED, 0, -1, -1, 1, -1, 0); }
extern "C" __device__ void fleet_qkv_fused(const etx_ctx* c) { fleet_call(c, fleet::TASK_QKV_FUSED, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_qkv_fused_fold(const etx_ctx* c) { fleet_call(c, fleet::TASK_QKV_FUSED, fleet::FLAG_FOLD_PARTIALS, -1, -1, 1, -1, CHIPLET_WORKER); }
// q_c publication: fleet's attention waits for the two q_abs slices INSIDE its body (after kv_post),
// through an XCD-local up-counter (rt.xcd_counters[xcd][local_event] >= epoch * slices). The qabs
// wrapper bumps that counter itself after its body, so the overlap fleet measured is preserved and
// ETX carries no E_qabs edge for attention (an in-body dependency, outside the IR; see graph.py).
#define QABS_EVENT(L, h) (256 + (L) * 16 + (h))
extern "C" __device__ void fleet_q_absorb(const etx_ctx* c) {
  fleet_call(c, fleet::TASK_Q_ABSORB, 0, c->coord[0], c->coord[1], fleet::kQAbsSlices, -1, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    const FleetParams* fp = (const FleetParams*)c->args[0];
    const int32_t L = *(const int32_t*)c->args[1];
    fleet::drain_stores();
    __hip_atomic_fetch_add(&fp->rt.xcd_counters[(size_t)c->domain * fleet::kMaxEvents + QABS_EVENT(L, c->coord[0])], 1u,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}
extern "C" __device__ void fleet_attention(const etx_ctx* c) {
  const FleetParams* fp = (const FleetParams*)c->args[0];
  const int32_t L = *(const int32_t*)c->args[1];
  fleet::TaskDescriptor t{};
  t.kind = fleet::TASK_ATTENTION; t.layer = (int16_t)L; t.xcd = (int16_t)c->domain; t.worker = 0;
  t.wait_event = -1; t.signal_event = -1; t.n_split = fleet::kMaxKvChunks; t.head = (int16_t)c->coord[0];
  t.kv_chunk = (int16_t)c->coord[1]; t.expert_slot = -1; t.wait_count = fleet::kQAbsSlices;
  t.local_event = (int16_t)QABS_EVENT(L, c->coord[0]); t.flags = fleet::FLAG_PUB_WAIT;
  fleet::TokenCtx tc;
  tc.tok = fp->tok; tc.epoch = fp->epoch; tc.pos = fp->pos; tc.seq_len = fp->pos + 1;
  tc.cos = fp->act.cos + (int64_t)fp->pos * fp->d.qk_rope; tc.sin = fp->act.sin + (int64_t)fp->pos * fp->d.qk_rope;
  tc.dump = fp->dump != 0; tc.trace = false;
  fleet::Activations& act = *const_cast<fleet::Activations*>(&fp->act);
  fleet::KVCache& kv = *const_cast<fleet::KVCache*>(&fp->kv);
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  fleet::run_task(t, fp->rt, fp->d, fp->w, act, kv, s, tc, (int)c->domain, 0);
}
extern "C" __device__ void fleet_merge_uv(const etx_ctx* c) { fleet_call(c, fleet::TASK_MERGE_UV, 0, c->coord[0], c->coord[1], fleet::kMaxKvChunks, -1, 0); }
extern "C" __device__ void fleet_o_proj(const etx_ctx* c) { fleet_call(c, fleet::TASK_O_PROJ, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_norm_router(const etx_ctx* c) { fleet_call(c, fleet::TASK_NORM_ROUTER, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_expert_gate_up(const etx_ctx* c) {
  const FleetParams* fp = (const FleetParams*)c->args[0];
  fleet_call(c, fleet::TASK_EXPERT_GATE_UP, fleet::FLAG_CHUNK_SIGNAL, -1, 1, 1, SLOT_OF_XCD(fp->d.top_k), CHIPLET_WORKER);
}
extern "C" __device__ void fleet_expert_down(const etx_ctx* c) {
  const FleetParams* fp = (const FleetParams*)c->args[0];
  fleet_call(c, fleet::TASK_EXPERT_DOWN, fleet::FLAG_CHUNK_WAIT, -1, 1, 1, SLOT_OF_XCD(fp->d.top_k), CHIPLET_WORKER);
}
extern "C" __device__ void fleet_expert_down_cached(const etx_ctx* c) {
  const FleetParams* fp = (const FleetParams*)c->args[0];
  fleet_call(c, fleet::TASK_EXPERT_DOWN, fleet::FLAG_CHUNK_WAIT | fleet::FLAG_ROUTING_CACHED, -1, 1, 1, SLOT_OF_XCD(fp->d.top_k), CHIPLET_WORKER);
}
extern "C" __device__ void fleet_dense_gate_up(const etx_ctx* c) { fleet_call(c, fleet::TASK_DENSE_GATE_UP, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_dense_down(const etx_ctx* c) { fleet_call(c, fleet::TASK_DENSE_DOWN, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_lm_head(const etx_ctx* c) { fleet_call(c, fleet::TASK_LM_HEAD, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_lm_head_fold(const etx_ctx* c) { fleet_call(c, fleet::TASK_LM_HEAD, fleet::FLAG_FOLD_PARTIALS, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_argmax(const etx_ctx* c) { fleet_call(c, fleet::TASK_ARGMAX, 0, -1, -1, 1, -1, 0); }

// exported for the host: the LDS the bodies need, so the graph's resource declaration can be checked
extern "C" size_t fleet_scratch_bytes() { return sizeof(fleet::Scratch); }
