// D5: this source replaces fleet_shim.hip ONLY in the ETX_DSV2_D5=1 graph.
// The default TU does not include it. No edits to the pinned fleet export.
#include "examples/dsv2lite/fleet_shim.hip"

// One 16 B trip per lane of waves 1..3 (192 * 16 = 3072 B/task), consumed
// here before the event wait. Wave 0 can poll while the other waves warm L2.
// No LDS, activation reads, stores, barriers, or registers retained across
// the wait. Unlike fleet::prefetch_issue, there are not eight live vectors.
static __device__ __forceinline__ void fleet_d5_warm(const fleet::PrefetchPlan& p) {
  if (threadIdx.x < 64 || p.rows <= 0) return;
  const int64_t trip = threadIdx.x - 64;
  const int64_t per_row = p.row_bytes / 16;
  if (per_row <= 0) return;
  const int64_t row = trip / per_row, col = trip % per_row;
  if (row >= p.rows) return;
  const fleet::fleet_u32x4 v = *reinterpret_cast<const fleet::fleet_u32x4*>(p.base + row * p.stride + col * 16);
  // All four components are needed: prevents narrowing/elision of the 16 B
  // load. The sink consumes them inside this hook, before the polling wait.
  asm volatile("" :: "v"(v.x), "v"(v.y), "v"(v.z), "v"(v.w) : "memory");
}

static __device__ __forceinline__ void fleet_d5_plan(const etx_ctx* c, uint16_t kind,
                                                   int head = -1, int chunk = -1, int splits = 1,
                                                   int slot = 0, int worker = -1) {
  if (threadIdx.x < 64) return;
  const FleetParams* fp = FP(c);
  fleet::TaskDescriptor t{};
  t.kind = kind; t.layer = LAYER(c); t.head = head; t.kv_chunk = chunk;
  t.n_split = splits; t.expert_slot = slot;
  // The pinned planner takes ONLY immutable dimensions/weights and static
  // task coordinates. It cannot see Activations, KVCache, or routing output.
  fleet_d5_warm(fleet::prefetch_plan(t, fp->d, fp->w, c->domain, worker < 0 ? c->coord[1] : worker));
}

extern "C" __device__ void fleet_d5_qkv(const etx_ctx* c) {
  fleet_d5_plan(c, fleet::TASK_QKV_FUSED);
}
extern "C" __device__ void fleet_d5_merge(const etx_ctx* c) {
  fleet_d5_plan(c, fleet::TASK_MERGE_UV, c->coord[0], c->coord[1], 16, 0, 0);
}
extern "C" __device__ void fleet_d5_oproj(const etx_ctx* c) {
  fleet_d5_plan(c, fleet::TASK_O_PROJ);
}
extern "C" __device__ void fleet_d5_router(const etx_ctx* c) {
  // The planner returns zero rows for layer 0's dense norm: no load there.
  fleet_d5_plan(c, fleet::TASK_NORM_ROUTER);
}
extern "C" __device__ void fleet_d5_shared_gate_up(const etx_ctx* c) {
  const int topk = FP(c)->d.top_k;
  if (c->coord[0] < topk) return;  // routing-dependent expert weights are forbidden before the wait
  fleet_d5_plan(c, fleet::TASK_EXPERT_GATE_UP, -1, -1, 1, -(c->coord[0] - topk + 1));
}
extern "C" __device__ void fleet_d5_shared_down(const etx_ctx* c) {
  const int topk = FP(c)->d.top_k;
  if (c->coord[0] < topk) return;
  fleet_d5_plan(c, fleet::TASK_EXPERT_DOWN, -1, -1, 1, -(c->coord[0] - topk + 1));
}
extern "C" __device__ void fleet_d5_lm_head(const etx_ctx* c) {
  fleet_d5_plan(c, fleet::TASK_LM_HEAD);
}
