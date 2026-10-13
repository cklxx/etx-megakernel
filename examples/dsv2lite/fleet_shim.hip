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
  uint64_t* pro_trace;  // optional (host: ETX_DS_PROLOGUE): per task {body start, end of staging, end of prologue}
};
// pro_trace slot of a task: layer (-1..26) x fleet task kind (< 32) x coordinate (coord[0] * 64 + coord[1] < 1024)
#ifndef ETX_DS_PRO
#define ETX_DS_PRO 0
#endif
#ifndef ETX_DSV2_STAMPS2
#define ETX_DSV2_STAMPS2 0
#endif
#define PRO_SLOT(L, kind, c) ((((int64_t)(L) + 1) * 32 + (kind)) * 1024 + (int64_t)(c)->coord[0] * 64 + (c)->coord[1])
// compiled in only with -DETX_DS_PRO=1 (the stamps cost registers: 286 -> 292 total per lane), so timing builds are unchanged
static __device__ __forceinline__ void pro_begin(const FleetParams* fp, fleet::TokenCtx& tc, fleet::Scratch& s, uint64_t& t0) {
#if !ETX_DS_PRO && !ETX_DSV2_STAMPS2
  return;
#endif
  tc.trace = fp->pro_trace != nullptr;
  if (tc.trace && threadIdx.x == 0) { s.stamp = 0; s.stamp_stage = 0; t0 = __builtin_amdgcn_s_memrealtime(); }
#if ETX_DSV2_STAMPS2
  if (tc.trace && threadIdx.x == 0) for (int i = 0; i < 13; ++i) s.stamps2[i] = 0;
#endif
}
static __device__ __forceinline__ void pro_end(const FleetParams* fp, const fleet::Scratch& s, int64_t slot, uint64_t t0) {
#if !ETX_DS_PRO && !ETX_DSV2_STAMPS2
  return;
#endif
  if (fp->pro_trace != nullptr && threadIdx.x == 0) {
#if ETX_DSV2_STAMPS2
    uint64_t* r = fp->pro_trace + slot * 16;
    r[0] = t0; r[1] = s.stamp_stage; r[2] = s.stamp;
    const int kind = (slot / 1024) % 32;
    for (int i = 0; i < 13; ++i)
      r[3 + i] = (kind == fleet::TASK_QKV_FUSED || kind == fleet::TASK_NORM_ROUTER || kind == fleet::TASK_MERGE_UV) ? s.stamps2[i] : 0;
#else
    uint64_t* r = fp->pro_trace + slot * 3;
    r[0] = t0; r[1] = s.stamp_stage; r[2] = s.stamp;
#endif
  }
}

// FleetParams location. FLEET_CONST_PARAMS=1 (default): a __constant__ copy the host refreshes per
// token, so the bodies read weights / dims / runtime state with scalar loads the way fleet's kernel
// reads its kernel arguments. 0: through the pointer in argument 0 (global memory, vector loads).
#ifndef FLEET_CONST_PARAMS
#define FLEET_CONST_PARAMS 1
#endif
// ETX_DS_REQUESTS (graph ETX_DSV2_REQUESTS): requests decoded together, one FleetParams each; a task's request is its
// grid's second constant (graph.py consts=(L, r)), 0 in the single-request graph
#ifndef ETX_DS_REQUESTS
#define ETX_DS_REQUESTS 1
#endif
#define REQ(c) ((c)->cst[1])
#if FLEET_CONST_PARAMS
__constant__ FleetParams etx_fleet_const[ETX_DS_REQUESTS];
#if ETX_DS_REQUESTS == 1
#define FP(c) (&etx_fleet_const[0])              // the single-request build compiles exactly as before (286 registers)
#else
#define FP(c) (&etx_fleet_const[REQ(c)])
#endif
extern "C" void fleet_set_params(const void* host_fp, hipStream_t s, int r) {
  (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(etx_fleet_const), host_fp, sizeof(FleetParams), (size_t)r * sizeof(FleetParams),
                               hipMemcpyHostToDevice, s);
  (void)hipStreamSynchronize(s);
}
#else
#define FP(c) ((const FleetParams*)(c)->args[0])
extern "C" void fleet_set_params(const void*, hipStream_t, int) {}
#endif
// the layer index is a per-grid constant (graph.py consts=(L,)): an immediate in ctx.cst, not a load
#define LAYER(c) ((c)->cst[0])

static __device__ __forceinline__ void fleet_call(const etx_ctx* c, uint16_t kind, int16_t flags,
                                                  int head, int kv_chunk, int n_split, int expert_slot,
                                                  int worker) {
  const FleetParams* fp = FP(c);
  const int32_t L = LAYER(c);
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
  uint64_t t0 = 0;
  pro_begin(fp, tc, s, t0);
  fleet::run_task(t, fp->rt, fp->d, fp->w, act, kv, s, tc, (int)c->domain, worker);
  pro_end(fp, s, PRO_SLOT(L, kind, c), t0);
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
    const FleetParams* fp = FP(c);
    const int32_t L = LAYER(c);
    fleet::drain_stores();
#ifdef FLEET_COH_LOADS
    // D2 (lever coherent_loads): attention reads q_absorbed with loads past this XCD's L2, and this in-body hand-off has
    // no ETX consumer edge to carry the write-back release, so write the L2 back before publishing
    fleet::fence_release();
#endif
    __hip_atomic_fetch_add(&fp->rt.xcd_counters[(size_t)c->domain * fleet::kMaxEvents + QABS_EVENT(L, c->coord[0])], 1u,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}
extern "C" __device__ void fleet_attention(const etx_ctx* c) {
  const FleetParams* fp = FP(c);
  const int32_t L = LAYER(c);
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
  uint64_t t0 = 0;
  pro_begin(fp, tc, s, t0);
  fleet::run_task(t, fp->rt, fp->d, fp->w, act, kv, s, tc, (int)c->domain, 0);
  pro_end(fp, s, PRO_SLOT(L, fleet::TASK_ATTENTION, c), t0);
}
#if defined(ETX_DSV2_KVA_SHARED) && ETX_DSV2_KVA_SHARED
// K1 (docs/DSV2-D1-D5.md, ETX_DSV2_KVA=shared; the arm passes -DETX_DSV2_KVA_SHARED=1): fleet's FLAG_KVA_SHARED -- each XCD
// computes 1/8 of the 576 kv_a rows into ONE buffer instead of its own full copy, graph.py makes the qkv event global and
// attention reads that buffer. Compiled only in that build: the default translation unit stays byte-identical (with the
// entry points always present, the unchanged default ISA moved in 78 lines on ROCm 7.2.4). A plan that names these
// symbols without the define fails to link.
extern "C" __device__ void fleet_qkv_fused_kvs(const etx_ctx* c) { fleet_call(c, fleet::TASK_QKV_FUSED, fleet::FLAG_KVA_SHARED, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_qkv_fused_fold_kvs(const etx_ctx* c) { fleet_call(c, fleet::TASK_QKV_FUSED, fleet::FLAG_FOLD_PARTIALS | fleet::FLAG_KVA_SHARED, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_attention_kvs(const etx_ctx* c) {   // fleet_attention plus FLAG_KVA_SHARED
  const FleetParams* fp = FP(c);
  const int32_t L = LAYER(c);
  fleet::TaskDescriptor t{};
  t.kind = fleet::TASK_ATTENTION; t.layer = (int16_t)L; t.xcd = (int16_t)c->domain; t.worker = 0;
  t.wait_event = -1; t.signal_event = -1; t.n_split = fleet::kMaxKvChunks; t.head = (int16_t)c->coord[0];
  t.kv_chunk = (int16_t)c->coord[1]; t.expert_slot = -1; t.wait_count = fleet::kQAbsSlices;
  t.local_event = (int16_t)QABS_EVENT(L, c->coord[0]); t.flags = fleet::FLAG_PUB_WAIT | fleet::FLAG_KVA_SHARED;
  fleet::TokenCtx tc;
  tc.tok = fp->tok; tc.epoch = fp->epoch; tc.pos = fp->pos; tc.seq_len = fp->pos + 1;
  tc.cos = fp->act.cos + (int64_t)fp->pos * fp->d.qk_rope; tc.sin = fp->act.sin + (int64_t)fp->pos * fp->d.qk_rope;
  tc.dump = fp->dump != 0; tc.trace = false;
  fleet::Activations& act = *const_cast<fleet::Activations*>(&fp->act);
  fleet::KVCache& kv = *const_cast<fleet::KVCache*>(&fp->kv);
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  uint64_t t0 = 0;
  pro_begin(fp, tc, s, t0);
  fleet::run_task(t, fp->rt, fp->d, fp->w, act, kv, s, tc, (int)c->domain, 0);
  pro_end(fp, s, PRO_SLOT(L, fleet::TASK_ATTENTION, c), t0);
}
#endif
extern "C" __device__ void fleet_merge_uv(const etx_ctx* c) { fleet_call(c, fleet::TASK_MERGE_UV, 0, c->coord[0], c->coord[1], fleet::kMaxKvChunks, -1, 0); }
extern "C" __device__ void fleet_o_proj(const etx_ctx* c) { fleet_call(c, fleet::TASK_O_PROJ, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_norm_router(const etx_ctx* c) { fleet_call(c, fleet::TASK_NORM_ROUTER, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_expert_gate_up(const etx_ctx* c) {
  const FleetParams* fp = FP(c);
  fleet_call(c, fleet::TASK_EXPERT_GATE_UP, fleet::FLAG_CHUNK_SIGNAL, -1, 1, 1, SLOT_OF_XCD(fp->d.top_k), CHIPLET_WORKER);
}
extern "C" __device__ void fleet_expert_down(const etx_ctx* c) {
  const FleetParams* fp = FP(c);
  fleet_call(c, fleet::TASK_EXPERT_DOWN, fleet::FLAG_CHUNK_WAIT, -1, 1, 1, SLOT_OF_XCD(fp->d.top_k), CHIPLET_WORKER);
}
extern "C" __device__ void fleet_expert_down_cached(const etx_ctx* c) {
  const FleetParams* fp = FP(c);
  fleet_call(c, fleet::TASK_EXPERT_DOWN, fleet::FLAG_CHUNK_WAIT | fleet::FLAG_ROUTING_CACHED, -1, 1, 1, SLOT_OF_XCD(fp->d.top_k), CHIPLET_WORKER);
}
extern "C" __device__ void fleet_dense_gate_up(const etx_ctx* c) { fleet_call(c, fleet::TASK_DENSE_GATE_UP, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_dense_down(const etx_ctx* c) { fleet_call(c, fleet::TASK_DENSE_DOWN, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_lm_head(const etx_ctx* c) { fleet_call(c, fleet::TASK_LM_HEAD, 0, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_lm_head_fold(const etx_ctx* c) { fleet_call(c, fleet::TASK_LM_HEAD, fleet::FLAG_FOLD_PARTIALS, -1, -1, 1, -1, CHIPLET_WORKER); }
extern "C" __device__ void fleet_argmax(const etx_ctx* c) { fleet_call(c, fleet::TASK_ARGMAX, 0, -1, -1, 1, -1, 0); }

#if defined(ETX_DSV2_KCHUNKS) && ETX_DSV2_KCHUNKS
// D3 (docs/DSV2-D1-D5.md): gate_up -> down handed over in 4 K-chunks of 352 rows of h. gate_up signals chunk c on the
// XCD's counter KCH_EVENT(L) + c from every wave as its rows land (fleet's CHUNK_SIGNAL); down -- whose ETX edge is the
// routing, not gate_up -- waits chunk 0 here and chunks 1..3 inside fleet's body (CHUNK_WAIT), each followed by
// buffer_inv sc1 (lever kchunk352). A worker's gate_up precedes its down in its queue, so the waits cannot deadlock.
static_assert(fleet::kExpertKChunk == 352, "ETX_DSV2_KCHUNKS needs lever kchunk352 (fleet kExpertKChunk = 352)");
static_assert(fleet::kWorkersPerXCD == 37 && fleet::kWaves == 4, "148 chunk signals per chunk per token");
#define KCH_N 4
#define KCH_EVENT(L) (1024 + KCH_N * (L))   // xcd_counters slots (qabs 256 + 16 L + h < 768, D4 768 + L)
#define KCH_SIGNALS (fleet::kWorkersPerXCD * fleet::kWaves)
static __device__ __forceinline__ void fleet_call_k(const etx_ctx* c, uint16_t kind, int16_t flags) {
  const FleetParams* fp = FP(c);
  const int32_t L = LAYER(c);
  fleet::TaskDescriptor t{};
  t.kind = kind; t.layer = (int16_t)L; t.xcd = (int16_t)c->domain; t.worker = (int16_t)CHIPLET_WORKER;
  t.wait_event = -1; t.signal_event = -1; t.signal_scope = 0; t.n_split = 1;
  t.head = -1; t.kv_chunk = KCH_N; t.expert_slot = (int16_t)SLOT_OF_XCD(fp->d.top_k);
  t.wait_scope = 0; t.wait_count = KCH_SIGNALS; t.local_event = (int16_t)KCH_EVENT(L); t.flags = flags;
  t.signal_xcd_count = 0; t.index = 0;
  fleet::TokenCtx tc;
  tc.tok = fp->tok; tc.epoch = fp->epoch; tc.pos = fp->pos; tc.seq_len = fp->pos + 1;
  tc.cos = fp->act.cos + (int64_t)fp->pos * fp->d.qk_rope;
  tc.sin = fp->act.sin + (int64_t)fp->pos * fp->d.qk_rope;
  tc.dump = fp->dump != 0; tc.trace = false;
  fleet::Activations& act = *const_cast<fleet::Activations*>(&fp->act);
  fleet::KVCache& kv = *const_cast<fleet::KVCache*>(&fp->kv);
  fleet::Scratch& s = *reinterpret_cast<fleet::Scratch*>(c->lds);
  uint64_t t0 = 0;
  pro_begin(fp, tc, s, t0);
  fleet::run_task(t, fp->rt, fp->d, fp->w, act, kv, s, tc, (int)c->domain, CHIPLET_WORKER);
  pro_end(fp, s, PRO_SLOT(L, kind, c), t0);
}
extern "C" __device__ void fleet_expert_gate_up_k(const etx_ctx* c) {
  fleet_call_k(c, fleet::TASK_EXPERT_GATE_UP, fleet::FLAG_CHUNK_SIGNAL);
}
extern "C" __device__ void fleet_expert_down_k(const etx_ctx* c) {
  const FleetParams* fp = FP(c);
  __shared__ int k0_ok;
  if (threadIdx.x == 0) {
    k0_ok = fleet::wait_event(fp->rt, fp->epoch, KCH_EVENT(LAYER(c)), fleet::SCOPE_XCD_LOCAL, (int)c->domain, KCH_SIGNALS) ? 1 : 0;
    if (k0_ok) asm volatile("buffer_inv sc1" ::: "memory");   // the CU's L1 too (sc0 would not, run 5)
  }
  __syncthreads();
  if (!k0_ok) return;
  fleet_call_k(c, fleet::TASK_EXPERT_DOWN, fleet::FLAG_CHUNK_WAIT | fleet::FLAG_ROUTING_CACHED);
}
#endif

#if defined(ETX_DSV2_D4) && ETX_DSV2_D4
// D4 (docs/DSV2-D1-D5.md; placement and its build-time check: examples/dsv2lite/d4_placement.py, graph.py
// build_moe_layer_d4). Grid constants (L, request, rows, spx). Routed grid coordinate k = routing slot k = XCD k;
// shared grid coordinate x = shared half x on XCD top_k + x, which also runs rows [0, rows) of slots [x*spx, x*spx+spx).
// A slice's h lives in its slot's h buffer (rows [0, rows) by the shared XCD, [rows, inter) by the routed XCD; rows is a
// multiple of 32, so no h cache line is written by two XCDs).
#define D4_ROWS(c) ((int)(c)->cst[2])
#define D4_SPX(c) ((int)(c)->cst[3])
#define D4_EVENT(L) (768 + (L))      // xcd_counters slot of the routing publication (qabs uses 256 + 16 L + h < 768)
struct D4Task {
  const FleetParams* fp; int L, xcd, worker, nw;
  fleet::Activations* act; fleet::Scratch* s;
};
static __device__ __forceinline__ D4Task d4_task(const etx_ctx* c) {
  D4Task t;
  t.fp = FP(c); t.L = LAYER(c); t.xcd = (int)c->domain; t.worker = (int)c->coord[1]; t.nw = fleet::kWorkersPerXCD;
  t.act = const_cast<fleet::Activations*>(&t.fp->act); t.s = reinterpret_cast<fleet::Scratch*>(c->lds);
  return t;
}
static __device__ __forceinline__ fleet::ExpertUnit d4_unit(const D4Task& t, int slot, const int32_t* ids, const float* ws) {
  const fleet::ModelDims& d = t.fp->d;
  return fleet::resolve_unit(slot, t.fp->w.layer_experts[t.L], t.fp->w.layer_shared[t.L], ids, ws, d.routed_scaling,
                             d.hidden, d.moe_inter, d.moe_inter * 2, t.fp->w.expert_stride);
}
// a routed slot's slice for the shared XCD: (id, weight) as the routed XCD published them (coherent loads)
static __device__ __forceinline__ fleet::ExpertUnit d4_slice_unit(const D4Task& t, int slot, int rows, float* weight) {
  const fleet::ModelDims& d = t.fp->d;
  const int id = fleet::ld_coh(t.act->topk_ids + slot, true);
  *weight = fleet::ld_coh(t.act->topk_w + slot, true) * d.routed_scaling;
  fleet::ExpertUnit u;
  const __hip_bfloat16* base = t.fp->w.layer_experts[t.L] + (int64_t)id * t.fp->w.expert_stride;
  u.gate_up = base; u.down = base + (int64_t)2 * d.moe_inter * d.hidden; u.down_ld = d.moe_inter;
  u.weight = 1.f; u.inter = rows;          // the routing weight is folded into h (d4 shared down)
  return u;
}

// The measured 32-row case: three slices fit in one wave-row batch (96 rows, 37 workers x 4 waves).
// Keep the shared half first; only then wait for routing. ids/ws survive into sdown, just as routed gate_up/down:
// graph.build() requires cached routing, and the two shared tasks are consecutive on the same pinned worker.
static __device__ __forceinline__ void d4_slices_gate_up_32(const D4Task& t, int x) {
  const fleet::ModelDims& d = t.fp->d;
  __shared__ int ready;
  int ok = 0;
  if (threadIdx.x == 0) {
    ok = fleet::wait_event(t.fp->rt, t.fp->epoch, D4_EVENT(t.L), fleet::SCOPE_XCD_LOCAL, t.xcd, 3) ? 1 : 0;
    ready = ok;
  }
  if (threadIdx.x < fleet::kWaveLanes) {   // uniform wave-0 branch; broadcast success before reading routing
    ok = __shfl(ok, 0, fleet::kWaveLanes);
    if (ok && threadIdx.x < 3) {
      // Six scalar atomics in thread 0 serialize in ROCm's ISA; three lanes fetch one pair each in two rounds.
      const int j = threadIdx.x;
      t.s->ids[j] = fleet::ld_coh(t.act->topk_ids + x * 3 + j, true);
      t.s->ws[j] = fleet::ld_coh(t.act->topk_w + x * 3 + j, true) * d.routed_scaling;
    }
  }
  __syncthreads();                       // one routing broadcast, shared-half GEMV already complete
  if (!ready) return;
  const int wave = threadIdx.x / fleet::kWaveLanes, lane = threadIdx.x % fleet::kWaveLanes;
  for (int n = t.worker + wave * t.nw; n < 3 * 32; n += fleet::kWaves * t.nw) {
    const int j = n / 32, row = n % 32;
    const __hip_bfloat16* base = t.fp->w.layer_experts[t.L] + (int64_t)t.s->ids[j] * t.fp->w.expert_stride;
    const __hip_bfloat16* pair[2] = {base + (int64_t)2 * row * d.hidden,
                                    base + (int64_t)(2 * row + 1) * d.hidden};
    float sums[2];
    fleet::wave_dot<2, 4>(pair, t.s->vec, d.hidden, sums);
    if (lane == 0)
      t.act->expert_h[(int64_t)(x * 3 + j) * d.moe_inter + row] =
          fleet::bf16_round(fleet::bf16_round(fleet::silu(fleet::bf16_round(sums[0]))) * fleet::bf16_round(sums[1]));
  }
}

// K=32 needs four 16-byte loads per output row, not a half-wave with 28 idle lanes. All three slices' weight loads
// are independent and issued together. Keep three separate reductions and the original shared+s0+s1+s2 order;
// concatenating into one K=96 dot or weighting after reduction would change the floating-point computation.
static __device__ __forceinline__ void d4_shared_down_32(const D4Task& t, int x, const fleet::ExpertUnit& sh,
                                                       __hip_bfloat16* out) {
  const fleet::ModelDims& d = t.fp->d;
  // row is otherwise unused by these two expert tasks; vec and part keep their existing GEMV roles.
  if (threadIdx.x < 3 * 32) {
    const int j = threadIdx.x / 32, i = threadIdx.x % 32;
    t.s->row[threadIdx.x] = t.act->expert_h[(int64_t)(x * 3 + j) * d.moe_inter + i] * t.s->ws[j];
  }
  fleet::stage_vector(t.act->expert_h + (int64_t)t.xcd * d.moe_inter, t.s->vec, sh.inter);
  // stage_vector's barrier also publishes all 96 weighted h values. No per-slice barrier or vec overwrite.
  fleet::expert_down_chunk(sh, t.s->vec, t.s->part, out, d.hidden, t.worker, t.nw, 0, sh.inter, true, false);
  __syncthreads();                       // row ownership changes: all shared-half accumulators must be visible
  const int lane4 = threadIdx.x & 3;
  for (int n = t.worker + (threadIdx.x / 4) * t.nw; n < d.hidden; n += (fleet::kBlockThreads / 4) * t.nw) {
    uint4 weights[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const __hip_bfloat16* base = t.fp->w.layer_experts[t.L] + (int64_t)t.s->ids[j] * t.fp->w.expert_stride;
      const __hip_bfloat16* w = base + (int64_t)2 * d.moe_inter * d.hidden + (int64_t)n * d.moe_inter;
      weights[j] = fleet::load_weight(reinterpret_cast<const uint4*>(w) + lane4);
    }
    float sums[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float acc = 0.f;
      fleet::fma8(weights[j], t.s->row + j * 32 + lane4 * 8, acc);
      // The old half-wave reduction has 28 zero lanes: offsets 16,8,4 add zero, then offsets 2,1 do this tree.
      acc += 0.f;                         // retain the old tree's signed-zero normalization
      acc += __shfl_xor(acc, 2, fleet::kWaveLanes);
      acc += __shfl_xor(acc, 1, fleet::kWaveLanes);
      sums[j] = acc;
    }
    if (lane4 == 0) {
      float acc = t.s->part[n];
#pragma unroll
      for (int j = 0; j < 3; ++j) acc += sums[j];
      out[n] = __float2bfloat16(fleet::bf16_round(acc));
    }
  }
}

// routed XCD k: fleet's routing (per task, as fleet_expert_gate_up), worker 0 publishes slot k's (id, weight) for the
// shared XCD that runs slot k's slice, then rows [rows, inter) of slot k
extern "C" __device__ void fleet_d4_routed_gate_up(const etx_ctx* c) {
  const D4Task t = d4_task(c);
  const fleet::ModelDims& d = t.fp->d;
  const int slot = (int)c->coord[0], rows = D4_ROWS(c);
#if ETX_DSV2_STAMPS2
  // stamped builds only: fleet's gate_up stamps (start, routing done, x staged) in fleet's gate_up slot for this XCD,
  // so the stage table splits topk / moe_up for D4 arms as for the default arm. Kept in LDS (stamps2[0], stamp_stage,
  // stamp), nothing live in registers across the GEMV: a register-held start time spilled 96 B/lane. Unstamped
  // builds are unchanged.
  if (t.fp->pro_trace != nullptr && threadIdx.x == 0) t.s->stamps2[0] = __builtin_amdgcn_s_memrealtime();
#endif
  fleet::select_experts(d, t.act->router_logits + (int64_t)t.xcd * d.n_routed, t.s->ids, t.s->ws);
#if ETX_DSV2_STAMPS2
  if (t.fp->pro_trace != nullptr && threadIdx.x == 0) t.s->stamp_stage = __builtin_amdgcn_s_memrealtime();
#endif
  if (t.worker == 0 && threadIdx.x == 0) {
    fleet::st_coh(t.act->topk_ids + slot, t.s->ids[slot], true);
    fleet::st_coh(t.act->topk_w + slot, t.s->ws[slot], true);
    fleet::drain_stores();
    fleet::fence_release();
    __hip_atomic_fetch_add(&t.fp->rt.xcd_counters[(size_t)(d.top_k + slot / D4_SPX(c)) * fleet::kMaxEvents + D4_EVENT(t.L)],
                           1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  fleet::ExpertUnit u = d4_unit(t, slot, t.s->ids, t.s->ws);
  fleet::stage_vector(t.act->x_norm + (int64_t)t.xcd * d.hidden, t.s->vec, d.hidden);
#if ETX_DSV2_STAMPS2
  if (t.fp->pro_trace != nullptr && threadIdx.x == 0) t.s->stamp = __builtin_amdgcn_s_memrealtime();
#endif
  u.gate_up += (int64_t)2 * rows * d.hidden;
  u.inter -= rows;
  fleet::expert_gate_up(u, t.s->vec, t.act->expert_h + (int64_t)slot * d.moe_inter + rows, d.hidden, t.worker, t.nw);
#if ETX_DSV2_STAMPS2
  if (t.fp->pro_trace != nullptr && threadIdx.x == 0) {      // pro_end's STAMPS2 row: {start, staging, prologue, 0...}
    uint64_t* r = t.fp->pro_trace + PRO_SLOT(t.L, fleet::TASK_EXPERT_GATE_UP, c) * 16;
    r[0] = t.s->stamps2[0]; r[1] = t.s->stamp_stage; r[2] = t.s->stamp;
    for (int i = 3; i < 16; ++i) r[i] = 0;
  }
#endif
}

// routed XCD k: rows [rows, inter) of slot k's down, with the routing its own gate_up left in LDS (as ROUTING_CACHED)
extern "C" __device__ void fleet_d4_routed_down(const etx_ctx* c) {
  const D4Task t = d4_task(c);
  const fleet::ModelDims& d = t.fp->d;
  const int slot = (int)c->coord[0], rows = D4_ROWS(c);
  fleet::ExpertUnit u = d4_unit(t, slot, t.s->ids, t.s->ws);
  u.down += rows;
  u.inter -= rows;
  fleet::stage_vector(t.act->expert_h + (int64_t)slot * d.moe_inter + rows, t.s->vec, u.inter);
  fleet::expert_down_chunk(u, t.s->vec, nullptr, t.act->expert_out + (int64_t)t.xcd * d.hidden, d.hidden, t.worker, t.nw,
                           0, u.inter, true, true);
}

// shared XCD top_k + x: the router's prologue without the router (fold the o_proj partials, norm), the shared half's
// gate_up -- no routing, so this starts when o_proj completes -- then, after the routed XCDs published their routing,
// rows [0, rows) of slots [x*spx, x*spx + spx) from the same staged x
extern "C" __device__ void fleet_d4_shared_gate_up(const etx_ctx* c) {
  const D4Task t = d4_task(c);
  const fleet::ModelDims& d = t.fp->d;
  const int x = (int)c->coord[0], rows = D4_ROWS(c), spx = D4_SPX(c);
  fleet::stage_residual_f32(d, t.act->x[0], t.act->oproj_partial, t.act->x[1], t.s->vec, t.xcd, t.worker, false);
  fleet::rms_norm_lds(d, t.fp->w.layer_norm_post[t.L], t.s->vec, t.s->smem);
  const fleet::ExpertUnit sh = d4_unit(t, -(x + 1), nullptr, nullptr);
  fleet::expert_gate_up(sh, t.s->vec, t.act->expert_h + (int64_t)t.xcd * d.moe_inter, d.hidden, t.worker, t.nw);
  if (rows == 0) return;
  if (rows == 32 && spx == 3) { d4_slices_gate_up_32(t, x); return; }
  __shared__ int d4_ok;
  if (threadIdx.x == 0)
    d4_ok = fleet::wait_event(t.fp->rt, t.fp->epoch, D4_EVENT(t.L), fleet::SCOPE_XCD_LOCAL, t.xcd, spx) ? 1 : 0;
  __syncthreads();
  if (!d4_ok) return;
  for (int j = 0; j < spx; ++j) {
    const int slot = x * spx + j;
    float weight;
    const fleet::ExpertUnit u = d4_slice_unit(t, slot, rows, &weight);
    fleet::expert_gate_up(u, t.s->vec, t.act->expert_h + (int64_t)slot * d.moe_inter, d.hidden, t.worker, t.nw);
  }
}

// shared XCD top_k + x: one partial = bf16(shared half's down + sum over its slices of weight * down . h), as fleet's
// K-chunked down (fp32 accumulator in LDS, one bf16 store); the slices' routing weights are folded into their h
extern "C" __device__ void fleet_d4_shared_down(const etx_ctx* c) {
  const D4Task t = d4_task(c);
  const fleet::ModelDims& d = t.fp->d;
  const int x = (int)c->coord[0], rows = D4_ROWS(c), spx = D4_SPX(c);
  __hip_bfloat16* out = t.act->expert_out + (int64_t)t.xcd * d.hidden;
  const fleet::ExpertUnit sh = d4_unit(t, -(x + 1), nullptr, nullptr);
  if (rows == 32 && spx == 3) { d4_shared_down_32(t, x, sh, out); return; }
  fleet::stage_vector(t.act->expert_h + (int64_t)t.xcd * d.moe_inter, t.s->vec, sh.inter);
  fleet::expert_down_chunk(sh, t.s->vec, t.s->part, out, d.hidden, t.worker, t.nw, 0, sh.inter, true, rows == 0);
  for (int j = 0; rows > 0 && j < spx; ++j) {
    const int slot = x * spx + j;
    float weight;
    const fleet::ExpertUnit u = d4_slice_unit(t, slot, rows, &weight);
    __syncthreads();                       // every wave is done with s.vec (and its accumulator rows are in LDS)
    const float* h = t.act->expert_h + (int64_t)slot * d.moe_inter;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) t.s->vec[i] = h[i] * weight;
    __syncthreads();
    fleet::expert_down_chunk(u, t.s->vec, t.s->part, out, d.hidden, t.worker, t.nw, 0, rows, false, j + 1 == spx);
  }
}
#endif

// exported for the host: the LDS the bodies need, so the graph's resource declaration can be checked
extern "C" size_t fleet_scratch_bytes() { return sizeof(fleet::Scratch); }
