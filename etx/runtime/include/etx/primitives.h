// Device-side runtime primitives, built only from the macros in the generated
// etx_lowering.h. Nothing here names an architecture.
#pragma once
#include "etx/abi.h"

#if defined(ETX_VENDOR_AMD)
#include "etx/primitives_amdgcn.h"
#else
#include "etx/primitives_nvptx.h"
#endif

static __device__ __forceinline__ int etx_cdiv(int a, int b) { return (a + b - 1) / b; }
static __device__ __forceinline__ int etx_ev_numel(const int32_t* shape) {
  int n = 1; for (int i = 0; i < 4; ++i) if (shape[i] > 0) n *= shape[i]; return n;
}

#define ETX_DEFINE_SCOPE(S)                                                              \
  static __device__ __forceinline__ void etx_wait_##S(etx_event* e) {                    \
    while (ETX_POLL_##S(e) > 0) { ETX_BACKOFF(); }                                        \
  }                                                                                       \
  static __device__ __forceinline__ int32_t etx_arrive_##S(etx_event* e) {               \
    return ETX_ARRIVE_##S(e) - 1;                                                         \
  }
ETX_DEFINE_SCOPE(DOMAIN)
ETX_DEFINE_SCOPE(DEVICE)
ETX_DEFINE_SCOPE(SYSTEM)
#undef ETX_DEFINE_SCOPE

// queues: many-producer many-consumer ring; slots hold task ids, -1 = empty
static __device__ __forceinline__ int32_t etx_pop(etx_queue* q) {
  if (q->capacity == 0) return -1;
  int32_t h = atomicAdd(q->head, 0);
  int32_t t = atomicAdd(q->tail, 0);
  if (h >= t) return -1;
  if (atomicCAS(q->head, h, h + 1) != h) return -1;       // lost the race; caller retries next loop
  int32_t* slot = q->slots + (h % q->capacity);
  int32_t v;
  while ((v = atomicExch(slot, -1)) < 0) { ETX_BACKOFF(); }  // wait for the pusher's store
  return v;
}

static __device__ __forceinline__ void etx_push(etx_queue* q, int32_t task) {
  int32_t t = atomicAdd(q->tail, 1);
  int32_t* slot = q->slots + (t % q->capacity);
  while (atomicCAS(slot, -1, task) != -1) { ETX_BACKOFF(); } // slot still owned by a slow popper
}

static __device__ __forceinline__ void etx_push_consumers(const etx_params& p, int ev_id, int lin, uint32_t worker) {
  const int32_t base = p.push_index[ev_id] + lin;
  const int32_t b = p.push_offsets[base], e = p.push_offsets[base + 1];
  for (int32_t i = b; i < e; ++i) {
    const int32_t task = p.push_lists[i];
    const etx_task d = p.descs[task];
    (void)d;
    etx_push(p.local_queue + p.worker_domain[worker], task);   // domain-local push (hybrid); global handled by host layout
  }
}

// deps_ready: static-head probe without blocking. Conservative: the generated
// wait code re-checks; this only decides whether to try the queues first.
static __device__ __forceinline__ bool etx_deps_ready(const etx_params& p, int32_t task) {
  (void)p; (void)task;
  return true;   // v0: always take the static head; a per-task ready bitmap is the planned refinement
}

static __device__ __forceinline__ bool etx_step_done(const etx_params& p) {
  return atomicAdd(p.ctrl_done, 0) >= p.n_tasks;
}

static __device__ __forceinline__ uint32_t etx_discover_domain(const etx_params& p, uint32_t worker) {
#if ETX_CAP_WG_TO_DOMAIN_MAP_DISCOVER
  (void)p; (void)worker;
  return (uint32_t)ETX_DOMAIN_ID();
#else
  return p.worker_domain ? (uint32_t)p.worker_domain[worker] : 0u;
#endif
}
