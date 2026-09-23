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

// waits check the abort word every 256 polls so the host watchdog can stop a hung step
#define ETX_DEFINE_SCOPE(S)                                                              \
  static __device__ __forceinline__ void etx_wait_##S(etx_event* e, const int32_t* abort_flag) { \
    uint32_t n = 0;                                                                       \
    while (ETX_POLL_##S(e) > 0) {                                                         \
      ETX_BACKOFF();                                                                      \
      if ((++n & 255u) == 0 && ETX_POLL_DEVICE(abort_flag) != 0) return;                  \
    }                                                                                     \
  }                                                                                       \
  static __device__ __forceinline__ int32_t etx_arrive_##S(etx_event* e) {               \
    return ETX_ARRIVE_##S(e) - 1;                                                         \
  }
ETX_DEFINE_SCOPE(DOMAIN)
ETX_DEFINE_SCOPE(DEVICE)
ETX_DEFINE_SCOPE(SYSTEM)
#undef ETX_DEFINE_SCOPE

// Last-arriver flush (fleet's scheme, measured 13 -> 6 us per layer on MI300X): for a
// DEVICE-scope event with several producers on one domain, every producer drains its
// stores and bumps the domain's sub-counter; only the arrival that completes the
// domain's share performs the L2 write-back and subtracts the whole share from the
// global counter. Returns the global counter's new value when this call updated it,
// else a positive dummy (the caller only acts on 0).
static __device__ __forceinline__ int32_t etx_arrive_flush_DEVICE(const etx_params& p, int32_t idx, uint32_t domain) {
  const int32_t share = p.ev_share ? p.ev_share[(size_t)domain * p.event_words + idx] : 0;
  if (share <= 1) { ETX_RELEASE_DEVICE(); return ETX_ARRIVE_DEVICE(p.events + idx) - 1; }
  ETX_RELEASE_DOMAIN();                                              // drain this workgroup's stores into the domain's L2
  const int32_t old = atomicAdd(p.ev_sub + (size_t)domain * p.event_words + idx, 1);
  if (old + 1 != share) return 1;
  ETX_RELEASE_DEVICE();                                              // one write-back covers every earlier arrival's stores
  return __hip_atomic_fetch_sub(p.events + idx, share, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - share;
}

// Queues: many-producer many-consumer *ticket* ring. Measured on MI300X
// (bench/calib/queue_contention): a CAS-based pop costs ~1.1 us alone and its
// retry storm makes aggregate throughput fall past 8 poppers. The ticket ring
// pays one atomic per push (tail) and one per pop (head); the popper then polls
// its own slot with a scoped load. Slots are written at most once per step
// (the plan sizes capacity >= pushes per step; the host resets between steps),
// so poppers never clear them. A ticket taken beyond the final tail is harmless:
// the worker keeps servicing its static queue and exits at step_done.
// Slot values carry a pass tag in the top bits: value = (t / capacity) << 24 | task.
// A worker that over-popped (its ticket is beyond the final tail) wraps onto a
// slot consumed in an earlier pass; the tag mismatch makes it keep waiting
// instead of re-executing that task (this exact bug was caught by the trace
// on MI300X: 267 of 320 tasks ran twice). Pushes never wrap because the plan
// sizes capacity >= pushes per step. Task ids must be < 2^24.
#define ETX_TAG_SHIFT 24
#define ETX_TASK_MASK 0x00FFFFFF

static __device__ __forceinline__ void etx_push(const etx_queue* q, int32_t task) {
  const int32_t t = atomicAdd(q->tail, 1);
  atomicExch(q->slots + (t % q->capacity), ((t / q->capacity) << ETX_TAG_SHIFT) | task);
}

// Returns a task id, or -1 if nothing is available yet. `ticket` is per-worker
// state (-1 = none held) and must persist across calls.
static __device__ __forceinline__ int32_t etx_try_pop(const etx_queue* q, int32_t* ticket) {
  if (q->capacity == 0) return -1;
  if (*ticket < 0) {
    const int32_t head = ETX_POLL_DEVICE(q->head);
    const int32_t tail = ETX_POLL_DEVICE(q->tail);
    if (head >= tail) return -1;                            // empty right now: do not reserve
    *ticket = atomicAdd(q->head, 1);
  }
  const int32_t v = ETX_POLL_DEVICE(q->slots + (*ticket % q->capacity));
  if (v < 0 || (v >> ETX_TAG_SHIFT) != (*ticket / q->capacity)) return -1;
  *ticket = -1;
  return v & ETX_TASK_MASK;
}

// Event coordinate (ev_id, lin) reached zero: for every dynamic / hybrid consumer
// decrement its remaining-dependency counter; the arrival that brings it to zero
// pushes the task exactly once (a task with several in-events is not pushed per
// event -- caught on MI300X: combine tasks ran twice). Push-list encoding:
//   enc >= 0 : hybrid consumer, its OWN domain in bits 24.., task id in bits 0..23
//   enc <  0 : dynamic consumer, ~task id -> device-global queue
static __device__ __forceinline__ void etx_push_consumers(const etx_params& p, int ev_id, int lin, uint32_t domain) {
  (void)domain;
  const int32_t base = p.push_index[ev_id] + lin;
  const int32_t b = p.push_offsets[base], e = p.push_offsets[base + 1];
  for (int32_t i = b; i < e; ++i) {
    const int32_t enc = p.push_lists[i];
    const int32_t task = enc >= 0 ? (enc & ETX_TASK_MASK) : ~enc;
    if (atomicSub(p.task_remaining + task, 1) != 1) continue;
    if (enc >= 0) etx_push(p.local_queue + (enc >> ETX_TAG_SHIFT), task);
    else          etx_push(&p.global_queue, task);
  }
}

// deps_ready: non-blocking readiness probe of a task's in-events, generated per
// task type by the kernel emitter from the same edge maps as the wait code.
// The worker loop uses it to prefer queue work while its static head is blocked.
static __device__ bool etx_deps_ready(const etx_params& p, int32_t task);

// Step termination for a worker whose static queue is drained: only dynamic /
// hybrid tasks are counted (one atomic per such task; static tasks cost none),
// and the count is polled with a plain scoped load, not an atomic. Measured on
// MI300X (bench/calib/step_overhead): a per-task atomic on one word from 608
// workers plus atomic idle polling cost ~6 us per task slot and doubled the
// step time; without them the loop costs ~0.7 us per slot.
static __device__ __forceinline__ bool etx_step_done(const etx_params& p) {
  return p.n_dynamic == 0 || ETX_POLL_DEVICE(p.ctrl_done) >= p.n_dynamic;
}

static __device__ __forceinline__ uint32_t etx_discover_domain(const etx_params& p, uint32_t worker) {
#if ETX_CAP_WG_TO_DOMAIN_MAP_DISCOVER
  (void)p; (void)worker;
  return (uint32_t)ETX_DOMAIN_ID();
#else
  return p.worker_domain ? (uint32_t)p.worker_domain[worker] : 0u;
#endif
}
