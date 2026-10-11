// ETX runtime ABI (design §06 5.2). Shared by generated kernels, tile bodies and
// the host launcher. Deliberately six things: coordinate, shape scalars, argument
// table, events, position, LDS. A tile that needs more is trying to schedule.
#pragma once
#include <stdint.h>
#include <stddef.h>

// Words between consecutive event counters in p.events. 1 packs them (the 8 coordinates of an XCD-local event share one
// cache line and every hand-off contends on it: 3.3 us vs 1.0 us padded, bench/hw/hw_bench.hip handoff, docs/MODEL-dsv2.md
// H6); 32 gives every counter its own 128 B line. Device code addresses a counter as p.events + ETX_EVW(word).
#ifndef ETX_EV_STRIDE
#define ETX_EV_STRIDE 1
#endif
#if ETX_EV_STRIDE > 1 && defined(__HIP__) && !defined(ETX_EVW_FOLD)   // -DETX_EVW_FOLD: the old folded form (A/B)
// Device code: the word index goes through an empty asm, so p.events + ETX_EVW(const) is computed where it is used.
// With a constant index LLVM hoists every event's address out of the persistent loop; at stride 1 the offsets fit the
// load/atomic immediate (4 KB) and cost nothing, at stride 32 they do not, and the hoisted 64-bit addresses spilled
// (Qwen3-8B megakernel: 2 -> 35 VGPR spills, ROCm 7.2.4 clang, examples/vllm_llm/rocm_build_check.sh). Host code
// (etx/host.h, compiled in both passes of a HIP translation unit) calls the same function; the asm is device-only.
static inline __attribute__((host, device, always_inline)) int etx_evw(int x) {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("" : "+v"(x));
#endif
  return x * ETX_EV_STRIDE;
}
#define ETX_EVW(x) etx_evw(x)
#else
#define ETX_EVW(x) ((x) * ETX_EV_STRIDE)
#endif

typedef int32_t etx_event;            // one counter; an ETensor is a contiguous run of them

struct etx_task {
  int32_t type;                        // task-type id == index into the dispatch switch
  int32_t coord[4];
};

struct etx_queue {                     // ring of task ids, single-producer-many / many-many via atomics
  int32_t* slots;
  int32_t  capacity;
  int32_t* head;                       // pop cursor
  int32_t* tail;                       // push cursor
};

struct etx_ctx {
  int32_t         coord[4];
  const int32_t*  shape;               // symbolic scalars of this step, in graph.symbols order
  void* const*    args;                // tensor pointers in THIS grid's call_device(args=[...]) order
  etx_event*      events;              // event buffer base
  const int32_t*  ev_offset;           // per-event-tensor offset into events
  const int32_t*  ev_shape;            // per-event-tensor shape, 4 ints each
  uint32_t        domain;
  uint32_t        worker;
  void*           lds;
  int32_t         cst[4];              // per-grid compile-time constants (TaskGrid.consts): immediates, no load after the wait
};

// one entry of the device-side per-step reset (ETX_DEV_RESET); `src == nullptr` means fill with `fill`
struct etx_reset_op { int32_t* dst; const int32_t* src; int32_t words; int32_t fill; };

struct etx_params {
  const int32_t*   shape;
  void* const*     args;               // global tensor table (plan order)
  void* const*     type_args;          // per task type: max_args pointers in the grid's own argument order
  int32_t          max_args;
  etx_event*       events;
  const int32_t*   ev_offset;
  const int32_t  (*ev_shape)[4];
  const etx_task*  descs;
  const int32_t*   static_queue;       // concatenated per-worker queues
  const etx_task*  static_descs;       // descs[static_queue[i]], so a static worker takes its head with one load
  const int32_t*   static_begin;       // per worker
  const int32_t*   static_end;
  etx_queue*       local_queue;        // one per domain
  etx_queue        global_queue;       // capacity 0 when unused
  const int32_t*   worker_domain;      // optional host-provided map (when discovery is not needed)
  int32_t*         domain_slots;       // per domain, zeroed per step: workers claim slots -> logical worker id
  int32_t          workers_per_domain;
  int32_t          n_workers;
  const int32_t*   push_offsets;       // CSR: (event id, linear) -> consumer task ids
  const int32_t*   push_lists;
  const int32_t*   push_index;         // per event id: base offset into push_offsets
  int32_t*         task_remaining;     // per task: in-event coordinates not yet at zero (dynamic/hybrid tasks); pushed at 0
  int32_t*         ev_sub;             // [n_domains][event words]: per-domain arrival sub-counters (last-arriver flush), zeroed per step
  const int32_t*   ev_share;           // [n_domains][event words]: producers of each coordinate on each domain (0 = plain arrive)
  int32_t          event_words;
  int32_t*         trace_exec;         // optional: per-task execution counter (debug); nullptr to disable
  uint64_t*        trace_time;         // optional: per task 4 x ETX_TIMER ticks {taken, deps ready, body done, worker}; nullptr to disable
  int32_t*         ctrl_abort;         // host writes 1 to abort; kernel writes 2 on spin timeout
  int32_t*         ctrl_done;          // tasks completed this step
  int32_t          n_tasks;            // this device's tasks (diagnostics)
  int32_t          n_dynamic;          // this device's dynamic + hybrid tasks: the only ones counted in ctrl_done
  uint32_t         spin_limit;
  int32_t          n_events;
  // relay (P5): one workgroup per domain mirrors the DEVICE-scope words into ev_mirror[domain][word];
  // DEVICE-scope waits poll their domain's mirror. nullptr = off (waits poll the global words).
  etx_event*       ev_mirror;
  const int32_t*   relay_words;        // this device's mirrored word offsets
  int32_t          n_relay;
  int32_t          relay_local_acquire; // 1: the relay's acquire covers the domain, consumers only drop L1
  // ETX_DEV_RESET: the per-step state reset, done by the grid at the end of the kernel instead of ~17 host enqueues.
  // Each op either copies `src` or fills with `fill`; the host builds the table once at setup. ctrl is deliberately
  // NOT in it -- the host reads ctrl_done after the kernel, so resetting it on the device would race that read.
  const etx_reset_op* reset_ops;
  int32_t          n_reset_ops;
#if defined(ETX_INBODY) && ETX_INBODY
  // Diagnostic ABI only. No fields or changed offsets in ordinary builds.
  uint64_t* inbody_poll;              // [task], post-success issue ticks
  uint64_t* inbody_event;             // [event word][before, after final arrive, producer task]
  uint64_t* inbody_wave;              // [task][wave][tile return, DOMAIN drain]
#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER
  const int32_t* arrival_offsets;      // CSR [task+1]
  const int32_t* arrival_words;        // original event words for every arrival
  uint64_t* arrival_stamps;            // [incidence][before, after original call]
  int32_t arrival_tasks;               // global task-id capacity of arrival_offsets
  int32_t arrival_slots;               // logical capacity of words and stamp pairs
#endif
#endif
};

#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER && !(defined(ETX_INBODY) && ETX_INBODY)
#error "producer ledger requires the diagnostic ETX_INBODY build"
#endif
