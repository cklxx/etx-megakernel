// ETX runtime ABI (design §06 5.2). Shared by generated kernels, tile bodies and
// the host launcher. Deliberately six things: coordinate, shape scalars, argument
// table, events, position, LDS. A tile that needs more is trying to schedule.
#pragma once
#include <stdint.h>
#include <stddef.h>

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
};
