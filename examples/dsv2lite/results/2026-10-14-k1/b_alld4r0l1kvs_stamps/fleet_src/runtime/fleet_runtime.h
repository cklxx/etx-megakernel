// Fleet runtime: AOT task queues, hierarchical events, XCD discovery.
//
// docs/design.md §4 and §8. Task code sees only three operations —
// fetch_task, signal_event, wait_event — so the tasks stay runtime-agnostic
// and can later be moved onto the upstream ROCm/fleet-chiplet-megakernel
// without being rewritten.
//
// The protocol, in one paragraph: descriptors are built on the host, never
// mutated, and pre-assigned to per-XCD worker queues (ahead-of-time launch, so
// a worker waits on its own next event rather than being dispatched by a
// scheduler after the fact — one synchronisation hop instead of two). Each XCD
// dedicates one CU to a scheduler whose only job is to mirror global event
// counters into that XCD's flag array, so 37 workers poll a local array
// instead of 296 workers polling the global counters.
//
// Memory model. MI300X L2 is per XCD and not coherent across XCDs for
// ordinary device memory, so every cross-workgroup handshake here is written
// against the LLVM AMDGPU memory model for gfx942 rather than against cache
// folklore: producers release with an agent-scope fence (buffer_wbl2), the
// counter is an agent-scope atomic, consumers acquire with an agent-scope
// fence (buffer_inv), and polling uses agent-scope atomic loads so a stale
// L1/L2 line can never satisfy the wait. XCD-local events acquire with an
// L1-only invalidate (the payload is in the producer's own L2). With
// coherent_acts every cross-XCD activation is accessed through agent-scope
// relaxed atomics (global_load/store with sc1, which bench (d''') checks
// word for word; MTYPE-UC allocation does NOT give this on the VM, bench
// (d'')), so global waits drop the L2 invalidate except where a cached
// payload still crosses XCDs: the KV cache at the token boundary and the
// dense layer's h (fenced_events). Producers keep the writeback: measured
// necessary (see event_release_fenced).
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace fleet {

constexpr int kXCDs = 8;
constexpr int kCUsPerXCD = 38;
// Workgroups per CU. Two were tried (the kernel fits: 252 VGPRs, 24 KB LDS)
// on the theory that one wave per SIMD leaves memory requests on the table;
// measured 6.5 ms vs 5.4 ms per token — every GEMV task ran ~2x slower with
// half the rows, so per-CU parallelism is not what bounds those phases.
#ifndef FLEET_BLOCKS_PER_CU
#define FLEET_BLOCKS_PER_CU 1
#endif
constexpr int kBlocksPerCU = FLEET_BLOCKS_PER_CU;
constexpr int kBlocksPerXCD = kCUsPerXCD * kBlocksPerCU;
constexpr int kWorkersPerXCD = kBlocksPerXCD - 1;   // one workgroup is the scheduler
constexpr int kGrid = kXCDs * kBlocksPerXCD;        // 608 workgroups, 2 per CU
constexpr int kBlock = 256;                         // 4 waves (kWaves lives in gemv.h)
constexpr int kMaxEvents = 4096;

// Task kinds — must match TaskKind in src/host/taskgraph.py.
enum TaskKind : uint16_t {
    TASK_QKV_FUSED = 0,
    TASK_ATTENTION = 1,
    TASK_MERGE_UV = 2,
    TASK_O_PROJ = 3,
    TASK_NORM_ROUTER = 4,
    TASK_EXPERT_GATE_UP = 5,
    TASK_EXPERT_DOWN = 6,
    TASK_REDUCE = 7,
    TASK_DENSE_GATE_UP = 8,
    TASK_DENSE_DOWN = 9,
    TASK_EMBED = 10,
    TASK_LM_HEAD = 11,
    TASK_ARGMAX = 12,
    TASK_PREFETCH = 13,   // stream routing-independent weights into the Infinity Cache
    TASK_Q_ABSORB = 14,   // publish one row-slice of a head's absorbed q_c
};

enum EventScope : int16_t {
    SCOPE_NONE = 0,
    SCOPE_XCD_LOCAL = 1,
    SCOPE_GLOBAL = 2,
};

// 64 bytes, laid out to match taskgraph.py's struct.pack("<Hhhh hhhh hhhh hhhh i 28x").
// Any drift here is silent and catastrophic, so static_assert guards the size
// and tests/test_descriptor_layout.py re-checks the field offsets from Python.
//
// A wait needs the *waited* event's scope and producer count, which are
// properties of that event, not of this task's own signal — so both sides are
// stored explicitly. Passing the signal-side pair to wait_event was the bug
// that made every wait target wrong.
struct TaskDescriptor {
    uint16_t kind;
    int16_t  layer;           // -1 for embed / lm_head / argmax
    int16_t  xcd;
    int16_t  worker;
    int16_t  wait_event;      // -1 = no dependency
    int16_t  signal_event;    // -1 = signals nothing
    int16_t  signal_scope;    // EventScope of signal_event
    int16_t  n_split;         // task-kind specific: kv_chunks for attention/merge
    int16_t  head;            // attention / merge, else -1
    int16_t  kv_chunk;        // split-KV index; experts: number of K chunks
    int16_t  expert_slot;     // >=0 routed slot k, <0 shared half, else -1
    int16_t  wait_scope;      // EventScope of wait_event
    int16_t  wait_count;      // producers of wait_event (per epoch)
    int16_t  local_event;     // FLAG_SIGNAL_LAST: XCD-local arrival counter, else -1
    int16_t  flags;           // TaskFlags
    int16_t  signal_xcd_count; // producers of signal_event on this XCD
    int32_t  index;           // position in the global task list (> 32767)
    uint8_t  _pad[28];
};
static_assert(sizeof(TaskDescriptor) == 64, "descriptor must stay 64 B");

enum TaskFlags : int16_t {
    FLAG_FOLD_PARTIALS = 1,   // prologue folds the 8 expert partials into the residual
    FLAG_SIGNAL_LAST = 2,     // signal signal_event only as the last of n_split
                              // arrivals on local_event (the merging KV chunk)
    FLAG_CHUNK_SIGNAL = 8,    // gate_up: each wave bumps XCD-local counter
                              // local_event + c when its rows of K-chunk c are
                              // stored; kv_chunk = number of chunks
    FLAG_CHUNK_WAIT = 16,     // down: waits local_event + c (c >= 1) in the
                              // body before consuming chunk c; chunk 0 is the
                              // descriptor's wait_event
    FLAG_KVA_SHARED = 32,     // q/kv_a: kv_a rows split over the 8 XCDs into
                              // one buffer; attention: read that buffer
    FLAG_WORKER_GROUP = 64,   // rows split over the n_split workers starting
                              // at worker `head`, not over all 37
    FLAG_ROUTING_CACHED = 128, // expert down: this worker's previous task was
                              // the layer's gate_up; its top-k is still in LDS
    FLAG_TOPK_PUBLISH = 256,  // router: the last of the XCD's 37 workers to
                              // arrive on local_event turns the 64 logits into
                              // the top-k and publishes it for the experts
    FLAG_TOPK_READ = 512,     // expert: read that published top-k (64 B) rather
                              // than repeating softmax + top-k per task
    FLAG_OPROJ_ROW_SPLIT = 2048, // o_proj splits W_o's output rows over the 8 XCDs
                              // against the full o[2048] (so the merge boundary is
                              // global) instead of K-splitting into 8 fp32 partials
    FLAG_PUB_WAIT = 1024,     // the task waits on local_event *inside its body*, after
                              // the work that does not depend on the publication, and
                              // then reads a published buffer instead of recomputing.
                              // wait_count carries the publisher count.
};

// Device-side state, allocated once by the host.
struct RuntimeState {
    const TaskDescriptor* __restrict__ tasks;   // all descriptors, immutable
    const int32_t* __restrict__ queue_offset;   // [kXCDs * kWorkersPerXCD + 1]
    const int32_t* __restrict__ queue_index;    // task indices, grouped per worker

    uint32_t* __restrict__ global_events;       // [kMaxEvents], device scope
    uint32_t* __restrict__ xcd_flags;           // [kXCDs][kMaxEvents], mirror
    uint32_t* __restrict__ xcd_counters;        // [kXCDs][kMaxEvents], local events

    uint32_t* __restrict__ xcd_arrivals;        // [kXCDs] role tickets, reset per launch
    uint32_t* __restrict__ grid_arrivals;       // [1] grid barrier, reset per launch
    uint32_t* __restrict__ abort;               // [1] 0 = fine, else a reason code

    // The ids of the GLOBAL-scope events, compact. XCD-local events never
    // touch global_events, so the scheduler must not poll their slots: with
    // 165 global among 805 events that would be 80% wasted fabric reads.
    const int32_t* __restrict__ global_event_ids;
    int32_t  n_global_events;

    // Launch model v2 (§4): one launch decodes n_tokens tokens. Token t of the
    // launch runs the whole graph at epoch epoch0 + t; counters are never
    // reset (§12), so wait targets are epoch x producers. The embed task of
    // token t > 0 additionally waits for the previous token's final event.
    uint32_t epoch0;       // epoch of the launch's first token (1 for a fresh state)
    int32_t  n_tokens;     // tokens decoded by this launch
    int32_t  teacher;      // 1 = token_in is pre-filled by the host (teacher forcing)
    int32_t  n_events;
    int32_t  done_event;   // the graph's final event: schedulers stop when it
                           // reaches epoch0 + n_tokens - 1
    int32_t  use_uncached_counters;  // event scheme (ii) from §4, set by D1 (a)
    int32_t  smoke;        // 1 = run the protocol only, skip every task body
    int32_t  direct_poll;  // 1 = workers poll global counters themselves,
                           //     bypassing the scheduler mirror (A/B on D1)
    int32_t  dump_layers;  // 1 = copy every layer's output x to act.layer_dump
                           //     on the launch's first token
    int32_t  trace_token;  // token of the launch to trace, or -1
    uint32_t spin_limit;   // polls before a wait declares a deadlock
    int32_t  coherent_acts; // 1 = cross-XCD activations use agent-scope atomic
                            //     loads/stores and global waits skip the L2
                            //     invalidate except for done_event and
                            //     fenced_events (producers still write back)
    int32_t  prefetch_nt;   // 1 = TASK_PREFETCH uses non-temporal loads
    int32_t  prefetch_next; // 1 = before waiting on a task's event, touch the
                            //     first 32 KB of the weights it will stream
                            //     (cross-task prefetch, design.md §12)
    int32_t  fenced_events[4];  // events whose payload is cached and cross-XCD
                                // even with coherent_acts (dense gate_up); -1 = none

    // Optional per-descriptor trace (design.md §10): [n_descriptors][4] of
    // s_memrealtime ticks — began waiting, wait satisfied, signalled — plus
    // the XCD id. nullptr disables it. Written by thread 0 of the worker.
    uint64_t* __restrict__ trace;
};

// Abort reason codes, read back by the host after the launch.
constexpr uint32_t kAbortNone = 0;
constexpr uint32_t kAbortXcdDistribution = 0x10000000u;  // | xcd << 8 | count
constexpr uint32_t kAbortWaitTimeout = 0x20000000u;      // | event id
constexpr uint32_t kAbortBarrierTimeout = 0x30000000u;

// ---------------------------------------------------------------- primitives

// gfx942 exposes the chiplet id in HW_REG_XCC_ID. A workgroup uses it to find
// its own queue and flags; MI300X dispatches workgroups round-robin over XCDs,
// so this is the only reliable way to know where you are.
__device__ __forceinline__ int xcd_id() {
#if defined(__gfx942__)
    unsigned id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
    return (int)(id & 0xf);
#else
    return (int)(blockIdx.x % kXCDs);
#endif
}

// Agent-scope atomic load: never satisfied by a stale L1/L2 line.
__device__ __forceinline__ uint32_t poll(uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void backoff() {
    // Keep polling off the fabric between reads; s_sleep 1 is ~64 cycles.
    __builtin_amdgcn_s_sleep(1);
}

__device__ __forceinline__ void fence_release() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // buffer_wbl2 + waitcnt
}

__device__ __forceinline__ void fence_acquire() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // buffer_inv sc1: L1 + L2
}

// Acquire for a consumer on the producer's own XCD: the drained stores are
// in this L2 already, so only this CU's L1 (write-through TCP) can be stale.
// `buffer_inv sc0` invalidates L1 without touching the L2 that the other 36
// workers are streaming through; an agent-scope buffer_inv here costs the
// phase several microseconds (bench/microbench.hip (f)), and (d') checks a
// same-XCD consumer reads every word with this acquire.
__device__ __forceinline__ void fence_acquire_local() {
#if defined(__gfx942__) || defined(__gfx940__) || defined(__gfx941__)
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\tbuffer_inv sc0\n\ts_waitcnt vmcnt(0)" ::: "memory");
#else
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#endif
}

// coherent_acts: which L2 fences does a global event still need?
// Measured (STATUS.md): dropping the consumer-side invalidate is safe (every
// cross-XCD activation is read through an agent-scope atomic load, and the
// KV cache row is never in a reader's L2 before it is written); dropping the
// producer-side writeback is NOT — 18/18 launches correct with the release
// kept, most launches wrong without it, although bench (d''') passes. So the
// release stays on every global event; the acquire only where a cached
// payload crosses XCDs (done_event: the KV cache; fenced_events: the dense
// layer's h). Bit 1 of coherent_acts re-enables every acquire (diagnostic).
__device__ __forceinline__ bool event_fenced(const RuntimeState& rt, int event) {
    if (!rt.coherent_acts || event == rt.done_event) return true;
#pragma unroll
    for (int i = 0; i < 4; ++i) if (rt.fenced_events[i] == event) return true;
    return false;
}
__device__ __forceinline__ bool event_release_fenced(const RuntimeState& rt, int event) {
    return true;   // see above: measured necessary
}
__device__ __forceinline__ bool event_acquire_fenced(const RuntimeState& rt, int event) {
    return (rt.coherent_acts & 2) || event_fenced(rt, event);
}

// Agent-scope coherent accesses for cross-XCD activations (coherent_acts).
__device__ __forceinline__ float ld_coh(const float* p, bool coh) {
    return coh ? __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : *p;
}
__device__ __forceinline__ int32_t ld_coh(const int32_t* p, bool coh) {
    return coh ? __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : *p;
}
__device__ __forceinline__ void st_coh(float* p, float v, bool coh) {
    if (coh) __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); else *p = v;
}
__device__ __forceinline__ void st_coh(int32_t* p, int32_t v, bool coh) {
    if (coh) __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); else *p = v;
}

__device__ __forceinline__ bool aborted(const RuntimeState& rt) {
    return poll(rt.abort) != kAbortNone;
}

__device__ __forceinline__ void raise_abort(const RuntimeState& rt, uint32_t code) {
    // First reason wins; later ones are consequences of it.
    uint32_t expected = kAbortNone;
    __hip_atomic_compare_exchange_strong(rt.abort, &expected, code,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT);
}

// ---------------------------------------------------------------- roles
//
// Which workgroup is the scheduler and which are workers is decided by an
// XCD-local ticket, not by blockIdx: the hardware places blocks on XCDs
// round-robin, but nothing guarantees blockIdx 0..7 land one per XCD, and an
// XCD with no scheduler would hang every worker on it. The ticket makes the
// role follow the physical placement. Returns -1 for the scheduler, else the
// worker id; the caller must then pass claim_role's grid barrier, which
// converts a wrong distribution (an XCD with != 38 blocks) into an abort.
__device__ __forceinline__ int claim_role(const RuntimeState& rt, int xcd,
                                          int* slot_smem) {
    if (threadIdx.x == 0) {
        *slot_smem = (int)atomicAdd(&rt.xcd_arrivals[xcd], 1u);
        atomicAdd(rt.grid_arrivals, 1u);
    }
    __syncthreads();
    const int slot = *slot_smem;

    // Grid barrier: cooperative launch guarantees residency, so this cannot
    // deadlock unless fewer than kGrid blocks were launched at all.
    if (threadIdx.x == 0) {
        uint32_t spins = 0;
        while (poll(rt.grid_arrivals) < (uint32_t)kGrid) {
            backoff();
            if (++spins > rt.spin_limit) { raise_abort(rt, kAbortBarrierTimeout); break; }
        }
        const uint32_t here = poll(&rt.xcd_arrivals[xcd]);
        if (here != (uint32_t)kBlocksPerXCD) {
            raise_abort(rt, kAbortXcdDistribution | ((uint32_t)xcd << 8) | (here & 0xff));
        }
    }
    __syncthreads();
    return slot - 1;
}

// ---------------------------------------------------------------- the three ops

// fetch_task: with AOT queues a worker already knows its whole task list, so
// this is an index lookup, not a dispatch. Returns nullptr when drained.
__device__ __forceinline__ const TaskDescriptor* fetch_task(
        const RuntimeState& rt, int xcd, int worker, int slot) {
    const int q = xcd * kWorkersPerXCD + worker;
    const int32_t begin = rt.queue_offset[q];
    const int32_t end = rt.queue_offset[q + 1];
    if (begin + slot >= end) return nullptr;
    return &rt.tasks[rt.queue_index[begin + slot]];
}

// Store draining. On gfx942 the L1 (TCP) is write-through, so once a wave's
// stores have completed (vscnt = 0) they are in this XCD's L2; no cache
// writeback is needed for a consumer on the same L2.
__device__ __forceinline__ void drain_stores() {
    __builtin_amdgcn_s_waitcnt(0);
}

// arrive_last: bump an XCD-local arrival counter after draining this
// workgroup's stores, and report whether this arrival completed the group
// for this epoch (the last KV chunk of a head then merges the partials, on
// the same XCD). The caller's fence_acquire() makes the others' data visible.
__device__ __forceinline__ bool arrive_last(
        const RuntimeState& rt, uint32_t epoch, int counter, int xcd, int members) {
    drain_stores();
    const uint32_t old = __hip_atomic_fetch_add(
        &rt.xcd_counters[xcd * kMaxEvents + counter], 1u,
        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    return old + 1 == epoch * (uint32_t)members;
}

// signal_event: publish this task's stores, then count. Called by one thread
// after a __syncthreads(), so every wave's stores precede the drain.
//
// XCD-local events: drain, then bump the XCD-local counter. The consumer is
// on the same L2, so the drained stores are already where it reads.
//
// Global events (design.md §4, the Fleet scheme): drain, then bump this
// XCD's arrival counter for the event; the arrival that completes this
// XCD's share (xcd_count producers per epoch) performs the one L2 writeback
// — which covers every earlier arrival's drained stores, as their atomics
// are ordered before it in L2 — and adds the whole share to the global
// counter. One buffer_wbl2 per XCD per event instead of one per producer:
// measured on the expert phase, per-producer flushes cost 13 us per layer,
// the last-arriver flush 6 us (bench/microbench.hip (f)); the payload test
// (d) checks the scheme publishes every word. Scheme (ii) (uncached
// counters) uses the same code path.
// Returns the global counter's value after this XCD's share was added, if
// this call added it; else 0. A caller that sees the epoch's full producer
// count knows it is the globally last producer (unused today; kept for diagnostics).
__device__ __forceinline__ uint32_t signal_event(
        const RuntimeState& rt, uint32_t epoch, int event, EventScope scope,
        int xcd, int xcd_count) {
    if (event < 0) return 0;
    drain_stores();
    uint32_t* local = &rt.xcd_counters[xcd * kMaxEvents + event];
    const uint32_t old = __hip_atomic_fetch_add(local, 1u, __ATOMIC_RELAXED,
                                                __HIP_MEMORY_SCOPE_AGENT);
    if (scope == SCOPE_XCD_LOCAL) return 0;
    if (old + 1 == epoch * (uint32_t)xcd_count) {        // last of this XCD's share
        if (event_release_fenced(rt, event)) fence_release();
        const uint32_t g = __hip_atomic_fetch_add(&rt.global_events[event], (uint32_t)xcd_count,
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        return g + (uint32_t)xcd_count;
    }
    return 0;
}

// wait_event: block until the event has `producers` signals for this epoch.
// Global events are read from this XCD's mirror (written by its scheduler);
// XCD-local events are read from the local counter directly. Called by one
// thread; the caller's __syncthreads() releases the other waves, and the
// acquire fence here is what makes the producers' data visible to them.
//
// Returns false if the wait was abandoned because of an abort (this worker's
// own timeout or someone else's), so the worker can drain and exit instead of
// hanging the launch.
__device__ __forceinline__ bool wait_event(
        const RuntimeState& rt, uint32_t epoch, int event, EventScope scope,
        int xcd, int producers) {
    if (event < 0) return true;
    const uint32_t target = epoch * (uint32_t)producers;
    uint32_t* p = (scope == SCOPE_XCD_LOCAL)
        ? &rt.xcd_counters[xcd * kMaxEvents + event]
        : (rt.direct_poll ? &rt.global_events[event]
                          : &rt.xcd_flags[xcd * kMaxEvents + event]);
    uint32_t spins = 0;
    while (poll(p) < target) {
        backoff();
        if ((++spins & 1023u) == 0) {
            if (aborted(rt)) return false;
            if (spins > rt.spin_limit) {
                raise_abort(rt, kAbortWaitTimeout | (uint32_t)event);
                return false;
            }
        }
    }
    if (scope == SCOPE_XCD_LOCAL) fence_acquire_local();
    else if (event_acquire_fenced(rt, event)) fence_acquire();
    return true;
}

// ---------------------------------------------------------------- scheduler
//
// The per-XCD scheduler's entire job: mirror the global counters (only the
// global-scope events, from the compact id list) into this XCD's flag array
// so its 37 workers poll that instead of the global counters. Global-counter
// polling drops from 296 pollers to 8. It stops once the graph's final event
// reaches this epoch — every task precedes that event transitively, so
// nothing can still be waiting.
__device__ __forceinline__ void run_scheduler(const RuntimeState& rt, int xcd) {
    asm volatile("s_setprio 3");     // scheduler waves win arbitration
    const int lane = threadIdx.x;
    uint32_t* mirror = rt.xcd_flags + xcd * kMaxEvents;
    const uint32_t done_target = rt.epoch0 + (uint32_t)rt.n_tokens - 1;
    __shared__ int stop;
    if (lane == 0) stop = 0;
    __syncthreads();
    while (!stop) {
        // Relay with a formal happens-before chain: the producer's release
        // fence -> its counter increment -> our relaxed read -> our acquire
        // fence -> our release fence -> our mirror store -> the worker's
        // relaxed read of the mirror -> the worker's acquire fence. Without
        // the two fences here the worker would acquire on an object the
        // producer never released on, and the payload's visibility would rest
        // on cache behaviour rather than on the memory model.
        bool changed = false;
        for (int k = lane; k < rt.n_global_events; k += blockDim.x) {
            const int e = rt.global_event_ids[k];
            const uint32_t g = poll(&rt.global_events[e]);
            if (poll(&mirror[e]) != g) {
                if (!changed) {
                    if (!rt.coherent_acts) { fence_acquire(); fence_release(); }
                    changed = true;
                }
                __hip_atomic_store(&mirror[e], g, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        backoff();
        if (lane == 0 && (poll(&rt.global_events[rt.done_event]) >= done_target ||
                          aborted(rt))) {
            stop = 1;
        }
        __syncthreads();
    }
    asm volatile("s_setprio 0");
}

}  // namespace fleet
