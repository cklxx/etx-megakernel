// Generic host launcher for one ETX step (HIP). Include plan_data.h before this
// file. Allocates the argument table from the plan, uploads descriptors and
// queues, initialises events, launches the persistent kernel cooperatively,
// waits with a watchdog, and on timeout dumps every event counter that did not
// reach zero (design §13: hangs must be diagnosable, never silent).
#pragma once
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include "etx/abi.h"

#define ETX_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(_e)); exit(1); } } while (0)

extern "C" __global__ void etx_megakernel(etx_params p);

struct etx_host {
  std::vector<void*> args;                 // device pointers, plan order
  etx_params p{};
  int32_t *d_events = nullptr, *d_ctrl = nullptr, *d_slots = nullptr, *d_trace = nullptr;
  int32_t *d_lq_slots = nullptr, *d_lq_head = nullptr, *d_lq_tail = nullptr, *d_gq_slots = nullptr;
  etx_queue* d_local_queues = nullptr;
  double last_ms = 0.0;

  void alloc_args() {
    args.resize(ETX_N_ARGS);
    for (int i = 0; i < ETX_N_ARGS; ++i) {
      ETX_CHECK(hipMalloc(&args[i], etx_arg_bytes[i] ? etx_arg_bytes[i] : 16));
      ETX_CHECK(hipMemset(args[i], 0, etx_arg_bytes[i] ? etx_arg_bytes[i] : 16));
    }
  }
  int arg_index(const char* name) const {
    for (int i = 0; i < ETX_N_ARGS; ++i) if (!strcmp(etx_arg_names[i], name)) return i;
    fprintf(stderr, "unknown arg %s\n", name); exit(1);
  }
  template <class T> static T* upload(const T* src, size_t n) {
    T* d; ETX_CHECK(hipMalloc(&d, n * sizeof(T))); ETX_CHECK(hipMemcpy(d, src, n * sizeof(T), hipMemcpyHostToDevice)); return d;
  }
  void setup() {
    if (args.empty()) alloc_args();
    ETX_CHECK(hipMalloc(&p.args, ETX_N_ARGS * sizeof(void*)));
    ETX_CHECK(hipMemcpy((void*)p.args, args.data(), ETX_N_ARGS * sizeof(void*), hipMemcpyHostToDevice));
    {   // per-type argument tables in each grid's own order
      std::vector<void*> ta(ETX_N_TYPES * ETX_MAX_ARGS, nullptr);
      for (int t = 0; t < ETX_N_TYPES; ++t)
        for (int k = 0; k < ETX_MAX_ARGS; ++k)
          if (etx_type_arg_index[t][k] >= 0) ta[t * ETX_MAX_ARGS + k] = args[etx_type_arg_index[t][k]];
      p.type_args = upload(ta.data(), ta.size());
      p.max_args = ETX_MAX_ARGS;
    }
    p.shape = upload(etx_shape, ETX_N_SYMBOLS ? ETX_N_SYMBOLS : 1);
    p.descs = upload(etx_descs, ETX_N_TASKS);
    p.static_queue = upload(etx_static_queue, ETX_STATIC_LEN);
    p.static_begin = upload(etx_static_begin, ETX_N_WORKERS);
    p.static_end = upload(etx_static_end, ETX_N_WORKERS);
    p.ev_offset = upload(etx_ev_offset, ETX_N_EVENTS ? ETX_N_EVENTS : 1);
    p.ev_shape = (const int32_t (*)[4])upload(&etx_ev_shape[0][0], 4 * (ETX_N_EVENTS ? ETX_N_EVENTS : 1));
    d_events = upload(etx_ev_counts, ETX_EVENT_WORDS);
    p.events = d_events;
    p.push_index = upload(etx_push_index, sizeof(etx_push_index) / sizeof(int32_t));
    p.push_offsets = upload(etx_push_offsets, sizeof(etx_push_offsets) / sizeof(int32_t));
    p.push_lists = upload(etx_push_lists, sizeof(etx_push_lists) / sizeof(int32_t));
    // local queues: one ring per domain
    int32_t total = 0; for (int d = 0; d < ETX_N_DOMAINS; ++d) total += etx_local_capacity[d];
    ETX_CHECK(hipMalloc(&d_lq_slots, (total ? total : 1) * sizeof(int32_t)));
    ETX_CHECK(hipMemset(d_lq_slots, 0xFF, (total ? total : 1) * sizeof(int32_t)));          // -1 = empty
    ETX_CHECK(hipMalloc(&d_lq_head, 2 * ETX_N_DOMAINS * sizeof(int32_t)));
    ETX_CHECK(hipMemset(d_lq_head, 0, 2 * ETX_N_DOMAINS * sizeof(int32_t)));
    d_lq_tail = d_lq_head + ETX_N_DOMAINS;
    std::vector<etx_queue> lq(ETX_N_DOMAINS);
    int32_t off = 0;
    for (int d = 0; d < ETX_N_DOMAINS; ++d) { lq[d] = etx_queue{d_lq_slots + off, etx_local_capacity[d], d_lq_head + d, d_lq_tail + d}; off += etx_local_capacity[d]; }
    d_local_queues = upload(lq.data(), ETX_N_DOMAINS);
    p.local_queue = d_local_queues;
    ETX_CHECK(hipMalloc(&d_gq_slots, (ETX_GLOBAL_CAPACITY ? ETX_GLOBAL_CAPACITY : 1) * sizeof(int32_t)));
    ETX_CHECK(hipMemset(d_gq_slots, 0xFF, (ETX_GLOBAL_CAPACITY ? ETX_GLOBAL_CAPACITY : 1) * sizeof(int32_t)));
    int32_t* gq_ht; ETX_CHECK(hipMalloc(&gq_ht, 2 * sizeof(int32_t))); ETX_CHECK(hipMemset(gq_ht, 0, 8));
    p.global_queue = etx_queue{d_gq_slots, ETX_GLOBAL_CAPACITY, gq_ht, gq_ht + 1};
    // control words in fine-grained device memory so the host can read/write them while the kernel runs
    ETX_CHECK(hipExtMallocWithFlags((void**)&d_ctrl, 64, hipDeviceMallocFinegrained));
    ETX_CHECK(hipMemset(d_ctrl, 0, 64));
    p.ctrl_abort = d_ctrl; p.ctrl_done = d_ctrl + 1;
    ETX_CHECK(hipMalloc(&d_slots, ETX_N_DOMAINS * sizeof(int32_t)));
    ETX_CHECK(hipMemset(d_slots, 0, ETX_N_DOMAINS * sizeof(int32_t)));
    p.domain_slots = d_slots;
    p.worker_domain = nullptr;
    ETX_CHECK(hipMalloc(&d_trace, ETX_N_TASKS * sizeof(int32_t)));
    p.trace_exec = d_trace;
    p.workers_per_domain = ETX_WORKERS_PER_DOMAIN;
    p.n_workers = ETX_N_WORKERS;
    p.n_tasks = ETX_N_TASKS;
    p.n_events = ETX_N_EVENTS;
    p.spin_limit = 200000000u;
  }
  // reset per-step state (events, queues, ctrl, slots)
  void reset_step() {
    ETX_CHECK(hipMemcpy(d_events, etx_ev_counts, ETX_EVENT_WORDS * sizeof(int32_t), hipMemcpyHostToDevice));
    ETX_CHECK(hipMemset(d_ctrl, 0, 64));
    ETX_CHECK(hipMemset(d_trace, 0, ETX_N_TASKS * sizeof(int32_t)));
    ETX_CHECK(hipMemset(d_slots, 0, ETX_N_DOMAINS * sizeof(int32_t)));
    // rings: clear slots, then seed the initially-ready dynamic/hybrid tasks and set the tails
    int32_t total = 0; for (int d = 0; d < ETX_N_DOMAINS; ++d) total += etx_local_capacity[d];
    if (total) ETX_CHECK(hipMemset(d_lq_slots, 0xFF, total * sizeof(int32_t)));
    std::vector<int32_t> heads(ETX_N_DOMAINS, 0), tails(ETX_N_DOMAINS, 0);
    int32_t off = 0, src = 0;
    for (int d = 0; d < ETX_N_DOMAINS; ++d) {
      const int32_t n = etx_init_local_len[d];
      if (n) ETX_CHECK(hipMemcpy(d_lq_slots + off, etx_init_local + src, n * sizeof(int32_t), hipMemcpyHostToDevice));
      tails[d] = n; src += n; off += etx_local_capacity[d];
    }
    ETX_CHECK(hipMemcpy(d_lq_head, heads.data(), ETX_N_DOMAINS * sizeof(int32_t), hipMemcpyHostToDevice));
    ETX_CHECK(hipMemcpy(d_lq_tail, tails.data(), ETX_N_DOMAINS * sizeof(int32_t), hipMemcpyHostToDevice));
    if (ETX_GLOBAL_CAPACITY) ETX_CHECK(hipMemset(d_gq_slots, 0xFF, ETX_GLOBAL_CAPACITY * sizeof(int32_t)));
    if (ETX_INIT_GLOBAL_LEN) ETX_CHECK(hipMemcpy(d_gq_slots, etx_init_global, ETX_INIT_GLOBAL_LEN * sizeof(int32_t), hipMemcpyHostToDevice));
    int32_t gq[2] = {0, ETX_INIT_GLOBAL_LEN};
    ETX_CHECK(hipMemcpy((void*)p.global_queue.head, gq, 8, hipMemcpyHostToDevice));
  }
  // returns true on success; false on watchdog timeout (with event dump)
  bool run(double timeout_s = 5.0) {
    reset_step();
    hipStream_t s; ETX_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    void* kargs[] = {&p};
    dim3 grid(ETX_N_WORKERS), block(ETX_THREADS);
    int dev = 0; ETX_CHECK(hipGetDevice(&dev));
    int max_blocks = 0;
    ETX_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (const void*)etx_megakernel, ETX_THREADS, 0));
    hipDeviceProp_t prop; ETX_CHECK(hipGetDeviceProperties(&prop, dev));
    if ((long)max_blocks * prop.multiProcessorCount < ETX_N_WORKERS) {
      fprintf(stderr, "co-residency: %d workers requested, only %d x %d resident possible\n", ETX_N_WORKERS, max_blocks, prop.multiProcessorCount);
      return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    ETX_CHECK(hipLaunchCooperativeKernel((const void*)etx_megakernel, grid, block, kargs, 0, s));
    // watchdog
    while (hipStreamQuery(s) == hipErrorNotReady) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > timeout_s) {
        int32_t one = 1; ETX_CHECK(hipMemcpy(d_ctrl, &one, 4, hipMemcpyHostToDevice));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        dump_events();
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ETX_CHECK(hipStreamSynchronize(s));
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    int32_t ctrl[2]; ETX_CHECK(hipMemcpy(ctrl, d_ctrl, 8, hipMemcpyDeviceToHost));
    if (ctrl[0] & 2) { fprintf(stderr, "kernel hit spin limit (abort=%d, done=%d/%d)\n", ctrl[0], ctrl[1], ETX_N_TASKS); dump_events(); return false; }
    if (ctrl[1] != ETX_N_TASKS) { fprintf(stderr, "done=%d of %d tasks\n", ctrl[1], ETX_N_TASKS); dump_events(); return false; }
    return true;
  }
  void dump_events() {
    dump_trace();
    std::vector<int32_t> ev(ETX_EVENT_WORDS);
    ETX_CHECK(hipMemcpy(ev.data(), d_events, ETX_EVENT_WORDS * sizeof(int32_t), hipMemcpyDeviceToHost));
    int shown = 0;
    for (int e = 0; e < ETX_N_EVENTS; ++e) {
      int n = 1; for (int d = 0; d < 4; ++d) if (etx_ev_shape[e][d] > 0) n *= etx_ev_shape[e][d];
      for (int i = 0; i < n; ++i) if (ev[etx_ev_offset[e] + i] != 0 && shown++ < 32)
        fprintf(stderr, "  event %d[%d] = %d (initial %d)\n", e, i, ev[etx_ev_offset[e] + i], etx_ev_counts[etx_ev_offset[e] + i]);
    }
    if (shown > 32) fprintf(stderr, "  ... %d non-zero events\n", shown);
    if (shown == 0) fprintf(stderr, "  all events reached zero\n");
  }
  // execution-count histogram + queue state (debug)
  void dump_trace() {
    std::vector<int32_t> tr(ETX_N_TASKS);
    ETX_CHECK(hipMemcpy(tr.data(), d_trace, ETX_N_TASKS * sizeof(int32_t), hipMemcpyDeviceToHost));
    int hist[4] = {0, 0, 0, 0}; int first_dup = -1;
    for (int i = 0; i < ETX_N_TASKS; ++i) { int c = tr[i] < 3 ? tr[i] : 3; hist[c]++; if (tr[i] > 1 && first_dup < 0) first_dup = i; }
    fprintf(stderr, "  exec counts: 0x=%d 1x=%d 2x=%d 3x+=%d%s", hist[0], hist[1], hist[2], hist[3], first_dup >= 0 ? " first dup task " : "\n");
    if (first_dup >= 0) fprintf(stderr, "%d (type %d)\n", first_dup, etx_descs[first_dup].type);
    std::vector<int32_t> ht(2 * ETX_N_DOMAINS);
    ETX_CHECK(hipMemcpy(ht.data(), d_lq_head, 2 * ETX_N_DOMAINS * sizeof(int32_t), hipMemcpyDeviceToHost));
    fprintf(stderr, "  local queues head/tail:");
    for (int d = 0; d < ETX_N_DOMAINS; ++d) fprintf(stderr, " %d/%d", ht[d], ht[ETX_N_DOMAINS + d]);
    int32_t gq[2]; ETX_CHECK(hipMemcpy(gq, p.global_queue.head, 8, hipMemcpyDeviceToHost));
    fprintf(stderr, "  global %d/%d (cap %d)\n", gq[0], gq[1], ETX_GLOBAL_CAPACITY);
  }
};
