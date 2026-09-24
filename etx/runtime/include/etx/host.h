// Generic host launcher for one ETX step (HIP), one instance per device.
// Include plan_data.h before this file. Allocates the argument table from the
// plan (or takes pre-filled device pointers), uploads descriptors and queues,
// initialises events (a shared buffer across devices when events are
// system-scope), launches the persistent kernel cooperatively, waits with a
// watchdog, and on timeout dumps every event counter that did not reach zero
// plus a per-task execution histogram (design §13: hangs must be diagnosable).
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

struct etx_host {
  int device = 0;
  const void* kernel = nullptr;            // etx_megakernel_d<device>, set by the example host
  std::vector<void*> args;                 // device pointers, plan order (pre-fill to control placement)
  etx_params p{};
  int32_t *d_events = nullptr, *d_ctrl = nullptr, *d_slots = nullptr, *d_trace = nullptr, *d_remaining = nullptr;
  uint64_t* d_trace_time = nullptr;
  int32_t* d_ev_sub = nullptr;
  int32_t* d_mirror = nullptr;             // relay mirrors [n_domains][event words]
  std::vector<int32_t> mirror_init;
  int32_t *d_lq_slots = nullptr, *d_lq_head = nullptr, *d_lq_tail = nullptr, *d_gq_slots = nullptr;
  etx_queue* d_local_queues = nullptr;
  bool owns_events = true;
  hipStream_t stream = nullptr;
  double last_ms = 0.0;

  explicit etx_host(int dev = 0) : device(dev) {}

  void alloc_args() {
    ETX_CHECK(hipSetDevice(device));
    args.resize(ETX_N_ARGS);
    for (int i = 0; i < ETX_N_ARGS; ++i) {
      ETX_CHECK(hipMalloc(&args[i], etx_arg_bytes[i] ? etx_arg_bytes[i] : 16));
      ETX_CHECK(hipMemset(args[i], 0, etx_arg_bytes[i] ? etx_arg_bytes[i] : 16));
    }
  }
  static int arg_index(const char* name) {
    for (int i = 0; i < ETX_N_ARGS; ++i) if (!strcmp(etx_arg_names[i], name)) return i;
    fprintf(stderr, "unknown arg %s\n", name); exit(1);
  }
  template <class T> static T* upload(const T* src, size_t n) {
    T* d; ETX_CHECK(hipMalloc(&d, n * sizeof(T))); ETX_CHECK(hipMemcpy(d, src, n * sizeof(T), hipMemcpyHostToDevice)); return d;
  }
  // events: allocate the shared buffer (fine-grained if any event is system scope)
  static int32_t* alloc_events(int on_device) {
    ETX_CHECK(hipSetDevice(on_device));
    int32_t* e;
    if (ETX_EVENTS_FINE_GRAINED) ETX_CHECK(hipExtMallocWithFlags((void**)&e, ETX_EVENT_WORDS * sizeof(int32_t), hipDeviceMallocFinegrained));
    else ETX_CHECK(hipMalloc(&e, ETX_EVENT_WORDS * sizeof(int32_t)));
    return e;
  }
  void setup(int32_t* shared_events = nullptr) {
    ETX_CHECK(hipSetDevice(device));
    ETX_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    if (args.empty()) alloc_args();
    ETX_CHECK(hipMalloc(&p.args, ETX_N_ARGS * sizeof(void*)));
    ETX_CHECK(hipMemcpy((void*)p.args, args.data(), ETX_N_ARGS * sizeof(void*), hipMemcpyHostToDevice));
    {
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
    {
      std::vector<etx_task> sd(ETX_STATIC_LEN);
      for (int i = 0; i < ETX_STATIC_LEN; ++i) sd[i] = etx_descs[etx_static_queue[i] >= 0 ? etx_static_queue[i] : 0];
      p.static_descs = upload(sd.data(), sd.size());
    }
    p.static_begin = upload(etx_static_begin + device * ETX_N_WORKERS, ETX_N_WORKERS);
    p.static_end = upload(etx_static_end + device * ETX_N_WORKERS, ETX_N_WORKERS);
    p.ev_offset = upload(etx_ev_offset, ETX_N_EVENTS ? ETX_N_EVENTS : 1);
    p.ev_shape = (const int32_t (*)[4])upload(&etx_ev_shape[0][0], 4 * (ETX_N_EVENTS ? ETX_N_EVENTS : 1));
    if (shared_events) { d_events = shared_events; owns_events = false; }
    else d_events = alloc_events(device);
    ETX_CHECK(hipSetDevice(device));
    p.events = d_events;
    p.push_index = upload(etx_push_index, sizeof(etx_push_index) / sizeof(int32_t));
    p.push_offsets = upload(etx_push_offsets, sizeof(etx_push_offsets) / sizeof(int32_t));
    p.push_lists = upload(etx_push_lists, sizeof(etx_push_lists) / sizeof(int32_t));
    int32_t total = 0; for (int d = 0; d < ETX_N_DOMAINS; ++d) total += etx_local_capacity[device * ETX_N_DOMAINS + d];
    ETX_CHECK(hipMalloc(&d_lq_slots, (total ? total : 1) * sizeof(int32_t)));
    ETX_CHECK(hipMalloc(&d_lq_head, 2 * ETX_N_DOMAINS * sizeof(int32_t)));
    d_lq_tail = d_lq_head + ETX_N_DOMAINS;
    std::vector<etx_queue> lq(ETX_N_DOMAINS);
    int32_t off = 0;
    for (int d = 0; d < ETX_N_DOMAINS; ++d) {
      const int32_t cap = etx_local_capacity[device * ETX_N_DOMAINS + d];
      lq[d] = etx_queue{d_lq_slots + off, cap, d_lq_head + d, d_lq_tail + d}; off += cap;
    }
    d_local_queues = upload(lq.data(), ETX_N_DOMAINS);
    p.local_queue = d_local_queues;
    const int32_t gcap = etx_global_capacity[device];
    ETX_CHECK(hipMalloc(&d_gq_slots, (gcap ? gcap : 1) * sizeof(int32_t)));
    int32_t* gq_ht; ETX_CHECK(hipMalloc(&gq_ht, 2 * sizeof(int32_t)));
    p.global_queue = etx_queue{d_gq_slots, gcap, gq_ht, gq_ht + 1};
    ETX_CHECK(hipExtMallocWithFlags((void**)&d_ctrl, 64, hipDeviceMallocFinegrained));
    p.ctrl_abort = d_ctrl; p.ctrl_done = d_ctrl + 1;
    ETX_CHECK(hipMalloc(&d_slots, ETX_N_DOMAINS * sizeof(int32_t)));
    p.domain_slots = d_slots;
    p.worker_domain = nullptr;
    ETX_CHECK(hipMalloc(&d_trace, ETX_N_TASKS * sizeof(int32_t)));
    p.trace_exec = d_trace;
    ETX_CHECK(hipMalloc(&d_remaining, ETX_N_TASKS * sizeof(int32_t)));
    p.task_remaining = d_remaining;
    p.trace_time = nullptr;
    if (getenv("ETX_TRACE")) { ETX_CHECK(hipMalloc(&d_trace_time, (size_t)ETX_N_TASKS * 4 * sizeof(uint64_t))); p.trace_time = d_trace_time; }
    // last-arriver flush tables: this device's share rows and zeroed sub-counters
    p.event_words = ETX_EVENT_WORDS;
    p.ev_share = upload(etx_ev_share + (size_t)device * ETX_N_DOMAINS * ETX_EVENT_WORDS, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS);
    ETX_CHECK(hipMalloc(&d_ev_sub, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS * sizeof(int32_t)));
    p.ev_sub = d_ev_sub;
    if (getenv("ETX_NO_LASTFLUSH")) p.ev_share = nullptr;      // A/B: per-producer write-back
    p.workers_per_domain = ETX_WORKERS_PER_DOMAIN;
    // relay (P5): ETX_RELAY=0 turns it off at run time (waits then poll the global words; A/B);
    // ETX_RELAY_ACQ=consumer makes every consumer do the full device acquire itself (fleet's choice)
    p.ev_mirror = nullptr; p.n_relay = 0; p.relay_words = nullptr; p.relay_local_acquire = 1;
    {
      const char* r = getenv("ETX_RELAY");
      const int32_t b = etx_relay_begin[device], e = etx_relay_begin[device + 1];
      if (ETX_RELAY && e > b && !(r && !strcmp(r, "0"))) {
        ETX_CHECK(hipMalloc(&d_mirror, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS * sizeof(int32_t)));
        mirror_init.resize((size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS);
        for (int d = 0; d < ETX_N_DOMAINS; ++d) memcpy(mirror_init.data() + (size_t)d * ETX_EVENT_WORDS, etx_ev_counts, ETX_EVENT_WORDS * sizeof(int32_t));
        p.ev_mirror = d_mirror; p.n_relay = e - b; p.relay_words = upload(etx_relay_words + b, (size_t)(e - b));
        const char* a = getenv("ETX_RELAY_ACQ");
        p.relay_local_acquire = (a && !strcmp(a, "consumer")) ? 0 : 1;
      }
    }
    p.n_workers = ETX_N_WORKERS;
    p.n_tasks = etx_n_tasks_dev[device];
    p.n_dynamic = etx_n_dynamic_dev[device];
    p.n_events = ETX_N_EVENTS;
    p.spin_limit = 200000000u;
    const char* lm = getenv("ETX_LAUNCH");                 // "ordinary": plain launch when the grid is known to fit
    cooperative = !(lm && !strcmp(lm, "ordinary"));
  }
  bool cooperative = true;
  // reset per-step state (events if owned, queues, ctrl, slots, counters)
  void reset_step() {
    ETX_CHECK(hipSetDevice(device));
    if (owns_events) ETX_CHECK(hipMemcpy(d_events, etx_ev_counts, ETX_EVENT_WORDS * sizeof(int32_t), hipMemcpyHostToDevice));
    ETX_CHECK(hipMemset(d_ctrl, 0, 64));
    ETX_CHECK(hipMemset(d_slots, 0, ETX_N_DOMAINS * sizeof(int32_t)));
    ETX_CHECK(hipMemset(d_trace, 0, ETX_N_TASKS * sizeof(int32_t)));
    ETX_CHECK(hipMemcpy(d_remaining, etx_task_remaining, ETX_N_TASKS * sizeof(int32_t), hipMemcpyHostToDevice));
    ETX_CHECK(hipMemset(d_ev_sub, 0, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS * sizeof(int32_t)));
    if (d_mirror) ETX_CHECK(hipMemcpy(d_mirror, mirror_init.data(), mirror_init.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    int32_t total = 0; for (int d = 0; d < ETX_N_DOMAINS; ++d) total += etx_local_capacity[device * ETX_N_DOMAINS + d];
    if (total) ETX_CHECK(hipMemset(d_lq_slots, 0xFF, total * sizeof(int32_t)));
    std::vector<int32_t> heads(ETX_N_DOMAINS, 0), tails(ETX_N_DOMAINS, 0);
    int32_t off = 0, src = 0;
    for (int dd = 0; dd < device * ETX_N_DOMAINS; ++dd) src += etx_init_local_len[dd];
    for (int d = 0; d < ETX_N_DOMAINS; ++d) {
      const int32_t n = etx_init_local_len[device * ETX_N_DOMAINS + d];
      if (n) ETX_CHECK(hipMemcpy(d_lq_slots + off, etx_init_local + src, n * sizeof(int32_t), hipMemcpyHostToDevice));
      tails[d] = n; src += n; off += etx_local_capacity[device * ETX_N_DOMAINS + d];
    }
    ETX_CHECK(hipMemcpy(d_lq_head, heads.data(), ETX_N_DOMAINS * sizeof(int32_t), hipMemcpyHostToDevice));
    ETX_CHECK(hipMemcpy(d_lq_tail, tails.data(), ETX_N_DOMAINS * sizeof(int32_t), hipMemcpyHostToDevice));
    const int32_t gcap = etx_global_capacity[device];
    if (gcap) ETX_CHECK(hipMemset(d_gq_slots, 0xFF, gcap * sizeof(int32_t)));
    int32_t gsrc = 0; for (int dd = 0; dd < device; ++dd) gsrc += etx_init_global_len[dd];
    const int32_t gn = etx_init_global_len[device];
    if (gn) ETX_CHECK(hipMemcpy(d_gq_slots, etx_init_global + gsrc, gn * sizeof(int32_t), hipMemcpyHostToDevice));
    int32_t gq[2] = {0, gn};
    ETX_CHECK(hipMemcpy((void*)p.global_queue.head, gq, 8, hipMemcpyHostToDevice));
  }
  bool check_residency() {
    ETX_CHECK(hipSetDevice(device));
    int max_blocks = 0;
    ETX_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, ETX_THREADS, 0));
    hipDeviceProp_t prop; ETX_CHECK(hipGetDeviceProperties(&prop, device));
    if ((long)max_blocks * prop.multiProcessorCount < ETX_N_LAUNCH) {
      fprintf(stderr, "device %d co-residency: %d workgroups requested, only %d x %d resident possible\n", device, ETX_N_LAUNCH, max_blocks, prop.multiProcessorCount);
      return false;
    }
    return true;
  }
  hipEvent_t ev_start = nullptr, ev_stop = nullptr;
  // asynchronous launch (call reset_step first; for multi-device, reset all then launch all, then wait all)
  void launch() {
    ETX_CHECK(hipSetDevice(device));
    if (!ev_start) { ETX_CHECK(hipEventCreate(&ev_start)); ETX_CHECK(hipEventCreate(&ev_stop)); }
    ETX_CHECK(hipEventRecord(ev_start, stream));
    void* kargs[] = {&p};
    if (cooperative) ETX_CHECK(hipLaunchCooperativeKernel(kernel, dim3(ETX_N_LAUNCH), dim3(ETX_THREADS), kargs, 0, stream));
    else ETX_CHECK(hipLaunchKernel(kernel, dim3(ETX_N_LAUNCH), dim3(ETX_THREADS), kargs, 0, stream));   // residency checked by check_residency()
    ETX_CHECK(hipEventRecord(ev_stop, stream));
  }
  // wait with watchdog; returns false on timeout / spin limit / incomplete
  bool wait(double timeout_s, std::chrono::steady_clock::time_point t0) {
    ETX_CHECK(hipSetDevice(device));
    while (hipStreamQuery(stream) == hipErrorNotReady) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > timeout_s) {
        int32_t one = 1; ETX_CHECK(hipMemcpy(d_ctrl, &one, 4, hipMemcpyHostToDevice));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        fprintf(stderr, "device %d: watchdog timeout after %.1f s\n", device, el);
        dump_events();
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ETX_CHECK(hipStreamSynchronize(stream));
    float ms = 0.f; ETX_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));   // device timestamps: launch to kernel end, no host polling jitter
    last_ms = ms;
    int32_t ctrl[2]; ETX_CHECK(hipMemcpy(ctrl, d_ctrl, 8, hipMemcpyDeviceToHost));
    if (ctrl[0] & 2) { fprintf(stderr, "device %d: kernel hit spin limit (abort=%d, done=%d/%d)\n", device, ctrl[0], ctrl[1], p.n_tasks); dump_events(); return false; }
    if (ctrl[1] != p.n_dynamic) { fprintf(stderr, "device %d: dynamic done=%d of %d\n", device, ctrl[1], p.n_dynamic); dump_events(); return false; }
    // static tasks are not counted on the device; the trace (one increment per executed task) verifies them
    std::vector<int32_t> tr(ETX_N_TASKS);
    ETX_CHECK(hipMemcpy(tr.data(), d_trace, ETX_N_TASKS * sizeof(int32_t), hipMemcpyDeviceToHost));
    int executed = 0;
    for (int i = 0; i < ETX_N_TASKS; ++i) if (tr[i] == 1) executed++;
    if (executed != p.n_tasks) { fprintf(stderr, "device %d: %d of %d tasks executed exactly once\n", device, executed, p.n_tasks); dump_events(); return false; }
    return true;
  }
  // single-device convenience
  bool run(double timeout_s = 5.0) {
    reset_step();
    if (!check_residency()) return false;
    auto t0 = std::chrono::steady_clock::now();
    launch();
    return wait(timeout_s, t0);
  }
  void dump_events() {
    dump_trace();
    ETX_CHECK(hipSetDevice(device));
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
  // Per-phase attribution from the timestamp trace (ETX_TRACE=1): for every task type
  // (grid) the mean wait (taken -> deps ready) and body (ready -> done) time, and the
  // phase span (first task taken -> last task done). Ticks are ETX_TIMER units
  // (100 MHz on AMD: 10 ns). `type_names[t]` labels the types (nullptr -> ids).
  void report_phases(const char* const* type_names, int n_types, double tick_ns = 10.0, FILE* out = stdout) {
    if (!d_trace_time) return;
    ETX_CHECK(hipSetDevice(device));
    std::vector<uint64_t> tr((size_t)ETX_N_TASKS * 4);
    ETX_CHECK(hipMemcpy(tr.data(), d_trace_time, tr.size() * sizeof(uint64_t), hipMemcpyDeviceToHost));
    if (const char* raw = getenv("ETX_TRACE_RAW")) {   // per task: type, coord[0..3], taken, ready, done, worker (int64 x 9)
      FILE* f = fopen(raw, "wb");
      if (f) {
        for (int i = 0; i < ETX_N_TASKS; ++i) {
          int64_t row[9] = {etx_descs[i].type, etx_descs[i].coord[0], etx_descs[i].coord[1], etx_descs[i].coord[2], etx_descs[i].coord[3],
                            (int64_t)tr[i * 4], (int64_t)tr[i * 4 + 1], (int64_t)tr[i * 4 + 2], (int64_t)tr[i * 4 + 3]};
          fwrite(row, sizeof(row), 1, f);
        }
        fclose(f);
      }
    }
    std::vector<double> wait(n_types, 0), body(n_types, 0); std::vector<int> cnt(n_types, 0);
    std::vector<uint64_t> first_taken(n_types, ~0ull), first_ready(n_types, ~0ull), last_ready(n_types, 0), last_done(n_types, 0);
    uint64_t t_min = ~0ull, t_max = 0;
    for (int i = 0; i < ETX_N_TASKS; ++i) {
      const uint64_t* r = &tr[(size_t)i * 4];
      if (r[0] == 0 || r[2] == 0) continue;
      const int ty = etx_descs[i].type; if (ty < 0 || ty >= n_types) continue;
      wait[ty] += (double)(r[1] - r[0]); body[ty] += (double)(r[2] - r[1]); cnt[ty]++;
      first_taken[ty] = std::min(first_taken[ty], r[0]); first_ready[ty] = std::min(first_ready[ty], r[1]);
      last_ready[ty] = std::max(last_ready[ty], r[1]); last_done[ty] = std::max(last_done[ty], r[2]);
      t_min = std::min(t_min, r[0]); t_max = std::max(t_max, r[2]);
    }
    const double k = tick_ns / 1000.0;
    fprintf(out, "phase trace (us): total span %.1f; per type: mean wait/body, then first-ready, last-ready, last-done relative to the type's first-ready (fleet's timeline columns)\n", (double)(t_max - t_min) * k);
    fprintf(out, "  %-22s %6s %9s %9s %11s %10s %10s\n", "type", "tasks", "wait", "body", "first-ready", "last-ready", "last-done");
    for (int ty = 0; ty < n_types; ++ty) if (cnt[ty])
      fprintf(out, "  %-22s %6d %9.2f %9.2f %11.1f %10.1f %10.1f\n", type_names ? type_names[ty] : "", cnt[ty],
              wait[ty] / cnt[ty] * k, body[ty] / cnt[ty] * k, (double)(first_ready[ty] - t_min) * k,
              (double)(last_ready[ty] - first_ready[ty]) * k, (double)(last_done[ty] - first_ready[ty]) * k);
  }
  void dump_trace() {
    ETX_CHECK(hipSetDevice(device));
    std::vector<int32_t> tr(ETX_N_TASKS);
    ETX_CHECK(hipMemcpy(tr.data(), d_trace, ETX_N_TASKS * sizeof(int32_t), hipMemcpyDeviceToHost));
    int hist[4] = {0, 0, 0, 0}; int first_dup = -1;
    for (int i = 0; i < ETX_N_TASKS; ++i) { int c = tr[i] < 3 ? tr[i] : 3; hist[c]++; if (tr[i] > 1 && first_dup < 0) first_dup = i; }
    fprintf(stderr, "  device %d exec counts: 0x=%d 1x=%d 2x=%d 3x+=%d%s", device, hist[0], hist[1], hist[2], hist[3], first_dup >= 0 ? " first dup task " : "\n");
    if (first_dup >= 0) fprintf(stderr, "%d (type %d)\n", first_dup, etx_descs[first_dup].type);
    std::vector<int32_t> ht(2 * ETX_N_DOMAINS);
    ETX_CHECK(hipMemcpy(ht.data(), d_lq_head, 2 * ETX_N_DOMAINS * sizeof(int32_t), hipMemcpyDeviceToHost));
    fprintf(stderr, "  local queues head/tail:");
    for (int d = 0; d < ETX_N_DOMAINS; ++d) fprintf(stderr, " %d/%d", ht[d], ht[ETX_N_DOMAINS + d]);
    int32_t gq[2]; ETX_CHECK(hipMemcpy(gq, p.global_queue.head, 8, hipMemcpyDeviceToHost));
    fprintf(stderr, "  global %d/%d (cap %d)\n", gq[0], gq[1], etx_global_capacity[device]);
  }
};
