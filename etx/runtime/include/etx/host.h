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

#define ETX_SLOT_WORDS (ETX_N_DOMAINS * (2 + 2 * 256))

struct etx_host {
  int device = 0;
  const void* kernel = nullptr;            // etx_megakernel_d<device>, set by the example host
  std::vector<void*> args;                 // device pointers, plan order (pre-fill to control placement)
  etx_params p{};
  std::vector<int32_t> ev_strided;   // initial counts at their strided positions (ETX_EV_STRIDE > 1)
  int32_t *d_events = nullptr, *d_ctrl = nullptr, *d_slots = nullptr, *d_trace = nullptr, *d_remaining = nullptr;
  bool exec_count = false;
  int32_t gq_init[2] = {0, -1};             // persistent source for the global-queue head/tail copy
  std::vector<int32_t> lq_heads, lq_tails;   // persistent: hipMemcpyAsync sources must outlive the call          // ETX_EXEC_COUNT / ETX_TRACE: per-task execution counter (fused path only)
  uint64_t* d_trace_time = nullptr;
#if defined(ETX_INBODY) && ETX_INBODY
  uint64_t *d_inbody_poll = nullptr, *d_inbody_event = nullptr, *d_inbody_wave = nullptr;
#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER
  uint64_t* d_arrival_stamps = nullptr;
#endif
#endif
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
    if (ETX_EVENTS_FINE_GRAINED) ETX_CHECK(hipExtMallocWithFlags((void**)&e, (size_t)ETX_EVENT_WORDS * ETX_EV_STRIDE * sizeof(int32_t), hipDeviceMallocFinegrained));
    else ETX_CHECK(hipMalloc(&e, (size_t)ETX_EVENT_WORDS * ETX_EV_STRIDE * sizeof(int32_t)));
    ETX_CHECK(hipMemset(e, 0, (size_t)ETX_EVENT_WORDS * ETX_EV_STRIDE * sizeof(int32_t)));
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
    // domain slots, then (ETX_SLOT_BY_CU) per-domain CU count, per-(domain, CU key) arrival count and CU key -> dense index
    ETX_CHECK(hipMalloc(&d_slots, ETX_SLOT_WORDS * sizeof(int32_t)));
    p.domain_slots = d_slots;
    p.worker_domain = nullptr;
    // Per-task execution counter: one device-scope atomic per task, on the FUSED path only (the unfused baseline
    // has none), and 32 task counters share a 128 B line while their tasks run concurrently on different XCDs -- so
    // it is instrumentation that taxes only one side of the comparison. Opt in with ETX_EXEC_COUNT=1 (the run then
    // verifies every task ran exactly once); off, correctness still rests on --check / --check-dsl and 32/32 tokens.
    ETX_CHECK(hipMalloc(&d_trace, ETX_N_TASKS * sizeof(int32_t)));
    exec_count = getenv("ETX_EXEC_COUNT") != nullptr || getenv("ETX_TRACE") != nullptr;
    p.trace_exec = exec_count ? d_trace : nullptr;
    ETX_CHECK(hipMalloc(&d_remaining, ETX_N_TASKS * sizeof(int32_t)));
    p.task_remaining = d_remaining;
    p.trace_time = nullptr;
    if (getenv("ETX_TRACE")) { ETX_CHECK(hipMalloc(&d_trace_time, (size_t)ETX_N_TASKS * 4 * sizeof(uint64_t))); p.trace_time = d_trace_time; }
#if defined(ETX_INBODY) && ETX_INBODY
    if (getenv("ETX_INBODY_RAW")) {
      if (!d_trace_time) { ETX_CHECK(hipMalloc(&d_trace_time, (size_t)ETX_N_TASKS * 4 * sizeof(uint64_t))); p.trace_time = d_trace_time; }
      ETX_CHECK(hipMalloc(&d_inbody_poll, (size_t)ETX_N_TASKS * sizeof(uint64_t)));
      ETX_CHECK(hipMalloc(&d_inbody_event, (size_t)ETX_EVENT_WORDS * 3 * sizeof(uint64_t)));
      ETX_CHECK(hipMalloc(&d_inbody_wave, (size_t)ETX_N_TASKS * (ETX_THREADS / 64) * 2 * sizeof(uint64_t)));
      p.inbody_poll = d_inbody_poll; p.inbody_event = d_inbody_event; p.inbody_wave = d_inbody_wave;
#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER
      p.arrival_offsets = upload(etx_arrival_offsets, (size_t)ETX_N_TASKS + 1);
      p.arrival_words = upload(etx_arrival_words, (size_t)(ETX_ARRIVAL_SLOTS ? ETX_ARRIVAL_SLOTS : 1));
      ETX_CHECK(hipMalloc(&d_arrival_stamps, (size_t)(ETX_ARRIVAL_SLOTS ? ETX_ARRIVAL_SLOTS : 1) * 2 * sizeof(uint64_t)));
      p.arrival_stamps = d_arrival_stamps;
#endif
    }
#endif
    // last-arriver flush tables: this device's share rows and zeroed sub-counters
    p.event_words = ETX_EVENT_WORDS;
    p.ev_share = upload(etx_ev_share + (size_t)device * ETX_N_DOMAINS * ETX_EVENT_WORDS, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS);
    ETX_CHECK(hipMalloc(&d_ev_sub, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS * sizeof(int32_t)));
    p.ev_sub = d_ev_sub;
    if (getenv("ETX_NO_LASTFLUSH")) p.ev_share = nullptr;      // A/B: per-producer write-back
    p.workers_per_domain = ETX_WORKERS_PER_DOMAIN;
    // relay (P5): ETX_RELAY=0 turns it off at run time (waits then poll the global words; A/B);
    // consumers do the full device acquire themselves by default; ETX_RELAY_ACQ=local enables the hierarchical
    // acquire (relay invalidates the domain's L2, consumers drop only L1) -- measured UNSAFE on MI300X 2026-09-24
    p.ev_mirror = nullptr; p.n_relay = 0; p.relay_words = nullptr; p.relay_local_acquire = 0;
    {
      const char* r = getenv("ETX_RELAY");
      const int32_t b = etx_relay_begin[device], e = etx_relay_begin[device + 1];
      if (ETX_RELAY && e > b && r && !strcmp(r, "1")) {     // opt-in at run time; otherwise the reserved workgroup idles
        ETX_CHECK(hipMalloc(&d_mirror, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS * sizeof(int32_t)));
        mirror_init.resize((size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS);
        for (int d = 0; d < ETX_N_DOMAINS; ++d) memcpy(mirror_init.data() + (size_t)d * ETX_EVENT_WORDS, etx_ev_counts, ETX_EVENT_WORDS * sizeof(int32_t));
        p.ev_mirror = d_mirror; p.n_relay = e - b; p.relay_words = upload(etx_relay_words + b, (size_t)(e - b));
        const char* a = getenv("ETX_RELAY_ACQ");
        p.relay_local_acquire = (a && !strcmp(a, "local")) ? 1 : 0;
      }
    }
    p.n_workers = ETX_N_WORKERS;
    p.n_tasks = etx_n_tasks_dev[device];
    p.n_dynamic = etx_n_dynamic_dev[device];
    p.n_events = ETX_N_EVENTS;
    p.spin_limit = 200000000u;
    const char* lm = getenv("ETX_LAUNCH");                 // "ordinary": plain launch when the grid is known to fit
    cooperative = !(lm && !strcmp(lm, "ordinary"));
    const char* sm = getenv("ETX_SYNC");
    sync_block = sm && !strcmp(sm, "block");
    // The emitter's decision, not an independent one: ETX_EMIT_DEV_RESET comes from plan_data.h, which the same
    // compile produced from the same plan, so the host cannot believe in a tail the kernel does not carry. The env
    // var only turns OFF what was emitted; it can never turn on what was not.
    dev_reset = ETX_EMIT_DEV_RESET && !(getenv("ETX_DEV_RESET") && !strcmp(getenv("ETX_DEV_RESET"), "0"));
    build_reset_ops();
  }
  bool cooperative = true;
  bool sync_block = false;        // ETX_SYNC=block: hipStreamSynchronize instead of the watchdog poll
  bool dev_reset = false;         // ETX_EMIT_DEV_RESET: the kernel rewrites its own per-step state (ETX_DEV_RESET=0 disables)
  bool dev_reset_valid = false;   // the device did the reset for the step about to run
  bool dev_reset_checked = false; // the first step verified that the kernel's tail actually ran
  etx_reset_op* d_reset_ops = nullptr;
  int32_t n_reset_ops = 0;
  // The ops the device can do; the host keeps ctrl (read back after the kernel) and, when the counter is on, d_trace.
  void build_reset_ops() {
    if (!dev_reset) return;
    std::vector<etx_reset_op> ops;
    auto fill = [&](int32_t* dst, size_t words, int32_t v) { if (words) ops.push_back({dst, nullptr, (int32_t)words, v}); };
    auto copy = [&](int32_t* dst, const int32_t* host_src, size_t words) {
      if (!words) return;
      int32_t* d = nullptr; ETX_CHECK(hipMalloc(&d, words * sizeof(int32_t)));
      ETX_CHECK(hipMemcpy(d, host_src, words * sizeof(int32_t), hipMemcpyHostToDevice));
      ops.push_back({dst, d, (int32_t)words, 0});
    };
    if (owns_events) {
      if (ETX_EV_STRIDE == 1) copy((int32_t*)d_events, etx_ev_counts, ETX_EVENT_WORDS);
      else { if (ev_strided.empty()) { ev_strided.assign((size_t)ETX_EVENT_WORDS * ETX_EV_STRIDE, 0); for (int i = 0; i < ETX_EVENT_WORDS; ++i) ev_strided[(size_t)ETX_EVW(i)] = etx_ev_counts[i]; }
             copy((int32_t*)d_events, ev_strided.data(), ev_strided.size()); }
    }
    fill(d_slots, ETX_N_DOMAINS * 2 + ETX_N_DOMAINS * 256, 0);
    fill(d_slots + ETX_N_DOMAINS * 2 + ETX_N_DOMAINS * 256, ETX_N_DOMAINS * 256, -1);     // 0xFF bytes
    copy(d_remaining, etx_task_remaining, ETX_N_TASKS);
    fill(d_ev_sub, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS, 0);
    if (d_mirror) copy((int32_t*)d_mirror, mirror_init.data(), mirror_init.size());
    int32_t total = 0; for (int d = 0; d < ETX_N_DOMAINS; ++d) total += etx_local_capacity[device * ETX_N_DOMAINS + d];
    fill(d_lq_slots, total, -1);
    if (lq_heads.empty()) { lq_heads.assign(ETX_N_DOMAINS, 0); lq_tails.assign(ETX_N_DOMAINS, 0); }
    int32_t off = 0, src = 0;
    for (int dd = 0; dd < device * ETX_N_DOMAINS; ++dd) src += etx_init_local_len[dd];
    for (int d = 0; d < ETX_N_DOMAINS; ++d) {
      const int32_t n = etx_init_local_len[device * ETX_N_DOMAINS + d];
      if (n) copy(d_lq_slots + off, etx_init_local + src, n);
      lq_tails[d] = n; src += n; off += etx_local_capacity[device * ETX_N_DOMAINS + d];
    }
    copy(d_lq_head, lq_heads.data(), ETX_N_DOMAINS);
    copy(d_lq_tail, lq_tails.data(), ETX_N_DOMAINS);
    const int32_t gcap = etx_global_capacity[device];
    fill(d_gq_slots, gcap, -1);
    int32_t gsrc = 0; for (int dd = 0; dd < device; ++dd) gsrc += etx_init_global_len[dd];
    const int32_t gn = etx_init_global_len[device];
    if (gn) copy(d_gq_slots, etx_init_global + gsrc, gn);
    gq_init[0] = 0; gq_init[1] = gn;
    copy((int32_t*)p.global_queue.head, gq_init, 2);
    n_reset_ops = (int32_t)ops.size();
    ETX_CHECK(hipMalloc(&d_reset_ops, ops.size() * sizeof(etx_reset_op)));
    ETX_CHECK(hipMemcpy(d_reset_ops, ops.data(), ops.size() * sizeof(etx_reset_op), hipMemcpyHostToDevice));
    p.reset_ops = d_reset_ops; p.n_reset_ops = n_reset_ops;
  }
  // reset per-step state (events if owned, queues, ctrl, slots, counters)
  void reset_step() {
    ETX_CHECK(hipSetDevice(device));
#if defined(ETX_INBODY) && ETX_INBODY
    // Host-only diagnostic reset, including when the production device reset
    // is active. Never add diagnostic buffers to the timed reset tail.
    if (d_inbody_poll) {
      ETX_CHECK(hipMemsetAsync(d_trace_time, 0, (size_t)ETX_N_TASKS * 4 * sizeof(uint64_t), stream));
      ETX_CHECK(hipMemsetAsync(d_inbody_poll, 0, (size_t)ETX_N_TASKS * sizeof(uint64_t), stream));
      ETX_CHECK(hipMemsetAsync(d_inbody_event, 0, (size_t)ETX_EVENT_WORDS * 3 * sizeof(uint64_t), stream));
      ETX_CHECK(hipMemsetAsync(d_inbody_wave, 0, (size_t)ETX_N_TASKS * (ETX_THREADS / 64) * 2 * sizeof(uint64_t), stream));
#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER
      ETX_CHECK(hipMemsetAsync(d_arrival_stamps, 0, (size_t)(ETX_ARRIVAL_SLOTS ? ETX_ARRIVAL_SLOTS : 1) * 2 * sizeof(uint64_t), stream));
#endif
    }
#endif
    ETX_CHECK(hipMemsetAsync(d_ctrl, 0, 64, stream));      // never device-side: the host reads ctrl_done after the kernel
    if (dev_reset && dev_reset_valid) {                    // the kernel rewrote everything else on its way out
      if (exec_count) ETX_CHECK(hipMemsetAsync(d_trace, 0, ETX_N_TASKS * sizeof(int32_t), stream));
      return;
    }
    if (owns_events) {
      if (ETX_EV_STRIDE == 1) ETX_CHECK(hipMemcpyAsync(d_events, etx_ev_counts, ETX_EVENT_WORDS * sizeof(int32_t), hipMemcpyHostToDevice, stream));
      else {                                             // counters at word * ETX_EV_STRIDE (one cache line each)
        if (ev_strided.empty()) { ev_strided.assign((size_t)ETX_EVENT_WORDS * ETX_EV_STRIDE, 0); for (int i = 0; i < ETX_EVENT_WORDS; ++i) ev_strided[(size_t)ETX_EVW(i)] = etx_ev_counts[i]; }
        ETX_CHECK(hipMemcpyAsync(d_events, ev_strided.data(), ev_strided.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream));
      }
    }
    ETX_CHECK(hipMemsetAsync(d_slots, 0, (ETX_N_DOMAINS * 2 + ETX_N_DOMAINS * 256) * sizeof(int32_t), stream));
    ETX_CHECK(hipMemsetAsync(d_slots + ETX_N_DOMAINS * 2 + ETX_N_DOMAINS * 256, 0xFF, ETX_N_DOMAINS * 256 * sizeof(int32_t), stream));
    if (exec_count) ETX_CHECK(hipMemsetAsync(d_trace, 0, ETX_N_TASKS * sizeof(int32_t), stream));   // 241 KB; dead work when the counter is off
    ETX_CHECK(hipMemcpyAsync(d_remaining, etx_task_remaining, ETX_N_TASKS * sizeof(int32_t), hipMemcpyHostToDevice, stream));
    ETX_CHECK(hipMemsetAsync(d_ev_sub, 0, (size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS * sizeof(int32_t), stream));
    if (d_mirror) ETX_CHECK(hipMemcpyAsync(d_mirror, mirror_init.data(), mirror_init.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream));
    int32_t total = 0; for (int d = 0; d < ETX_N_DOMAINS; ++d) total += etx_local_capacity[device * ETX_N_DOMAINS + d];
    if (total) ETX_CHECK(hipMemsetAsync(d_lq_slots, 0xFF, total * sizeof(int32_t), stream));
    if (lq_heads.empty()) { lq_heads.assign(ETX_N_DOMAINS, 0); lq_tails.assign(ETX_N_DOMAINS, 0); }
    std::vector<int32_t>& heads = lq_heads; std::vector<int32_t>& tails = lq_tails;
    int32_t off = 0, src = 0;
    for (int dd = 0; dd < device * ETX_N_DOMAINS; ++dd) src += etx_init_local_len[dd];
    for (int d = 0; d < ETX_N_DOMAINS; ++d) {
      const int32_t n = etx_init_local_len[device * ETX_N_DOMAINS + d];
      if (n) ETX_CHECK(hipMemcpyAsync(d_lq_slots + off, etx_init_local + src, n * sizeof(int32_t), hipMemcpyHostToDevice, stream));
      tails[d] = n; src += n; off += etx_local_capacity[device * ETX_N_DOMAINS + d];
    }
    ETX_CHECK(hipMemcpyAsync(d_lq_head, heads.data(), ETX_N_DOMAINS * sizeof(int32_t), hipMemcpyHostToDevice, stream));
    ETX_CHECK(hipMemcpyAsync(d_lq_tail, tails.data(), ETX_N_DOMAINS * sizeof(int32_t), hipMemcpyHostToDevice, stream));
    const int32_t gcap = etx_global_capacity[device];
    if (gcap) ETX_CHECK(hipMemsetAsync(d_gq_slots, 0xFF, gcap * sizeof(int32_t), stream));
    int32_t gsrc = 0; for (int dd = 0; dd < device; ++dd) gsrc += etx_init_global_len[dd];
    const int32_t gn = etx_init_global_len[device];
    if (gn) ETX_CHECK(hipMemcpyAsync(d_gq_slots, etx_init_global + gsrc, gn * sizeof(int32_t), hipMemcpyHostToDevice, stream));
    if (gq_init[1] != gn) { gq_init[0] = 0; gq_init[1] = gn; }
    ETX_CHECK(hipMemcpyAsync((void*)p.global_queue.head, gq_init, 8, hipMemcpyHostToDevice, stream));
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
  void launch(bool record_events = true) {
    ETX_CHECK(hipSetDevice(device));
    if (!ev_start) { ETX_CHECK(hipEventCreate(&ev_start)); ETX_CHECK(hipEventCreate(&ev_stop)); }
    if (record_events) ETX_CHECK(hipEventRecord(ev_start, stream));
    void* kargs[] = {&p};
    if (cooperative) ETX_CHECK(hipLaunchCooperativeKernel(kernel, dim3(ETX_N_LAUNCH), dim3(ETX_THREADS), kargs, 0, stream));
    else ETX_CHECK(hipLaunchKernel(kernel, dim3(ETX_N_LAUNCH), dim3(ETX_THREADS), kargs, 0, stream));   // residency checked by check_residency()
    if (record_events) ETX_CHECK(hipEventRecord(ev_stop, stream));
  }
  // wait with watchdog; returns false on timeout / spin limit / incomplete
  bool wait(double timeout_s, std::chrono::steady_clock::time_point t0) {
    ETX_CHECK(hipSetDevice(device));
    // ETX_SYNC=block: block in the driver instead of polling. hipStreamQuery costs microseconds on this driver, so the
    // poll loop's granularity became ~71 us of overshoot per step against a 5 ms kernel (measured 2026-10-01). The
    // device's own spin_limit still aborts a hung kernel (ctrl[0] |= 2); what is given up is the host-side watchdog
    // for a hang the device cannot see, so the poll path stays the default.
    if (sync_block) {
      ETX_CHECK(hipStreamSynchronize(stream));
      return after_wait();
    }
    while (hipStreamQuery(stream) == hipErrorNotReady) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > timeout_s) {
        int32_t one = 1; ETX_CHECK(hipMemcpy(d_ctrl, &one, 4, hipMemcpyHostToDevice));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        fprintf(stderr, "device %d: watchdog timeout after %.1f s\n", device, el);
        dump_events();
        return false;
      }
      std::this_thread::yield();   // a 50 us sleep overshot the 5 ms kernel by ~100 us/step (2026-10-01)
    }
    ETX_CHECK(hipStreamSynchronize(stream));
    return after_wait();
  }
  // everything the host checks once the kernel has retired; shared by the polling and blocking waits
  bool after_wait() {
    dev_reset_valid = false;          // a failed step leaves the state half-written; the host resets it fully
    float ms = 0.f; ETX_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));   // device timestamps: launch to kernel end, no host polling jitter
    last_ms = ms;
    int32_t ctrl[2]; ETX_CHECK(hipMemcpy(ctrl, d_ctrl, 8, hipMemcpyDeviceToHost));
    if (ctrl[0] & 2) { fprintf(stderr, "device %d: kernel hit spin limit (abort=%d, done=%d/%d)\n", device, ctrl[0], ctrl[1], p.n_tasks); dump_events(); return false; }
    if (ctrl[1] != p.n_dynamic) { fprintf(stderr, "device %d: dynamic done=%d of %d\n", device, ctrl[1], p.n_dynamic); dump_events(); return false; }
    // Defence in depth behind the manifest: the first step reads back the exit counter the tail increments and
    // falls back permanently if it is short. The manifest makes an emitter/host mismatch impossible, but the tail
    // also has to actually reach its exit (a surplus worker returns early), and a silent no-op step would cost
    // 32/32 with no diagnostic, so this stays.
    if (dev_reset && !dev_reset_checked) {
      dev_reset_checked = true;
      int32_t c[4] = {0, 0, 0, 0};
      ETX_CHECK(hipMemcpy(c, d_ctrl, 16, hipMemcpyDeviceToHost));
      if (c[2] != ETX_N_LAUNCH) {
        fprintf(stderr, "ETX_DEV_RESET: the kernel's reset tail did not run (%d of %d workgroups reported); "
                        "falling back to host resets.\n", c[2], ETX_N_LAUNCH);
        dev_reset = false;
      }
    }
    dev_reset_valid = dev_reset;      // the kernel retired, so its end-of-kernel reset ran
    // static tasks are not counted on the device; the trace (one increment per executed task) verifies them
    if (!exec_count) return true;                     // ETX_EXEC_COUNT=0: the counter's atomics are off, nothing to verify
    std::vector<int32_t> tr(ETX_N_TASKS);
    ETX_CHECK(hipMemcpy(tr.data(), d_trace, ETX_N_TASKS * sizeof(int32_t), hipMemcpyDeviceToHost));
    int executed = 0;
    for (int i = 0; i < ETX_N_TASKS; ++i) if (tr[i] == 1) executed++;
    if (executed != p.n_tasks) { fprintf(stderr, "device %d: %d of %d tasks executed exactly once\n", device, executed, p.n_tasks); dump_events(); return false; }
    return true;
  }
  // Unfused baseline: the same tiles, one ordinary launch per task grid in plan order (etx_unfused_d<dev>),
  // captured once into a HIP graph and replayed per step (ETX_UNFUSED_NOGRAPH=1: plain stream launches).
  // Kernel boundaries are the only synchronisation; events are not used.
  const void* unfused_kernel = nullptr;
  hipGraphExec_t unfused_exec = nullptr;
  void enqueue_unfused() {
    // ETX_UNFUSED_SYNC=1: synchronize after every launch and name the one that faults (debug; forces graph off)
    static const bool dbg_sync = [] { const char* e = getenv("ETX_UNFUSED_SYNC"); return e && *e == '1'; }();
    for (int i = etx_launch_begin[device]; i < etx_launch_begin[device + 1]; ++i) {
      int32_t first = etx_launch_first[i];
      void* kargs[] = {&p, &first};
      ETX_CHECK(hipLaunchKernel(unfused_kernel, dim3(etx_launch_count[i]), dim3(ETX_THREADS), kargs, 0, stream));
      if (dbg_sync) {
        const hipError_t e = hipStreamSynchronize(stream);
        if (e != hipSuccess) {
          fprintf(stderr, "ETX_UNFUSED_SYNC: launch %d type %s (%d tasks, first task %d) failed: %s\n",
                  i, etx_type_names[etx_descs[first].type], etx_launch_count[i], first, hipGetErrorString(e));
          abort();
        }
      }
    }
  }
  bool run_unfused() {
    ETX_CHECK(hipSetDevice(device));
    if (!ev_start) { ETX_CHECK(hipEventCreate(&ev_start)); ETX_CHECK(hipEventCreate(&ev_stop)); }
    const bool graph = !getenv("ETX_UNFUSED_NOGRAPH");
    if (graph && !unfused_exec) {
      hipGraph_t g;
      ETX_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
      enqueue_unfused();
      ETX_CHECK(hipStreamEndCapture(stream, &g));
      ETX_CHECK(hipGraphInstantiate(&unfused_exec, g, nullptr, nullptr, 0));
    }
    ETX_CHECK(hipEventRecord(ev_start, stream));
    if (graph) ETX_CHECK(hipGraphLaunch(unfused_exec, stream)); else enqueue_unfused();
    ETX_CHECK(hipEventRecord(ev_stop, stream));
    ETX_CHECK(hipStreamSynchronize(stream));
    float ms = 0.f; ETX_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop)); last_ms = ms;
    return true;
  }
  // ETX_HOST_PROF=1: accumulate the per-step host cost of each stage (the wall-vs-device gap, 2026-10-01)
  struct HostProf { double reset = 0, resid = 0, launch = 0, wait = 0; long n = 0;
    ~HostProf() { if (n) fprintf(stderr, "HOSTPROF steps %ld: reset %.3f us  residency %.3f  launch %.3f  wait %.3f  total %.3f\n",
                                 n, reset / n, resid / n, launch / n, wait / n, (reset + resid + launch + wait) / n); } };
  static HostProf& host_prof() { static HostProf hp; return hp; }
  static bool host_prof_on() { static bool on = getenv("ETX_HOST_PROF") != nullptr; return on; }
  // ETX_VL_GRAPH=1: capture reset_step + launch once and replay the graph per step. MEASURED 2026-10-01 and it does
  // NOT pay once reset_step is async: wall 5.053 against 5.044 for a plain ordinary launch and 5.060 cooperative
  // (interleaved medians, same binary). The host cost left after the async conversion is the wait() poll and the
  // dispatch, neither of which a graph removes. Kept, off by default, so nobody re-tries it blind. The ~17 state copies and the
  // dispatch are identical every step -- every hipMemcpyAsync source is a stable buffer (the etx_* tables are static,
  // ev_strided / mirror_init / lq_heads / lq_tails / gq_init are members) -- so only the replay crosses to the driver.
  // Requires ETX_LAUNCH_MODE=ordinary: a cooperative launch cannot be captured.
  hipGraphExec_t step_exec = nullptr;
  bool graph_step() {
    if (!step_exec) {
      if (cooperative) { fprintf(stderr, "ETX_VL_GRAPH needs ETX_LAUNCH_MODE=ordinary (cooperative launches are not capturable)\n"); return false; }
      if (!check_residency()) return false;
      hipGraph_t g;
      ETX_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
      reset_step();
      launch(false);                       // the timing events stay outside: captured, they are graph nodes and
      ETX_CHECK(hipStreamEndCapture(stream, &g));   // hipEventElapsedTime then fails with an invalid handle
      ETX_CHECK(hipGraphInstantiate(&step_exec, g, nullptr, nullptr, 0));
      if (!ev_start) { ETX_CHECK(hipEventCreate(&ev_start)); ETX_CHECK(hipEventCreate(&ev_stop)); }
    }
    ETX_CHECK(hipEventRecord(ev_start, stream));
    ETX_CHECK(hipGraphLaunch(step_exec, stream));
    ETX_CHECK(hipEventRecord(ev_stop, stream));
    return true;
  }
  // single-device convenience
  bool run(double timeout_s = 5.0) {
    static const bool use_graph = getenv("ETX_VL_GRAPH") != nullptr;
    if (use_graph) {
      auto t0 = std::chrono::steady_clock::now();
      if (!graph_step()) return false;
      return wait(timeout_s, t0);
    }
    if (!host_prof_on()) {
      reset_step();
      if (!check_residency()) return false;
      auto t0 = std::chrono::steady_clock::now();
      launch();
      return wait(timeout_s, t0);
    }
    auto& hp = host_prof(); using C = std::chrono::steady_clock;
    const auto a = C::now(); reset_step();
    const auto b = C::now(); if (!check_residency()) return false;
    const auto c = C::now(); launch();
    const auto d = C::now(); const bool ok = wait(timeout_s, c);
    const auto e = C::now();
    hp.reset += std::chrono::duration<double, std::micro>(b - a).count();
    hp.resid += std::chrono::duration<double, std::micro>(c - b).count();
    hp.launch += std::chrono::duration<double, std::micro>(d - c).count();
    hp.wait += std::chrono::duration<double, std::micro>(e - d).count();
    ++hp.n;
    return ok;
  }
  void dump_events() {
    dump_trace();
    ETX_CHECK(hipSetDevice(device));
    std::vector<int32_t> evs((size_t)ETX_EVENT_WORDS * ETX_EV_STRIDE), ev(ETX_EVENT_WORDS);
    ETX_CHECK(hipMemcpy(evs.data(), d_events, evs.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < ETX_EVENT_WORDS; ++i) ev[i] = evs[(size_t)ETX_EVW(i)];
    int shown = 0;
    for (int e = 0; e < ETX_N_EVENTS; ++e) {
      int n = 1; for (int d = 0; d < 4; ++d) if (etx_ev_shape[e][d] > 0) n *= etx_ev_shape[e][d];
      for (int i = 0; i < n; ++i) if (ev[etx_ev_offset[e] + i] != 0 && shown++ < 32)
        fprintf(stderr, "  event %d[%d] = %d (initial %d)\n", e, i, ev[etx_ev_offset[e] + i], etx_ev_counts[etx_ev_offset[e] + i]);
    }
    if (shown > 32) fprintf(stderr, "  ... %d non-zero events\n", shown);
    if (shown == 0) fprintf(stderr, "  all events reached zero\n");
#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER
    if (const char* path = getenv("ETX_WATCHDOG_RAW")) {
      write_inbody(path);
      fprintf(stderr, "  producer snapshot: %s (observations only; cause unknown)\n", path);
    }
#endif
  }
  // Per-phase attribution from the timestamp trace (ETX_TRACE=1): for every task type
  // (grid) the mean wait (taken -> deps ready) and body (ready -> done) time, and the
  // phase span (first task taken -> last task done). Ticks are ETX_TIMER units
  // (100 MHz on AMD: 10 ns). `type_names[t]` labels the types (nullptr -> ids).
#if defined(ETX_INBODY) && ETX_INBODY
  void write_inbody(const char* path) {
    if (!d_inbody_poll) { fprintf(stderr, "inbody disabled\n"); exit(1); }
    // Versioned little-endian u64 header, then complete legacy/task/event/wave
    // arrays. Reader resolves task arrival brackets from plan event instances.
    uint64_t version = 1;
#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER
    version = 2;
#endif
    uint64_t header[6] = {0x3142444f424e4954ull, version, ETX_N_TASKS, ETX_EVENT_WORDS, ETX_THREADS / 64, 10};
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    if (fwrite(header, sizeof(header), 1, f) != 1) { perror("inbody header"); exit(1); }
    auto write = [&](uint64_t* d, size_t n) {
      std::vector<uint64_t> h(n);
      ETX_CHECK(hipMemcpy(h.data(), d, n * sizeof(uint64_t), hipMemcpyDeviceToHost));
      if (fwrite(h.data(), sizeof(uint64_t), n, f) != n) { perror("inbody records"); exit(1); }
    };
    write(d_trace_time, (size_t)ETX_N_TASKS * 4);
    write(d_inbody_poll, ETX_N_TASKS);
    write(d_inbody_event, (size_t)ETX_EVENT_WORDS * 3);
    write(d_inbody_wave, (size_t)ETX_N_TASKS * (ETX_THREADS / 64) * 2);
#if defined(ETX_PRODUCER_LEDGER) && ETX_PRODUCER_LEDGER
    uint64_t extra[3] = {ETX_ARRIVAL_SLOTS, ETX_EVENT_WORDS, ETX_N_DOMAINS};
    if (fwrite(extra, sizeof(extra), 1, f) != 1) { perror("ledger header"); exit(1); }
    auto write_words = [&](const int32_t* a, size_t n) {
      std::vector<uint64_t> h(n);
      for (size_t i = 0; i < n; ++i) h[i] = (uint64_t)(int64_t)a[i];
      if (fwrite(h.data(), sizeof(uint64_t), n, f) != n) { perror("ledger words"); exit(1); }
    };
    write_words(etx_arrival_offsets, (size_t)ETX_N_TASKS + 1);
    write_words(etx_arrival_words, ETX_ARRIVAL_SLOTS);
    write(d_arrival_stamps, (size_t)ETX_ARRIVAL_SLOTS * 2);
    std::vector<int32_t> counters((size_t)ETX_EVENT_WORDS * ETX_EV_STRIDE), logical(ETX_EVENT_WORDS);
    ETX_CHECK(hipMemcpy(counters.data(), d_events, counters.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < ETX_EVENT_WORDS; ++i) logical[i] = counters[(size_t)ETX_EVW(i)];
    write_words(logical.data(), logical.size());
    std::vector<int32_t> sub((size_t)ETX_N_DOMAINS * ETX_EVENT_WORDS);
    ETX_CHECK(hipMemcpy(sub.data(), d_ev_sub, sub.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    write_words(sub.data(), sub.size());
#endif
    if (fclose(f)) { perror("inbody close"); exit(1); }
  }
#endif
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
    if (!exec_count) { fprintf(stderr, "  device %d exec counts: off (set ETX_EXEC_COUNT=1)\n", device); return; }
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
