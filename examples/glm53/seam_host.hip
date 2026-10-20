// Opt-in production-shaped synthetic handoff, not full GLM layer cost.
#include "plan_data.h"
#include "etx/host.h"
#include <atomic>
#include <cerrno>
#include <climits>

extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_megakernel_d1(etx_params);

static int number(const char* s, int low, int high) {
  char* end = nullptr; errno = 0; long x = strtol(s, &end, 10);
  if (errno || !*s || *end || x < low || x > high) {
    fprintf(stderr, "SEAM-FAIL invalid argument %s\n", s); exit(2);
  }
  return (int)x;
}
static int fail(const char* reason, int round = -1, int iteration = -1, int row = -1) {
  fprintf(stderr, "SEAM-FAIL %s round=%d iteration=%d row=%d\n", reason, round, iteration, row);
  return 4;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
  if (argc > 5) return fail("usage: run [iterations=1000] [rounds=5] [warmup=20] [trace=0]");
  const int iterations = argc > 1 ? number(argv[1], 1, 1000000) : 1000;
  const int rounds = argc > 2 ? number(argv[2], 1, 100) : 5;
  const int warmup = argc > 3 ? number(argv[3], 1, 10000) : 20;
  const bool trace = argc > 4 ? number(argv[4], 0, 1) : false;
  // Own instrumentation/reset/synchronization policy; ambient env cannot change labels.
  unsetenv("ETX_TRACE"); unsetenv("ETX_EXEC_COUNT"); unsetenv("ETX_INBODY_RAW");
  setenv("ETX_SYNC", "poll", 1); setenv("ETX_DEV_RESET", "0", 1);
  if (trace) setenv("ETX_TRACE", "1", 1);
  const int batch = etx_shape[ETX_SYM_B], n = batch * 16384;
  if (batch != 1 && batch != 2 && batch != 4 && batch != 8) return fail("invalid batch");
  printf("SEAM-CONFIG {\"schema\":1,\"label\":\"production-shaped synthetic handoff; not full layer cost\",\"batch\":%d,\"iterations\":%d,\"rounds\":%d,\"warmup\":%d,\"trace\":%s,\"row_bytes\":32768}\n",
         batch, iterations, rounds, warmup, trace ? "true" : "false");
  // Independent watchdog uses NO HIP calls: even a driver copy/sync hang is bounded.
  // gpu_jobs / Python also impose a hard outer process deadline.
  std::atomic<int64_t> heartbeat{0};
  std::atomic<int> sequence{-1};
  std::atomic<const char*> stage{"setup"};
  const auto epoch_time = std::chrono::steady_clock::now();
  auto touch = [&](const char* s) {
    stage.store(s);
    heartbeat.store(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-epoch_time).count());
  };
  std::thread([&]() {
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-epoch_time).count();
      if (ms-heartbeat.load() > 10000) {
        fprintf(stderr, "SEAM-FAIL hard watchdog stage=%s sequence=%d stalled_ms=%lld (no HIP calls attempted)\n",
                stage.load(), sequence.load(), (long long)(ms-heartbeat.load()));
        fflush(nullptr); std::_Exit(3);
      }
    }
  }).detach();
  int count = 0; ETX_CHECK(hipGetDeviceCount(&count));
  if (count != 2) return fail("expose exactly two gfx942 GPUs");
  for (int rank = 0; rank < 2; ++rank) {
    touch("peer qualification");
    hipDeviceProp_t prop; ETX_CHECK(hipGetDeviceProperties(&prop, rank));
    printf("SEAM-DEVICE rank=%d name=%s arch=%s bytes=%zu\n", rank, prop.name, prop.gcnArchName, prop.totalGlobalMem);
    if (strncmp(prop.gcnArchName, "gfx942", 6)) return fail("compiled for gfx942");
    int access = 0, atomics = 0;
    ETX_CHECK(hipDeviceCanAccessPeer(&access, rank, 1-rank));
    ETX_CHECK(hipDeviceGetP2PAttribute(&atomics, hipDevP2PAttrNativeAtomicSupported, rank, 1-rank));
    if (!access || !atomics) return fail("peer access and native peer integer atomics required");
    ETX_CHECK(hipSetDevice(rank));
    const hipError_t peer = hipDeviceEnablePeerAccess(1-rank, 0);
    if (peer == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError(); else ETX_CHECK(peer);
  }
  std::vector<void*> buffers(ETX_N_ARGS, nullptr);
  for (int a = 0; a < ETX_N_ARGS; ++a) {
    touch("argument allocation");
    int owner = etx_arg_writer_device[a]; if (owner < 0) owner = 0;
    ETX_CHECK(hipSetDevice(owner));
    const bool shared = !strcmp(etx_arg_names[a], "mailbox") || !strcmp(etx_arg_names[a], "token") || !strcmp(etx_arg_names[a], "control");
    if (shared) ETX_CHECK(hipExtMallocWithFlags(&buffers[a], etx_arg_bytes[a], hipDeviceMallocFinegrained));
    else ETX_CHECK(hipMalloc(&buffers[a], etx_arg_bytes[a]));
    ETX_CHECK(hipMemset(buffers[a], 0, etx_arg_bytes[a]));
  }
  int32_t* events = etx_host::alloc_events(0);
  std::vector<etx_host> hosts; hosts.reserve(2);
  for (int rank = 0; rank < 2; ++rank) {
    touch("runtime setup");
    hosts.emplace_back(rank);
    hosts.back().kernel = rank ? (const void*)etx_megakernel_d1 : (const void*)etx_megakernel_d0;
    hosts.back().args = buffers; hosts.back().setup(events); hosts.back().owns_events = rank == 0;
    if (!hosts.back().check_residency()) return fail("persistent kernel not resident");
  }
  const int mailbox_arg = etx_host::arg_index("mailbox"), received_arg = etx_host::arg_index("received");
  const int epoch_arg = etx_host::arg_index("epoch"), control_arg = etx_host::arg_index("control");
  const int returned_arg = etx_host::arg_index("returned"), token_arg = etx_host::arg_index("token");
  std::vector<uint16_t> poison(n, 0xDEAD), received(n);
  std::vector<int32_t> returned(batch);
  std::vector<uint64_t> stamps[2];
  for (int rank = 0; rank < 2; ++rank) if (trace) stamps[rank].resize(ETX_N_TASKS * 4);
  // Locate row task ids from generated descriptors; never assume plan ordering.
  std::vector<int> ids[3]; for (auto& v : ids) v.assign(batch, -1);
  const char* task_names[] = {"seam_send", "seam_consume", "seam_feedback"};
  for (int t = 0; t < ETX_N_TASKS; ++t) for (int k = 0; k < 3; ++k)
    if (!strcmp(etx_type_names[etx_descs[t].type], task_names[k])) {
      int r = etx_descs[t].coord[0];
      if (r < 0 || r >= batch || ids[k][r] != -1) return fail("invalid row descriptor");
      ids[k][r] = t;
    }
  if (ETX_N_TASKS != batch * 3 || ETX_THREADS != 256) return fail("wrong production row/task shape");
  for (auto& v : ids) for (int id : v) if (id < 0) return fail("missing row task");
  int seq = 0;
  for (int round = 0; round < rounds; ++round) for (int order = 0; order < 2; ++order) {
    const int control = (order + round) % 2;  // reverse order each round
    const char* arm = control ? "control" : "payload";
    for (int step = -warmup; step < iterations; ++step, ++seq) {
      sequence.store(seq); touch("step setup/reset");
      // Bounded epoch avoids int overflow even at maximum CLI iteration counts.
      const int32_t epoch = seq % 1000000;
      ETX_CHECK(hipSetDevice(0));
      ETX_CHECK(hipMemcpy(buffers[epoch_arg], &epoch, 4, hipMemcpyHostToDevice));
      ETX_CHECK(hipMemcpy(buffers[control_arg], &control, 4, hipMemcpyHostToDevice));
      ETX_CHECK(hipMemcpy(buffers[mailbox_arg], poison.data(), n*2, hipMemcpyHostToDevice));
      ETX_CHECK(hipMemset(buffers[returned_arg], 0xFF, batch*4));
      ETX_CHECK(hipSetDevice(1));
      ETX_CHECK(hipMemcpy(buffers[received_arg], poison.data(), n*2, hipMemcpyHostToDevice));
      ETX_CHECK(hipMemset(buffers[token_arg], 0xFF, batch*4));
      for (auto& host : hosts) {
        host.reset_step();
        if (trace) ETX_CHECK(hipMemsetAsync(host.d_trace_time, 0, ETX_N_TASKS*4*sizeof(uint64_t), host.stream));
      }
      // BOTH resets must complete before either rank sees an event from this epoch.
      for (auto& host : hosts) { ETX_CHECK(hipSetDevice(host.device)); ETX_CHECK(hipStreamSynchronize(host.stream)); }
      touch("timed launch/stream retirement");
      const auto start = std::chrono::steady_clock::now();
      for (auto& host : hosts) host.launch();
      for (auto& host : hosts) if (!host.wait_stream(5.0, start)) return fail("runtime watchdog", round, step);
      const double wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count();
      touch("untimed runtime validation/D2H");
      // after_wait does synchronous copies: it MUST stay outside host wall timing.
      for (auto& host : hosts) {
        ETX_CHECK(hipSetDevice(host.device));
        if (!host.after_wait()) return fail("incomplete/aborted runtime", round, step);
        if (trace) ETX_CHECK(hipMemcpy(stamps[host.device].data(), host.d_trace_time, ETX_N_TASKS*4*sizeof(uint64_t), hipMemcpyDeviceToHost));
      }
      ETX_CHECK(hipSetDevice(1));
      ETX_CHECK(hipMemcpy(received.data(), buffers[received_arg], n*2, hipMemcpyDeviceToHost));
      ETX_CHECK(hipSetDevice(0));
      ETX_CHECK(hipMemcpy(returned.data(), buffers[returned_arg], batch*4, hipMemcpyDeviceToHost));
      touch("every-row correctness");
      for (int r = 0; r < batch; ++r) {
        for (int j = 0; j < 16384; ++j) {
          const int i = r*16384+j;
          const uint16_t expected = control && j != 0 && j != 16383 ? 0xDEAD : (i+17*epoch) % 0x7F00;
          if (received[i] != expected) return fail("payload mismatch", round, step, r);
        }
        const int expected = (r*16384+17*epoch) % 0x7F00 + ((r+1)*16384-1+17*epoch) % 0x7F00;
        if (returned[r] != expected) return fail("token mismatch", round, step, r);
        if (trace) {
          for (int k = 0; k < 3; ++k) {
            const auto* tr = stamps[k == 1 ? 1 : 0].data()+ids[k][r]*4;
            if (!tr[0] || tr[0] > tr[1] || tr[1] > tr[2]) return fail("missing/nonmonotonic trace", round, step, r);
          }
          // Same rank only: feedback cannot become ready before that row's producer.
          if (stamps[0][ids[2][r]*4+1] < stamps[0][ids[0][r]*4+1]) return fail("invalid rank0 roundtrip", round, step, r);
        }
      }
      printf("SEAM-STEP {\"round\":%d,\"arm\":\"%s\",\"phase\":\"%s\",\"iteration\":%d,\"correct\":true,\"rows_checked\":%d,\"elements_checked\":%d,\"wall_us\":%.6f,\"rank0_span_us\":%.6f,\"rank1_span_us\":%.6f",
             round, arm, step < 0 ? "warmup" : "sample", step < 0 ? step+warmup : step, batch, n, wall_us, hosts[0].last_ms*1000, hosts[1].last_ms*1000);
      if (trace) {
        printf(",\"trace_rows\":[");
        for (int r = 0; r < batch; ++r) {
          printf("%s{\"row\":%d", r ? "," : "", r);
          const char* names[] = {"send", "consume", "feedback"};
          for (int k = 0; k < 3; ++k) {
            const auto* tr = stamps[k == 1 ? 1 : 0].data()+ids[k][r]*4;
            printf(",\"%s\":[%llu,%llu,%llu]", names[k], (unsigned long long)tr[0], (unsigned long long)tr[1], (unsigned long long)tr[2]);
          }
          printf("}");
        }
        printf("]");
      }
      printf("}\n");
      if (step < 0 || (step+1) % 100 == 0 || step+1 == iterations)
        printf("SEAM-PROGRESS round=%d arm=%s step=%d/%d trace=%d every_row=OK\n", round, arm, step+1, iterations, trace);
    }
  }
  printf("SEAM-RESULT {\"status\":\"PASS\",\"steps_checked\":%d}\n", seq);
  // Process teardown avoids freeing peer memory while runtime objects still refer to it.
  fflush(nullptr); std::_Exit(0);
}
