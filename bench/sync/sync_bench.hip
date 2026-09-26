// Synchronisation microbenchmarks for MI300X (docs/RESEARCH-sync-hardware.md section 3). One question:
// can a synchronisation inside a persistent kernel be cheaper than a kernel boundary, and with which memory type?
//
//   sync_bench pingpong   payload hand-off between two workgroups on DIFFERENT XCDs (and a same-XCD pair):
//                         every round the receiver checks every payload word, so staleness is counted, not assumed
//   sync_bench barrier    whole-GPU barrier with one 1024-thread workgroup per CU, optional L2 load (dirty / clean
//                         lines per workgroup between barriers), with a staleness check across XCDs
//   sync_bench boundary   the kernel-boundary gap: K tiny kernels in one HIP graph
//   sync_bench gws        hardware barrier (DS_GWS_*) in a cooperative launch; run last, under a deadline
//
// Memory/fence variants (the "mode" column):
//   dev    cached memory, today's ETX device scope: release s_waitcnt + buffer_wbl2 sc1, acquire buffer_inv sc1
//   uc     payload in uncached memory (hipDeviceMallocUncached, MTYPE UC), release s_waitcnt, acquire buffer_inv sc0
//          (L1 only); counters and flags stay in ordinary memory in every mode, as ETX's event words do
//   light  cached memory with the uc fences: expected to go STALE across XCDs (proves the check detects it)
//   xcd    same-XCD pair, cached memory, XCD fences (s_waitcnt / buffer_inv sc0): the floor
// Output lines start with RESULT; a hand-off that cannot complete reports NO PROGRESS instead of hanging.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(1); } } while (0)

enum Mode { DEV = 0, UC = 1, LIGHT = 2, XCD = 3 };
static const char* mode_name[] = {"dev", "uc", "light", "xcd"};

__device__ __forceinline__ uint32_t xcc_id() { return __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20) & 0xF; }
__device__ __forceinline__ void release(int mode) {
  if (mode == DEV) asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\tbuffer_wbl2 sc1\n\ts_waitcnt vmcnt(0)" ::: "memory");
  else asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}
__device__ __forceinline__ void acquire(int mode) {
  if (mode == DEV) asm volatile("buffer_inv sc1" ::: "memory");
  else asm volatile("buffer_inv sc0" ::: "memory");
}
__device__ __forceinline__ int poll(const int* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ void post(int* p, int v) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
#define SPIN_LIMIT 20000000u

// ------------------------------------------------------------------ ping-pong
// Workgroup `a` (on XCD xa) and `b` (on XCD xb) take turns: write `words` ints of payload = round, release, post
// the flag; the other polls, acquires, checks every word. roles[] is decided on the host from a discovery pass.
__global__ void __launch_bounds__(1024) k_discover(uint32_t* xcc) { if (threadIdx.x == 0) xcc[blockIdx.x] = xcc_id(); }

__global__ void __launch_bounds__(1024) k_pingpong(int* pay_ab, int* pay_ba, int* flags, int words, int rounds, int mode,
                                                   int wa, int wb, int* errors, int* progress, unsigned long long* ticks) {
  const int me = blockIdx.x == wa ? 0 : blockIdx.x == wb ? 1 : -1;
  if (me < 0) return;
  int* out = me == 0 ? pay_ab : pay_ba; const int* in = me == 0 ? pay_ba : pay_ab;
  int* my_flag = flags + 64 * me; const int* other_flag = flags + 64 * (1 - me);
  __shared__ int s_ok, s_err;
  if (threadIdx.x == 0) { s_err = 0; }
  __syncthreads();
  const unsigned long long t0 = wall_clock64();
  for (int r = 1; r <= rounds; ++r) {
    const bool send_first = me == 0;
    for (int phase = 0; phase < 2; ++phase) {
      if ((phase == 0) == send_first) {                         // send
        for (int i = threadIdx.x; i < words; i += blockDim.x) out[i] = r;
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");           // every wave drains its stores (to L2, or memory for UC)
        __syncthreads();
        if (threadIdx.x == 0) { release(mode); post(my_flag, r); }
      } else {                                                  // receive
        if (threadIdx.x == 0) {
          unsigned n = 0; int ok = 1;
          while (poll(other_flag) < r) { if (++n > SPIN_LIMIT) { ok = 0; break; } }
          if (ok) acquire(mode);
          s_ok = ok;
        }
        __syncthreads();
        if (!s_ok) { if (threadIdx.x == 0) { progress[0] = r; } return; }
        int bad = 0;
        for (int i = threadIdx.x; i < words; i += blockDim.x) bad += in[i] != r;
        if (bad) atomicAdd(&s_err, bad);
        __syncthreads();
      }
    }
  }
  if (threadIdx.x == 0) { ticks[me] = wall_clock64() - t0; errors[me] = s_err; progress[0] = rounds + 1; ticks[2 + me] = xcc_id(); }
}

// ------------------------------------------------------------------ barrier
// Every workgroup: optional L2 load (write `dirty` KB, read `clean` KB of its own scratch), write its slot = gen,
// release, arrive; one waiter per XCD (relay) or every workgroup polls; acquire; read the slot of a workgroup on
// another XCD and check it. arrive is combined per XCD: the last arriver of an XCD arrives once on the global word.
__global__ void __launch_bounds__(1024) k_barrier(int* slots, int* ctr_xcd, int* ctr_all, int* local_flag, int iters, int mode,
                                                  int relay, int dirty_kb, int clean_kb, int* scratch, const uint32_t* xcc_of,
                                                  int nwg, int* errors, int* progress) {
  const int w = blockIdx.x; const uint32_t x = xcc_id();
  const int peer = (w + nwg / 2 + 1) % nwg;                     // a workgroup that is on another XCD with round-robin placement
  __shared__ int s_ok;
  int* my_scratch = scratch + (size_t)w * 64 * 1024;           // 256 KB per workgroup
  int err = 0;
  // workgroups per XCD: dispatch is round-robin over the 8 XCDs and nwg is a multiple of 8
  __shared__ int s_nx; if (threadIdx.x == 0) s_nx = nwg / 8; __syncthreads();
  (void)xcc_of;
  for (int g = 1; g <= iters; ++g) {
    for (int i = threadIdx.x; i < dirty_kb * 256; i += blockDim.x) my_scratch[i] = g;               // dirty lines
    int sink = 0; for (int i = threadIdx.x; i < clean_kb * 256; i += blockDim.x) sink += my_scratch[64 * 1024 / 2 + i];
    if (sink == -12345) slots[0] = sink;
    for (int i = threadIdx.x; i < 16; i += blockDim.x) slots[w * 16 + i] = g;
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      release(mode);
      const int prev = __hip_atomic_fetch_add(ctr_xcd + 64 * x, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (prev + 1 == g * s_nx) __hip_atomic_fetch_add(ctr_all, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      unsigned n = 0; int ok = 1;
      const int target = g * 8;                                  // eight XCDs
      if (relay) {
        const bool poller = prev == (g - 1) * s_nx;              // the first arriver of each XCD polls the global word
        if (poller) {
          while (poll(ctr_all) < target) { if (++n > SPIN_LIMIT) { ok = 0; break; } }
          post(local_flag + 64 * x, g);                          // XCD-local flag (same L2 for the XCD's waiters)
        } else {
          while (poll(local_flag + 64 * x) < g) { if (++n > SPIN_LIMIT) { ok = 0; break; } }
        }
      } else {
        while (poll(ctr_all) < target) { if (++n > SPIN_LIMIT) { ok = 0; break; } }
      }
      if (ok) acquire(mode);
      s_ok = ok;
    }
    __syncthreads();
    if (!s_ok) return;                                           // counted as not finished
    for (int i = threadIdx.x; i < 16; i += blockDim.x) err += slots[peer * 16 + i] != g && slots[peer * 16 + i] < g;
    __syncthreads();
  }
  if (err) atomicAdd(errors, err);
  if (threadIdx.x == 0) atomicAdd(progress, 1);                // workgroups that finished every barrier
}

// ------------------------------------------------------------------ kernel boundary
__global__ void __launch_bounds__(1024) k_tiny(int* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1; }

// ------------------------------------------------------------------ GWS
__global__ void __launch_bounds__(1024) __attribute__((target("gws"))) k_gws(int iters, int* progress) {
#if defined(__gfx942__) || defined(__gfx950__)
  if (threadIdx.x == 0 && blockIdx.x == 0) __builtin_amdgcn_ds_gws_init(gridDim.x - 1, 0);
  // the other waves must not use the resource before it is initialised: a grid-wide software barrier first
  __syncthreads();
  for (int g = 0; g < iters; ++g) {
    if (threadIdx.x == 0) __builtin_amdgcn_ds_gws_barrier(gridDim.x - 1, 0);
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(progress, 1);
#endif
}

// ------------------------------------------------------------------ host
static bool wait_stream(hipStream_t s, double seconds) {
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const hipError_t e = hipStreamQuery(s);
    if (e == hipSuccess) return true;
    if (e != hipErrorNotReady) { fprintf(stderr, "stream error %s\n", hipGetErrorString(e)); return false; }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > seconds) return false;
  }
}
static int* alloc(size_t bytes, bool uncached) {
  int* p;
  if (uncached) CHECK(hipExtMallocWithFlags((void**)&p, bytes, hipDeviceMallocUncached));
  else CHECK(hipMalloc(&p, bytes));
  CHECK(hipMemset(p, 0, bytes));
  return p;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string what = argc > 1 ? argv[1] : "all";
  hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, 0));
  const int ncu = prop.multiProcessorCount;
  printf("device %s, %d CUs, clock tick 100 MHz for wall_clock64\n", prop.gcnArchName, ncu);
  hipStream_t st; CHECK(hipStreamCreate(&st));

  // one 1024-thread workgroup per CU and each workgroup's XCD
  uint32_t* dxcc; CHECK(hipMalloc(&dxcc, ncu * 4));
  hipLaunchKernelGGL(k_discover, dim3(ncu), dim3(1024), 0, st, dxcc);
  CHECK(hipStreamSynchronize(st));
  std::vector<uint32_t> xcc(ncu); CHECK(hipMemcpy(xcc.data(), dxcc, ncu * 4, hipMemcpyDeviceToHost));
  int wa = 0, wb_cross = -1, wb_same = -1;
  for (int i = 1; i < ncu; ++i) { if (wb_cross < 0 && xcc[i] != xcc[0]) wb_cross = i; if (wb_same < 0 && xcc[i] == xcc[0]) wb_same = i; }
  printf("workgroup 0 on XCD %u; cross partner %d (XCD %u), same partner %d\n", xcc[0], wb_cross, wb_cross >= 0 ? xcc[wb_cross] : 99, wb_same);

  if (what == "all" || what == "pingpong") {
    const int rounds = 2000;
    for (int mode : {DEV, UC, LIGHT, XCD}) for (int words : {16, 2048, 16384}) {
      const bool uc = mode == UC;
      int* ab = alloc(words * 4, uc); int* ba = alloc(words * 4, uc); int* fl = alloc(128 * 4, false);   // flags: ordinary memory, like ETX's event words
      int *err = alloc(8, false), *prog = alloc(4, false); unsigned long long* ticks; CHECK(hipMalloc(&ticks, 32));
      const int wb = mode == XCD ? wb_same : wb_cross;
      hipLaunchKernelGGL(k_pingpong, dim3(ncu), dim3(1024), 0, st, ab, ba, fl, words, rounds, mode, wa, wb, err, prog, ticks);
      if (!wait_stream(st, 30.0)) { printf("RESULT pingpong mode=%s bytes=%d HUNG\n", mode_name[mode], words * 4); return 3; }
      int e[2], p; unsigned long long t[4];
      CHECK(hipMemcpy(e, err, 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(&p, prog, 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(t, ticks, 32, hipMemcpyDeviceToHost));
      if (p <= rounds) printf("RESULT pingpong mode=%-5s bytes=%-6d NO PROGRESS at round %d\n", mode_name[mode], words * 4, p);
      else printf("RESULT pingpong mode=%-5s bytes=%-6d one-way %.3f us  stale words %d  (XCDs %llu -> %llu%s)\n", mode_name[mode], words * 4,
                  t[0] * 10.0 / 1000.0 / (2.0 * rounds), e[0] + e[1], t[2], t[3],
                  (mode == XCD) == (t[2] == t[3]) ? "" : ", PLACEMENT NOT AS INTENDED");
      CHECK(hipFree(ab)); CHECK(hipFree(ba)); CHECK(hipFree(fl)); CHECK(hipFree(err)); CHECK(hipFree(prog)); CHECK(hipFree(ticks));
    }
  }

  if (what == "all" || what == "barrier") {
    const int iters = 500;
    int* scratch; CHECK(hipMalloc(&scratch, (size_t)ncu * 256 * 1024));
    for (int mode : {DEV, UC}) for (int relay : {0, 1}) for (int load : {0, 1}) {
      if (mode == DEV && relay) continue;                      // the relay's cheap acquire is only valid with uncached data
      const bool uc = mode == UC;
      int* slots = alloc(ncu * 16 * 4, uc); int* cx = alloc(8 * 64 * 4, false); int* ca = alloc(64 * 4, false); int* lf = alloc(8 * 64 * 4, false);
      int *err = alloc(4, false), *prog = alloc(4, false);
      const int dkb = load ? 64 : 0, ckb = load ? 128 : 0;
      hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
      CHECK(hipEventRecord(e0, st));
      hipLaunchKernelGGL(k_barrier, dim3(ncu), dim3(1024), 0, st, slots, cx, ca, lf, iters, mode, relay, dkb, ckb, scratch, dxcc, ncu, err, prog);
      CHECK(hipEventRecord(e1, st));
      if (!wait_stream(st, 60.0)) { printf("RESULT barrier mode=%s relay=%d HUNG\n", mode_name[mode], relay); return 3; }
      float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
      int e, p; CHECK(hipMemcpy(&e, err, 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(&p, prog, 4, hipMemcpyDeviceToHost));
      printf("RESULT barrier  mode=%-5s relay=%d L2 load %3d KB dirty + %3d KB clean per WG: %.2f us per barrier%s  stale %d\n",
             mode_name[mode], relay, dkb, ckb, ms * 1000.0 / iters, p < ncu ? " (NO PROGRESS: some workgroups timed out)" : "", e);
      CHECK(hipFree(slots)); CHECK(hipFree(cx)); CHECK(hipFree(ca)); CHECK(hipFree(lf)); CHECK(hipFree(err)); CHECK(hipFree(prog));
    }
  }

  if (what == "all" || what == "boundary") {
    int* p = alloc(4, false);
    for (int nwg : {1, ncu}) {
      const int K = 400;
      hipGraph_t g; hipGraphExec_t ge;
      CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
      for (int k = 0; k < K; ++k) hipLaunchKernelGGL(k_tiny, dim3(nwg), dim3(1024), 0, st, p);
      CHECK(hipStreamEndCapture(st, &g)); CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
      CHECK(hipGraphLaunch(ge, st)); CHECK(hipStreamSynchronize(st));        // warm
      hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
      CHECK(hipEventRecord(e0, st)); CHECK(hipGraphLaunch(ge, st)); CHECK(hipEventRecord(e1, st)); CHECK(hipStreamSynchronize(st));
      float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
      printf("RESULT boundary %d kernels of %d x 1024 threads in one HIP graph: %.2f us per kernel\n", K, nwg, ms * 1000.0 / K);
    }
  }

  if (what == "gws") {                                          // separate: may be unsupported on the queue
    int* prog = alloc(4, false); const int iters = 500;
    void* args[] = {(void*)&iters, (void*)&prog};
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    CHECK(hipEventRecord(e0, st));
    const hipError_t le = hipLaunchCooperativeKernel((const void*)k_gws, dim3(ncu), dim3(1024), args, 0, st);
    if (le != hipSuccess) { printf("RESULT gws launch failed: %s\n", hipGetErrorString(le)); return 0; }
    CHECK(hipEventRecord(e1, st));
    if (!wait_stream(st, 30.0)) { printf("RESULT gws HUNG (no GWS on this queue?)\n"); return 3; }
    float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); int p; CHECK(hipMemcpy(&p, prog, 4, hipMemcpyDeviceToHost));
    printf("RESULT gws %d workgroups: %.2f us per barrier, %d of %d finished\n", ncu, ms * 1000.0 / iters, p, ncu);
  }
  return 0;
}
