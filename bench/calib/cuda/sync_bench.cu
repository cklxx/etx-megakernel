// CUDA syntax port of bench/sync/sync_bench.hip. Status: compiled, never run.
// Source comments below describe the HIP experiment; see README.md for CUDA mappings and exclusions.
// Synchronisation microbenchmarks for MI300X (docs/core/RESEARCH-sync-hardware.md section 3). One question:
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
//          (measured stale 2026-09-27: sc0 leaves the consumer's L1 stale; uc1 / xcd1 use sc1 and are clean)
//          (L1 only); counters and flags stay in ordinary memory in every mode, as ETX's event words do
//   light  cached memory with the uc fences: expected to go STALE across XCDs (proves the check detects it)
//   xcd    same-XCD pair, cached memory, XCD fences (s_waitcnt / buffer_inv sc0): the floor
// Output lines start with RESULT; a hand-off that cannot complete reports NO PROGRESS instead of hanging.
#include "port.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

enum Mode { DEV = 0, UC = 1, LIGHT = 2, XCD = 3, UC1 = 4, XCD1 = 5, XN = 6, XL0 = 7, XL1 = 8, CL1 = 9, CL01 = 10 };
static const char* mode_name[] = {"dev", "uc", "light", "xcd", "uc1", "xcd1", "xn", "xl0", "xl1", "cl1", "cl01"};
// Load-scope modes (2026-10-13, `sync_bench loadscope`): NO acquire invalidate; the consumer reads the payload with
//   xn   plain loads, same XCD (control: must read stale, as xcd did)
//   xl0  global_load sc0, same XCD          xl1  global_load sc1, same XCD
//   cl1  global_load sc1, cross XCD, producer release buffer_wbl2 sc1
//   cl01 global_load sc0 sc1, cross XCD, producer release buffer_wbl2 sc1 (D2's coherent_loads form)
// The question: which load skips the consumer's stale L1 line (and, same XCD, still hits the shared L2), so that the
// per-task buffer_inv sc1 (D1's safe acquire, 232 us/token) can go without D2's per-task write-back.
// uc1 / xcd1: as uc / xcd but the acquire is buffer_inv sc1 (agent scope). Added 2026-09-27 after uc and xcd read stale
// data at 64 B and 8 KB (not at 64 KB): the hypothesis is that buffer_inv sc0 (workgroup scope) does not invalidate
// the vector L1 outside threadgroup-split mode, so a consumer re-reading the same lines sees its own stale L1 copy.
static bool is_uc(int m) { return m == UC || m == UC1; }
static bool is_xcd(int m) { return m == XCD || m == XCD1 || m == XN || m == XL0 || m == XL1; }
__host__ __device__ static bool no_inv(int m) { return m >= XN; }

__device__ __forceinline__ uint32_t xcc_id() { return cuda_domain(); }
__device__ __forceinline__ void release(int mode) {
  if (mode == DEV || mode == CL1 || mode == CL01) __threadfence();
  else __threadfence();
}
__device__ __forceinline__ void acquire(int mode) {
  if (no_inv(mode)) return;
  if (mode == DEV || mode == UC1 || mode == XCD1) __threadfence();
  else __threadfence();
}
// payload load with the mode's cache bits (each load waited for: correctness test, so its one-way time is not
// comparable with the plain-load modes)
__device__ __forceinline__ int ld_payload(const int* p, int mode) {
  int v;
  if (mode == XL0) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  else if (mode == XL1 || mode == CL1) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  else if (mode == CL01) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  else v = *p;
  return v;
}
__device__ __forceinline__ int poll(const int* p) { return port_load(p); }
__device__ __forceinline__ void post(int* p, int v) { port_store(p, v); }
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
        __threadfence();           // every wave drains its stores (to L2, or memory for UC)
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
        for (int i = threadIdx.x; i < words; i += blockDim.x) bad += ld_payload(in + i, mode) != r;
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
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      release(mode);
      const int prev = atomicAdd(ctr_xcd + 64 * x, 1);
      if (prev + 1 == g * s_nx) atomicAdd(ctr_all, 1);
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
__global__ void __launch_bounds__(1024)  k_gws(int iters, int* progress) {
  // Host rejects GWS: CUDA has no DS_GWS resource. Trap if called accidentally.
  asm volatile("trap;");
}

// ------------------------------------------------------------------ host
static bool wait_stream(cudaStream_t s, double seconds) {
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const cudaError_t e = cudaStreamQuery(s);
    if (e == cudaSuccess) return true;
    if (e != cudaErrorNotReady) { fprintf(stderr, "stream error %s\n", cudaGetErrorString(e)); return false; }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > seconds) return false;
  }
}
static int* alloc(size_t bytes, bool uncached) {
  int* p;
  if (uncached) CHECK(cudaMalloc((void**)&p, bytes));
  else CHECK(cudaMalloc(&p, bytes));
  CHECK(cudaMemset(p, 0, bytes));
  return p;
}



int main(int argc, char** argv) {
  if (port_help(argc, argv, "sync_bench boundary (AMD cache/XCD/GWS modes unsupported)")) return 0;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string what = argc > 1 ? argv[1] : "all";

  if (what != "boundary") { fprintf(stderr, "Unsupported CUDA experiment: %s (see README.md)\n", what.c_str()); return 2; }
  cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
  const int ncu = prop.multiProcessorCount;
  printf("device %s, %d CUs, clock tick 100 MHz for wall_clock64\n", prop.name, ncu);
  cudaStream_t st; CHECK(cudaStreamCreate(&st));

  // one 1024-thread workgroup per CU and each workgroup's XCD
  uint32_t* dxcc; CHECK(cudaMalloc(&dxcc, ncu * 4));
  PORT_LAUNCH(k_discover, dim3(ncu), dim3(1024), 0, st, dxcc);
  CHECK(cudaStreamSynchronize(st));
  std::vector<uint32_t> xcc(ncu); CHECK(cudaMemcpy(xcc.data(), dxcc, ncu * 4, cudaMemcpyDeviceToHost));
  int wa = 0, wb_cross = -1, wb_same = -1;
  for (int i = 1; i < ncu; ++i) { if (wb_cross < 0 && xcc[i] != xcc[0]) wb_cross = i; if (wb_same < 0 && xcc[i] == xcc[0]) wb_same = i; }
  printf("workgroup 0 on XCD %u; cross partner %d (XCD %u), same partner %d\n", xcc[0], wb_cross, wb_cross >= 0 ? xcc[wb_cross] : 99, wb_same);

  if (what == "all" || what == "pingpong" || what == "loadscope") {
    const int rounds = 2000;
    const std::vector<int> modes = what == "loadscope" ? std::vector<int>{XCD, XCD1, XN, XL0, XL1, DEV, CL1, CL01}
                                                       : std::vector<int>{DEV, UC, LIGHT, XCD, UC1, XCD1};
    for (int mode : modes) for (int words : {16, 2048, 16384}) {
      const bool uc = is_uc(mode);
      int* ab = alloc(words * 4, uc); int* ba = alloc(words * 4, uc); int* fl = alloc(128 * 4, false);   // flags: ordinary memory, like ETX's event words
      int *err = alloc(8, false), *prog = alloc(4, false); unsigned long long* ticks; CHECK(cudaMalloc(&ticks, 32));
      const int wb = is_xcd(mode) ? wb_same : wb_cross;
      PORT_LAUNCH(k_pingpong, dim3(ncu), dim3(1024), 0, st, ab, ba, fl, words, rounds, mode, wa, wb, err, prog, ticks);
      if (!wait_stream(st, 30.0)) { printf("RESULT pingpong mode=%s bytes=%d HUNG\n", mode_name[mode], words * 4); return 3; }
      int e[2], p; unsigned long long t[4];
      CHECK(cudaMemcpy(e, err, 8, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(&p, prog, 4, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(t, ticks, 32, cudaMemcpyDeviceToHost));
      if (p <= rounds) printf("RESULT pingpong mode=%-5s bytes=%-6d NO PROGRESS at round %d\n", mode_name[mode], words * 4, p);
      else printf("RESULT pingpong mode=%-5s bytes=%-6d one-way %.3f us  stale words %d  (XCDs %llu -> %llu%s)\n", mode_name[mode], words * 4,
                  t[0] * 10.0 / 1000.0 / (2.0 * rounds), e[0] + e[1], t[2], t[3],
                  is_xcd(mode) == (t[2] == t[3]) ? "" : ", PLACEMENT NOT AS INTENDED");
      CHECK(cudaFree(ab)); CHECK(cudaFree(ba)); CHECK(cudaFree(fl)); CHECK(cudaFree(err)); CHECK(cudaFree(prog)); CHECK(cudaFree(ticks));
    }
  }

  if (what == "all" || what == "barrier") {
    const int iters = 500;
    int* scratch; CHECK(cudaMalloc(&scratch, (size_t)ncu * 256 * 1024));
    for (int mode : {DEV, UC, UC1}) for (int relay : {0, 1}) for (int load : {0, 1}) {
      if (mode == DEV && relay) continue;                      // the relay's cheap acquire is only valid with uncached data
      const bool uc = is_uc(mode);
      int* slots = alloc(ncu * 16 * 4, uc); int* cx = alloc(8 * 64 * 4, false); int* ca = alloc(64 * 4, false); int* lf = alloc(8 * 64 * 4, false);
      int *err = alloc(4, false), *prog = alloc(4, false);
      const int dkb = load ? 64 : 0, ckb = load ? 128 : 0;
      cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
      CHECK(cudaEventRecord(e0, st));
      PORT_LAUNCH(k_barrier, dim3(ncu), dim3(1024), 0, st, slots, cx, ca, lf, iters, mode, relay, dkb, ckb, scratch, dxcc, ncu, err, prog);
      CHECK(cudaEventRecord(e1, st));
      if (!wait_stream(st, 60.0)) { printf("RESULT barrier mode=%s relay=%d HUNG\n", mode_name[mode], relay); return 3; }
      float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
      int e, p; CHECK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(&p, prog, 4, cudaMemcpyDeviceToHost));
      printf("RESULT barrier  mode=%-5s relay=%d L2 load %3d KB dirty + %3d KB clean per WG: %.2f us per barrier%s  stale %d\n",
             mode_name[mode], relay, dkb, ckb, ms * 1000.0 / iters, p < ncu ? " (NO PROGRESS: some workgroups timed out)" : "", e);
      CHECK(cudaFree(slots)); CHECK(cudaFree(cx)); CHECK(cudaFree(ca)); CHECK(cudaFree(lf)); CHECK(cudaFree(err)); CHECK(cudaFree(prog));
    }
  }

  if (what == "all" || what == "boundary") {
    int* p = alloc(4, false);
    for (int nwg : {1, ncu}) {
      const int K = 400;
      cudaGraph_t g; cudaGraphExec_t ge;
      CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      for (int k = 0; k < K; ++k) PORT_LAUNCH(k_tiny, dim3(nwg), dim3(1024), 0, st, p);
      CHECK(cudaStreamEndCapture(st, &g)); CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
      CHECK(cudaGraphLaunch(ge, st)); CHECK(cudaStreamSynchronize(st));        // warm
      cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
      CHECK(cudaEventRecord(e0, st)); CHECK(cudaGraphLaunch(ge, st)); CHECK(cudaEventRecord(e1, st)); CHECK(cudaStreamSynchronize(st));
      float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
      printf("RESULT boundary %d kernels of %d x 1024 threads in one HIP graph: %.2f us per kernel\n", K, nwg, ms * 1000.0 / K);
    }
  }

  if (what == "gws") {                                          // separate: may be unsupported on the queue
    int* prog = alloc(4, false); const int iters = 500;
    void* args[] = {(void*)&iters, (void*)&prog};
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0, st));
    const cudaError_t le = cudaLaunchCooperativeKernel((const void*)k_gws, dim3(ncu), dim3(1024), args, 0, st);
    if (le != cudaSuccess) { printf("RESULT gws launch failed: %s\n", cudaGetErrorString(le)); return 0; }
    CHECK(cudaEventRecord(e1, st));
    if (!wait_stream(st, 30.0)) { printf("RESULT gws HUNG (no GWS on this queue?)\n"); return 3; }
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1)); int p; CHECK(cudaMemcpy(&p, prog, 4, cudaMemcpyDeviceToHost));
    printf("RESULT gws %d workgroups: %.2f us per barrier, %d of %d finished\n", ncu, ms * 1000.0 / iters, p, ncu);
  }
  return 0;
}
