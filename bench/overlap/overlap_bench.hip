// Overlap microbenchmarks for MI300X: can idle memory bandwidth be filled without slowing the latency-bound chain?
// (docs/qwen/PLAN-v3.md section 4.4, the DeepSeek-V2-Lite schedule work). Two questions, each one number:
//
//   overlap_bench stream   read rate of S MB (304 x 256 threads, 8 x 16 B in flight per lane), three passes:
//                          cold (after a 1 GiB flush), warm (the same workgroup reads the same bytes again: L2 + MALL),
//                          mall (each workgroup reads the bytes its neighbour read: another XCD's L2 has them, so they
//                          come from the Infinity Cache). Answers: does a prefetch into the Infinity Cache make the
//                          later expert stream faster than HBM?
//   overlap_bench fold     the latency-bound prologue under load: a foreground workgroup repeatedly invalidates its L2
//                          (buffer_inv sc1, as a cross-XCD consumer does) and reads 64 KB (fleet's 8 x 2048 fp32
//                          partials: one round trip, 16 x 16 B per lane), while background workgroups stream HBM at a
//                          chosen depth (0 = idle). Placement: `same` = one foreground and one background workgroup on
//                          every CU (two requests interleaved); `split` = per XCD, the first F CUs foreground and the
//                          rest background (a latency phase on fewer CUs, a stream on the others); `solo` = one
//                          foreground workgroup per XCD, nothing else (the idle round trip).
//                          Answers: how much does filling the idle bandwidth slow the chain it is meant to hide behind?
//
// Output lines start with RESULT. Every configuration runs under a host deadline and a device stop flag.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(1); } } while (0)

__device__ __forceinline__ uint32_t xcc_id() { return __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20) & 0xF; }
// HW_ID (hwreg 4): CU_ID [11:8], SH_ID [12], SE_ID [15:13] -> one key per CU within an XCC
__device__ __forceinline__ uint32_t cu_key() { return (xcc_id() << 8) | ((__builtin_amdgcn_s_getreg((31 << 11) | (0 << 6) | 4) >> 8) & 0xFF); }
__device__ __forceinline__ int poll(const int* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
typedef unsigned int u4 __attribute__((ext_vector_type(4)));    // native vector: usable as an inline-asm operand
__device__ __forceinline__ uint32_t x4(u4 v) { return v.x ^ v.y ^ v.z ^ v.w; }
__device__ __forceinline__ u4 ld4(const uint4* p) { return *reinterpret_cast<const u4*>(p); }
// All N loads in flight at once: the empty asm needs every value at the same point, so the compiler must issue them
// all before one wait (without it, it folds the xor as the data arrives and keeps only 4 loads in flight).
template <int N> __device__ __forceinline__ void all_live(u4* v);
template <> __device__ __forceinline__ void all_live<1>(u4* v) { asm volatile("" : "+v"(v[0])); }
template <> __device__ __forceinline__ void all_live<2>(u4* v) { asm volatile("" : "+v"(v[0]), "+v"(v[1])); }
template <> __device__ __forceinline__ void all_live<4>(u4* v) { asm volatile("" : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3])); }
template <> __device__ __forceinline__ void all_live<8>(u4* v) {
  asm volatile("" : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3]), "+v"(v[4]), "+v"(v[5]), "+v"(v[6]), "+v"(v[7]));
}
template <> __device__ __forceinline__ void all_live<16>(u4* v) {
  asm volatile("" : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3]), "+v"(v[4]), "+v"(v[5]), "+v"(v[6]), "+v"(v[7]),
                    "+v"(v[8]), "+v"(v[9]), "+v"(v[10]), "+v"(v[11]), "+v"(v[12]), "+v"(v[13]), "+v"(v[14]), "+v"(v[15]));
}

static bool wait_stream(hipStream_t s, double seconds) {
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const hipError_t e = hipStreamQuery(s);
    if (e == hipSuccess) return true;
    if (e != hipErrorNotReady) { fprintf(stderr, "stream error %s\n", hipGetErrorString(e)); return false; }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > seconds) return false;
  }
}

// ------------------------------------------------------------------ stream
constexpr int kThreads = 256, kDepth = 8;
// Workgroup w reads chunks c = w + shift, w + shift + nwg, ... (a chunk = kThreads * kDepth x 16 B = 32 KB).
__global__ void __launch_bounds__(kThreads) k_stream(const uint4* __restrict__ buf, size_t n_chunks, int shift, uint32_t* sink,
                                                     uint32_t* xcc) {
  if (threadIdx.x == 0 && xcc) xcc[blockIdx.x] = xcc_id();
  uint32_t acc = 0;
  for (size_t c = (blockIdx.x + shift) % gridDim.x; c < n_chunks; c += gridDim.x) {
    const uint4* p = buf + c * (kThreads * kDepth) + threadIdx.x;
    u4 v[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d) v[d] = ld4(p + d * kThreads);
    all_live<kDepth>(v);
#pragma unroll
    for (int d = 0; d < kDepth; ++d) acc ^= x4(v[d]);
  }
  if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;          // practically never: keeps the loads alive
}

// ------------------------------------------------------------------ fold under load
// role per workgroup: decided on the device from the CU it landed on (placement `same`) or its slot within the XCD
// (`split`, `solo`); written to roles[] so the host can report what actually happened.
enum Role { NONE = 0, FG = 1, BG = 2 };
enum Place { SAME = 0, SPLIT = 1, SOLO = 2 };
static const char* place_name[] = {"same", "split", "solo"};

template <int D>
__device__ __forceinline__ uint32_t stream_step(const uint4* __restrict__ w, size_t& c, size_t n_chunks, size_t stride) {
  const uint4* p = w + c * (kThreads * 8) + threadIdx.x;
  u4 v[D];
#pragma unroll
  for (int d = 0; d < D; ++d) v[d] = ld4(p + d * kThreads);
  all_live<D>(v);
  uint32_t a = 0;
#pragma unroll
  for (int d = 0; d < D; ++d) a ^= x4(v[d]);
  c += stride; if (c >= n_chunks) c -= n_chunks;
  return a;
}

__global__ void __launch_bounds__(kThreads, 2) k_fold(const uint4* partials, const uint4* __restrict__ weights,
                                                      size_t n_chunks, int place, int fg_per_xcd, int bg_depth, int iters,
                                                      int* cu_slots, int* xcd_slots, int* roles, int* ctr,
                                                      unsigned long long* fold_ticks, unsigned long long* bg_bytes,
                                                      unsigned long long* bg_ticks, uint32_t* sink) {
  __shared__ int s_role;
  __shared__ uint32_t s_red[kThreads / 64];
  if (threadIdx.x == 0) {
    int role = NONE;
    if (place == SAME) {
      const int slot = atomicAdd(&cu_slots[cu_key()], 1);
      role = slot == 0 ? FG : slot == 1 ? BG : NONE;
    } else {
      const int slot = atomicAdd(&xcd_slots[xcc_id()], 1);
      role = slot < fg_per_xcd ? FG : (place == SPLIT ? BG : NONE);
    }
    if (role == BG && bg_depth == 0) role = NONE;
    s_role = role; roles[blockIdx.x] = role;
    if (role == FG) atomicAdd(&ctr[1], 1);          // ctr: [0] claimed, [1] foreground count, [2] foreground done
    __hip_atomic_fetch_add(&ctr[0], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  const int role = s_role;
  uint32_t acc = 0;
  if (role == FG) {
    // one fold per iteration: acquire (invalidate this XCD's L2 of lines from elsewhere), read 64 KB, reduce, barrier
    for (int it = 0; it < iters; ++it) {
      const unsigned long long t0 = wall_clock64();
      if (threadIdx.x == 0) asm volatile("buffer_inv sc1" ::: "memory");
      __syncthreads();
      u4 v[16];
#pragma unroll
      for (int d = 0; d < 16; ++d) v[d] = ld4(partials + d * kThreads + threadIdx.x);
      all_live<16>(v);
      uint32_t a = 0;
#pragma unroll
      for (int d = 0; d < 16; ++d) a ^= x4(v[d]);
      a ^= __shfl_xor(a, 1); a ^= __shfl_xor(a, 2); a ^= __shfl_xor(a, 4); a ^= __shfl_xor(a, 8); a ^= __shfl_xor(a, 16); a ^= __shfl_xor(a, 32);
      if ((threadIdx.x & 63) == 0) s_red[threadIdx.x >> 6] = a;
      __syncthreads();
      acc ^= s_red[0] ^ s_red[1] ^ s_red[2] ^ s_red[3];
      if (threadIdx.x == 0) fold_ticks[(size_t)blockIdx.x * iters + it] = wall_clock64() - t0;
    }
    if (threadIdx.x == 0) atomicAdd(&ctr[2], 1);
  } else if (role == BG) {
    const unsigned long long t0 = wall_clock64();
    size_t c = ((size_t)blockIdx.x * 7919u) % n_chunks, bytes = 0;
    const size_t stride = gridDim.x;
    __shared__ int s_stop;
    for (int k = 0;; ++k) {
      if ((k & 15) == 0) {
        // stop once every workgroup has claimed its role and every foreground one has finished (1 s cap)
        if (threadIdx.x == 0) s_stop = (poll(&ctr[0]) == (int)gridDim.x && poll(&ctr[2]) >= poll(&ctr[1])) || wall_clock64() - t0 > 100000000ull;
        __syncthreads();
        if (s_stop) break;
      }
      switch (bg_depth) {
        case 1: acc ^= stream_step<1>(weights, c, n_chunks, stride); bytes += 1 * 16 * kThreads; break;
        case 2: acc ^= stream_step<2>(weights, c, n_chunks, stride); bytes += 2 * 16 * kThreads; break;
        case 4: acc ^= stream_step<4>(weights, c, n_chunks, stride); bytes += 4 * 16 * kThreads; break;
        default: acc ^= stream_step<8>(weights, c, n_chunks, stride); bytes += 8 * 16 * kThreads; break;
      }
    }
    if (threadIdx.x == 0) { bg_bytes[blockIdx.x] = bytes; bg_ticks[blockIdx.x] = wall_clock64() - t0; }
  }
  if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
}

// ------------------------------------------------------------------ host
static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0; std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string what = argc > 1 ? argv[1] : "all";
  hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, 0));
  const int ncu = prop.multiProcessorCount;
  printf("device %s, %d CUs; wall_clock64 at 100 MHz\n", prop.gcnArchName, ncu);
  hipStream_t st; CHECK(hipStreamCreate(&st));
  uint32_t* sink; CHECK(hipMalloc(&sink, 4096 * 4));
  const size_t kFlush = 1ull << 30, kWeights = 1ull << 30;
  uint4* flush; CHECK(hipMalloc(&flush, kFlush)); CHECK(hipMemset(flush, 1, kFlush));
  const size_t chunk_bytes = (size_t)kThreads * kDepth * 16;

  if (what == "all" || what == "stream") {
    uint32_t* dxcc; CHECK(hipMalloc(&dxcc, ncu * 4));
    for (double mb : {8.0, 17.3, 34.6, 64.0, 104.0, 128.0, 192.0}) {
      const size_t n_chunks = std::max<size_t>(ncu, (size_t)(mb * 1e6 / chunk_bytes));
      uint4* buf; CHECK(hipMalloc(&buf, n_chunks * chunk_bytes)); CHECK(hipMemset(buf, 3, n_chunks * chunk_bytes));
      std::vector<double> cold, warm, mall;
      hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
      for (int rep = 0; rep < 5; ++rep) {
        hipLaunchKernelGGL(k_stream, dim3(ncu), dim3(kThreads), 0, st, flush, kFlush / chunk_bytes, 0, sink, nullptr);   // flush L2 + MALL
        for (int pass = 0; pass < 3; ++pass) {
          CHECK(hipEventRecord(e0, st));
          hipLaunchKernelGGL(k_stream, dim3(ncu), dim3(kThreads), 0, st, buf, n_chunks, pass == 2 ? 1 : 0, sink, pass == 0 ? dxcc : nullptr);
          CHECK(hipEventRecord(e1, st));
          if (!wait_stream(st, 30.0)) { printf("RESULT stream %.1f MB HUNG\n", mb); return 3; }
          float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
          (pass == 0 ? cold : pass == 1 ? warm : mall).push_back(n_chunks * chunk_bytes / (ms * 1e-3) / 1e12);
        }
      }
      std::vector<uint32_t> xcc(ncu); CHECK(hipMemcpy(xcc.data(), dxcc, ncu * 4, hipMemcpyDeviceToHost));
      int same = 0; for (int i = 0; i < ncu; ++i) same += xcc[i] == xcc[(i + 1) % ncu];
      printf("RESULT stream %6.1f MB  cold %.2f TB/s  warm %.2f TB/s  mall %.2f TB/s  (median of 5; mall pass: %d of %d neighbours share an XCD%s)\n",
             n_chunks * chunk_bytes / 1e6, pct(cold, 0.5), pct(warm, 0.5), pct(mall, 0.5), same, ncu, same > ncu / 8 ? ", MALL PASS PARTLY L2" : "");
      CHECK(hipFree(buf));
    }
  }

  if (what == "all" || what == "fold") {
    uint4* partials; CHECK(hipMalloc(&partials, 64 * 1024)); CHECK(hipMemset(partials, 5, 64 * 1024));
    uint4* weights; CHECK(hipMalloc(&weights, kWeights)); CHECK(hipMemset(weights, 7, kWeights));
    const size_t n_chunks = kWeights / ((size_t)kThreads * 8 * 16);
    const int iters = 200, warm = 20, nwg_max = 2 * ncu;
    int *cu_slots, *xcd_slots, *roles, *ctr;
    CHECK(hipMalloc(&cu_slots, 4096 * 4)); CHECK(hipMalloc(&xcd_slots, 64 * 4)); CHECK(hipMalloc(&roles, nwg_max * 4)); CHECK(hipMalloc(&ctr, 16));
    unsigned long long *fold_ticks, *bg_bytes, *bg_ticks;
    CHECK(hipMalloc(&fold_ticks, (size_t)nwg_max * iters * 8)); CHECK(hipMalloc(&bg_bytes, nwg_max * 8)); CHECK(hipMalloc(&bg_ticks, nwg_max * 8));
    struct Cfg { int place, fg_per_xcd, depth; };
    std::vector<Cfg> cfgs;
    cfgs.push_back({SOLO, 1, 0});                                   // idle round trip, 8 folds
    for (int d : {0, 1, 2, 4, 8}) cfgs.push_back({SAME, 0, d});     // every CU folds; d = background depth on the same CUs
    for (int f : {8, 19, 30}) for (int d : {0, 8}) cfgs.push_back({SPLIT, f, d});   // f CUs per XCD fold, the rest stream
    for (const Cfg& c : cfgs) {
      const int nwg = c.place == SAME ? 2 * ncu : ncu;
      CHECK(hipMemset(cu_slots, 0, 4096 * 4)); CHECK(hipMemset(xcd_slots, 0, 64 * 4)); CHECK(hipMemset(roles, 0, nwg_max * 4));
      CHECK(hipMemset(ctr, 0, 16)); CHECK(hipMemset(fold_ticks, 0, (size_t)nwg_max * iters * 8));
      CHECK(hipMemset(bg_bytes, 0, nwg_max * 8)); CHECK(hipMemset(bg_ticks, 0, nwg_max * 8));
      // dynamic LDS pins the residency: 28 KB -> at most two workgroups per CU (same: exactly one fg + one bg per CU),
      // 40 KB -> one per CU (split / solo: one workgroup per CU, as ETX's persistent workers)
      const size_t lds = c.place == SAME ? 28 * 1024 : 40 * 1024;
      hipLaunchKernelGGL(k_fold, dim3(nwg), dim3(kThreads), lds, st, partials, weights, n_chunks, c.place, c.fg_per_xcd, c.depth, iters,
                         cu_slots, xcd_slots, roles, ctr, fold_ticks, bg_bytes, bg_ticks, sink);
      if (!wait_stream(st, 20.0)) { printf("RESULT fold place=%s HUNG\n", place_name[c.place]); return 3; }
      std::vector<int> r(nwg); CHECK(hipMemcpy(r.data(), roles, nwg * 4, hipMemcpyDeviceToHost));
      std::vector<unsigned long long> ft((size_t)nwg * iters), bb(nwg), bt(nwg);
      CHECK(hipMemcpy(ft.data(), fold_ticks, ft.size() * 8, hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(bb.data(), bg_bytes, nwg * 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(bt.data(), bg_ticks, nwg * 8, hipMemcpyDeviceToHost));
      int fg = 0, bg = 0; std::vector<double> us; double bytes = 0, tmax = 0;
      for (int w = 0; w < nwg; ++w) {
        if (r[w] == FG) { ++fg; for (int it = warm; it < iters; ++it) us.push_back(ft[(size_t)w * iters + it] * 0.01); }
        if (r[w] == BG) { ++bg; bytes += bb[w]; tmax = std::max(tmax, bt[w] * 0.01); }
      }
      const double tbs = tmax > 0 ? bytes / (tmax * 1e-6) / 1e12 : 0;
      char where[64] = ""; if (c.place == SPLIT) snprintf(where, sizeof where, "fg %d CUs/XCD", c.fg_per_xcd);
      printf("RESULT fold place=%-5s %-14s bg depth %d: fold median %6.2f us  p90 %6.2f us  | %3d fg  %3d bg workgroups, bg %.2f TB/s%s\n",
             place_name[c.place], where, c.depth, pct(us, 0.5), pct(us, 0.9), fg, bg, tbs,
             (c.place == SAME && c.depth > 0 && bg < ncu * 9 / 10) ? "  (PLACEMENT: fewer than 90% of CUs got a background workgroup)" : "");
    }
  }
  return 0;
}
