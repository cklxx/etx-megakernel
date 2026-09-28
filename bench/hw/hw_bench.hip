// Hardware primitives of the DeepSeek decode model (docs/MODEL-dsv2.md, parameters H2, H5-H11, H13, pairs E1-E4). Every
// mode prints RESULT lines; times from s_memrealtime (100 MHz, one clock for the whole GPU) unless noted.
//
//   hw_bench stream    phase replica: A workgroups (spread evenly over the 8 XCDs) each stream B bytes of weights,
//                      contiguous per workgroup or as fleet's rows (row r of the phase on worker r mod W), 16 B loads,
//                      8 in flight per lane; all start together (device barrier). Reports the phase span (first start to
//                      last end), the median workgroup time and the achieved TB/s. Sizes = the real phases' bytes per task.
//   hw_bench shared    staging replica: 8 producers (one per XCD) each write 8 KB of a 64 KB vector and release (write-back),
//                      then N readers (N / 8 per XCD) acquire and read all 64 KB (16 x 16 B per lane in flight), vs the same
//                      vector already in the reader's own L2. Reports the median and last reader time after the release.
//   hw_bench handoff   ETX event replica: 37 producers per XCD arrive on an XCD-local counter (or all 296 on the device
//                      protocol: per-XCD sub-counter, last arriver writes back and bumps the device word); the same
//                      workgroups then wait (poll, s_sleep S between polls) and acquire. Reports last arrival -> first /
//                      median / last wake-up, for S in {0, 1, 8, 32}, acquire sc0 / sc1, with and without 64 KB of dirty
//                      lines per producer.
//   hw_bench wg        workgroup primitives, 1 workgroup per CU on every CU: __syncthreads, LDS 8 KB write+read, a 64-lane
//                      shuffle reduction, a 4-wave LDS reduction, fp32 FMA rate, bf16->fp32 dot rate, expf / rsqrtf rate.
//   hw_bench clock     shader clock (s_memtime cycles per s_memrealtime tick) idle and while every CU streams.
//   hw_bench pair      two workgroups per CU: a foreground body X timed beside a background body Y on the same CU, for
//                      X, Y in {stream, chain (fold-like: 64 KB shared read + LDS reduce + 2 barriers + a 2 KB norm), alu,
//                      lds, idle}. Reports X's time with Y / X's time alone.
// Added 2026-09-28 for the vLLM-kernel model (docs/MODEL-vllm-kernels.md: every parameter the simulator used to fit):
//   hw_bench lat       dependent round trip (pointer chase, one wave, 2000 hops over distinct 128 B lines in random order),
//                      vector (global_load) and scalar (s_load) loads, the line in L1 / K$, L2, MALL or HBM, optionally a
//                      buffer_inv sc0 / sc1 before every hop (the state after an ETX acquire), with the other 303 CUs idle
//                      or streaming at full rate (loaded latency). Replaces the fitted t_v, t_s and the argument path.
//   hw_bench lf        Little's law: A workgroups (256 threads) each stream B bytes with D x 16 B per lane in flight
//                      (D = 1..16: 4-64 KB per workgroup); rate against bytes in flight. Replaces the fitted bandwidth.
//   hw_bench launch    the kernel boundary with work: 200 kernels in one HIP graph, each reads the previous kernel's output
//                      and chases H dependent loads (H = -1: an empty kernel). Splits the fitted F into boundary + hops.
//   hw_bench evx       ETX's own event protocol (the gfx942 lowering: fetch_sub arrive, poll = buffer_inv sc0 + load with
//                      s_sleep 1, acquire sc0 / sc1, the last-arriver flush for DEVICE scope) with P producers and C
//                      consumers, as the plan's events are shaped; also the wait on an event already complete.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(1); } } while (0)
typedef unsigned int u4 __attribute__((ext_vector_type(4)));
constexpr int kT = 256;                                   // threads per workgroup, as fleet's tiles

__device__ __forceinline__ uint32_t xcc_id() { return __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20) & 0xF; }
__device__ __forceinline__ uint32_t cu_key() { return (xcc_id() << 8) | ((__builtin_amdgcn_s_getreg((31 << 11) | (0 << 6) | 4) >> 8) & 0xFF); }
__device__ __forceinline__ int ld_agent(const int* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ uint64_t now() { return wall_clock64(); }
__device__ __forceinline__ uint32_t xs(u4 v) { return v.x ^ v.y ^ v.z ^ v.w; }
__device__ __forceinline__ void live8(u4* v) {
  asm volatile("" : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3]), "+v"(v[4]), "+v"(v[5]), "+v"(v[6]), "+v"(v[7]));
}
__device__ __forceinline__ void live16(u4* v) { live8(v); live8(v + 8); }

static bool wait_stream(hipStream_t s, double sec) {
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const hipError_t e = hipStreamQuery(s);
    if (e == hipSuccess) return true;
    if (e != hipErrorNotReady) { fprintf(stderr, "stream error %s\n", hipGetErrorString(e)); return false; }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > sec) return false;
  }
}
static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))]; }

// slot within this workgroup's XCD (claimed once) and a device barrier over all workgroups of the launch
struct Ctl { int xslot[64]; int bar; int pad[63]; int err; };
__device__ int claim_xslot(Ctl* c) { return atomicAdd(&c->xslot[xcc_id()], 1); }
__device__ void grid_barrier(Ctl* c, int gen) {           // gen = 1, 2, 3 ...; every workgroup of the grid
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    atomicAdd(&c->bar, 1);
    while (ld_agent(&c->bar) < gen * (int)gridDim.x) __builtin_amdgcn_s_sleep(1);
    asm volatile("buffer_inv sc1" ::: "memory");
  }
  __syncthreads();
}

// ------------------------------------------------------------------ stream
// active workgroup k (per XCD slot s < A/8) streams `bytes` from region base; rows mode: 4 KB rows (K = 2048 bf16), row r of
// the phase belongs to worker r % W where W = A, so worker w reads rows w, w + A, ... (fleet's distribution)
__global__ void __launch_bounds__(kT) k_stream(const u4* __restrict__ w, size_t buf_u4, int active, int64_t bytes, int rows_mode,
                                               int iters, Ctl* ctl, uint64_t* t_start, uint64_t* t_end, uint32_t* sink) {
  __shared__ int s_slot;
  if (threadIdx.x == 0) s_slot = claim_xslot(ctl);
  __syncthreads();
  const int per_x = active / 8, slot = s_slot, wid = (int)xcc_id() * per_x + slot;
  const bool on = slot < per_x;
  uint32_t acc = 0;
  for (int it = 0; it < iters; ++it) {
    grid_barrier(ctl, it + 1);
    if (!on) continue;
    const uint64_t t0 = now();
    const size_t region = ((size_t)it * (size_t)active * (size_t)bytes / 16) % (buf_u4 / 2);   // a fresh region per iteration
    const int64_t n16 = bytes / 16;
    if (!rows_mode) {
      const u4* p = w + region + (size_t)wid * n16;
      for (int64_t i = threadIdx.x; i < n16; i += kT * 8) {
        u4 v[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) v[d] = (i + d * kT < n16) ? p[i + d * kT] : u4{0, 0, 0, 0};
        live8(v);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc ^= xs(v[d]);
      }
    } else {
      const int64_t row16 = 4096 / 16, my_rows = bytes / 4096 > 0 ? bytes / 4096 : 1;
      for (int64_t r = 0; r < my_rows; r += 8) {               // 8 rows in flight, one 16 B chunk per lane per row
        u4 v[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) v[d] = (r + d < my_rows) ? w[region + (size_t)((r + d) * active + wid) * row16 + threadIdx.x] : u4{0, 0, 0, 0};
        live8(v);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc ^= xs(v[d]);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) { t_start[(size_t)it * active + wid] = t0; t_end[(size_t)it * active + wid] = now(); }
  }
  if (acc == 0x12345u) sink[blockIdx.x] = acc;
}

// ------------------------------------------------------------------ shared (staging replica)
__global__ void __launch_bounds__(kT) k_shared(float* vec, int readers, int remote, int iters, Ctl* ctl,
                                               uint64_t* t_rel, uint64_t* t_done, uint32_t* sink) {
  __shared__ int s_slot;
  if (threadIdx.x == 0) s_slot = claim_xslot(ctl);
  __syncthreads();
  const int x = (int)xcc_id(), slot = s_slot, per_x = readers / 8;
  const bool producer = slot == 0, reader = slot >= 1 && slot <= per_x;
  uint32_t acc = 0;
  for (int it = 0; it < iters; ++it) {
    grid_barrier(ctl, 2 * it + 1);
    // produce: each XCD's producer writes its 8 KB of the 64 KB vector (remote mode), or XCD 0 writes all (local mode for XCD 0)
    if (producer && (remote || x == 0)) {
      const int lo = remote ? x * 2048 : 0, hi = remote ? lo + 2048 : 16384;
      for (int i = lo + threadIdx.x; i < hi; i += kT) vec[i] = (float)(it + i);
      __syncthreads();
      if (threadIdx.x == 0) asm volatile("s_waitcnt vmcnt(0)\n\tbuffer_wbl2 sc1\n\ts_waitcnt vmcnt(0)" ::: "memory");
    }
    grid_barrier(ctl, 2 * it + 2);          // the release point (every reader starts from here)
    const uint64_t t0 = now();
    if (reader && (remote || x == 0)) {
      if (threadIdx.x == 0) asm volatile("buffer_inv sc1" ::: "memory");
      __syncthreads();
      const u4* p = reinterpret_cast<const u4*>(vec);
      u4 v[16];
#pragma unroll
      for (int d = 0; d < 16; ++d) v[d] = p[d * kT + threadIdx.x];
      live16(v);
#pragma unroll
      for (int d = 0; d < 16; ++d) acc ^= xs(v[d]);
      __syncthreads();
      if (threadIdx.x == 0) { t_rel[(size_t)it * 8 * 64 + x * 64 + slot] = t0; t_done[(size_t)it * 8 * 64 + x * 64 + slot] = now(); }
    }
  }
  if (acc == 0x12345u) sink[blockIdx.x] = acc;
}

// ------------------------------------------------------------------ handoff (ETX event replica)
// scope 0: XCD-local (37 producers -> the same 37 workgroups wait); scope 1: device (per-XCD sub-counter, last arriver
// writes back and adds to the device word; all 296 wait on it). Producers finish at a common time (after a barrier + a
// staggered spin to mimic a phase tail of `skew` us).
__global__ void __launch_bounds__(kT) k_handoff(int* counters, int pad, int scope, int sleep, int acq_sc1, int dirty_kb, int skew_ticks,
                                                int iters, Ctl* ctl, float* scratch, uint64_t* t_arr, uint64_t* t_wake) {
  __shared__ int s_slot;
  if (threadIdx.x == 0) s_slot = claim_xslot(ctl);
  __syncthreads();
  const int x = (int)xcc_id(), slot = s_slot;
  const bool on = slot < 37;
  const int wid = x * 37 + slot;
  for (int it = 0; it < iters; ++it) {
    grid_barrier(ctl, it + 1);
    if (!on) continue;
    // "work": dirty lines, then a per-worker delay spreading the arrivals over skew_ticks
    float* my = scratch + (size_t)wid * 16384;
    for (int i = threadIdx.x; i < dirty_kb * 256; i += kT) my[i] = (float)(it + i);
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint64_t until = now() + (uint64_t)((slot * 7919 + x * 131) % (skew_ticks + 1));
      while (now() < until) {}
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      int* xc = counters + (size_t)it * 64 * pad + x * pad;    // per-iteration words: no reset needed; pad 1 = 8 XCDs in one line
      int* dev = counters + (size_t)it * 64 * pad + 32 * pad;
      if (scope == 0) {
        t_arr[(size_t)it * 296 + wid] = now();
        atomicAdd(xc, 1);
      } else {
        const int prev = atomicAdd(xc, 1);
        if (prev == 36) { asm volatile("buffer_wbl2 sc1\n\ts_waitcnt vmcnt(0)" ::: "memory"); t_arr[(size_t)it * 296 + wid] = now(); atomicAdd(dev, 1); }
        else t_arr[(size_t)it * 296 + wid] = 0;
      }
      const int* wword = scope == 0 ? xc : dev;
      const int target = scope == 0 ? 37 : 8;
      while (ld_agent(wword) < target) {
        if (sleep == 1) __builtin_amdgcn_s_sleep(1);
        else if (sleep == 8) __builtin_amdgcn_s_sleep(8);
        else if (sleep == 32) __builtin_amdgcn_s_sleep(32);
      }
      if (acq_sc1) asm volatile("buffer_inv sc1" ::: "memory"); else asm volatile("buffer_inv sc0" ::: "memory");
      t_wake[(size_t)it * 296 + wid] = now();
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------ wg primitives and clock
__global__ void __launch_bounds__(kT) k_wg(int what, int reps, uint64_t* t, float* sink) {
  __shared__ float lds[2048 + 64];
  float a = threadIdx.x * 1e-3f, b = 1.0001f;
  const uint64_t t0 = now();
  const uint64_t c0 = __builtin_readcyclecounter();
  for (int r = 0; r < reps; ++r) {
    if (what == 0) { __syncthreads(); }
    else if (what == 1) { for (int i = threadIdx.x; i < 2048; i += kT) lds[i] = a + i; __syncthreads(); a += lds[(threadIdx.x * 5 + r) & 2047]; __syncthreads(); }
    else if (what == 2) { float v = a; v += __shfl_xor(v, 1); v += __shfl_xor(v, 2); v += __shfl_xor(v, 4); v += __shfl_xor(v, 8); v += __shfl_xor(v, 16); v += __shfl_xor(v, 32); a = v * 1e-6f; }
    else if (what == 3) { float v = a; v += __shfl_xor(v, 1); v += __shfl_xor(v, 2); v += __shfl_xor(v, 4); v += __shfl_xor(v, 8); v += __shfl_xor(v, 16); v += __shfl_xor(v, 32);
                          if ((threadIdx.x & 63) == 0) lds[2048 + (threadIdx.x >> 6)] = v; __syncthreads(); a = (lds[2048] + lds[2049] + lds[2050] + lds[2051]) * 1e-6f; __syncthreads(); }
    else if (what == 4) {
#pragma unroll
      for (int i = 0; i < 64; ++i) a = __builtin_fmaf(a, b, 1e-7f);
    }
    else if (what == 5) { float s = 0; const unsigned u = __float_as_uint(a);
#pragma unroll
      for (int i = 0; i < 32; ++i) { const unsigned w = u + i * 0x10001u; s += __uint_as_float(w << 16) * a + __uint_as_float(w & 0xFFFF0000u) * b; } a += s * 1e-9f; }
    else if (what == 6) {
#pragma unroll
      for (int i = 0; i < 16; ++i) a = __expf(a * -1e-3f) + rsqrtf(a * a + 1.0f);
    }
  }
  const uint64_t c1 = __builtin_readcyclecounter();
  __syncthreads();
  if (threadIdx.x == 0) { t[blockIdx.x * 2] = now() - t0; t[blockIdx.x * 2 + 1] = c1 - c0; }
  if (a == 123.456f) sink[blockIdx.x] = a;
}

// ------------------------------------------------------------------ pair (two workgroups per CU)
enum Body { B_IDLE = 0, B_STREAM = 1, B_CHAIN = 2, B_ALU = 3, B_LDS = 4 };
static const char* body_name[] = {"idle", "stream", "chain", "alu", "lds"};
__device__ uint32_t run_body(int body, const u4* w, size_t buf_u4, const float* shared_vec, float* lds, int rep, int wid) {
  uint32_t acc = 0;
  if (body == B_STREAM) {                                     // 128 KB per call, 8 x 16 B in flight per lane
    const u4* p = w + ((size_t)(rep * 608 + wid) * 8192) % (buf_u4 - 8192);
    for (int i = threadIdx.x; i < 8192; i += kT * 8) {
      u4 v[8];
#pragma unroll
      for (int d = 0; d < 8; ++d) v[d] = p[i + d * kT];
      live8(v);
      float s = 0;
#pragma unroll
      for (int d = 0; d < 8; ++d) s += __uint_as_float(v[d].x << 16) * __uint_as_float(v[d].y & 0xFFFF0000u);   // a bf16 dot's ALU
      acc ^= __float_as_uint(s);
    }
  } else if (body == B_CHAIN) {                               // fold-like prologue: 64 KB shared read, reduce, LDS, 2 barriers, norm
    if (threadIdx.x == 0) asm volatile("buffer_inv sc1" ::: "memory");
    __syncthreads();
    const u4* p = reinterpret_cast<const u4*>(shared_vec);
    u4 v[16];
#pragma unroll
    for (int d = 0; d < 16; ++d) v[d] = p[d * kT + threadIdx.x];
    live16(v);
    float s = 0;
#pragma unroll
    for (int d = 0; d < 16; ++d) s += __uint_as_float(v[d].x) + __uint_as_float(v[d].y) + __uint_as_float(v[d].z) + __uint_as_float(v[d].w);
    for (int i = threadIdx.x; i < 2048; i += kT) lds[i] = s + i;
    __syncthreads();
    float ss = 0;
    for (int i = threadIdx.x; i < 2048; i += kT) ss += lds[i] * lds[i];
    ss += __shfl_xor(ss, 1); ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 4); ss += __shfl_xor(ss, 8); ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 32);
    if ((threadIdx.x & 63) == 0) lds[2048 + (threadIdx.x >> 6)] = ss;
    __syncthreads();
    acc = __float_as_uint(rsqrtf(lds[2048] + lds[2049] + lds[2050] + lds[2051] + 1.0f));
  } else if (body == B_ALU) {
    float a = threadIdx.x * 1e-3f;
#pragma unroll 8
    for (int i = 0; i < 4096; ++i) a = __builtin_fmaf(a, 1.0001f, 1e-7f);
    acc = __float_as_uint(a);
  } else if (body == B_LDS) {
    float a = 0;
    for (int r = 0; r < 64; ++r) { for (int i = threadIdx.x; i < 2048; i += kT) lds[i] = a + i + r; __syncthreads(); a += lds[(threadIdx.x * 33 + r) & 2047]; __syncthreads(); }
    acc = __float_as_uint(a);
  }
  return acc;
}
__global__ void __launch_bounds__(kT, 2) k_pair(const u4* w, size_t buf_u4, const float* shared_vec, int fg, int bg, int reps,
                                                int* cu_slots, int* ctr, uint64_t* t_fg, uint32_t* sink) {
  __shared__ float lds[2048 + 64];
  __shared__ int s_role;
  if (threadIdx.x == 0) s_role = atomicAdd(&cu_slots[cu_key()], 1);    // 0: foreground, 1: background, else idle
  __syncthreads();
  const int role = s_role;
  uint32_t acc = 0;
  if (role == 0) {
    for (int r = 0; r < reps; ++r) {
      __syncthreads();
      const uint64_t t0 = now();
      acc ^= run_body(fg, w, buf_u4, shared_vec, lds, r, blockIdx.x);
      __syncthreads();
      if (threadIdx.x == 0) t_fg[(size_t)blockIdx.x * reps + r] = now() - t0;
    }
    if (threadIdx.x == 0) atomicAdd(ctr, 1);
  } else if (role == 1 && bg != B_IDLE) {
    __shared__ int s_stop;
    for (int r = 0;; ++r) {
      if ((r & 3) == 0) { if (threadIdx.x == 0) s_stop = ld_agent(ctr) >= (int)(gridDim.x / 2) - 8; __syncthreads(); if (s_stop) break; }
      acc ^= run_body(bg, w, buf_u4, shared_vec, lds, r + 1000, blockIdx.x);
      if (r > 200000) break;
    }
  }
  if (acc == 0x12345u) sink[blockIdx.x] = acc;
}


// ------------------------------------------------------------------ lat (dependent round trips)
// one node per 128 B line: its first 8 bytes hold the next node's address
__device__ __forceinline__ uint64_t ld_v(uint64_t p) {
  uint64_t r;
  asm volatile("global_load_dwordx2 %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=v"(r) : "v"(p) : "memory");
  return r;
}
__device__ __forceinline__ uint64_t ld_s(uint64_t p) {
  uint64_t r;
  asm volatile("s_load_dwordx2 %0, %1, 0x0\n\ts_waitcnt lgkmcnt(0)" : "=s"(r) : "s"(p) : "memory");
  return r;
}
__device__ __forceinline__ void inv_hop(int inv) {
  if (inv == 1) asm volatile("buffer_inv sc0" ::: "memory");
  else if (inv == 2) asm volatile("buffer_inv sc1" ::: "memory");
}
__global__ void k_nodes(uint64_t* base, const int64_t* offs, int n) {       // node i -> node i + 1 (a cycle)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) base[offs[i] / 8] = (uint64_t)base + (uint64_t)offs[(i + 1) % n];
}
// the chaser: the workgroup with XCD slot 0 on XCD 0; wave 0 chases, the workgroup evicts its L2 first when asked.
// loaded: every other workgroup streams (8 x 16 B per lane in flight) until the chaser is done
__global__ void __launch_bounds__(kT) k_lat(uint64_t start, int kind, int inv, int hops, int nwarm, int64_t evict_bytes,
                                            const u4* w, size_t buf_u4, int loaded, Ctl* ctl, int* done, uint64_t* out, uint32_t* sink) {
  __shared__ int s_slot;
  if (threadIdx.x == 0) s_slot = claim_xslot(ctl);
  __syncthreads();
  const bool chaser = s_slot == 0 && xcc_id() == 0;
  uint32_t acc = 0;
  if (chaser) {
    // warm: one pass over the whole cycle (nwarm = its length), so the timed pass (from the start again) hits
    if (threadIdx.x < 64 && nwarm) {
      uint64_t q = start;
      if (kind == 1) for (int h = 0; h < nwarm; ++h) q = ld_s(q);
      else for (int h = 0; h < nwarm; ++h) q = ld_v(q);
      acc ^= (uint32_t)q;
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < evict_bytes / 16; i += kT * 8) {      // evict this XCD's L2 (4 MB) with other lines
      u4 v[8];
#pragma unroll
      for (int d = 0; d < 8; ++d) v[d] = w[(buf_u4 / 2) + i + d * kT];
      live8(v);
#pragma unroll
      for (int d = 0; d < 8; ++d) acc ^= xs(v[d]);
    }
    __syncthreads();
    if (threadIdx.x < 64) {
      uint64_t t0, t1, c0, c1, r = 0;
      if (kind == 0) { uint64_t q = start; t0 = now(); c0 = __builtin_readcyclecounter();
        for (int h = 0; h < hops; ++h) { inv_hop(inv); q = ld_v(q); }
        t1 = now(); c1 = __builtin_readcyclecounter(); r = q; }
      else if (kind == 1) { uint64_t q = start; t0 = now(); c0 = __builtin_readcyclecounter();
        for (int h = 0; h < hops; ++h) { inv_hop(inv); q = ld_s(q); }
        t1 = now(); c1 = __builtin_readcyclecounter(); r = q; }
      else { t0 = now(); c0 = __builtin_readcyclecounter();                  // the invalidate alone
        for (int h = 0; h < hops; ++h) { inv_hop(inv); asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
        t1 = now(); c1 = __builtin_readcyclecounter(); }
      if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = c1 - c0; out[2] = r; __hip_atomic_store(done, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
    }
  } else if (loaded) {
    __shared__ int s_stop;
    for (int r = 0;; ++r) {
      if (threadIdx.x == 0) s_stop = ld_agent(done);
      __syncthreads();
      if (s_stop || r > 100000) break;
      const u4* p = w + ((size_t)(r * 304 + blockIdx.x) * 8192) % (buf_u4 / 2 - 8192);
      for (int i = threadIdx.x; i < 8192; i += kT * 8) {
        u4 v[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) v[d] = p[i + d * kT];
        live8(v);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc ^= xs(v[d]);
      }
      __syncthreads();
    }
  }
  if (acc == 0x12345u) sink[blockIdx.x] = acc;
}
__global__ void __launch_bounds__(kT) k_evict(const u4* w, size_t n16, uint32_t* sink) {   // every CU streams w[0, n16)
  uint32_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * kT + threadIdx.x; i < n16; i += (size_t)gridDim.x * kT) acc ^= xs(w[i]);
  if (acc == 0x12345u) sink[blockIdx.x] = acc;
}

// ------------------------------------------------------------------ lf (bytes in flight -> rate)
template <int D>
__global__ void __launch_bounds__(kT) k_lf(const u4* __restrict__ w, size_t buf_u4, int active, int64_t bytes, int iters, Ctl* ctl,
                                           uint64_t* t_start, uint64_t* t_end, uint32_t* sink) {
  __shared__ int s_slot;
  if (threadIdx.x == 0) s_slot = claim_xslot(ctl);
  __syncthreads();
  const int per_x = active / 8, slot = s_slot, wid = (int)xcc_id() * per_x + slot;
  const bool on = slot < per_x;
  uint32_t acc = 0;
  for (int it = 0; it < iters; ++it) {
    grid_barrier(ctl, it + 1);
    if (!on) continue;
    const uint64_t t0 = now();
    const size_t region = ((size_t)it * (size_t)active * (size_t)bytes / 16) % (buf_u4 / 2);
    const int64_t n16 = bytes / 16;
    const u4* p = w + region + (size_t)wid * n16;
    for (int64_t i = threadIdx.x; i < n16; i += kT * D) {
      u4 v[D];
#pragma unroll
      for (int d = 0; d < D; ++d) v[d] = (i + d * kT < n16) ? p[i + d * kT] : u4{0, 0, 0, 0};
#pragma unroll
      for (int d = 0; d < D; ++d) asm volatile("" : "+v"(v[d]));
#pragma unroll
      for (int d = 0; d < D; ++d) acc ^= xs(v[d]);
    }
    __syncthreads();
    if (threadIdx.x == 0) { t_start[(size_t)it * active + wid] = t0; t_end[(size_t)it * active + wid] = now(); }
  }
  if (acc == 0x12345u) sink[blockIdx.x] = acc;
}

// ------------------------------------------------------------------ launch (kernel boundary with work)
__global__ void k_tiny(const uint64_t* in, uint64_t* out, int hops) {
  if (hops < 0) return;
  if (threadIdx.x == 0) {
    uint64_t p = ld_v((uint64_t)(in + blockIdx.x));                         // the previous kernel's output
    for (int h = 0; h < hops; ++h) p = ld_v(p);
    out[blockIdx.x] = p;
  }
}

// ------------------------------------------------------------------ evx (ETX's event protocol, gfx942 lowering)
// the macros of the generated etx_lowering.h for gfx942 (etx/machines, build/*/etx_lowering.h), spelled out
__device__ __forceinline__ int evx_poll(const int* p) { asm volatile("buffer_inv sc0" ::: "memory"); return *(volatile const int*)p; }
__device__ __forceinline__ void evx_rel_domain() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
__device__ __forceinline__ void evx_rel_device() { asm volatile("buffer_wbl2 sc1\n\ts_waitcnt vmcnt(0)" ::: "memory"); }
__device__ __forceinline__ int evx_arrive(int* p, int n) { return __hip_atomic_fetch_sub(p, n, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
// scope 0 (DOMAIN): per XCD, slots < P produce and slots < C consume one event (all 8 XCDs at once, one event each).
// scope 1 (DEVICE): P / 8 producers and C / 8 consumers per XCD on one device event; every producer drains and bumps its
// XCD's sub-counter, the XCD's last one writes back L2 and subtracts the XCD's share (etx_arrive_flush_DEVICE_s).
// A workgroup can be both: it arrives, then waits (a task's end, then the next task's wait). late: consumers start their
// wait `late_ticks` after the producers' latest possible arrival (the event is already complete: the cost of a ready wait).
__global__ void __launch_bounds__(kT) k_evx(int* words, int* subs, int stride, int P, int C, int scope, int dirty_kb, int skew_ticks,
                                            int late_ticks, int iters, Ctl* ctl, float* scratch, uint64_t* t_arr, uint64_t* t_w0, uint64_t* t_wake,
                                            int ts_after) {
  __shared__ int s_slot;
  if (threadIdx.x == 0) s_slot = claim_xslot(ctl);
  __syncthreads();
  const int x = (int)xcc_id(), slot = s_slot, wid = x * 38 + slot;
  const bool prod = scope == 0 ? slot < P : slot < P / 8;
  const bool cons = scope == 0 ? slot < C : slot < C / 8;
  const int share = P / 8;
  for (int it = 0; it < iters; ++it) {
    grid_barrier(ctl, it + 1);
    const uint64_t tb = now();
    int* word = words + ((size_t)it * 16 + (scope == 0 ? x : 8)) * stride;
    int* sub = subs + ((size_t)it * 8 + x) * stride;
    if (prod) {
      float* my = scratch + (size_t)wid * 16384;
      for (int i = threadIdx.x; i < dirty_kb * 256; i += kT) my[i] = (float)(it + i);
      __syncthreads();
      if (threadIdx.x == 0) {
        const uint64_t until = tb + (uint64_t)((slot * 7919 + x * 131) % (skew_ticks + 1));
        while (now() < until) {}
        // ts_after: the arrival time stays in a register until the arrive is issued (the store of a timestamp before
        // the release would be drained by it: a cost of the measurement, not of the protocol)
        const uint64_t ta = now();
        if (!ts_after) t_arr[(size_t)it * 304 + wid] = ta;
        if (scope == 0) { evx_rel_domain(); evx_arrive(word, 1); }
        else if (share <= 1) { evx_rel_device(); evx_arrive(word, 1); }
        else { evx_rel_domain(); if (atomicAdd(sub, 1) + 1 == share) { evx_rel_device(); evx_arrive(word, share); } }
        if (ts_after) t_arr[(size_t)it * 304 + wid] = ta;
      }
    }
    if (cons) {
      if (threadIdx.x == 0) {
        if (late_ticks) { const uint64_t until = tb + skew_ticks + late_ticks; while (now() < until) {} }
        t_w0[(size_t)it * 304 + wid] = now();
        while (evx_poll(word) > 0) __builtin_amdgcn_s_sleep(1);
        if (scope == 0) asm volatile("buffer_inv sc0" ::: "memory"); else asm volatile("buffer_inv sc1" ::: "memory");
      }
      __syncthreads();
      if (threadIdx.x == 0) t_wake[(size_t)it * 304 + wid] = now();
    }
  }
}

// ------------------------------------------------------------------ host
int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string what = argc > 1 ? argv[1] : "all";
  hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, 0));
  const int ncu = prop.multiProcessorCount;
  printf("device %s, %d CUs, clockRate %d kHz\n", prop.gcnArchName, ncu, prop.clockRate);
  hipStream_t st; CHECK(hipStreamCreate(&st));
  const size_t buf_bytes = 2ull << 30, buf_u4 = buf_bytes / 16;
  u4* w; CHECK(hipMalloc(&w, buf_bytes)); CHECK(hipMemset(w, 1, buf_bytes));
  Ctl* ctl; CHECK(hipMalloc(&ctl, sizeof(Ctl)));
  uint32_t* sink; CHECK(hipMalloc(&sink, 4096 * 4));
  const size_t lds1 = 40 * 1024;                      // one workgroup per CU
  auto reset = [&]() { CHECK(hipMemset(ctl, 0, sizeof(Ctl))); };

  if (what == "all" || what == "stream") {
    const int iters = 12;
    uint64_t *ts, *te; CHECK(hipMalloc(&ts, (size_t)iters * 304 * 8)); CHECK(hipMalloc(&te, (size_t)iters * 304 * 8));
    // bytes per task of the real phases: router 0.9 KB, merge 8 KB, o_proj 28 KB, q/kv_a 50 KB, down 156 KB, gate_up 312 KB
    struct P { int active; int64_t bytes; const char* phase; };
    std::vector<P> ps = {{296, 896, "router"}, {256, 8192, "merge"}, {296, 28672, "o_proj"}, {296, 51200, "q/kv_a"},
                         {296, 159744, "down"}, {296, 319488, "gate_up"}, {296, 1 << 20, "1 MB"}, {8, 319488, "gate_up x8"},
                         {37, 319488, "gate_up x37"}, {74, 319488, "gate_up x74"}, {148, 319488, "gate_up x148"}};
    for (int rows = 0; rows < 2; ++rows) for (const P& p : ps) {
      if (p.active % 8) continue;
      reset(); CHECK(hipMemset(ts, 0, (size_t)iters * 304 * 8)); CHECK(hipMemset(te, 0, (size_t)iters * 304 * 8));
      hipLaunchKernelGGL(k_stream, dim3(ncu), dim3(kT), lds1, st, w, buf_u4, p.active, p.bytes, rows, iters, ctl, ts, te, sink);
      if (!wait_stream(st, 30)) { printf("RESULT stream HUNG\n"); return 3; }
      std::vector<uint64_t> hs((size_t)iters * 304), he((size_t)iters * 304);
      CHECK(hipMemcpy(hs.data(), ts, hs.size() * 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(he.data(), te, he.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> spans, wg;
      for (int it = 2; it < iters; ++it) {
        uint64_t lo = ~0ull, hi = 0;
        for (int k = 0; k < p.active; ++k) { const uint64_t a = hs[(size_t)it * p.active + k], b = he[(size_t)it * p.active + k];
          if (!a) continue; lo = std::min(lo, a); hi = std::max(hi, b); wg.push_back((b - a) * 0.01); }
        if (hi > lo) spans.push_back((hi - lo) * 0.01);
      }
      const double sp = pct(spans, 0.5);
      printf("RESULT stream %-12s %s %3d WGs x %7lld B: phase span %7.2f us  WG median %7.2f us  p90 %7.2f  -> %.2f TB/s\n", p.phase,
             rows ? "rows  " : "contig", p.active, (long long)p.bytes, sp, pct(wg, 0.5), pct(wg, 0.9), p.active * (double)p.bytes / (sp * 1e-6) / 1e12);
    }
  }

  if (what == "all" || what == "shared") {
    const int iters = 20;
    float* vec; CHECK(hipMalloc(&vec, 64 * 1024));
    uint64_t *tr, *td; CHECK(hipMalloc(&tr, (size_t)iters * 512 * 8)); CHECK(hipMalloc(&td, (size_t)iters * 512 * 8));
    for (int remote = 0; remote < 2; ++remote) for (int readers : {8, 64, 152, 296}) {
      reset(); CHECK(hipMemset(tr, 0, (size_t)iters * 512 * 8)); CHECK(hipMemset(td, 0, (size_t)iters * 512 * 8));
      hipLaunchKernelGGL(k_shared, dim3(ncu), dim3(kT), lds1, st, vec, readers, remote, iters, ctl, tr, td, sink);
      if (!wait_stream(st, 30)) { printf("RESULT shared HUNG\n"); return 3; }
      std::vector<uint64_t> a((size_t)iters * 512), b((size_t)iters * 512);
      CHECK(hipMemcpy(a.data(), tr, a.size() * 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(b.data(), td, b.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> d, last;
      for (int it = 2; it < iters; ++it) { double mx = 0; for (int k = 0; k < 512; ++k) { const size_t i = (size_t)it * 512 + k; if (!a[i]) continue; d.push_back((b[i] - a[i]) * 0.01); mx = std::max(mx, (b[i] - a[i]) * 0.01); } if (mx > 0) last.push_back(mx); }
      printf("RESULT shared %s readers %3d (64 KB each): median %6.2f us  p90 %6.2f  last %6.2f\n",
             remote ? "remote (8 XCDs wrote, wbl2 + inv sc1)" : "local  (own XCD wrote)             ", remote ? readers : readers / 8, pct(d, 0.5), pct(d, 0.9), pct(last, 0.5));
    }
  }

  if (what == "all" || what == "handoff") {
    const int iters = 24;
    const int pad_max = 32;                                  // 32 ints = 128 B: every XCD's counter in its own cache line
    int* counters; CHECK(hipMalloc(&counters, (size_t)iters * 64 * pad_max * 4));
    float* scratch; CHECK(hipMalloc(&scratch, (size_t)296 * 16384 * 4));
    uint64_t *ta, *tw; CHECK(hipMalloc(&ta, (size_t)iters * 296 * 8)); CHECK(hipMalloc(&tw, (size_t)iters * 296 * 8));
    for (int pad : {1, pad_max}) for (int scope = 0; scope < 2; ++scope) for (int sleep : {0, 1, 8, 32}) for (int sc1 = 0; sc1 < 2; ++sc1) for (int dirty : {0, 64}) {
      if (dirty && sleep != 8) continue;
      if (pad != 1 && (sleep != 8 || sc1)) continue;          // padded: the ETX setting only (sleep 8, sc0), clean and dirty
      reset(); CHECK(hipMemset(counters, 0, (size_t)iters * 64 * pad_max * 4)); CHECK(hipMemset(ta, 0, (size_t)iters * 296 * 8)); CHECK(hipMemset(tw, 0, (size_t)iters * 296 * 8));
      hipLaunchKernelGGL(k_handoff, dim3(ncu), dim3(kT), lds1, st, counters, pad, scope, sleep, sc1, dirty, 100 /* 1 us of arrival skew */, iters, ctl, scratch, ta, tw);
      if (!wait_stream(st, 30)) { printf("RESULT handoff HUNG\n"); return 3; }
      std::vector<uint64_t> a((size_t)iters * 296), b((size_t)iters * 296);
      CHECK(hipMemcpy(a.data(), ta, a.size() * 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(b.data(), tw, b.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> first, med, last;
      for (int it = 2; it < iters; ++it) for (int x = 0; x < (scope ? 1 : 8); ++x) {
        uint64_t arr = 0; std::vector<double> wk;
        for (int k = 0; k < 296; ++k) { const size_t i = (size_t)it * 296 + k; if (scope == 0 && k / 37 != x) continue; arr = std::max(arr, a[i]); }
        for (int k = 0; k < 296; ++k) { const size_t i = (size_t)it * 296 + k; if (scope == 0 && k / 37 != x) continue; if (b[i]) wk.push_back(((double)b[i] - (double)arr) * 0.01); }
        if (wk.empty() || !arr) continue;
        std::sort(wk.begin(), wk.end()); first.push_back(wk.front()); med.push_back(wk[wk.size() / 2]); last.push_back(wk.back());
      }
      printf("RESULT handoff %-6s %s sleep %2d acquire %s dirty %2d KB: last arrival -> first %5.2f  median %5.2f  last %5.2f us\n",
             scope ? "device" : "xcd", pad == 1 ? "shared-line" : "padded    ", sleep, sc1 ? "sc1" : "sc0", dirty, pct(first, 0.5), pct(med, 0.5), pct(last, 0.5));
    }
  }

  if (what == "all" || what == "wg" || what == "clock") {
    uint64_t* t; CHECK(hipMalloc(&t, (size_t)ncu * 16)); float* fs; CHECK(hipMalloc(&fs, ncu * 4));
    const char* names[] = {"__syncthreads", "LDS 8 KB write+read (2 barriers)", "64-lane shuffle reduce", "4-wave reduce (shuffle + LDS)",
                           "64 fp32 FMA / lane", "32 bf16->fp32 pair products / lane", "16 expf + rsqrtf / lane"};
    for (int k = 0; k < 7; ++k) {
      const int reps = 2000;
      hipLaunchKernelGGL(k_wg, dim3(ncu), dim3(kT), 0, st, k, reps, t, fs);
      if (!wait_stream(st, 30)) { printf("RESULT wg HUNG\n"); return 3; }
      std::vector<uint64_t> h((size_t)ncu * 2); CHECK(hipMemcpy(h.data(), t, h.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> us, cyc; for (int i = 0; i < ncu; ++i) { us.push_back(h[i * 2] * 0.01 / reps * 1000.0); cyc.push_back((double)h[i * 2 + 1] / reps); }
      printf("RESULT wg %-38s %8.1f ns per rep (%6.0f cycles), all CUs busy\n", names[k], pct(us, 0.5), pct(cyc, 0.5));
    }
    // clock: cycles per 10 ns tick, idle-ish (FMA loop) and while other CUs stream
    std::vector<uint64_t> h((size_t)ncu * 2);
    hipLaunchKernelGGL(k_wg, dim3(ncu), dim3(kT), 0, st, 4, 20000, t, fs); CHECK(hipStreamSynchronize(st));
    CHECK(hipMemcpy(h.data(), t, h.size() * 8, hipMemcpyDeviceToHost));
    std::vector<double> ghz; for (int i = 0; i < ncu; ++i) ghz.push_back((double)h[i * 2 + 1] / (h[i * 2] * 10.0));
    printf("RESULT clock under a full-GPU FMA load: %.3f GHz (median over CUs; min %.3f)\n", pct(ghz, 0.5), pct(ghz, 0.0));
  }

  if (what == "all" || what == "pair") {
    const int reps = 60;
    float* vec; CHECK(hipMalloc(&vec, 64 * 1024)); CHECK(hipMemset(vec, 0, 64 * 1024));
    int *cu_slots, *ctr; CHECK(hipMalloc(&cu_slots, 4096 * 4)); CHECK(hipMalloc(&ctr, 4));
    uint64_t* tf; CHECK(hipMalloc(&tf, (size_t)2 * ncu * reps * 8));
    const size_t lds2 = 20 * 1024;                   // at most two workgroups per CU (static 8 KB + 20 KB each)
    std::vector<std::vector<double>> base(5);
    for (int fg : {B_STREAM, B_CHAIN, B_ALU, B_LDS}) for (int bg : {B_IDLE, B_STREAM, B_CHAIN, B_ALU, B_LDS}) {
      CHECK(hipMemset(cu_slots, 0, 4096 * 4)); CHECK(hipMemset(ctr, 0, 4)); CHECK(hipMemset(tf, 0, (size_t)2 * ncu * reps * 8));
      CHECK(hipMemset(w, (fg * 7 + bg) & 0xFF, buf_bytes / 2));      // evict: 1 GB written between configurations (MALL is 256 MB)
      CHECK(hipStreamSynchronize(st));
      hipLaunchKernelGGL(k_pair, dim3(2 * ncu), dim3(kT), lds2, st, w, buf_u4, vec, fg, bg, reps, cu_slots, ctr, tf, sink);
      if (!wait_stream(st, 60)) { printf("RESULT pair HUNG\n"); return 3; }
      std::vector<uint64_t> h((size_t)2 * ncu * reps); CHECK(hipMemcpy(h.data(), tf, h.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> us; for (size_t i = 0; i < h.size(); ++i) if (h[i] && (i % reps) >= 5) us.push_back(h[i] * 0.01);
      const double m = pct(us, 0.5);
      if (bg == B_IDLE) base[fg] = {m};
      printf("RESULT pair fg %-6s bg %-6s: fg median %8.2f us  x%.2f of alone\n", body_name[fg], body_name[bg], m, base[fg].empty() ? 1.0 : m / base[fg][0]);
    }
  }

  if (what == "all" || what == "lat") {
    // node lists: n distinct 128 B lines of a span, in random order; states as the ETX / vLLM round trips see them
    const size_t nb = 1ull << 30;
    uint64_t* nodes; CHECK(hipMalloc(&nodes, nb)); CHECK(hipMemset(nodes, 0, nb));
    int* done; CHECK(hipMalloc(&done, 4)); uint64_t* o; CHECK(hipMalloc(&o, 32));
    int64_t* doffs; CHECK(hipMalloc(&doffs, 4096 * 8));
    struct S { const char* state; int n; int64_t span; int warm; int64_t evict; int cold; };
    const std::vector<S> states = {{"L1/K$", 64, 8192, 1, 0, 0}, {"L2", 4096, 2 << 20, 1, 0, 0},
                                   {"MALL", 4096, 64 << 20, 1, 16 << 20, 0}, {"HBM", 4096, 1 << 30, 0, 0, 1}};
    const char* kinds[] = {"vector", "scalar", "inv-only"};
    const char* invs[] = {"-      ", "inv sc0", "inv sc1"};
    const int hops = 2000;
    uint64_t seed = 88172645463325252ull;
    auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (const S& sv : states) {
      const int64_t lines = sv.span / 128;
      std::vector<int64_t> offs; std::vector<char> used(lines, 0);
      if (sv.n == lines) { for (int64_t i = 0; i < lines; ++i) offs.push_back(i * 128); for (int64_t i = lines - 1; i > 0; --i) std::swap(offs[i], offs[rnd() % (i + 1)]); }
      else while ((int)offs.size() < sv.n) { const int64_t l = rnd() % lines; if (!used[l]) { used[l] = 1; offs.push_back(l * 128); } }
      CHECK(hipMemcpy(doffs, offs.data(), offs.size() * 8, hipMemcpyHostToDevice));
      hipLaunchKernelGGL(k_nodes, dim3((sv.n + 255) / 256), dim3(256), 0, st, nodes, doffs, sv.n); CHECK(hipStreamSynchronize(st));
      const uint64_t start = (uint64_t)nodes + (uint64_t)offs[0];
      for (int kind = 0; kind < 3; ++kind) for (int inv = 0; inv < 3; ++inv) for (int loaded = 0; loaded < 2; ++loaded) {
        if (kind == 2 && (inv == 0 || sv.n != 64)) continue;              // the invalidate alone: once
        if (inv && sv.n != 64 && sv.span != (2 << 20)) continue;          // invalidates: on the L1 / K$ and L2 lists only
        std::vector<double> ns, cyc;
        for (int rep = 0; rep < 3; ++rep) {
          if (sv.cold) { hipLaunchKernelGGL(k_evict, dim3(ncu), dim3(kT), 0, st, w, buf_u4 / 2, sink); }   // 1 GB through MALL
          reset(); CHECK(hipMemset(done, 0, 4));
          hipLaunchKernelGGL(k_lat, dim3(ncu), dim3(kT), lds1, st, start, kind, inv, hops, sv.warm ? sv.n : 0, sv.evict, w, buf_u4, loaded, ctl, done, o, sink);
          if (!wait_stream(st, 30)) { printf("RESULT lat HUNG\n"); return 3; }
          uint64_t h[3]; CHECK(hipMemcpy(h, o, 24, hipMemcpyDeviceToHost));
          ns.push_back(h[0] * 10.0 / hops); cyc.push_back((double)h[1] / hops);
        }
        printf("RESULT lat %-8s %-6s %s %s: %7.1f ns per hop (%5.0f cycles; min %7.1f)\n", kinds[kind], sv.state, invs[inv],
               loaded ? "loaded" : "idle  ", pct(ns, 0.5), pct(cyc, 0.5), pct(ns, 0.0));
      }
    }
  }

  if (what == "all" || what == "lf") {
    const int iters = 10;
    uint64_t *ts, *te; CHECK(hipMalloc(&ts, (size_t)iters * 304 * 8)); CHECK(hipMalloc(&te, (size_t)iters * 304 * 8));
    for (int64_t bytes : {98304LL, 262144LL}) for (int active : {64, 128, 192, 256, 296}) for (int D : {1, 2, 4, 8, 16}) {
      reset(); CHECK(hipMemset(ts, 0, (size_t)iters * 304 * 8)); CHECK(hipMemset(te, 0, (size_t)iters * 304 * 8));
      switch (D) {
        case 1: hipLaunchKernelGGL(k_lf<1>, dim3(ncu), dim3(kT), lds1, st, w, buf_u4, active, bytes, iters, ctl, ts, te, sink); break;
        case 2: hipLaunchKernelGGL(k_lf<2>, dim3(ncu), dim3(kT), lds1, st, w, buf_u4, active, bytes, iters, ctl, ts, te, sink); break;
        case 4: hipLaunchKernelGGL(k_lf<4>, dim3(ncu), dim3(kT), lds1, st, w, buf_u4, active, bytes, iters, ctl, ts, te, sink); break;
        case 8: hipLaunchKernelGGL(k_lf<8>, dim3(ncu), dim3(kT), lds1, st, w, buf_u4, active, bytes, iters, ctl, ts, te, sink); break;
        default: hipLaunchKernelGGL(k_lf<16>, dim3(ncu), dim3(kT), lds1, st, w, buf_u4, active, bytes, iters, ctl, ts, te, sink); break;
      }
      if (!wait_stream(st, 30)) { printf("RESULT lf HUNG\n"); return 3; }
      std::vector<uint64_t> hs((size_t)iters * 304), he((size_t)iters * 304);
      CHECK(hipMemcpy(hs.data(), ts, hs.size() * 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(he.data(), te, he.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> spans, wg;
      for (int it = 2; it < iters; ++it) {
        uint64_t lo = ~0ull, hi = 0;
        for (int k = 0; k < active; ++k) { const uint64_t a = hs[(size_t)it * active + k], b = he[(size_t)it * active + k];
          if (!a) continue; lo = std::min(lo, a); hi = std::max(hi, b); wg.push_back((b - a) * 0.01); }
        if (hi > lo) spans.push_back((hi - lo) * 0.01);
      }
      const double med = pct(wg, 0.5), trips = (double)bytes / (D * 16.0 * kT);
      printf("RESULT lf bytes %6lld active %3d inflight %2d KB/WG (%5.2f MB total): WG median %7.2f us  span %7.2f  per trip %5.3f us  "
             "-> %.2f TB/s (WG median), %.2f TB/s (span)\n", (long long)bytes, active, D * 4, active * D * 4 / 1024.0, med, pct(spans, 0.5),
             med / trips, active * (double)bytes / (med * 1e-6) / 1e12, active * (double)bytes / (pct(spans, 0.5) * 1e-6) / 1e12);
    }
  }

  if (what == "all" || what == "launch") {
    // kernel k + 1 reads what kernel k wrote (ping-pong), then H dependent hops over a 4096-node list (L2-sized span)
    const int K = 200;
    uint64_t* nodes; CHECK(hipMalloc(&nodes, 2 << 20)); CHECK(hipMemset(nodes, 0, 2 << 20));
    int64_t* doffs; CHECK(hipMalloc(&doffs, 4096 * 8));
    std::vector<int64_t> offs(4096); for (int i = 0; i < 4096; ++i) offs[i] = (int64_t)((i * 2654435761u) % 16384) * 128;
    std::sort(offs.begin(), offs.end()); offs.erase(std::unique(offs.begin(), offs.end()), offs.end());
    for (size_t i = offs.size() - 1; i > 0; --i) std::swap(offs[i], offs[(i * 7919) % (i + 1)]);
    const int n = (int)offs.size();
    CHECK(hipMemcpy(doffs, offs.data(), n * 8, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_nodes, dim3((n + 255) / 256), dim3(256), 0, st, nodes, doffs, n);
    uint64_t *bufa, *bufb; CHECK(hipMalloc(&bufa, 4096 * 8)); CHECK(hipMalloc(&bufb, 4096 * 8));
    std::vector<uint64_t> init(4096); for (int i = 0; i < 4096; ++i) init[i] = (uint64_t)nodes + (uint64_t)offs[i % n];
    for (int grid : {1, 304}) for (int thr : {256, 1024}) for (int hops : {-1, 0, 1, 4}) {
      CHECK(hipMemcpy(bufa, init.data(), 4096 * 8, hipMemcpyHostToDevice)); CHECK(hipMemcpy(bufb, init.data(), 4096 * 8, hipMemcpyHostToDevice));
      hipGraph_t g; hipGraphExec_t ge;
      CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
      for (int k = 0; k < K; ++k) hipLaunchKernelGGL(k_tiny, dim3(grid), dim3(thr), 0, st, (k & 1) ? bufb : bufa, (k & 1) ? bufa : bufb, hops);
      CHECK(hipStreamEndCapture(st, &g)); CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
      hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
      std::vector<double> us;
      for (int rep = 0; rep < 6; ++rep) {
        CHECK(hipEventRecord(e0, st)); CHECK(hipGraphLaunch(ge, st)); CHECK(hipEventRecord(e1, st));
        if (!wait_stream(st, 30)) { printf("RESULT launch HUNG\n"); return 3; }
        float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); if (rep) us.push_back(ms * 1e3 / K);
      }
      printf("RESULT launch grid %3d threads %4d hops %2d: %6.2f us per kernel (min %6.2f)\n", grid, thr, hops, pct(us, 0.5), pct(us, 0.0));
      CHECK(hipGraphExecDestroy(ge)); CHECK(hipGraphDestroy(g));
    }
  }

  if (what == "all" || what == "evx") {
    const int iters = 24, stride = 32;                        // ETX_EV_STRIDE 32: every event word in its own 128 B line
    int *words, *subs; CHECK(hipMalloc(&words, (size_t)iters * 16 * stride * 4)); CHECK(hipMalloc(&subs, (size_t)iters * 8 * stride * 4));
    float* scratch; CHECK(hipMalloc(&scratch, (size_t)304 * 16384 * 4));
    uint64_t *ta, *t0, *tw; CHECK(hipMalloc(&ta, (size_t)iters * 304 * 8)); CHECK(hipMalloc(&t0, (size_t)iters * 304 * 8)); CHECK(hipMalloc(&tw, (size_t)iters * 304 * 8));
    struct E { int scope, P, C, dirty, late; };
    const std::vector<E> es = {{0, 1, 1, 0, 0}, {0, 1, 8, 0, 0}, {0, 1, 37, 0, 0}, {0, 8, 8, 0, 0}, {0, 37, 1, 0, 0}, {0, 37, 37, 0, 0},
                               {0, 37, 37, 64, 0}, {0, 1, 1, 0, 500}, {1, 8, 8, 0, 0}, {1, 8, 296, 0, 0}, {1, 64, 64, 0, 0},
                               {1, 296, 8, 0, 0}, {1, 296, 296, 0, 0}, {1, 296, 296, 64, 0}, {1, 8, 8, 0, 500}};
    for (int tsa = 0; tsa < 2; ++tsa) for (const E& e : es) {
      std::vector<int> hw((size_t)iters * 16 * stride, 0);
      for (int it = 0; it < iters; ++it) for (int k = 0; k < 9; ++k) hw[((size_t)it * 16 + k) * stride] = e.P;   // DOMAIN words: P each; DEVICE: P
      CHECK(hipMemcpy(words, hw.data(), hw.size() * 4, hipMemcpyHostToDevice)); CHECK(hipMemset(subs, 0, (size_t)iters * 8 * stride * 4));
      reset(); CHECK(hipMemset(ta, 0, (size_t)iters * 304 * 8)); CHECK(hipMemset(t0, 0, (size_t)iters * 304 * 8)); CHECK(hipMemset(tw, 0, (size_t)iters * 304 * 8));
      hipLaunchKernelGGL(k_evx, dim3(ncu), dim3(kT), lds1, st, words, subs, stride, e.P, e.C, e.scope, e.dirty, 100, e.late, iters, ctl, scratch, ta, t0, tw, tsa);
      if (!wait_stream(st, 30)) { printf("RESULT evx HUNG\n"); return 3; }
      std::vector<uint64_t> a((size_t)iters * 304), b((size_t)iters * 304), c((size_t)iters * 304);
      CHECK(hipMemcpy(a.data(), ta, a.size() * 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(b.data(), tw, b.size() * 8, hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(c.data(), t0, c.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> first, med, last, ready;
      for (int it = 2; it < iters; ++it) for (int x = 0; x < (e.scope ? 1 : 8); ++x) {
        uint64_t arr = 0; std::vector<double> wk;
        for (int k = 0; k < 304; ++k) { const size_t i = (size_t)it * 304 + k; if (e.scope == 0 && k / 38 != x) continue; arr = std::max(arr, a[i]); }
        for (int k = 0; k < 304; ++k) { const size_t i = (size_t)it * 304 + k; if (e.scope == 0 && k / 38 != x) continue;
          if (b[i]) { wk.push_back(((double)b[i] - (double)arr) * 0.01); ready.push_back(((double)b[i] - (double)c[i]) * 0.01); } }
        if (wk.empty() || !arr) continue;
        std::sort(wk.begin(), wk.end()); first.push_back(wk.front()); med.push_back(wk[wk.size() / 2]); last.push_back(wk.back());
      }
      printf("RESULT evx%s %-6s P %3d C %3d dirty %2d KB%s: last arrival -> first %5.2f  median %5.2f  last %5.2f us; wait start -> wake median %5.2f us\n",
             tsa ? "" : "-tsbefore", e.scope ? "device" : "xcd", e.P, e.C, e.dirty, e.late ? " late " : "      ", pct(first, 0.5), pct(med, 0.5), pct(last, 0.5), pct(ready, 0.5));
    }
  }
  return 0;
}
