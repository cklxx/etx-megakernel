// Includes (lowering, adapters, DSL prep) are emitted by build.py.
#include "examples/vllm_llm/attention_early_merge/probe.h"
__shared__ unsigned c6_broadcast[3];

static __device__ __forceinline__ void c6_fail(int* abort) {
  if (threadIdx.x == 0) __hip_atomic_store(abort, 2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ bool c6_wait64(const unsigned long long* p, unsigned long long epoch, const C6Args& a) {
  if (threadIdx.x == 0) {
    c6_broadcast[0] = 0;
    for (unsigned n = 0;; ++n) {
      const auto value = __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
      if (value == epoch) break;
      if (value > epoch || n >= a.spin_limit || __hip_atomic_load(a.abort, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
        c6_broadcast[0] = 1; c6_fail(a.abort); break;
      }
      ETX_BACKOFF();
    }
  }
  __syncthreads();
  return !c6_broadcast[0];
}
static __device__ __forceinline__ bool c6_wait_count(const unsigned* p, unsigned count, const C6Args& a) {
  if (threadIdx.x == 0) {
    c6_broadcast[0] = 0;
    for (unsigned n = 0;; ++n) {
      const auto value = __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
      if (value == count) break;
      if (value > count || n >= a.spin_limit || __hip_atomic_load(a.abort, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
        c6_broadcast[0] = 1; c6_fail(a.abort); break;
      }
      ETX_BACKOFF();
    }
  }
  __syncthreads();
  return !c6_broadcast[0];
}
static __device__ __forceinline__ void c6_stamp(const C6Args& a, unsigned h, unsigned p, unsigned phase) {
#ifdef C6_DIAGNOSTIC
  if (threadIdx.x == 0 && a.stamps) a.stamps[(h * 7 + p) * 8 + phase] = ETX_TIMER();
#endif
}
static __device__ __forceinline__ void c6_prep(const C6Args& a, unsigned h) {
  etx_ctx c = a.prep; c.coord[0] = h;
  etx_tile_dsl_attnprep(&c);
  ETX_RELEASE_DOMAIN(); // EVERY writer wave, then full-workgroup join
  __syncthreads();
}
static __device__ __forceinline__ void c6_attn(const C6Args& a, unsigned h, unsigned p) {
  if (threadIdx.x == 0) { ETX_ACQUIRE_DOMAIN(); }
  __syncthreads();
  etx_ctx c = a.attention; c.coord[0] = h * 5 + p;
  etx_tile_attn_mfma16_g8(&c); // all active and idle waves finish slot rounds
  ETX_RELEASE_DOMAIN();
  __syncthreads();
}
static __device__ __forceinline__ void c6_reduce(const C6Args& a, unsigned h) {
  if (threadIdx.x == 0) { ETX_ACQUIRE_DOMAIN(); }
  __syncthreads();
  etx_ctx c = a.reduce; c.coord[0] = h;
  etx_tile_attn_reduce(&c); // blocks 8*h+s; original partition accumulation order
  ETX_RELEASE_DOMAIN();
  __syncthreads();
  // One leader flush covers the XCD after all producer waves drain.
  if (threadIdx.x == 0) {
    ETX_RELEASE_DEVICE();
    __hip_atomic_store(&a.state[h].output, a.epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_fetch_sub(a.outputs, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// The host checks cooperative residency. Ticket 0 is always the owner.
// 304 resident blocks give 38 per XCD on a full MI300X. The probe verifies
// at least seven tickets on each active XCD. Surplus blocks never wait.
static __device__ __attribute__((always_inline)) void c6_run(C6Args a) {
  etx_import_init();
  if (threadIdx.x == 0) {
    c6_broadcast[1] = ETX_DOMAIN_ID();
    c6_broadcast[2] = c6_broadcast[1] < 8 ?
      __hip_atomic_fetch_add(a.tickets + c6_broadcast[1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : 0xffffffffu;
  }
  __syncthreads();
  const unsigned h = c6_broadcast[1], ticket = c6_broadcast[2];
  if (h >= 8 || a.mode > 1 || !a.epoch) { c6_fail(a.abort); return; }
  if (h == 7 && ticket == 0) {
    // Identical end event in both arms: all four head outputs published and
    // acquired by a different XCD, as required by the o_proj join.
    if (threadIdx.x == 0) {
      c6_broadcast[0] = 0;
      for (unsigned n = 0; __hip_atomic_load(a.outputs, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != 0; ++n) {
        if (n >= a.spin_limit || __hip_atomic_load(a.abort, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
          c6_broadcast[0] = 1; c6_fail(a.abort); break;
        }
        ETX_BACKOFF();
      }
      ETX_ACQUIRE_DEVICE();
    }
    __syncthreads();
    if (!c6_broadcast[0]) {
      const unsigned char* args=etx_imp_argblock(a.reduce.args,a.reduce.cst[0]);
      const unsigned short* output=*(const unsigned short* const*)args;
      for (unsigned i=threadIdx.x;i<32*128;i+=blockDim.x) a.observed[i]=output[i];
      ETX_RELEASE_DOMAIN();
      __syncthreads();
    }
    return;
  }
  if (h >= 4 || ticket >= (a.mode ? 5u : 7u)) return;
  C6State* s = a.state + h;
  if (s->epoch != a.epoch) { c6_fail(a.abort); return; }
  c6_stamp(a,h,ticket,0);
  const bool owner = ticket == 0;
  if (owner) {
    if (a.fault == 1) return; // leave siblings to exercise their bounded timeout
    if (threadIdx.x == 0) c6_broadcast[0] = __hip_atomic_exchange(&s->claim, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (c6_broadcast[0]) { c6_fail(a.abort); return; }
    c6_prep(a,h);
    c6_stamp(a,h,ticket,1);
    if (threadIdx.x == 0) __hip_atomic_store(&s->ready, a.epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    if (!a.mode) return; // control: whole prep task ends here
  } else if (a.mode || ticket <= 5) {
    if (!c6_wait64(&s->ready,a.epoch,a)) return;
  }
  if (!a.mode && ticket == 6) {
    if (!c6_wait_count(&s->arrived,5,a)) return;
    c6_stamp(a,h,ticket,5);
    c6_reduce(a,h); c6_stamp(a,h,ticket,6);
    return;
  }
  const unsigned p = a.mode ? ticket : ticket-1;
  c6_stamp(a,h,ticket,2);
  if (p == a.skew_partition && a.skew_cycles) {
    const unsigned long long start=ETX_TIMER();
    while (ETX_TIMER()-start < a.skew_cycles) { ETX_BACKOFF(); }
  }
  c6_attn(a,h,p);
  c6_stamp(a,h,ticket,3);
  if (threadIdx.x == 0) {
    const unsigned old = a.mode ? __hip_atomic_fetch_or(&s->mask,1u<<p,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT) : 0u;
    c6_broadcast[0] = (old & (1u<<p)) ? 2u :
      (__hip_atomic_fetch_add(&s->arrived,1u,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT) == 4 ? 1u : 0u);
    if (a.mode && c6_broadcast[0] == 1) __hip_atomic_store(&s->elected,p,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  c6_stamp(a,h,ticket,4);
  if (c6_broadcast[0] == 2) { c6_fail(a.abort); return; }
  if (a.mode && c6_broadcast[0] == 1) {
    c6_stamp(a,h,ticket,5);
    c6_reduce(a,h); c6_stamp(a,h,ticket,6);
  }
}
#ifndef C6_RETAINED_ENVELOPE
extern "C" __global__ __launch_bounds__(1024, 1) void c6_probe(C6Args a) { c6_run(a); }
#endif
// Used only by the full-dispatch resource envelope. This is NOT a full-model
// experimental plan: no host binds the extra C6 argument tensor there.
extern "C" __device__ __attribute__((always_inline)) void etx_tile_c6_resource(const etx_ctx* c) {
  c6_run(*(const C6Args*)c->args[0]);
}
