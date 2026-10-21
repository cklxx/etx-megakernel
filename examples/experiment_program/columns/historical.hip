// Historical arithmetic; historical_stream.h is extracted verbatim from the
// source-bound Python STREAM literal by build.py after SHA-256 verification.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "etx/abi.h"
#include "etx/tile_util.h"
#define DSL_NB 2
#define DSL_K_MAX 4096
#include "historical_stream.h"
struct HistoricalArgs {
 const __hip_bfloat16 *weights,*inputs;
 __hip_bfloat16 *outputs;
 const int *ids;
 const float *mix;
 unsigned int *completions;
 unsigned long long *stamps;
 int N,K,ld,workers,xcds,epoch,role,probe;
 long long expert_stride;
 int slots,rpt,tpe;
};
template<int B>
__device__ __forceinline__ void historical_dense_work(HistoricalArgs a,int block,float* arena) {
  // Both layouts reserve identical LDS. Actual registers/scratch must be
  // inspected; the driver never infers equal residency from this declaration.
  float *xs=arena, *part=arena+B*4096;
  for(int i=threadIdx.x;i<B*a.K;i+=1024) xs[(i/a.K)*4096+i%a.K]=__bfloat162float(a.inputs[i]);
  __syncthreads();
  const int r0=block*20, r1=min(a.N,r0+20);
  if(r0<r1) dsl_gemv_nb<B>(a.weights,a.K,4096,xs,r0,r1,part,[&](int r,const float* v) {
#pragma unroll
    for(int b=0;b<B;++b) a.outputs[(size_t)b*a.N+r]=__float2bfloat16(v[b]);
  });
  __builtin_amdgcn_s_waitcnt(0); __threadfence(); __syncthreads();
  if(threadIdx.x==0) {
    __hip_atomic_store(a.completions+block,(unsigned)a.epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
    if(a.probe) a.stamps[block]=__builtin_amdgcn_s_memrealtime();
  }
}
template<int B>
__device__ __forceinline__ void dense(HistoricalArgs a) { __shared__ __attribute__((aligned(16))) float arena[12570];historical_dense_work<B>(a,blockIdx.x,arena); }
extern "C" __global__ __launch_bounds__(1024) void historical_nb1(HistoricalArgs a) { dense<1>(a); }
extern "C" __global__ __launch_bounds__(1024) void historical_nb2(HistoricalArgs a) { dense<2>(a); }
#ifdef HISTORICAL_FF
// One column per task, same two-pass gate/up and separate-slot down. Control
// and intervention invoke these SAME kernel symbols; only event edges vary.
static __device__ __forceinline__ void historical_expert_up_work(HistoricalArgs a,int block,float* arena) {
 float* xs=arena;
 const int slot=block/a.tpe,j=block%a.tpe;
 const int r0=j*a.rpt,r1=min(a.N,r0+a.rpt);
 const __hip_bfloat16* W=a.weights+(size_t)a.ids[slot]*a.expert_stride;
 for(int i=threadIdx.x;i<a.K;i+=1024) xs[i]=__bfloat162float(a.inputs[i]);
 __syncthreads();
 float* g=xs+a.K; float* part=g+2*a.rpt;
 if(r0<r1) {
  dsl_gemv(W,a.K,xs,r0,r1,part,[&](int r,float v){g[r-r0]=dsl_bfr(v);});
  dsl_gemv(W+(size_t)a.N*a.K,a.K,xs,r0,r1,part,[&](int r,float v){g[a.rpt+r-r0]=dsl_bfr(v);});
  for(int r=r0+threadIdx.x;r<r1;r+=1024) a.outputs[(size_t)slot*a.N+r]=__float2bfloat16(dsl_silu(g[r-r0])*g[a.rpt+r-r0]);
 }
 __builtin_amdgcn_s_waitcnt(0); __threadfence(); __syncthreads();
 if(threadIdx.x==0) __hip_atomic_store(a.completions+block,(unsigned)a.epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ void historical_expert_down_work(HistoricalArgs a,int block,float* arena) {
 const int slot=block/a.tpe,j=block%a.tpe;
 const int r0=j*a.rpt,r1=min(a.N,r0+a.rpt);
 for(int i=threadIdx.x;i<a.K;i+=1024) arena[i]=__bfloat162float(a.inputs[(size_t)slot*a.K+i]);
 __syncthreads();
 const __hip_bfloat16* W=a.weights+(size_t)a.ids[slot]*a.expert_stride;
 if(r0<r1) dsl_gemv(W,a.K,arena,r0,r1,arena+a.K,[&](int r,float v){a.outputs[(size_t)slot*a.N+r]=__float2bfloat16(v*a.mix[slot]);});
 __builtin_amdgcn_s_waitcnt(0); __threadfence(); __syncthreads();
 if(threadIdx.x==0) __hip_atomic_store(a.completions+block,(unsigned)a.epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
}
extern "C" __global__ __launch_bounds__(1024) void historical_expert_up(HistoricalArgs a) {
 __shared__ __attribute__((aligned(16))) float arena[12570];historical_expert_up_work(a,blockIdx.x,arena);
}
extern "C" __global__ __launch_bounds__(1024) void historical_expert_down(HistoricalArgs a) {
 __shared__ __attribute__((aligned(16))) float arena[12570];historical_expert_down_work(a,blockIdx.x,arena);
}
#endif
