// Used by both arms of one ETX executable. The queue proof rejects combined
// completion behind request-first queues before anything reaches the GPU.
#define HISTORICAL_FF 1
#include "historical.hip"
struct G13State {
 HistoricalArgs up,down;
 unsigned long long readiness_ticks,spin_ticks;
 int combined;
 int* abort;
 unsigned long long* marks;
 int* remaining;int request;unsigned long long* placement;
};
extern "C" __device__ void g13_ready(const etx_ctx* c) {
 const G13State& state=*(const G13State*)c->args[0];
 __shared__ unsigned long long start;
 if(threadIdx.x==0) {start=__builtin_amdgcn_s_memrealtime();__hip_atomic_fetch_min(state.marks,start,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);}
 __syncthreads();
 while(__builtin_amdgcn_s_memrealtime()-__hip_atomic_load(state.marks,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT)<state.readiness_ticks)__builtin_amdgcn_s_sleep(1);
 __syncthreads();if(threadIdx.x==0){if(state.up.probe)state.placement[896+state.request]=((unsigned long long)(__builtin_amdgcn_s_getreg((3<<11)|20)&15)<<32)|c->worker;state.marks[1+state.request]=__builtin_amdgcn_s_memrealtime();}
}
extern "C" __device__ void g13_up(const etx_ctx* c) {
 const G13State& state=*(const G13State*)c->args[0];
 if(state.up.probe && threadIdx.x==0)state.placement[state.request*448+c->coord[0]*24+c->coord[1]]=((unsigned long long)(__builtin_amdgcn_s_getreg((3<<11)|20)&15)<<32)|c->worker;
 HistoricalArgs a=state.up;
 historical_expert_up_work(a,c->coord[0]*24+c->coord[1],(float*)c->lds);
}
extern "C" __device__ void g13_down(const etx_ctx* c) {
 const G13State& state=*(const G13State*)c->args[0];
 __shared__ int live;
 if(threadIdx.x==0) {
  live=1;
  if(state.combined) {
   const int offset=c->ev_offset[c->cst[1]]+c->coord[0];
   const auto start=__builtin_amdgcn_s_memrealtime();
   while(__hip_atomic_load(c->events+ETX_EVW(offset),__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT)>0) {
    if(__builtin_amdgcn_s_memrealtime()-start>state.spin_ticks) {
     __hip_atomic_store(state.abort,2,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);live=0;break;
    }
    __builtin_amdgcn_s_sleep(1);
   }
  }
 }
 __syncthreads();if(!live)return;
 if(state.down.probe && threadIdx.x==0)state.placement[state.request*448+192+c->coord[0]*32+c->coord[1]]=((unsigned long long)(__builtin_amdgcn_s_getreg((3<<11)|20)&15)<<32)|c->worker;
 HistoricalArgs a=state.down;
 historical_expert_down_work(a,c->coord[0]*32+c->coord[1],(float*)c->lds);
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
 if(threadIdx.x==0) {
  const int before=__hip_atomic_fetch_sub(state.remaining,1,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);
  if(before==1)state.marks[3+state.request]=__builtin_amdgcn_s_memrealtime();
  if(before<=0)__hip_atomic_store(state.abort,3,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
 }
 // Sparse native per-task/worker/XCD stamps are emitted by ETX_TRACE, outside
 // decisive pairs. The normal event release drains every writing wave.
}
