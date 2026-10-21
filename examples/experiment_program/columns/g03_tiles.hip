#include "historical.hip"
struct G13State {
 HistoricalArgs up,down;unsigned long long readiness_ticks,spin_ticks;
 int combined;int* abort;unsigned long long* marks;int* remaining;int request;unsigned long long* placement;
};
static __device__ __forceinline__ bool g03_wait(const unsigned* word,unsigned target,const G13State& s) {
 __shared__ int ok;
 if(threadIdx.x==0){ok=1;const auto start=__builtin_amdgcn_s_memrealtime();while(__hip_atomic_load(word,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT)!=target){
  if(__builtin_amdgcn_s_memrealtime()-start>s.spin_ticks){__hip_atomic_store(s.abort,1,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);ok=0;break;}__builtin_amdgcn_s_sleep(1);}}
 __syncthreads();return ok;
}
extern "C" __device__ void g03_ready(const etx_ctx* c) {
 const auto& s=*(const G13State*)c->args[0];
 if(threadIdx.x==0){auto start=__builtin_amdgcn_s_memrealtime();__hip_atomic_fetch_min(s.marks,start,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);
  while(__builtin_amdgcn_s_memrealtime()-__hip_atomic_load(s.marks,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT)<s.readiness_ticks)__builtin_amdgcn_s_sleep(1);if(s.up.probe)s.placement[896+s.request]=((unsigned long long)(__builtin_amdgcn_s_getreg((3<<11)|20)&15)<<32)|c->worker;s.marks[1+s.request]=__builtin_amdgcn_s_memrealtime();}
 __syncthreads();
}
extern "C" __device__ void g03_service(const etx_ctx* c) {
 const auto& s=*(const G13State*)c->args[0];const int block=c->cst[2]*32+c->coord[0];
 if(s.up.probe && threadIdx.x==0)s.placement[s.request*256+block]=((unsigned long long)(__builtin_amdgcn_s_getreg((3<<11)|20)&15)<<32)|c->worker;
 if(s.combined)historical_dense_work<1>(s.up,block,(float*)c->lds);
 else if(s.request==0){
  const unsigned* event=(const unsigned*)(c->events+ETX_EVW(c->ev_offset[c->cst[1]]));
  if(!g03_wait(event,0,s))return;
  historical_dense_work<2>(s.down,block,(float*)c->lds);
 }else{
  if(!g03_wait(s.down.completions+block,(unsigned)s.up.epoch,s))return;
  if(threadIdx.x==0)__hip_atomic_store(s.up.completions+block,(unsigned)s.up.epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
 }
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
 if(threadIdx.x==0){const int before=__hip_atomic_fetch_sub(s.remaining,1,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);
  if(before==1)s.marks[3+s.request]=__builtin_amdgcn_s_memrealtime();if(before<=0)__hip_atomic_store(s.abort,2,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);}
 __syncthreads();
}
