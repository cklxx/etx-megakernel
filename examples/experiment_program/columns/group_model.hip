// Actual optional G05 model intervention. Build only through winner-gated
// integration.prepare_grouped; the original singleton and shared units remain.
#ifndef FP
#include "examples/dsv2lite/fleet_shim.hip"
#endif
#include "examples/experiment_program/columns/kernels.hip"
#ifndef ETX_COLUMN_MODEL_BATCH
#define ETX_COLUMN_MODEL_BATCH 2
#endif
#include "group_state.h"
static __device__ __forceinline__ void group_abort(ColumnGroupState* s) {
 __hip_atomic_store(&s->abort,1,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
}
extern "C" __device__ void etx_group_metadata(const etx_ctx* c) {
 constexpr int B=ETX_COLUMN_MODEL_BATCH;auto* state=(ColumnGroupState*)c->args[0];
 fleet::Scratch& scratch=*reinterpret_cast<fleet::Scratch*>(c->lds);
 const int layer=c->cst[0];
 if(threadIdx.x==0){state->epoch=((unsigned long long)etx_fleet_const[0].epoch<<8)|(unsigned)(layer+1);}
 __syncthreads();
 for(int r=0;r<B;++r) {
  const FleetParams* fp=&etx_fleet_const[r];
  if(fp->epoch!=etx_fleet_const[0].epoch){if(threadIdx.x==0)group_abort(state);return;}
  for(int slot=0;slot<6;++slot) {
   fleet::select_experts(fp->d,fp->act.router_logits+(int64_t)slot*fp->d.n_routed,scratch.ids,scratch.ws);
   if(threadIdx.x==0) {
    const int index=r*6+slot;const int id=scratch.ids[slot];int owner=index;
    // Stable first encounter; native picks are unique within each request.
    for(int before=0;before<index;++before)if(state->group[before].expert==id){owner=state->group[before].owner;break;}
    ColumnGroup& current=state->group[index];current.owner=owner;current.expert=id;current.width=0;current.native_mix=scratch.ws[slot];
    if(owner==index){current.width=1;current.member[0]=index;current.mix[0]=scratch.ws[slot]*fp->d.routed_scaling;}
    else {ColumnGroup& group=state->group[owner];const int b=group.width++;
     if(b>=4){group_abort(state);return;}group.member[b]=index;group.mix[b]=scratch.ws[slot]*fp->d.routed_scaling;}
   }
   __syncthreads();
  }
 }
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
}
static __device__ __forceinline__ void group_wait(ColumnGroupState* state,const unsigned long long* flag,unsigned long long epoch) {
 __shared__ int ok;
 if(threadIdx.x==0){ok=1;const auto start=__builtin_amdgcn_s_memrealtime();
  while(__hip_atomic_load(flag,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT)!=epoch){
   if(__hip_atomic_load(&state->abort,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT) ||
      __builtin_amdgcn_s_memrealtime()-start>100000000ull){group_abort(state);ok=0;break;}
   __builtin_amdgcn_s_sleep(1);
  }}
 __syncthreads();
}
template<int B,bool Down>
static __device__ __forceinline__ void group_columns(const etx_ctx* c,ColumnGroupState* state,const ColumnGroup& group) {
 // Runtime group width1..4 instantiates only its real observed multiplicity.
 // The arena is the ETX LDS allocation; no extra hidden full-column array.
 float* xs=(float*)c->lds;ColumnsArgs a{};
 const int layer=c->cst[0],worker=c->coord[1];
 a.N=Down?2048:1408;a.K=Down?1408:2048;a.ld=a.K;a.workers=37;a.xcds=1;a.role=Down?2:1;
 a.explicit_geometry=1;a.task_worker=worker;a.task_xcd=0;a.coherent_output=1;a.scales=group.mix;
 for(int b=0;b<B;++b){const int member=group.member[b],r=member/6,slot=member%6;const FleetParams* fp=&etx_fleet_const[r];
  const float* input=Down?fp->act.expert_h+(int64_t)slot*1408:fp->act.x_norm+(int64_t)slot*2048;
  fleet::stage_vector(input,xs+b*a.K,a.K);
  if(Down)a.output_bf16_columns[b]=fp->act.expert_out+(int64_t)slot*2048;
  else a.output_columns[b]=fp->act.expert_h+(int64_t)slot*1408;
 }
 const FleetParams* fp=&etx_fleet_const[0];
 a.weights=fp->w.layer_experts[layer]+(int64_t)group.expert*fp->w.expert_stride+(Down?(int64_t)2*1408*2048:0);
 column_rows<B,2,4>(a,xs,a.N);
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
}
template<bool Down>
static __device__ __forceinline__ void group_task(const etx_ctx* c) {
 auto* state=(ColumnGroupState*)c->args[1];const int r=c->cst[1],slot=c->coord[0],worker=c->coord[1];
 const FleetParams* fp=&etx_fleet_const[r];
 if(__hip_atomic_load(&state->abort,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT))return;
 if(slot>=6) {if constexpr(Down)fleet_expert_down_cached(c);else fleet_expert_gate_up(c);return;}
 const int index=r*6+slot;const ColumnGroup& own=state->group[index];const ColumnGroup& group=state->group[own.owner];
 if(group.width==1) {
  if constexpr(Down) {
   // The new all-up join permits another request to reuse this worker's LDS.
   // Restore this slot's exact native ID/mix before the unchanged cached body.
   fleet::Scratch& scratch=*reinterpret_cast<fleet::Scratch*>(c->lds);
   if(threadIdx.x==0){scratch.ids[slot]=own.expert;scratch.ws[slot]=own.native_mix;}__syncthreads();
   fleet_expert_down_cached(c);
  } else fleet_expert_gate_up(c);
  return;
 }
 if(own.owner!=index){group_wait(state,Down?&state->down_done[own.owner][worker]:&state->up_done[own.owner][worker],state->epoch);return;}
 if(group.width==2)group_columns<2,Down>(c,state,group);
 else if(group.width==3)group_columns<3,Down>(c,state,group);
 else if(group.width==4)group_columns<4,Down>(c,state,group);
 else {if(threadIdx.x==0)group_abort(state);return;}
 if(threadIdx.x==0)__hip_atomic_store(Down?&state->down_done[index][worker]:&state->up_done[index][worker],state->epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
}
extern "C" __device__ void etx_group_up(const etx_ctx* c){group_task<false>(c);}
extern "C" __device__ void etx_group_down(const etx_ctx* c){group_task<true>(c);}
static_assert(sizeof(ColumnGroupState)<=16384,"graph group buffer bound");
