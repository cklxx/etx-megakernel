#include "etx/abi.h"
#include "kernels.hip"
#include "micro_state.h"
#ifndef ETX_COLUMN_NATIVE_BATCH
#error "exact native batch must be supplied by the offline build"
#endif
static __device__ __forceinline__ bool column_micro_wait(const unsigned* word,unsigned epoch,ColumnMicroState* s,float* arena) {
 int& ok=*(int*)arena;
 if(threadIdx.x==0){auto start=__builtin_amdgcn_s_memrealtime();ok=1;while(__hip_atomic_load(word,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT)!=epoch){
  if(__builtin_amdgcn_s_memrealtime()-start>s->deadline){__hip_atomic_store(s->abort,1,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);ok=0;break;}__builtin_amdgcn_s_sleep(1);}}
 __syncthreads();const int result=ok;__syncthreads();return result;
}
static __device__ __forceinline__ void column_micro_finish(ColumnMicroState* s,int column,int request) {
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
 if(threadIdx.x==0 && __hip_atomic_fetch_sub(s->remaining+request,1,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT)==1)
  s->marks[5+request]=__builtin_amdgcn_s_memrealtime();
 __syncthreads();
}
extern "C" __device__ void column_micro_ready(const etx_ctx* c) {
 auto* s=(ColumnMicroState*)c->args[0];const int r=c->cst[0];
 if(threadIdx.x==0){const auto now=__builtin_amdgcn_s_memrealtime();__hip_atomic_fetch_min(s->marks,now,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);
  while(__builtin_amdgcn_s_memrealtime()-__hip_atomic_load(s->marks,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT)<s->arrival[r])__builtin_amdgcn_s_sleep(1);
  if(s->probe)s->placement[s->batch*296+r]=((unsigned long long)(__builtin_amdgcn_s_getreg((3<<11)|20)&15)<<32)|c->worker;
  s->marks[1+r]=__builtin_amdgcn_s_memrealtime();__hip_atomic_store(s->ready+r,(unsigned)s->epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);}
 __syncthreads();
}
extern "C" __device__ void column_micro_service(const etx_ctx* c) {
 auto* s=(ColumnMicroState*)c->args[0];const int request=c->cst[0],B=c->cst[1],kind=c->cst[2],worker=c->coord[1],xcd=c->coord[0];
 const int column=kind==2?request*6+xcd:request;
 if(!column_micro_wait(s->ready+request,s->epoch,s,(float*)c->lds))return;
 if(s->probe && threadIdx.x==0){const int index=kind==0?request*296+xcd*37+worker:column*37+worker;
  s->placement[index]=((unsigned long long)(__builtin_amdgcn_s_getreg((3<<11)|20)&15)<<32)|c->worker;}
 int members[4],width=1,owner=column;members[0]=column;
 if(s->fused){
  if(kind==2){const auto& group=s->groups[s->group_for[column]];width=group.width;for(int b=0;b<width;++b)members[b]=group.members[b];owner=members[0];}
  else {width=B;owner=0;for(int b=0;b<B;++b)members[b]=b;}
  if(column!=owner){column_micro_wait(s->owner_done+owner*296+(kind==0?xcd*37:0)+worker,s->epoch,s,(float*)c->lds);return;}
  for(int b=0;b<width;++b)if(!column_micro_wait(s->ready+members[b]/s->slots,s->epoch,s,(float*)c->lds))return;
 }
 float* xs=(float*)c->lds;
 ColumnsArgs a=s->columns[column];a.explicit_geometry=1;a.task_worker=worker;a.task_xcd=kind==0?xcd:0;a.xcds=kind==0?8:1;
 for(int b=0;b<width;++b){const auto& original=s->columns[members[b]];
  for(int i=threadIdx.x;i<a.K;i+=256)xs[b*2048+i]=original.inputs[i];
  a.output_columns[b]=original.outputs;a.output_bf16_columns[b]=original.output_bf16_columns[0];}
 __syncthreads();
 // column_rows consumes tightly packed K columns; native K1408 uses a packed
 // staging layout, preserving its exact original float expansion.
 if(a.K!=2048){for(int b=1;b<width;++b){for(int i=threadIdx.x;i<a.K;i+=256)xs[b*a.K+i]=s->columns[members[b]].inputs[i];}__syncthreads();}
 float local_scales[4];for(int b=0;b<width;++b)local_scales[b]=*s->columns[members[b]].scales;
 // Scales are read in each producing thread; no numerical reconstruction.
 a.scales=local_scales;
 if(!s->fused || width==1){
  const auto& original=s->columns[column];
  if(a.role==1)fleet::gemv_gate_up_rows(a.weights,xs,original.outputs,a.N,a.K,a.task_xcd,a.xcds,worker,37);
  else if(a.role==0 && a.N==3648){fleet::gemv_rows(a.weights,a.ld,xs,original.outputs,nullptr,fleet::EPI_BF16,local_scales[0],3072,a.K,a.task_xcd,8,worker,37);fleet::gemv_rows(a.weights+(long long)3072*a.ld,a.ld,xs,original.outputs+3072,nullptr,fleet::EPI_BF16,local_scales[0],576,a.K,a.task_xcd,8,worker,37);}
  else fleet::gemv_rows(a.weights,a.ld,xs,original.outputs,nullptr,fleet::EPI_BF16,local_scales[0],a.N,a.K,a.task_xcd,a.xcds,worker,37,false,original.output_bf16_columns[0]);
 }else{
#define COLUMN_RUN(W) if(a.role==0 && a.N==3648){a.N=3072;column_rows<W,2,4>(a,xs,3072);a.weights+=(long long)3072*a.ld;a.N=576;for(int b=0;b<width;++b)a.output_columns[b]+=3072;column_rows<W,2,4>(a,xs,576);}else if(a.role==1 || a.K>1024)column_rows<W,2,4>(a,xs,a.N);else column_rows<W,1,8>(a,xs,a.N)
  if(width==2){COLUMN_RUN(2);}
#if ETX_COLUMN_NATIVE_BATCH >= 3
  else if(width==3){COLUMN_RUN(3);}else if(width==4){COLUMN_RUN(4);}
#endif
#undef COLUMN_RUN
 }
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
 if(threadIdx.x==0)__hip_atomic_store(s->owner_done+owner*296+(kind==0?xcd*37:0)+worker,(unsigned)s->epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
 for(int b=0;b<width;++b)column_micro_finish(s,members[b],members[b]/s->slots);
}
