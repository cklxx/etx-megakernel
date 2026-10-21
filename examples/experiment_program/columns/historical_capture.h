#pragma once
#include "native_capture.h"
struct EtxHistConfig {
 EtxColumnRow* rows;const unsigned char* argument;const int64_t* position;
 const float* mix;int mix_stride;
 int blocks,base_block,request,layer,role,selected_position;
};
extern "C" void etx_hist_configure(const EtxHistConfig*,int,hipStream_t);
extern "C" void etx_hist_enable(int,hipStream_t);
#ifdef ETX_HIST_DEVICE_DEFINITION
__constant__ EtxHistConfig etx_hist_configs[32];
__constant__ int etx_hist_active;
extern "C" void etx_hist_configure(const EtxHistConfig* input,int n,hipStream_t stream) {
 EtxHistConfig configs[32]{};if(n<0||n>32)abort();for(int i=0;i<n;++i)configs[i]=input[i];
 hipMemcpyToSymbolAsync(HIP_SYMBOL(etx_hist_configs),configs,sizeof(configs),0,hipMemcpyHostToDevice,stream);hipStreamSynchronize(stream);
}
extern "C" void etx_hist_enable(int value,hipStream_t stream) {
 hipMemcpyToSymbolAsync(HIP_SYMBOL(etx_hist_active),&value,sizeof(value),0,hipMemcpyHostToDevice,stream);hipStreamSynchronize(stream);
}
#ifdef __HIP_DEVICE_COMPILE__
static __device__ __forceinline__ EtxColumnRow* hist_ready(const etx_ctx* ctx,const unsigned char* argument,int block,
 const float* xs,int K,int KS,int B,const void* weights,int N,int ld,int slot,int request_offset,int expert,long long expert_stride,float native_mix=0.f,int use_native_mix=0) {
 if(!etx_hist_active)return nullptr;
 const EtxHistConfig* cfg=nullptr;
 for(int i=0;i<32;++i)if(etx_hist_configs[i].argument==argument && etx_hist_configs[i].position &&
   *etx_hist_configs[i].position==etx_hist_configs[i].selected_position)cfg=&etx_hist_configs[i];
 if(!cfg)return nullptr;const int index=block-cfg->base_block;
 if(index<0||index>=cfg->blocks)return nullptr;
 EtxColumnRow* row=cfg->rows+index;
 if(threadIdx.x==0){
  if(row->h[0])row->h[25]=1;
  row->h[0]=0x455458434f4c3031ull;row->h[1]=1;row->h[2]=cfg->request+request_offset;
  row->h[3]=cfg->selected_position;row->h[4]=*cfg->position;row->h[5]=cfg->layer;row->h[6]=cfg->role;
  row->h[7]=ctx->domain;row->h[8]=ctx->worker;row->h[9]=etx_column_hwid();row->h[10]=__builtin_amdgcn_s_memrealtime();
  row->h[12]=K;row->h[13]=N;row->h[14]=ld;row->h[15]=(uint64_t)weights;
  row->h[16]=(uint64_t)N*ld*(cfg->role==2?4:2);row->h[17]=(uint64_t)(int64_t)expert;row->h[18]=(uint64_t)(int64_t)slot;
  row->h[19]=__float_as_uint(slot<0?1.f:use_native_mix?native_mix:cfg->mix[request_offset*cfg->mix_stride+slot]);
  row->h[40]=__builtin_amdgcn_s_getreg((3<<11)|20)&15;row->h[41]=expert_stride;row->h[32]=B;row->h[33]=KS;row->h[35]=(uint64_t)argument;row->h[36]=block;
  if(B*K>4096 || !ctx || block<0)row->h[25]=1;
 }
 for(int i=threadIdx.x;i<B*K && i<4096;i+=blockDim.x)row->input[i]=xs[(i/K)*KS+i%K];
 __syncthreads();return row;
}
static __device__ __forceinline__ void hist_finish(EtxColumnRow* row,const __hip_bfloat16* output,int stride,int B,int begin,int end) {
 if(!row)return;__builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
 const int count=end-begin;
 for(int i=threadIdx.x;i<B*count && i<256;i+=blockDim.x){int b=i/count,r=begin+i%count;
  row->indices[i]=b*stride+r;row->output[i]=__hip_atomic_load((const unsigned short*)(output+(size_t)b*stride+r),__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);}
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
 if(threadIdx.x==0){row->h[11]=__builtin_amdgcn_s_memrealtime();row->h[20]=2;row->h[21]=B*count;row->h[22]=blockDim.x/64;row->h[23]=1;
  if(B*count>256)row->h[25]=1;}
}
#endif
#endif
