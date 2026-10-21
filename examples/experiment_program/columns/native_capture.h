// Layout is shared by the patched original host and original device source.
#pragma once
#include <stdint.h>
struct EtxColumnRow {
 uint64_t h[64];
 float input[4096];
 uint32_t indices[256];
 uint32_t output[256]; // fp32 bits or zero-extended original bf16 bits
};
struct EtxColumnSink {
 EtxColumnRow* rows;
 int request,layer,token,role,workers;
};
#ifdef __HIP_DEVICE_COMPILE__
static __device__ __forceinline__ unsigned etx_column_hwid() {
 unsigned id;asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(id));return id;
}
static __device__ __forceinline__ EtxColumnRow* etx_column_ready(
 EtxColumnSink* sink,int layer,int token,int pos,unsigned epoch,int role,int xcd,int worker,
 const float* input,int K,int N,int ld,const void* weights,uint64_t weight_bytes,
 int slot,int expert,float mix,unsigned kind,unsigned flags,float native_mix=1.f) {
 if(!sink || sink->layer!=layer || sink->token!=token || sink->role!=role) return nullptr;
 EtxColumnRow* row=sink->rows+xcd*sink->workers+worker;
 if(threadIdx.x==0) {
  if(row->h[0]) row->h[25]=1;
  row->h[0]=0x455458434f4c3031ull;row->h[1]=epoch;row->h[2]=sink->request;
  row->h[3]=token;row->h[4]=pos;row->h[5]=layer;row->h[6]=role;
  row->h[7]=xcd;row->h[8]=worker;row->h[9]=etx_column_hwid();
  row->h[10]=__builtin_amdgcn_s_memrealtime();row->h[12]=K;row->h[13]=N;row->h[14]=ld;
  row->h[15]=(uint64_t)weights;row->h[16]=weight_bytes;row->h[17]=(uint64_t)(int64_t)expert;
  row->h[18]=(uint64_t)(int64_t)slot;row->h[19]=__float_as_uint(mix);row->h[24]=sink->workers;
  row->h[42]=__float_as_uint(native_mix);row->h[40]=__builtin_amdgcn_s_getreg((3<<11)|20)&15;row->h[26]=kind;row->h[27]=flags;
  if(K>2048 || worker<0 || worker>=sink->workers)row->h[25]=1;
 }
 for(int i=threadIdx.x;i<K && i<2048;i+=blockDim.x)row->input[i]=input[i];
 __syncthreads();return row;
}
static __device__ __forceinline__ void etx_column_finish(
 EtxColumnSink* sink,int layer,int token,unsigned epoch,int role,int xcd,int worker,
 const void* first,int first_begin,int first_end,int first_stride,bool first_bf16,
 const void* second=nullptr,int second_begin=0,int second_end=0,int second_stride=1,int second_base=0) {
 if(!sink || sink->layer!=layer || sink->token!=token || sink->role!=role)return;
 EtxColumnRow* row=sink->rows+xcd*sink->workers+worker;
 // All original producing waves drain their writes, then the block copies only
 // this task's owned final values. It never reads another worker's unfinished rows.
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
 int count=first_begin<first_end?(first_end-first_begin+first_stride-1)/first_stride:0;
 int count2=second_begin<second_end?(second_end-second_begin+second_stride-1)/second_stride:0;
 for(int i=threadIdx.x;i<count+count2 && i<256;i+=blockDim.x) {
  int index=i<count?first_begin+i*first_stride:second_begin+(i-count)*second_stride;
  row->indices[i]=index+(i<count?0:second_base);
  if(i<count)row->output[i]=first_bf16?((const uint16_t*)first)[index]:__float_as_uint(((const float*)first)[index]);
  else row->output[i]=__float_as_uint(((const float*)second)[index]);
 }
 __builtin_amdgcn_s_waitcnt(0);__threadfence();__syncthreads();
 if(threadIdx.x==0) {
  row->h[11]=__builtin_amdgcn_s_memrealtime();row->h[20]=first_bf16?2:4;row->h[21]=count+count2;
  row->h[22]=blockDim.x/64;row->h[23]=epoch;
  if(count+count2>256 || row->h[1]!=epoch)row->h[25]=1;
 }
}
#endif
