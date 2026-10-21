// Standalone native tile service. No production flag is enabled by this file.
// Exact arithmetic helpers come from the sealed Fleet export, not the live checkout.
#include <hip/hip_runtime.h>
#include "kernels/gemv.h"
using namespace fleet;

struct ColumnsArgs {
  const __hip_bfloat16* weights;
  const float* inputs;
  float* outputs;
  const float* scales;
  unsigned int* completions;
  unsigned long long* stamps;
  int N, K, ld, workers, xcds, epoch, role, probe;
  float* output_columns[4];
  __hip_bfloat16* output_bf16_columns[4];
  int explicit_geometry,task_worker,task_xcd,coherent_output;
};
static __device__ __forceinline__ void column_store(const ColumnsArgs& a,int b,int row,int stride,float value) {
 if(a.output_bf16_columns[b]) {
  const __hip_bfloat16 v=__float2bfloat16(value);
  if(a.coherent_output){unsigned short bits;__builtin_memcpy(&bits,&v,2);__hip_atomic_store((unsigned short*)(a.output_bf16_columns[b]+row),bits,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);}
  else a.output_bf16_columns[b][row]=v;
 } else {
  float* out=a.output_columns[b]?a.output_columns[b]:a.outputs+(int64_t)b*stride;
  if(a.coherent_output)__hip_atomic_store(out+row,value,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);else out[row]=value;
 }
}
// 0:qkv, 1:interleaved gate/up, 2:down. Inputs are already-normalized native
// fp32 operands. Prologue is outside this tile claim and must be captured, not
// approximated on CPU. Each per-column accumulator retains fma8's operation order.
template<int B, int RPI, int D>
__device__ __forceinline__ void column_rows(ColumnsArgs a, float* xs, int out_stride) {
  const int worker = a.explicit_geometry?a.task_worker:blockIdx.x % a.workers, xcd = a.explicit_geometry?a.task_xcd:blockIdx.x / a.workers;
  const RowSlice slice = xcd_rows(a.N, xcd, a.xcds);
  const int first = slice.begin + worker;
  const int count = first < slice.end ? (slice.end-first+a.workers-1)/a.workers : 0;
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int n4 = a.K / 8;
  int idx[D]; bool valid[D];
#pragma unroll
  for(int d=0; d<D; ++d) { valid[d] = lane+d*64 < n4; idx[d] = valid[d] ? lane+d*64 : 0; }
  // Gate/up uses adjacent gate+up rows for one output. Linear uses RPI rows.
  constexpr int UNIT = RPI;
  int kk = a.role == 1 ? wave : UNIT*wave;
  const int stride = a.role == 1 ? 4 : UNIT*4;
  if (kk < count) {
    uint4 bufA[RPI][D], bufB[RPI][D];
#define COL_ROW(KK,R) (a.weights + (int64_t)(a.role==1 ? 2*(first+min((KK),count-1)*a.workers)+(R) : first+min((KK)+(R),count-1)*a.workers)*a.ld)
#define COL_ISSUE(BUF,KK) \
    _Pragma("unroll") for(int r=0;r<RPI;++r) { \
      const uint4* row = reinterpret_cast<const uint4*>(COL_ROW(KK,r)); \
      _Pragma("unroll") for(int d=0;d<D;++d) BUF[r][d] = load_weight(row+idx[d]); \
    }
#define COL_CONSUME(BUF,KK) \
    { float sums[B][RPI]; \
      _Pragma("unroll") for (int b = 0; b < B; ++b) { \
        _Pragma("unroll") for(int r=0;r<RPI;++r) { \
          float acc = 0.f; \
          _Pragma("unroll") for(int d=0;d<D;++d) if(valid[d]) fma8(BUF[r][d],xs+b*a.K+idx[d]*8,acc); \
          sums[b][r] = wave_sum(acc); \
        } \
        if(lane==0) { \
          if(a.role==1) { if constexpr(RPI>=2) if((KK)<count) column_store(a,b,first+(KK)*a.workers,out_stride,bf16_round(bf16_round(silu(bf16_round(sums[b][0])))*bf16_round(sums[b][1]))); } \
          else { _Pragma("unroll") for(int r=0;r<RPI;++r) if((KK)+r<count) { \
            const float scale=a.scales[b]; \
            column_store(a,b,first+((KK)+r)*a.workers,out_stride,bf16_round(scale*bf16_round(sums[b][r]))); \
          }} \
        } \
      } \
    }
    COL_ISSUE(bufA,kk)
    while(true) {
      const int k2=kk+stride; COL_ISSUE(bufB,k2) COL_CONSUME(bufA,kk)
      if(k2>=count) break;
      const int k3=k2+stride; COL_ISSUE(bufA,k3) COL_CONSUME(bufB,k2)
      if(k3>=count) break;
      kk=k3;
    }
#undef COL_ROW
#undef COL_ISSUE
#undef COL_CONSUME
  }
}

template<int B>
__device__ __forceinline__ void body(ColumnsArgs a) {
  // 4 columns x 2048 floats = 32 KiB. Guards reject larger captured shapes.
  __shared__ float xs[B*2048];
  for(int i=threadIdx.x; i<B*a.K; i+=256) xs[i]=a.inputs[i];
  __syncthreads();
  const int out_stride=a.N;
  if(a.role==0 && a.N==3648) {
    // Frozen K1 QKV: each XCD's 384 Q rows, then its 72 shared KVA
    // rows. Keeping the two calls preserves original task row assignment.
    ColumnsArgs q=a;q.N=3072;column_rows<B,2,4>(q,xs,out_stride);
    ColumnsArgs kv=a;kv.N=576;kv.weights=a.weights+(int64_t)3072*a.ld;
    kv.outputs=a.outputs+3072;column_rows<B,2,4>(kv,xs,out_stride);
  } else if(a.role==1 || a.K>1024) column_rows<B,2,4>(a,xs,out_stride);
  else if(a.K>512) column_rows<B,4,2>(a,xs,out_stride);
  else column_rows<B,8,1>(a,xs,out_stride);
  // Every writing wave drains before the workgroup releases completion.
  __builtin_amdgcn_s_waitcnt(0);
  __threadfence();
  __syncthreads();
  if(threadIdx.x==0) {
    __hip_atomic_store(a.completions+blockIdx.x, (unsigned)a.epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    if(a.probe) a.stamps[blockIdx.x]=__builtin_amdgcn_s_memrealtime();
  }
}
extern "C" __global__ __launch_bounds__(256) void fleet_columns_1(ColumnsArgs a) { body<1>(a); }
extern "C" __global__ __launch_bounds__(256) void fleet_columns_2(ColumnsArgs a) { body<2>(a); }
extern "C" __global__ __launch_bounds__(256) void fleet_columns_3(ColumnsArgs a) { body<3>(a); }
extern "C" __global__ __launch_bounds__(256) void fleet_columns_4(ColumnsArgs a) { body<4>(a); }
// Unchanged single-column numerical control, compiled from the exact pinned helper.
extern "C" __global__ __launch_bounds__(256) void fleet_original(ColumnsArgs a) {
  __shared__ float x[2048];
  stage_vector(a.inputs,x,a.K);
  const int worker=blockIdx.x%a.workers, xcd=blockIdx.x/a.workers;
  if(a.role==1) gemv_gate_up_rows(a.weights,x,a.outputs,a.N,a.K,xcd,a.xcds,worker,a.workers);
  else if(a.role==0 && a.N==3648) {
    gemv_rows(a.weights,a.ld,x,a.outputs,nullptr,EPI_BF16,a.scales[0],3072,a.K,xcd,a.xcds,worker,a.workers);
    gemv_rows(a.weights+(int64_t)3072*a.ld,a.ld,x,a.outputs+3072,nullptr,EPI_BF16,a.scales[0],576,a.K,xcd,a.xcds,worker,a.workers);
  } else gemv_rows(a.weights,a.ld,x,a.outputs,nullptr,EPI_BF16,a.scales[0],a.N,a.K,xcd,a.xcds,worker,a.workers,false,a.output_bf16_columns[0]);
  __builtin_amdgcn_s_waitcnt(0); __threadfence(); __syncthreads();
  if(threadIdx.x==0) __hip_atomic_store(a.completions+blockIdx.x,(unsigned)a.epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
}
extern "C" __global__ void readiness_delay(unsigned long long ticks) {
  const unsigned long long start=__builtin_amdgcn_s_memrealtime();
  while(__builtin_amdgcn_s_memrealtime()-start<ticks) __builtin_amdgcn_s_sleep(1);
}
// G05 gather/scatter uses original slot indices; fold order remains untouched.
extern "C" __global__ void gather_columns(const float* src,float* dst,const int* slots,int K,int B) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<B*K) dst[i]=src[(int64_t)slots[i/K]*K+i%K];
}
extern "C" __global__ void scatter_columns(const float* src,float* dst,const int* slots,int N,int B) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<B*N) dst[(int64_t)slots[i/N]*N+i%N]=src[i];
}

extern "C" __global__ void scatter_columns16(const __hip_bfloat16* src,__hip_bfloat16* dst,const int* slots,int N,int B) {
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<B*N)dst[(int64_t)slots[i/N]*N+i%N]=src[i];
}
