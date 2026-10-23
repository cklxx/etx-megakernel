"""G01 winner-only model integration. Disabled unless explicitly requested.

The model adds a real reset task (charged to the measured step) and retains all
256 router tasks. No raw unregistered event or host reset promise is used.
"""
import os


def enabled():return os.environ.get('ETX_VL_LAST_ROUTER','0')=='1'


def emit(machine="gfx942"):
    """Emit a fragment for the enclosing translation unit's explicit machine."""
    source = r'''
// Same existing KEY32 implementation; forward declaration for emitter order.
static __device__ __attribute__((always_inline)) void dsl_run_dsl_topk(const unsigned char* a);
extern "C" __device__ __attribute__((always_inline)) void etx_tile_g01_reset(const etx_ctx* c) {
  if(threadIdx.x==0)__hip_atomic_store((int*)c->args[0]+32*c->cst[0],128,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
  ETX_RELEASE_DOMAIN();__syncthreads();
}
static __device__ __forceinline__ void g01_model_router(const etx_ctx* c,const unsigned char* a,int blk) {
  dsl_run_dsl_gemv_router_fold(a,blk);
  ETX_RELEASE_DOMAIN();__syncthreads(); // every writing wave, including canonical norm owner0
  __shared__ int g01_winner;
  if(threadIdx.x==0){
    g01_winner=0;
    if(blk<128){
      int old=__hip_atomic_fetch_sub((int*)c->args[1]+32*c->cst[3],1,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_SYSTEM);
      g01_winner=old==1;
    }
  }
  __syncthreads();
  if(g01_winner){
    ETX_ACQUIRE_DEVICE();__syncthreads();
    __shared__ unsigned char g01_topk_args[64];
    if(threadIdx.x==0){
      *(const void**)(g01_topk_args+0)=*(const void*const*)(a+16);
      *(void**)(g01_topk_args+8)=*(void*const*)(a+32);
      *(void**)(g01_topk_args+16)=*(void*const*)(a+24);
      *(void**)(g01_topk_args+24)=*(void*const*)(a+112);
      *(int*)(g01_topk_args+32)=128;*(int*)(g01_topk_args+36)=8;
      *(int*)(g01_topk_args+40)=1;*(int*)(g01_topk_args+44)=1;
    }
    __syncthreads();dsl_run_dsl_topk(g01_topk_args);__syncthreads();
    int* ids=*(int*const*)(a+24);float* mix=*(float*const*)(a+32);int* tei=*(int*const*)(a+112);
    for(int r=1;r<8;++r)for(int j=threadIdx.x;j<8;j+=blockDim.x){
      dsl_a_sti(ids+8*r+j,dsl_a_ldi(ids+j));dsl_a_stf(mix+8*r+j,dsl_a_ldf(mix+j));dsl_a_sti(tei+8*r+j,dsl_a_ldi(tei+j));
    }
    ETX_RELEASE_DOMAIN();__syncthreads(); // normal router completion cannot precede all payloads
  }
}
extern "C" __device__ __attribute__((always_inline)) void etx_tile_x_dsl_gemv_router_fold(const etx_ctx* c){
  g01_model_router(c,etx_imp_argblock(c->args,c->cst[0]+c->coord[0]*c->cst[1]),c->coord[0]*c->cst[2]+c->coord[1]);
}
extern "C" __device__ __attribute__((always_inline)) void etx_tile_dsl_gemv_router_fold(const etx_ctx* c){
  g01_model_router(c,etx_imp_argblock(c->args,c->cst[0]),c->coord[0]);
}
'''

    from etx.ops import prims
    if "source_prelude" in prims.machine_or_default(machine).primitives:
        source = "#include <cuda/atomic>\n" + source
        source = source.replace(
            "__hip_atomic_store((int*)c->args[0]+32*c->cst[0],128,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM)",
            "cuda::atomic_ref<int, cuda::thread_scope_system>(*((int*)c->args[0]+32*c->cst[0])).store(128, cuda::memory_order_relaxed)")
        source = source.replace(
            "__hip_atomic_fetch_sub((int*)c->args[1]+32*c->cst[3],1,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_SYSTEM)",
            "cuda::atomic_ref<int, cuda::thread_scope_system>(*((int*)c->args[1]+32*c->cst[3])).fetch_sub(1, cuda::memory_order_acq_rel)")
    return source
