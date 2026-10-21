// G01 native subgraph: unchanged 256 router tasks and exact KEY32 routines.
// One common executable/resource union for control and candidate. No grid shrink.
struct G01State {
  unsigned next[8], workers[8], done, elected, published, selected;
  unsigned epoch, reverse, domain_mask, winner, readers[8], task_seen[256];
};
static __device__ __forceinline__ unsigned g01_load(unsigned* p) {
  return __hip_atomic_load(p,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
}
static __device__ __forceinline__ void g01_store(unsigned* p,unsigned v) {
  __hip_atomic_store(p,v,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
}
static __device__ __forceinline__ unsigned g01_add(unsigned* p) {
  return __hip_atomic_fetch_add(p,1u,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_SYSTEM);
}

// Optional diagnostic-only native landmarks. Never used by decisive pairs.
// Eight words per task/wave; extra reader records follow the256 router tasks.
static __device__ __forceinline__ void g01_stamp(unsigned long long* trace,unsigned task,unsigned phase,unsigned local_worker,unsigned epoch) {
#ifdef G01_PROBE
  if((threadIdx.x&63)==0){
    unsigned long long* r=trace+(task*16+threadIdx.x/64)*13;
    r[phase]=__builtin_amdgcn_s_memrealtime();
    r[8]=__builtin_amdgcn_s_getreg((3<<11)|20)&15;
    unsigned hwid;asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hwid));r[12]=hwid;
    r[9]=ETX_DOMAIN_ID();r[10]=local_worker;r[11]=epoch;
#ifdef G01_EMPTY
    r[phase]=__builtin_amdgcn_s_memrealtime(); // matched adjacent empty stamp
#endif
  }
#endif
}

extern "C" __global__ void g01_reset(G01State* s,unsigned epoch,unsigned reverse) {
  for(unsigned i=threadIdx.x;i<sizeof(G01State)/4;i+=blockDim.x)((unsigned*)s)[i]=0;
  __syncthreads();
  if(threadIdx.x==0){s->epoch=epoch;s->reverse=reverse;s->winner=~0u;}
}
extern "C" __global__ __launch_bounds__(1024,1)
void g01_run(const unsigned char* router_args,const unsigned char* topk_args,
             G01State* s,unsigned epoch,int candidate,unsigned char* observed,unsigned long long* trace) {
  etx_import_init();__syncthreads();
  const unsigned domain=ETX_DOMAIN_ID();
  if(domain>=8)return; // host rejects incomplete domain coverage, never certifies it
  __shared__ unsigned local_worker,local_task,win;
  if(threadIdx.x==0){
    local_worker=g01_add(s->workers+domain);
    __hip_atomic_fetch_or(&s->domain_mask,1u<<domain,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
  for(;;){
    if(threadIdx.x==0)local_task=g01_add(&s->next[domain]);
    __syncthreads();
    if(local_task>=32)break;
    const unsigned blk=domain * 32+(s->reverse?31-local_task:local_task);
    dsl_touch_plain(router_args,blk); // the kept pre-wait immutable weight touch
    dsl_run_dsl_gemv_router_fold(router_args,blk);
    g01_stamp(trace,blk,0,local_worker,epoch);
    // Every writing wave drains before thread zero elects; owner zero's
    // canonical norm/residual stores are included by the unchanged router body.
    ETX_RELEASE_DOMAIN();__syncthreads();
    g01_stamp(trace,blk,1,local_worker,epoch);
    if(threadIdx.x==0){
      win=0;
      __hip_atomic_fetch_add(s->task_seen+blk,1u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
      if(candidate && blk<128){unsigned old=g01_add(&s->elected);win=(old == 127);if(win)s->winner=domain;}
    }
    __syncthreads();
    g01_stamp(trace,blk,2,local_worker,epoch);
    if(win){
      ETX_ACQUIRE_DEVICE();__syncthreads();
      g01_stamp(trace,blk,3,local_worker,epoch);
      dsl_run_dsl_topk(topk_args);__syncthreads();
      g01_stamp(trace,blk,4,local_worker,epoch);
      const float* w0=*(float* const*)(topk_args+8);
      const int* i0=*(int* const*)(topk_args+16);const int* t0=*(int* const*)(topk_args+24);
      for(int r=1; r < 8; ++r)for(unsigned j=threadIdx.x;j<8;j+=blockDim.x){
        dsl_a_stf(const_cast<float*>(w0)+r*8+j,dsl_a_ldf(w0+j));
        dsl_a_sti(const_cast<int*>(i0)+r*8+j,dsl_a_ldi(i0+j));
        dsl_a_sti(const_cast<int*>(t0)+r*8+j,dsl_a_ldi(t0+j));
      }
      ETX_RELEASE_DOMAIN();__syncthreads();
      g01_stamp(trace,blk,5,local_worker,epoch);
      if(threadIdx.x==0)g01_store(&s->published,epoch);
    }
    if(threadIdx.x==0)g01_add(&s->done);
    __syncthreads();
  }
  if(local_worker!=0)return;
  if(threadIdx.x==0)while(g01_load(&s->done)!=256)__builtin_amdgcn_s_sleep(8);
  __syncthreads();ETX_ACQUIRE_DEVICE();__syncthreads();
  if(!candidate){
    dsl_run_dsl_topk(topk_args+domain*64);
    ETX_RELEASE_DOMAIN();__syncthreads();
    if(threadIdx.x==0)g01_add(&s->selected);
  }
  if(threadIdx.x==0){
    if(candidate)while(g01_load(&s->published)!=epoch)__builtin_amdgcn_s_sleep(8);
    else while(g01_load(&s->selected)!=8)__builtin_amdgcn_s_sleep(8);
  }
  __syncthreads();ETX_ACQUIRE_DEVICE();__syncthreads();
  g01_stamp(trace,256+domain,6,local_worker,epoch);
  const unsigned char* a=topk_args+domain*64;
  const void* ptrs[6]={*(void*const*)(a+16),*(void*const*)(a+8),*(void*const*)(a+24),
      *(void*const*)(router_args+8),*(void*const*)(router_args+80),*(void*const*)(router_args+16)};
  const unsigned sizes[6]={32,32,32,4096,4096,256};
  unsigned off=domain*8544;
  for(int b=0;b<6;++b){
    for(unsigned j=threadIdx.x;j<sizes[b]/2;j+=blockDim.x){
      auto u=__hip_atomic_load((const DSL_G unsigned short*)ptrs[b]+j,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
      ((unsigned short*)(observed+off))[j]=u;
    }off+=sizes[b];
  }
  ETX_RELEASE_DOMAIN();__syncthreads();
  g01_stamp(trace,256+domain,7,local_worker,epoch);
  if(threadIdx.x==0)g01_store(s->readers+domain,epoch);
}
