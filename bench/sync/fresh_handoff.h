// Included by sync_bench.hip. gfx942-only experimental protocol; NOT GPU-validated.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <thread>
#include <sys/stat.h>
namespace fresh {
constexpr int T = 256, X = 8, KEYS = 4096;
constexpr size_t LDS = 40 * 1024, BUSY_WORDS = 512 * 1024;
enum Mode { CACHED, BYPASS, FORCED_OLD, PRIMED_STALE };
struct alignas(128) Flag { int value; };
struct Control {
  Flag ready, prime, ack[X], primed[X], barrier, stop, abort;
  int seen[KEYS], xcc[X], keys[1024], roles[1024], invalidations[KEYS];
  unsigned long long busy_bytes[1024], timer_ticks[1024];
};
struct Observation { unsigned long long load, bulk, errors; };
__device__ __forceinline__ unsigned cu() {
  return (xcc_id() << 8) | ((__builtin_amdgcn_s_getreg((31 << 11) | (0 << 6) | 4) >> 8) & 255);
}
__device__ __forceinline__ uint32_t read(const uint32_t* p, bool bypass) {
  uint32_t v;
  if (bypass) asm volatile("global_load_dword %0, %1, off sc0 sc1\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  else asm volatile("global_load_dword %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  return v;
}
__device__ __forceinline__ int flag(const Flag* p) { return (int)read((const uint32_t*)&p->value, true); }
__device__ __forceinline__ void set(Flag* p, int v) { __hip_atomic_store(&p->value, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ uint32_t pattern(unsigned epoch, unsigned handoff, unsigned word) {
  // Epoch differs at every reused address; no sentinel or argmax-only check.
  return (word * 2654435761u) ^ (handoff * 2246822519u) ^ (epoch * 3266489917u);
}
__device__ bool wait(Flag* p, int target, Control* c, unsigned long long timeout) {
  const auto start = wall_clock64();
  while (flag(p) < target) {
    if (flag(&c->abort) || wall_clock64() - start > timeout) { set(&c->abort, 1); return false; }
    __builtin_amdgcn_s_sleep(1);
  }
  return !flag(&c->abort);
}
__device__ void publish(bool cross) {
  if (cross) asm volatile("buffer_wbl2 sc1\n\ts_waitcnt vmcnt(0)" ::: "memory");
  else asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}
// 40 KiB LDS reserves one block/CU. Runtime occupancy and observed CU identities
// are checked; block index from a previous launch is NEVER treated as a CU ID.
__global__ __launch_bounds__(T) void handoff(uint32_t* payload, size_t stride, int words, int rounds,
    int topology, int mode, int loaded, unsigned epoch, Control* c, uint32_t* traffic,
    Observation* obs, unsigned long long* full, unsigned long long timeout) {
  __shared__ int role, ok, finish;
  __shared__ unsigned long long errors;
  const unsigned key = cu(), x = xcc_id();
  if (threadIdx.x == 0) {
    role = -1; errors = 0;
    unsigned long long timer = ~0ull;
    for (int i=0; i<32; ++i) { auto a=wall_clock64(); auto b=wall_clock64(); timer=min(timer,b-a); }
    c->timer_ticks[blockIdx.x] = timer;
    if (key >= KEYS || x >= X || atomicAdd(c->seen + key, 1) != 0) set(&c->abort, 1);
    else {
      // Exactly one instruction per observed CU, before any payload read/write.
      asm volatile("buffer_inv sc1\n\ts_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
      atomicAdd(c->invalidations + key, 1);
      const int slot = atomicAdd(c->xcc + x, 1);
      if (x == 0 && slot == 0) role = 0;
      if (topology == 0 && x == 0 && slot == 1) role = 1;
      if (topology == 1 && x == 1 && slot == 0) role = 1;
      if (topology == 2 && x > 0 && slot == 0) role = x;
    }
    c->keys[blockIdx.x] = key; c->roles[blockIdx.x] = role;
    __hip_atomic_fetch_add(&c->barrier.value, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    ok = wait(&c->barrier, gridDim.x, c, timeout);
  }
  __syncthreads();
  if (!ok) return;
  const int readers = topology == 2 ? 7 : 1;
  if (role < 0) {
    unsigned long long bytes = 0; uint32_t sink = 0; size_t offset = 0;
    do {
      if (threadIdx.x == 0) finish = flag(&c->stop) || flag(&c->abort);
      __syncthreads();
      if (finish) break;
      if (loaded) {
        for (int j = 0; j < 32; ++j)
          sink ^= read(traffic + (size_t)blockIdx.x * BUSY_WORDS + ((offset + j*T + threadIdx.x) % BUSY_WORDS), false);
        offset = (offset + 32*T) % BUSY_WORDS; bytes += 32*T*sizeof(uint32_t);
      } else __builtin_amdgcn_s_sleep(4);
    } while (true);
    if (threadIdx.x == 0) c->busy_bytes[blockIdx.x] = bytes;
    if (sink == 0xdeadbeefu) traffic[(size_t)blockIdx.x * BUSY_WORDS + threadIdx.x] = sink;
    return;
  }
  for (int r = 1; r <= rounds; ++r) {
    // Positive slots are never reused within a launch. Negative controls use one
    // intentionally reused slot so one million large-payload handoffs fit easily.
    uint32_t* p = payload + (mode >= FORCED_OLD ? 0 : (size_t)(r-1)*stride);
    if (mode >= FORCED_OLD) {
      if (role == 0) {
        for (int i = threadIdx.x; i < words; i += T) p[i] = pattern(epoch-1, r, i);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); __syncthreads();
        if (threadIdx.x == 0) {
          publish(topology != 0); set(&c->prime, r); ok = 1;
          for (int j=1; j<=readers; ++j) ok &= wait(&c->primed[j], r, c, timeout);
        }
      } else {
        if (threadIdx.x == 0) ok = wait(&c->prime, r, c, timeout);
        __syncthreads(); if (!ok) return;
        for (int i = threadIdx.x; i < words; i += T) (void)read(p+i, false);
        __syncthreads(); if (threadIdx.x == 0) set(&c->primed[role], r);
      }
      __syncthreads(); if (!ok) return;
    }
    unsigned long long start = 0;
    if (role == 0) {
      if (threadIdx.x == 0) start = wall_clock64();
      __syncthreads();
      for (int i = threadIdx.x; i < words; i += T) p[i] = pattern(epoch, r, i);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); __syncthreads();
      if (threadIdx.x == 0) {
        publish(topology != 0); set(&c->ready, r); ok = 1;
        for (int j=1; j<=readers; ++j) ok &= wait(&c->ack[j], r, c, timeout);
        full[r-1] = wall_clock64() - start;  // same writer clock, includes ACK/validation
      }
      __syncthreads(); if (!ok) return;
    } else {
      if (threadIdx.x == 0) { ok = wait(&c->ready, r, c, timeout); errors = 0; }
      __syncthreads(); if (!ok) return;
      // No payload invalidate/acquire here. Signal polling uses sc0 sc1, not inv.
      unsigned long long first_ticks = 0, bulk_start = 0;
      uint32_t first = 0;
      if (threadIdx.x == 0) {
        bulk_start = wall_clock64(); const auto a = wall_clock64();
        first = read(p + (mode >= FORCED_OLD ? words-1 : 0), mode == BYPASS);
        first_ticks = wall_clock64() - a;
      }
      __syncthreads();
      unsigned long long bad = 0;
      for (int j=threadIdx.x; j<words; j+=T) {
        const int i = mode >= FORCED_OLD ? words-1-j : j;
        uint32_t got = j == 0 ? first : read(p+i, mode == BYPASS);
        if (mode == FORCED_OLD) got = pattern(epoch-1, r, i);
        bad += got != pattern(epoch, r, i);
      }
      if (bad) atomicAdd(&errors, bad);
      __syncthreads();
      if (threadIdx.x == 0) {
        obs[(size_t)(role-1)*rounds+r-1] = {first_ticks, wall_clock64()-bulk_start, errors};
        set(&c->ack[role], r);
      }
    }
    __syncthreads();
  }
  if (role == 0 && threadIdx.x == 0) set(&c->stop, 1);
}
static bool wait_fresh(hipStream_t stream) {
  auto start=std::chrono::steady_clock::now(); int next=10;
  for (;;) {
    auto e=hipStreamQuery(stream);
    if(e==hipSuccess) return true;
    if(e!=hipErrorNotReady) { fprintf(stderr,"fresh stream error: %s\n",hipGetErrorString(e)); return false; }
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if(elapsed>180) return false;
    if(elapsed>=next) { printf("WAIT fresh %.0fs\n",elapsed); fflush(stdout); next+=10; }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
static void usage() {
  fprintf(stderr, "fresh BYTES HANDOFFS same|cross|fanout cached|bypass|forced_old|primed_stale idle|synthetic_busy OUTDIR\n");
  exit(2);
}
static int run(int argc, char** argv) {
  if (argc != 8) usage();
  const int bytes=std::stoi(argv[2]), rounds=std::stoi(argv[3]);
  const std::string topo=argv[4], mode_s=argv[5], load=argv[6], out=argv[7];
  const int topology=topo=="same"?0:topo=="cross"?1:topo=="fanout"?2:-1;
  const int mode=mode_s=="cached"?CACHED:mode_s=="bypass"?BYPASS:mode_s=="forced_old"?FORCED_OLD:mode_s=="primed_stale"?PRIMED_STALE:-1;
  if ((bytes!=64 && bytes!=8192 && bytes!=65536) || rounds<1 || rounds>2000000 || topology<0 || mode<0 ||
      (load!="idle" && load!="synthetic_busy")) usage();
  if ((mode>=FORCED_OLD && rounds<1000000) || (mode<FORCED_OLD && (uint64_t)rounds*(bytes/4)<10000000)) usage();
  if (mkdir(out.c_str(), 0755) != 0) { perror("fresh output directory (must be new)"); return 2; }
  hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop,0));
  if (std::string(prop.gcnArchName).find("gfx942")!=0 || prop.multiProcessorCount>1024 || !prop.cooperativeLaunch) {
    fprintf(stderr,"fresh requires cooperative gfx942 with <=1024 CUs\n"); return 2;
  }
  int blocks=0, khz=0; CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,handoff,T,LDS));
  CHECK(hipDeviceGetAttribute(&khz,hipDeviceAttributeWallClockRate,0));
  if (blocks!=1 || khz<=0) { fprintf(stderr,"cannot establish one resident block/CU or wall clock rate\n"); return 2; }
  const int ncu=prop.multiProcessorCount, words=bytes/4, readers=topology==2?7:1, loaded=load=="synthetic_busy";
  const size_t stride=(bytes+127)/128*32, slots=mode>=FORCED_OLD?1:rounds;
  uint32_t *payload,*traffic; Control* ctl; Observation* obs; unsigned long long* full;
  CHECK(hipMalloc(&payload,slots*stride*4)); CHECK(hipMemset(payload,0,slots*stride*4));
  if ((uintptr_t)payload%128) { fprintf(stderr,"payload alignment error\n"); return 2; }
  CHECK(hipMalloc(&traffic,(size_t)ncu*BUSY_WORDS*4)); CHECK(hipMemset(traffic,1,(size_t)ncu*BUSY_WORDS*4));
  CHECK(hipMalloc(&ctl,sizeof(Control))); CHECK(hipMalloc(&obs,(size_t)readers*rounds*sizeof(Observation)));
  CHECK(hipMalloc(&full,(size_t)rounds*8)); hipStream_t stream; CHECK(hipStreamCreate(&stream));
  std::ofstream results(out+"/results.jsonl"); if(!results) return 2;
  int status=0;
  // Two distinct kernel launches, host-confirmed completion between them. Payload
  // allocation and VA stay unchanged; there is no host payload copy/reset between.
  for (unsigned epoch=1;epoch<=2;++epoch) {
    CHECK(hipMemset(ctl,0,sizeof(Control))); CHECK(hipMemset(obs,0,(size_t)readers*rounds*sizeof(Observation))); CHECK(hipMemset(full,0,(size_t)rounds*8));
    auto timeout=(unsigned long long)khz*1000*5;
    int w=words,r=rounds,t=topology,m=mode,l=loaded; size_t s=stride;
    void* args[]={&payload,&s,&w,&r,&t,&m,&l,&epoch,&ctl,&traffic,&obs,&full,&timeout};
    printf("START fresh epoch=%u bytes=%d rounds=%d topology=%s mode=%s load=%s\n",epoch,bytes,rounds,topo.c_str(),mode_s.c_str(),load.c_str()); fflush(stdout);
    CHECK(hipLaunchCooperativeKernel((void*)handoff,dim3(ncu),dim3(T),args,LDS,stream));
    if(!wait_fresh(stream)) { fprintf(stderr,"fresh watchdog timeout\n"); return 3; }
    Control h; CHECK(hipMemcpy(&h,ctl,sizeof(h),hipMemcpyDeviceToHost));
    std::vector<Observation> samples((size_t)readers*rounds); std::vector<unsigned long long> duration(rounds);
    CHECK(hipMemcpy(samples.data(),obs,samples.size()*sizeof(Observation),hipMemcpyDeviceToHost)); CHECK(hipMemcpy(duration.data(),full,rounds*8,hipMemcpyDeviceToHost));
    unsigned long long errors=0,busy=0; std::set<int> keys; int seen_roles[X]={}; bool placement=true;
    for(int i=0;i<ncu;++i) {
      int key=h.keys[i],role=h.roles[i]; keys.insert(key); busy+=h.busy_bytes[i];
      if(key<0 || key>=KEYS || h.seen[key]!=1 || h.invalidations[key]!=1) placement=false;
      if(role>=0 && role<X) { ++seen_roles[role]; int x=key>>8;
        if ((role==0 && x!=0) || (role>0 && x!=(topology==0?0:role))) placement=false;
      }
    }
    for(int j=0;j<=readers;++j) if(seen_roles[j]!=1) placement=false;
    placement &= keys.size()==(size_t)ncu;
    bool complete=!h.abort.value && h.stop.value==1 && h.ready.value==rounds;
    for(int j=1;j<=readers;++j) complete &= h.ack[j].value==rounds;
    if(loaded && busy==0) complete=false;
    std::ofstream raw(out+"/samples-"+std::to_string(epoch)+".csv"); raw<<"handoff,reader,first_load_ticks,bulk_checked_ticks,handoff_ack_ticks,wrong_words\n";
    for(int j=0;j<readers;++j) for(int i=0;i<rounds;++i) { const auto a=samples[(size_t)j*rounds+i]; errors+=a.errors;
      if(!a.load || !a.bulk || !duration[i]) complete=false;
      raw<<i+1<<","<<j+1<<","<<a.load<<","<<a.bulk<<","<<duration[i]<<","<<a.errors<<"\n";
    }
    std::ofstream map(out+"/placement-"+std::to_string(epoch)+".csv"); map<<"block,cu_key,role,invalidations\n";
    for(int i=0;i<ncu;++i) { int k=h.keys[i]; map<<i<<","<<k<<","<<h.roles[i]<<","<<(k>=0&&k<KEYS?h.invalidations[k]:0)<<"\n"; }
    raw.close(); map.close();
    if(!raw || !map) { fprintf(stderr,"cannot write complete sample files\n"); return 2; }
    unsigned long long timer=*std::min_element(h.timer_ticks,h.timer_ticks+ncu);
    bool valid=complete && placement && (mode>=FORCED_OLD?errors>0:errors==0);
    if(!valid) status=1;
    results<<"{\"schema_version\":1,\"epoch\":"<<epoch<<",\"bytes\":"<<bytes<<",\"handoffs\":"<<rounds
      <<",\"readers\":"<<readers<<",\"words_checked\":"<<(uint64_t)rounds*words*readers<<",\"wrong_words\":"<<errors
      <<",\"topology\":\""<<topo<<"\",\"mode\":\""<<mode_s<<"\",\"load\":\""<<load<<"\",\"wall_clock_khz\":"<<khz
      <<",\"gpu\":\""<<prop.gcnArchName<<"\",\"ncu\":"<<ncu<<",\"cu_coverage\":"<<keys.size()
      <<",\"timer_min_ticks\":"<<timer<<",\"busy_bytes\":"<<busy<<",\"payload_address\":"<<(uintptr_t)payload<<",\"slot_stride_bytes\":"<<stride*4
      <<",\"launch_id\":"<<epoch<<",\"complete\":"<<(complete?"true":"false")<<",\"placement_valid\":"<<(placement?"true":"false")
      <<",\"valid\":"<<(valid?"true":"false")<<",\"load_kind\":\"synthetic_only\"}\n"; results.flush();
    if(!results) { fprintf(stderr,"cannot write result record\n"); return 2; }
    printf("RESULT fresh epoch=%u complete=%d placement=%d wrong_words=%llu valid=%d\n",epoch,complete,placement,errors,valid); fflush(stdout);
    if(!complete || !placement) break;
  }
  CHECK(hipStreamDestroy(stream)); CHECK(hipFree(payload)); CHECK(hipFree(traffic)); CHECK(hipFree(ctl)); CHECK(hipFree(obs)); CHECK(hipFree(full));
  return status;
}
} // namespace fresh
