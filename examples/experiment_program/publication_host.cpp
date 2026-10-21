// G11 only. Real HIP module launch of the original restricted 512-task chain.
#include <hip/hip_runtime.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <array>
#include <stdexcept>
#if G11_LAYER == 1
#include "plan-L1.h"
#elif G11_LAYER == 24
#include "plan-L24.h"
#elif G11_LAYER == 47
#include "plan-L47.h"
#else
#error "explicit original representative layer required"
#endif
#include "etx/host.h"
using Clock=std::chrono::steady_clock;
static Clock::time_point deadline;
static void wait(hipStream_t stream) {
  auto next=Clock::now()+std::chrono::seconds(10);
  for(;;) {
    auto rc=hipStreamQuery(stream);if(rc==hipSuccess)return;
    if(rc!=hipErrorNotReady){fprintf(stderr,"HIP stream failure %d\n",int(rc));exit(2);}
    if(Clock::now()>deadline){fprintf(stderr,"NOT-COMPLETE G11 stream deadline\n");_Exit(3);}
    if(Clock::now()>next){puts("PROGRESS G11 waiting for integrated chain");next=Clock::now()+std::chrono::seconds(10);}
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}
static std::vector<char> read(const std::string& path,size_t offset,size_t n) {
  std::ifstream f(path,std::ios::binary);std::vector<char> data(n);f.seekg(offset);f.read(data.data(),n);
  if(!f || size_t(f.gcount())!=n)throw std::runtime_error("short input "+path);return data;
}
static void save(std::ofstream& f,const void* pointer,size_t bytes) {
  std::vector<char> h(bytes);ETX_CHECK(hipMemcpy(h.data(),pointer,bytes,hipMemcpyDeviceToHost));f.write(h.data(),bytes);
  if(!f)throw std::runtime_error("G11 raw write failed");
}
struct Buffer { void* live;void* pristine;size_t bytes; };
struct Reference { int buffer;size_t offset;std::vector<char> bytes;int poison; };
int main(int argc,char**argv) try {
  if(argc==2 && std::string(argv[1])=="--help"){puts("G11 integrated down -> QKV; real HIP; config only; no device calls for help");return 0;}
  if(argc!=2)return 1;setvbuf(stdout,nullptr,_IOLBF,0);
  std::ifstream f(argv[1]);std::string output,path;int layer,samples,nb,np,nr;double seconds;
  f>>std::quoted(output)>>layer>>samples>>seconds>>nb>>np>>nr;
  if(!f || layer!=G11_LAYER || samples!=4096 || seconds<=0 || nb<1 || np<1 || nr<1)return 1;
  deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
  ETX_CHECK(hipInit(0));ETX_CHECK(hipSetDevice(0));hipDeviceProp_t prop{};ETX_CHECK(hipGetDeviceProperties(&prop,0));
  if(std::string(prop.gcnArchName).rfind("gfx942",0)!=0 || prop.warpSize!=64 || prop.multiProcessorCount!=304)throw std::runtime_error("G11 requires actual eight-XCD/38-CU gfx942 placement");
  int clock_khz=0;ETX_CHECK(hipDeviceGetAttribute(&clock_khz,hipDeviceAttributeWallClockRate,0));if(clock_khz<=0)return 1;
  std::array<hipModule_t,3> modules;std::array<hipFunction_t,3> kernels;
  for(int a=0;a<3;++a){f>>std::quoted(path);ETX_CHECK(hipModuleLoad(&modules[a],path.c_str()));ETX_CHECK(hipModuleGetFunction(&kernels[a],modules[a],"etx_megakernel_d0"));int capacity=0;ETX_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,kernels[a],1024,0));if(capacity<1)throw std::runtime_error("native resident capacity failed");printf("RESOURCE arm=%d actual_blocks_per_CU=%d CU=%d clock_khz=%d\n",a,capacity,prop.multiProcessorCount,clock_khz);}
  std::array<std::vector<char>,9> blocks;for(auto&b:blocks){f>>std::quoted(path);b=read(path,0,160);}
  std::vector<Buffer> buffers;
  for(int i=0;i<nb;++i){size_t off,n;f>>std::quoted(path)>>off>>n;if(!n)return 1;auto h=read(path,off,n);Buffer b;b.bytes=n;ETX_CHECK(hipMalloc(&b.live,n));ETX_CHECK(hipMalloc(&b.pristine,n));ETX_CHECK(hipMemcpy(b.pristine,h.data(),n,hipMemcpyHostToDevice));buffers.push_back(b);}
  for(int i=0;i<np;++i){int block,offset,buffer;size_t add;f>>block>>offset>>buffer>>add;if(block<0 || block>=9 || offset<0 || offset+8>160 || buffer<0 || buffer>=nb || add>=buffers[buffer].bytes)return 1;auto p=(char*)buffers[buffer].live+add;memcpy(blocks[block].data()+offset,&p,8);}
  std::vector<Reference> references;
  for(int i=0;i<nr;++i){Reference r;size_t n;f>>r.buffer>>r.offset>>std::quoted(path)>>n>>r.poison;if(r.buffer<0 || r.buffer>=nb || n==0 || r.offset+n>buffers[r.buffer].bytes)return 1;r.bytes=read(path,0,n);references.push_back(r);}
  if(!f)return 1;std::string extra;if(f>>extra)return 1;
  // Disable ambient options: this host owns the exact reset/trace selection.
  unsetenv("ETX_DEV_RESET");unsetenv("ETX_NO_LASTFLUSH");unsetenv("ETX_INBODY_RAW");unsetenv("ETX_RELAY");unsetenv("ETX_TRACE");unsetenv("ETX_EXEC_COUNT");
  etx_host h;h.alloc_args();std::vector<char> down;for(int i=0;i<8;++i)down.insert(down.end(),blocks[i].begin(),blocks[i].end());
  ETX_CHECK(hipMemcpy(h.args[0],down.data(),1280,hipMemcpyHostToDevice));ETX_CHECK(hipMemcpy(h.args[1],blocks[8].data(),160,hipMemcpyHostToDevice));h.setup();
  auto trace=etx_host::upload(std::vector<uint64_t>(512*4).data(),512*4);auto poll=etx_host::upload(std::vector<uint64_t>(512).data(),512);auto event=etx_host::upload(std::vector<uint64_t>(3).data(),3);auto wave=etx_host::upload(std::vector<uint64_t>(512*16*3).data(),512*16*3);
  const int rowwords=16*ETX_LOAD_RECORD_WORDS;auto loads=etx_host::upload(std::vector<uint64_t>(rowwords).data(),rowwords);std::vector<int32_t> offsets(513);auto d_offsets=etx_host::upload(offsets.data(),offsets.size());
  std::ofstream timings(output+".timing.jsonl");
  auto invoke=[&](int arm,int selector,bool coverage) {
    for(auto&b:buffers)ETX_CHECK(hipMemcpyAsync(b.live,b.pristine,b.bytes,hipMemcpyDeviceToDevice,h.stream));
    for(auto&r:references)if(r.poison){auto poison=r.bytes;for(auto&c:poison)c=(char)~(unsigned char)c;ETX_CHECK(hipMemcpy((char*)buffers[r.buffer].live+r.offset,poison.data(),poison.size(),hipMemcpyHostToDevice));}
    h.reset_step();h.p.trace_exec=(coverage||arm)?h.d_trace:nullptr;h.p.trace_time=(coverage||arm)?trace:nullptr;
    h.p.inbody_poll=arm?poll:nullptr;h.p.inbody_event=arm?event:nullptr;h.p.inbody_wave=(coverage||arm)?wave:nullptr;h.p.inbody_load=arm?loads:nullptr;h.p.inbody_load_offsets=d_offsets;
    ETX_CHECK(hipMemsetAsync(trace,0,512*4*8,h.stream));ETX_CHECK(hipMemsetAsync(poll,0,512*8,h.stream));ETX_CHECK(hipMemsetAsync(event,0,3*8,h.stream));ETX_CHECK(hipMemsetAsync(wave,0,512*16*3*8,h.stream));ETX_CHECK(hipMemsetAsync(loads,0,rowwords*8,h.stream));ETX_CHECK(hipMemsetAsync(h.d_trace,0,512*4,h.stream));
    int target=256+selector/16;for(int t=0;t<=512;++t)offsets[t]=t>target?rowwords:0;
    ETX_CHECK(hipMemcpyAsync(d_offsets,offsets.data(),offsets.size()*4,hipMemcpyHostToDevice,h.stream));uint64_t sentinel=5;ETX_CHECK(hipMemcpyAsync(loads+(selector%16)*ETX_LOAD_RECORD_WORDS,&sentinel,8,hipMemcpyHostToDevice,h.stream));wait(h.stream);
    hipEvent_t begin,end;ETX_CHECK(hipEventCreate(&begin));ETX_CHECK(hipEventCreate(&end));void* av[]={&h.p};auto start=Clock::now();ETX_CHECK(hipEventRecord(begin,h.stream));
    ETX_CHECK(hipModuleLaunchKernel(kernels[arm],304,1,1,1024,1,1,0,h.stream,av,nullptr));ETX_CHECK(hipEventRecord(end,h.stream));wait(h.stream);double wall_us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();float ms;ETX_CHECK(hipEventElapsedTime(&ms,begin,end));
    auto base=output+".s"+std::to_string(selector)+".a"+std::to_string(arm)+(coverage?".coverage":"");std::ofstream refs(base+".references",std::ios::binary);
    for(auto&r:references){std::vector<char> got(r.bytes.size());ETX_CHECK(hipMemcpy(got.data(),(char*)buffers[r.buffer].live+r.offset,got.size(),hipMemcpyDeviceToHost));refs.write(got.data(),got.size());if(got!=r.bytes)throw std::runtime_error("original exact output/intermediate mismatch");}
    // Slot counters/key maps are populated by the original worker allocation,
    // including the uninstrumented arm. No diagnostic extra placement launch.
    std::ofstream raw(base+".raw",std::ios::binary);uint64_t header[]={0x314C425550313147ULL,1,(uint64_t)G11_LAYER,(uint64_t)selector,(uint64_t)arm,(uint64_t)clock_khz,(uint64_t)coverage};raw.write((char*)header,sizeof(header));
    save(raw,trace,512*4*8);save(raw,poll,512*8);save(raw,event,3*8);save(raw,wave,512*16*3*8);save(raw,loads,rowwords*8);save(raw,h.d_slots,ETX_SLOT_WORDS*4);save(raw,h.d_trace,512*4);
    if(!raw || !refs)throw std::runtime_error("G11 evidence write failure");
    if(!coverage)timings<<"{\"selector\":"<<selector<<",\"arm\":"<<arm<<",\"device_us\":"<<std::setprecision(12)<<ms*1000.0<<",\"host_wall_us\":"<<wall_us<<",\"exact\":true}\n";
    timings.flush();ETX_CHECK(hipEventDestroy(begin));ETX_CHECK(hipEventDestroy(end));
    printf("G11 selector=%d arm=%d coverage=%d exact=YES device_us=%.6f\n",selector,arm,int(coverage),ms*1000.0);
  };
  // Separate untimed control coverage; timed control has diagnostics disabled.
  for(int arm=0;arm<3;++arm)invoke(arm,0,true);
  for(int s=0;s<samples;++s)for(int j=0;j<3;++j)invoke((s+j)%3,s,false);
  puts("COMPLETE G11 all4096 selectors and all3 modes exact references");return 0;
} catch(const std::exception&e){fprintf(stderr,"NOT-COMPLETE G11 %s\n",e.what());return 4;}
