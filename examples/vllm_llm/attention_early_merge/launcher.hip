// Replay actual check-host captures. No fabricated attention inputs.
#include <hip/hip_runtime.h>
#include "examples/vllm_llm/attention_early_merge/probe.h"
#include "imports.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cstdlib>
#define HIP(call) do { auto c6_error=(call); if(c6_error!=hipSuccess) { std::cerr<<#call<<": "<<hipGetErrorString(c6_error)<<'\n'; std::exit(2); } } while(0)
extern "C" __global__ void c6_probe(C6Args);
static void require(bool ok,const std::string& why) { if(!ok) { std::cerr<<"BLOCKER "<<why<<'\n'; std::exit(3); } }
static std::vector<unsigned char> read(const std::string& p,size_t n) {
  std::ifstream f(p,std::ios::binary); require(bool(f),"missing "+p);
  std::vector<unsigned char> b(n); f.read((char*)b.data(),n);
  require((size_t)f.gcount()==n && f.peek()==EOF,"size "+p); return b;
}
template<class T> static T* alloc(size_t n) { T* p; HIP(hipMalloc(&p,std::max<size_t>(n,1)*sizeof(T))); return p; }
struct Buffer { std::string name; uint64_t original; size_t bytes; unsigned char* device; std::vector<unsigned char> before,after; };
static void wait(hipStream_t stream) {
  auto t=std::chrono::steady_clock::now();
  for (;;) {
    auto e=hipStreamQuery(stream); if(e==hipSuccess) return;
    HIP(e==hipErrorNotReady?hipSuccess:e);
    if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()>5) {
      std::cerr<<"BLOCKER device execution timeout; do not reset live state\n"<<std::flush; std::_Exit(4);
    }
  }
}
int main(int argc,char** argv) {
  require(argc>=2,"usage: run FRAME [--measure] [--fault duplicate|stale|timeout|future] [--repeats N]");
  std::string dir=argv[1],fault; bool measure=false; int repeats=4; unsigned skew=0;
  for(int i=2;i<argc;++i) {
    std::string k=argv[i];
    if(k=="--measure") measure=true;
    else if(k=="--fault" && i+1<argc) fault=argv[++i];
    else if(k=="--skew" && i+1<argc) skew=std::atoi(argv[++i]);
    else if(k=="--repeats" && i+1<argc) repeats=std::atoi(argv[++i]);
    else require(false,"unknown option "+k);
  }
  require(skew<=100000,"skew cap");
#ifndef C6_RETAINED_ENVELOPE
  require(!measure,"timing requires --footprint build; small protocol binary is correctness-only");
#endif
  require(repeats>0 && repeats<=1000,"repeat cap");
  require(fault.empty() || fault=="duplicate" || fault=="stale" || fault=="timeout" || fault=="future","unknown fault");
  require(!measure || (fault.empty() && !skew),"cannot time fault injection");
  std::ifstream meta(dir+"/frame.txt"); std::string magic; int layer,position,count;
  meta>>magic>>layer>>position>>count;
  require(magic=="C6CAP1" && layer>=0 && layer<48 && position>=0 && position<=1040 && count==15,"capture shape");
  std::vector<Buffer> buffers;
  for(int i=0;i<count;++i) {
    Buffer b; meta>>b.name>>b.original>>b.bytes;
    require(bool(meta) && b.bytes>0 && b.bytes<(1u<<27),"buffer descriptor");
    require(b.name.find_first_not_of("abcdefghijklmnopqrstuvwxyz_")==std::string::npos,"buffer name");
    for(const auto& old:buffers) require(old.name!=b.name,"duplicate buffer name");
    b.before=read(dir+"/"+b.name+".before",b.bytes); b.after=read(dir+"/"+b.name+".after",b.bytes);
    b.device=alloc<unsigned char>(b.bytes); buffers.push_back(std::move(b));
  }
  const char* exports[]={"dsl_attnprep","attn_mfma16_g8","attn_reduce"};
  std::vector<unsigned char> arguments; size_t offsets[3];
  for(int k=0;k<3;++k) {
    std::string name; size_t bytes; meta>>name>>bytes; const auto* desc=etx_imp_find(exports[k]);
    require(name==exports[k] && desc && bytes==(size_t)desc->arg_bytes,"argument layout");
    offsets[k]=arguments.size(); auto arg=read(dir+"/"+name+".args",bytes);
    for(int j=0;j<desc->nparams;++j) if(desc->ty[j]=='p') {
      uint64_t old; memcpy(&old,arg.data()+desc->off[j],8); if(!old) continue;
      unsigned char* ptr=nullptr; unsigned found=0;
      for(auto& b:buffers) if(old>=b.original && old-b.original<b.bytes) { ptr=b.device+(old-b.original); ++found; }
      require(found==1,"unresolved/ambiguous pointer "+name+" parameter "+std::to_string(j));
      memcpy(arg.data()+desc->off[j],&ptr,8);
    }
    const int expected[3][4]={{4,1,1,4},{1,5,4,20},{32,1,1,32}};
    require(memcmp(arg.data()+desc->grid_off,expected[k],16)==0,"import grid");
    const char* error=etx_spec_validate(desc,arg.data(),expected[k][0],expected[k][1],expected[k][2]);
    require(!error,error?error:"");
    arguments.insert(arguments.end(),arg.begin(),arg.end());
  }
  for(const auto& b:buffers) if(b.name=="seq_lens") {
    int n=0; require(b.bytes==4,"seq_lens size"); memcpy(&n,b.before.data(),4); require(n==position+1,"sequence position");
  }
  hipDeviceProp_t prop; HIP(hipGetDeviceProperties(&prop,0));
  require(std::string(prop.gcnArchName).find("gfx942")==0 && prop.multiProcessorCount==304 && prop.cooperativeLaunch,"one full MI300X required");
  int resident=0; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,c6_probe,1024,0));
  require(resident>=1,"cooperative residency");
  hipStream_t stream; HIP(hipStreamCreate(&stream));
  auto ab=alloc<unsigned char>(arguments.size()); HIP(hipMemcpy(ab,arguments.data(),arguments.size(),hipMemcpyHostToDevice));
  auto table=alloc<void*>(1); HIP(hipMemcpy(table,&ab,sizeof(void*),hipMemcpyHostToDevice));
  C6Args a{};
  etx_ctx* contexts[]={&a.prep,&a.attention,&a.reduce};
  for(int k=0;k<3;++k) { contexts[k]->args=table; contexts[k]->cst[0]=offsets[k]; }
  a.state=alloc<C6State>(4); a.tickets=alloc<unsigned>(8); a.abort=alloc<int>(1); a.outputs=alloc<unsigned>(1); a.observed=alloc<unsigned short>(32*128);
  a.spin_limit=1000000; a.skew_cycles=skew; a.skew_partition=0;
#ifdef C6_DIAGNOSTIC
  a.stamps=alloc<unsigned long long>(4*7*8);
#endif
  unsigned long long epoch=0;
  hipEvent_t begin,end; HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end));
  auto launch=[&](unsigned mode,bool timed) {
    require(epoch<std::numeric_limits<unsigned long long>::max()-1,"rollover needs full quiescent clear");
    a.epoch=++epoch; a.mode=mode;
    C6State state[4]{}; for(auto& s:state) { s.epoch=a.epoch; s.elected=0xffffffff; }
    if(fault=="duplicate") for(auto& s:state) s.mask=31;
    if(fault=="stale") for(auto& s:state) s.epoch=a.epoch-1;
    if(fault=="future") for(auto& s:state) { s.claim=1; s.ready=a.epoch+1; }
    if(fault=="timeout") { a.fault=1; a.spin_limit=100; }
    // ALL previous blocks retired at wait(), before reset or epoch reuse.
    for(const auto& b:buffers) HIP(hipMemcpyAsync(b.device,b.before.data(),b.bytes,hipMemcpyHostToDevice,stream));
    HIP(hipMemcpyAsync(a.state,state,sizeof(state),hipMemcpyHostToDevice,stream));
    HIP(hipMemsetAsync(a.tickets,0,8*4,stream)); HIP(hipMemsetAsync(a.abort,0,4,stream));
    if(a.stamps) HIP(hipMemsetAsync(a.stamps,0,4*7*8*8,stream));
    unsigned four=4; HIP(hipMemcpyAsync(a.outputs,&four,4,hipMemcpyHostToDevice,stream));
    wait(stream); // identical start: inputs and initial protocol state ready
    if(timed) HIP(hipEventRecord(begin,stream));
    void* params[]={&a}; HIP(hipLaunchCooperativeKernel((const void*)c6_probe,dim3(304),dim3(1024),params,0,stream));
    if(timed) HIP(hipEventRecord(end,stream));
    wait(stream);
    int aborted=0; unsigned outputs=4,tickets[8];
    HIP(hipMemcpy(&aborted,a.abort,4,hipMemcpyDeviceToHost));
    if(!fault.empty()) { require(aborted!=0,"fault was not rejected"); return; }
    HIP(hipMemcpy(&outputs,a.outputs,4,hipMemcpyDeviceToHost));
    HIP(hipMemcpy(state,a.state,sizeof(state),hipMemcpyDeviceToHost));
    HIP(hipMemcpy(tickets,a.tickets,sizeof(tickets),hipMemcpyDeviceToHost));
    require(!aborted && !outputs && tickets[7]>=1,"protocol abort/output/consumer discovery gate");
    for(int h=0;h<4;++h) require(tickets[h]>=7 && state[h].claim==1 && (!mode || state[h].mask==31) && state[h].arrived==5 && state[h].output==a.epoch && (!mode || state[h].elected<5),"exact-once/discovery state");
    for (const auto& b:buffers) if (b.name=="attn_out") {
      require(b.bytes==32*128*2,"attention output size");
      std::vector<unsigned char> observed(b.bytes); HIP(hipMemcpy(observed.data(),a.observed,b.bytes,hipMemcpyDeviceToHost));
      require(observed==b.after,"cross-XCD consumer payload mismatch");
    }
    for(const auto& b:buffers) {
      std::vector<unsigned char> actual(b.bytes); HIP(hipMemcpy(actual.data(),b.device,b.bytes,hipMemcpyDeviceToHost));
      require(actual==b.after,"byte mismatch "+b.name+" mode="+std::to_string(mode));
    }
    if(timed) { float ms=0; HIP(hipEventElapsedTime(&ms,begin,end));
      std::cout<<"C6-RESULT arm="<<(mode?"candidate":"control")<<" layer="<<layer<<" position="<<position<<" device_ms="<<std::fixed<<ms<<'\n'; }
  };
  for(int r=0;r<repeats;++r) {
    std::cout<<"STAGE replay layer="<<layer<<" position="<<position<<" round="<<r<<std::endl;
    if (!fault.empty()) launch(1,false);
    else { launch(r%2,false); launch(1-r%2,false); }
  }
  if(!fault.empty()) { std::cout<<"C6-FAULT rejected="<<fault<<'\n'; return 0; }
  std::cout<<"C6-CHECK OK layer="<<layer<<" position="<<position<<" epochs="<<epoch<<std::endl;
  if(measure) for(int r=0;r<5;++r) { launch(r%2,true); launch(1-r%2,true); }
  // Process teardown follows quiescence. No state or epochs survive a process.
  return 0;
}
