#include <hip/hip_runtime_api.h>
#include "examples/experiment_program/hip_dynamic.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>
using Clock=std::chrono::steady_clock;
static Clock::time_point deadline;
#define HIP(x) do{auto e=(x);if(e!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e));}while(0)
static void wait(hipStream_t stream){
  auto last=Clock::now();
  for(;;){auto e=hipStreamQuery(stream);if(e==hipSuccess)return;
    if(e!=hipErrorNotReady)throw std::runtime_error(hipGetErrorString(e));
    if(Clock::now()>deadline){fprintf(stderr,"NOT-COMPLETE deadline waiting G01\n");fflush(stderr);_Exit(3);}
    if(Clock::now()-last>std::chrono::seconds(10)){puts("PROGRESS G01 stream wait");last=Clock::now();}
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}
static std::vector<unsigned char> read(const std::string& name,size_t n){
  std::ifstream f(name,std::ios::binary);std::vector<unsigned char>b(n);f.read((char*)b.data(),n);
  if(!f||f.peek()!=EOF)throw std::runtime_error("invalid exact input length: "+name);return b;
}
static void* alloc(size_t n,const void* data=nullptr){void*p;HIP(hipMalloc(&p,n));if(data)HIP(hipMemcpy(p,data,n,hipMemcpyHostToDevice));return p;}
template<class T>static void put(unsigned char*a,int off,T v){memcpy(a+off,&v,sizeof(v));}
struct State{unsigned next[8],workers[8],done,elected,published,selected,epoch,reverse,domain_mask,winner,readers[8],task_seen[256];};
struct Case{
  std::string path;std::vector<void*> memory;void* args;void* topk;std::vector<unsigned char> expected,observed_poison;
  std::vector<std::vector<unsigned char>> poison;
  explicit Case(std::string p):path(p){
    const char*names[]={"router","o","residual","gamma","norm","resid2","logits","ids","mix","tei"};
    size_t sizes[]={524288,4096,4096,4096,4096,4096,256,256,256,256};
    std::vector<std::vector<unsigned char>>h;
    for(int i=0;i<10;++i){h.push_back(read(p+names[i]+".bin",sizes[i]));memory.push_back(alloc(sizes[i],h.back().data()));}
    unsigned char a[160]={};put(a,0,memory[0]);put(a,8,memory[4]);put(a,16,memory[6]);
    put(a,48,128);put(a,52,2048);put(a,56,1);put(a,60,128);
    put(a,64,memory[1]);put(a,72,memory[2]);put(a,80,memory[5]);put(a,88,memory[3]);
    put(a,96,1);put(a,100,1e-6f);put(a,104,1);put(a,108,128);put(a,120,1);put(a,156,256);
    args=alloc(160,a);unsigned char tk[8*64]={};
    for(int r=0;r<8;++r){auto z=tk+r*64;put(z,0,memory[6]);put(z,8,(char*)memory[8]+r*32);put(z,16,(char*)memory[7]+r*32);put(z,24,(char*)memory[9]+r*32);put(z,32,128);put(z,36,8);put(z,40,1);}
    topk=alloc(sizeof(tk),tk);
    for(int r=0;r<8;++r)for(int i:{7,8,9,4,5,6}){
      size_t off=i>=7?r*32:0,n=i>=7?32:sizes[i];expected.insert(expected.end(),h[i].begin()+off,h[i].begin()+off+n);
    }
    for(int i=4;i<10;++i){poison.push_back(h[i]);for(auto& b:poison.back())b^=0xff;}
    observed_poison=expected;for(auto&b:observed_poison)b^=0xff;
  }
  ~Case(){for(void*p:memory)hipFree(p);hipFree(args);hipFree(topk);}
};
int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IOLBF,0);
  if(argc==2&&std::string(argv[1])=="--help"){puts("G01 resident exact replay: replay <validated-config.txt>; never generates tensor payloads");return 0;}
  if(argc!=2){fprintf(stderr,"usage: replay <validated-config.txt>\n");return 1;}
  try{
    std::ifstream f(argv[1]);std::string module,output;int warmup,population,chosen,ncases,diagnostic,stress;double seconds;
    std::string orders[5];f>>std::quoted(module)>>std::quoted(output)>>warmup>>population>>chosen>>ncases>>seconds>>diagnostic>>stress;
    for(auto&o:orders)f>>o;
    if(!f||warmup<1||population<1||ncases!=96||chosen<0||chosen>=ncases||!std::isfinite(seconds)||seconds<=0)throw std::runtime_error("invalid native protocol config");
    std::vector<std::string> paths(ncases);for(auto&p:paths)f>>std::quoted(p);
    if(!f)throw std::runtime_error("truncated case list");
    // No raw synthetic inputs are accepted by the Python entry point which seals
    // this text. This host still checks every tensor size and every output byte.
    deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    load_hip();HIP(hipInit(0));hipDeviceProp_t prop;HIP(hipGetDeviceProperties(&prop,0));
    if(std::string(prop.gcnArchName).rfind("gfx942",0)!=0||prop.multiProcessorCount!=304||prop.warpSize!=64)throw std::runtime_error("requires one304-CU gfx942 wave64");
    hipModule_t m;HIP(hipModuleLoad(&m,module.c_str()));hipFunction_t reset,kernel;
    HIP(hipModuleGetFunction(&reset,m,"g01_reset"));HIP(hipModuleGetFunction(&kernel,m,"g01_run"));
    hipStream_t stream;HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    auto state=alloc(sizeof(State));auto observed=alloc(8544*8);unsigned epoch=stress?0xfffffffdu:0;unsigned reset_count=0;
    constexpr size_t trace_bytes=264*16*13*sizeof(unsigned long long);
    std::vector<unsigned long long> trace_zero(trace_bytes/8,0);
    void* trace=diagnostic?alloc(trace_bytes,trace_zero.data()):nullptr;
    hipEvent_t begin,end;HIP(hipEventCreate(&begin));HIP(hipEventCreate(&end));
    auto trial=[&](Case& c,int candidate,unsigned reverse,bool save)->std::pair<double,double>{
      if(epoch==0xffffffffu){
        wait(stream);HIP(hipFree(state));state=alloc(sizeof(State));epoch=0;++reset_count;
        printf("RESET G01 previous_epoch=4294967295 quiescent=1 fresh_allocation=1 reset_count=%u\n",reset_count);
      }
      ++epoch;
      const size_t output_sizes[]={4096,4096,256,256,256,256};
      for(int i=0;i<6;++i)HIP(hipMemcpyAsync(c.memory[i+4],c.poison[i].data(),output_sizes[i],hipMemcpyHostToDevice,stream));
      HIP(hipMemcpyAsync(observed,c.observed_poison.data(),8544*8,hipMemcpyHostToDevice,stream)); // every missing byte differs from golden
      if(diagnostic)HIP(hipMemcpyAsync(trace,trace_zero.data(),trace_bytes,hipMemcpyHostToDevice,stream));
      void*reset_args[]={&state,&epoch,&reverse};void*args[]={&c.args,&c.topk,&state,&epoch,&candidate,&observed,&trace};
      wait(stream); // diagnostic poisoning is outside both decisive timing brackets
      auto host=Clock::now();HIP(hipEventRecord(begin,stream));
      HIP(hipModuleLaunchKernel(reset,1,1,1,256,1,1,0,stream,reset_args,nullptr));
      HIP(hipModuleLaunchKernel(kernel,304,1,1,1024,1,1,0,stream,args,nullptr));
      HIP(hipEventRecord(end,stream));wait(stream);auto wall=std::chrono::duration<double,std::nano>(Clock::now()-host).count();
      float ms;HIP(hipEventElapsedTime(&ms,begin,end));
      State st;HIP(hipMemcpy(&st,state,sizeof(st),hipMemcpyDeviceToHost));
      if(st.done!=256||st.epoch!=epoch||st.domain_mask!=255)throw std::runtime_error("missing task/domain/epoch");
      for(int i=0;i<256;++i)if(st.task_seen[i]!=1)throw std::runtime_error("duplicate/missing router task");
      for(int d=0;d<8;++d)if(st.workers[d]!=38||st.readers[d]!=epoch)throw std::runtime_error("physical XCD worker/read coverage");
      if(candidate&&(st.elected!=128||st.published!=epoch||st.winner>=4))throw std::runtime_error("invalid finalization");
      std::vector<unsigned char>got(8544*8);HIP(hipMemcpy(got.data(),observed,got.size(),hipMemcpyDeviceToHost));
      if(got!=c.expected)throw std::runtime_error("G01 exact original logits/norm/residual/replica bytes differ: "+c.path);
      if(save)printf("PLACEMENT G01 epoch=%u arm=%d reverse=%u winner=%u reset_count=%u workers_per_xcd=38 readers=8\n",epoch,candidate,reverse,st.winner,reset_count);
      if(diagnostic&&save){
        std::vector<unsigned long long> stamps(trace_zero.size());HIP(hipMemcpy(stamps.data(),trace,trace_bytes,hipMemcpyDeviceToHost));
        std::ofstream t(output+"/landmarks_"+std::to_string(epoch)+".bin",std::ios::binary);t.write((char*)stamps.data(),trace_bytes);if(!t)throw std::runtime_error("landmark export failure");
      }
      if(save){std::ofstream o(output+"/observed_"+std::to_string(epoch)+".bin",std::ios::binary);o.write((char*)got.data(),got.size());if(!o)throw std::runtime_error("output export failure");}
      return {ms*1e6,wall};
    };
    // All48 layers, both retained positions, original and reverse completion.
    for(int i=0;i<ncases;++i){Case c(paths[i]);for(int reverse=0;reverse<2;++reverse)for(int arm=0;arm<2;++arm)trial(c,arm,reverse,true);printf("CHECK G01 case=%d exact=1\n",i);}
    Case c(paths[chosen]);for(int j=0;j<warmup;++j)for(int a=0;a<2;++a)trial(c,a,j%2,false);
    for(int r=0;r<5;++r){
      if(orders[r]!="AB"&&orders[r]!="BA")throw std::runtime_error("invalid pair order");
      for(char ch:orders[r]){int arm=ch=='B';double device=0,wall=0;
        for(int j=0;j<population;++j){auto v=trial(c,arm,0,false);device+=v.first;wall+=v.second;
          printf("SAMPLE G01 ordinal=%d arm=%d sample=%d device_ns=%.9f wall_ns=%.9f\n",r,arm,j,v.first,v.second);}
        printf("PAIR G01 ordinal=%d arm=%d device_ns=%.9f wall_ns=%.9f population=%d\n",r,arm,device/population,wall/population,population);
      }
    }
    puts(diagnostic?"G01 DIAGNOSTIC COMPLETE all96 exact; unqualified landmarks only":"G01 COMPLETE all96 exact; uninstrumented reset-through-eight-readers pairs; GPU outcome requires protocol decision");
    if(trace)HIP(hipFree(trace));HIP(hipFree(observed));HIP(hipFree(state));HIP(hipModuleUnload(m));return 0;
  }catch(const std::exception&e){fprintf(stderr,"REFUSED G01: %s\n",e.what());return 2;}
}
