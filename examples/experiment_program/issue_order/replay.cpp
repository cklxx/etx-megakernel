// One resident allocation population, unchanged import slot rounds, original
// capture endpoints. Host is real x86 code; --help never loads HIP or a GPU.
#include <hip/hip_runtime_api.h>
#include "examples/experiment_program/hip_dynamic.h"
#include "io_layout.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <cstdlib>
#define HIP(call) do { auto io_error=(call); if(io_error!=hipSuccess) {std::cerr<<#call<<": "<<hipGetErrorString(io_error)<<std::endl; std::exit(2);} } while(0)
static void need(bool b,const std::string& msg) {if(!b){std::cerr<<"BLOCKER "<<msg<<std::endl;std::exit(3);}}
static std::vector<unsigned char> read(const std::string& path,size_t bytes) {
  std::ifstream f(path,std::ios::binary); need(bool(f),"missing "+path);
  std::vector<unsigned char> v(bytes); f.read((char*)v.data(),bytes);need(size_t(f.gcount())==bytes&&f.peek()==EOF,"truncated/oversized "+path);return v;
}
static void wait(hipStream_t s) {
  auto start=std::chrono::steady_clock::now();
  for(;;){auto e=hipStreamQuery(s);if(e==hipSuccess)return;HIP(e==hipErrorNotReady?hipSuccess:e);
    if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>5){std::cerr<<"BLOCKER device deadline; state not reset"<<std::endl;std::_Exit(4);}}
}
struct Buffer {std::string name;uint64_t original;size_t bytes;void* device;std::vector<unsigned char> initial,expected;};
int main(int argc,char** argv) {
  std::cout<<std::setprecision(17);
  if(argc==2&&std::string(argv[1])=="--help") {std::cout<<"usage: replay MODULE FRAME [--warmup 8] [--samples 50] [--seed 617]\nRequires Python preflight/seal gate before GPU execution. Outputs native exact-byte checks and five untrimmed resident AB/BA pairs.\n";return 0;}
  need(argc>=3,"use --help");std::string module=argv[1],dir=argv[2];unsigned warmup=8,samples=50,seed=617;
  for(int i=3;i<argc;++i){std::string key=argv[i];need(i+1<argc,"missing option value");std::string value=argv[++i];need(!value.empty()&&value.find_first_not_of("0123456789")==std::string::npos,"invalid integer option");unsigned n=std::stoul(value);
    if(key=="--warmup")warmup=n;else if(key=="--samples")samples=n;else if(key=="--seed")seed=n;else need(false,"unknown option");}
  need(warmup>=1&&warmup<=100&&samples>=1&&samples<=1000,"sample/warmup budget");
  std::ifstream meta(dir+"/frame.txt");std::string magic;int layer,position,count;meta>>magic>>layer>>position>>count;
  need(magic=="C6CAP1"&&count==15&&layer>=0&&layer<48&&(position==0||position==1023||position==1024||position==1040),"capture identity");
  struct Raw {std::string name;uint64_t original;size_t bytes;};std::vector<Raw> raw;
  for(int i=0;i<count;++i){Raw r;meta>>r.name>>r.original>>r.bytes;need(bool(meta)&&r.bytes>0&&r.bytes<(1u<<27)&&r.original>0,"buffer bounds");need(r.name.find_first_not_of("abcdefghijklmnopqrstuvwxyz_")==std::string::npos,"buffer name");
    for(auto& old:raw)need(r.name!=old.name&&(r.original+r.bytes<=old.original||old.original+old.bytes<=r.original),"duplicate/aliased capture");raw.push_back(r);}
  size_t argbytes=0;for(int i=0;i<3;++i){std::string name;size_t n;meta>>name>>n;if(name==IO_EXPORT)argbytes=n;}
  std::string excess;need(!(meta>>excess)&&argbytes==IO_ARG_BYTES,"argument descriptor");
  auto arguments=read(dir+"/" IO_EXPORT ".args",argbytes);
  const std::string record=IO_RECORD;
  std::set<std::string> upstream,outputs;
  if(record=="G06")outputs={"qkv","kc","vc"};
  if(record=="G08"){upstream={"qkv","kc","vc"};outputs={"tmp_out","exp_sums","max_logits"};}
  if(record=="G07"){upstream={"qkv","kc","vc","tmp_out","exp_sums","max_logits"};outputs={"attn_out"};}
  load_hip();HIP(hipInit(0));int devices=0;HIP(hipGetDeviceCount(&devices));need(devices>=1,"no GPU");HIP(hipSetDevice(0));
  hipDeviceProp_t prop{};HIP(hipGetDeviceProperties(&prop,0));need(std::string(prop.gcnArchName).find("gfx942")==0&&prop.warpSize==64,"gfx942 wave64 required");
  std::vector<Buffer> buffers;
  for(auto& r:raw){Buffer b{r.name,r.original,r.bytes,nullptr,{},{}};
    auto before=read(dir+"/"+r.name+".before",r.bytes),after=read(dir+"/"+r.name+".after",r.bytes);
    b.initial=upstream.count(r.name)?after:before;b.expected=outputs.count(r.name)?after:b.initial;
    HIP(hipMalloc(&b.device,b.bytes));buffers.push_back(std::move(b));}
  for(int offset:io_ptr_offsets){uint64_t old;std::memcpy(&old,arguments.data()+offset,8);if(!old)continue;void* updated=nullptr;unsigned found=0;
    for(auto& b:buffers)if(old>=b.original&&old-b.original<b.bytes){updated=(char*)b.device+(old-b.original);++found;}need(found==1,"unresolved capture pointer");std::memcpy(arguments.data()+offset,&updated,8);}
  int grid[4];std::memcpy(grid,arguments.data()+IO_GRID_OFF,16);need(grid[0]>0&&grid[1]>0&&grid[2]>0&&grid[3]==grid[0]*grid[1]*grid[2],"grid identity");
  need((record=="G06"&&grid[3]==4)||(record=="G07"&&grid[3]==32)||(record=="G08"&&grid[3]==20),"registered grid");
  unsigned blocks=(grid[3]+IO_SLOTS-1)/IO_SLOTS;
  void* ab=nullptr;void* table=nullptr;HIP(hipMalloc(&ab,argbytes));HIP(hipMemcpy(ab,arguments.data(),argbytes,hipMemcpyHostToDevice));HIP(hipMalloc(&table,sizeof(void*)));HIP(hipMemcpy(table,&ab,sizeof(void*),hipMemcpyHostToDevice));
  hipModule_t mod;HIP(hipModuleLoad(&mod,module.c_str()));hipFunction_t kernels[2];HIP(hipModuleGetFunction(&kernels[0],mod,"io_control"));HIP(hipModuleGetFunction(&kernels[1],mod,"io_candidate"));
  hipStream_t stream;HIP(hipStreamCreate(&stream));hipEvent_t begin,end;HIP(hipEventCreate(&begin));HIP(hipEventCreate(&end));
  uint64_t epoch=0;
  auto launch=[&](unsigned arm,bool timed){
    ++epoch;need(epoch<100000,"epoch budget");
    // Reset only after prior completion; same resident pointers and cache inputs
    // in both arms. Reset host cost is recorded separately; native interval starts inputs-ready.
    auto resetstart=std::chrono::steady_clock::now();
    for(auto& b:buffers)HIP(hipMemcpyAsync(b.device,b.initial.data(),b.bytes,hipMemcpyHostToDevice,stream));
    wait(stream);auto wallstart=std::chrono::steady_clock::now();
    HIP(hipEventRecord(begin,stream));void* params[]={&table};HIP(hipModuleLaunchKernel(kernels[arm],blocks,1,1,1024,1,1,0,stream,params,nullptr));HIP(hipEventRecord(end,stream));wait(stream);
    auto wallend=std::chrono::steady_clock::now();
    for(auto& b:buffers){std::vector<unsigned char> actual(b.bytes);HIP(hipMemcpy(actual.data(),b.device,b.bytes,hipMemcpyDeviceToHost));need(actual==b.expected,"exact byte mismatch "+b.name+" arm="+std::to_string(arm));}
    float ms=0;HIP(hipEventElapsedTime(&ms,begin,end));need(std::isfinite(ms)&&ms>0,"invalid device time");
    double wall=std::chrono::duration<double,std::nano>(wallend-wallstart).count();
    if(timed)std::cout<<"IO_SAMPLE arm="<<(arm?"candidate":"control")<<" device_ns="<<double(ms)*1e6<<" host_ns="<<wall<<" reset_ns="<<std::chrono::duration<double,std::nano>(wallstart-resetstart).count()<<" epoch="<<epoch<<std::endl;
    return std::pair<double,double>{double(ms)*1e6,wall};};
  for(unsigned i=0;i<warmup;++i){std::cout<<"STAGE warmup="<<i<<std::endl;launch(i%2,false);launch(1-i%2,false);}
  std::mt19937 random(seed);std::vector<unsigned> ordinals={0,1,2,3,4};std::shuffle(ordinals.begin(),ordinals.end(),random);
  for(unsigned ordinal:ordinals){double device[2]={},wall[2]={};unsigned first=ordinal%2;std::cout<<"STAGE pair="<<ordinal<<" first="<<first<<std::endl;
    for(unsigned s=0;s<samples;++s)for(unsigned j=0;j<2;++j){unsigned arm=(first+j)%2;auto t=launch(arm,true);device[arm]+=t.first;wall[arm]+=t.second;}
    std::cout<<"IO_PAIR ordinal="<<ordinal<<" control_ns="<<device[0]/samples<<" candidate_ns="<<device[1]/samples<<" control_host_ns="<<wall[0]/samples<<" candidate_host_ns="<<wall[1]/samples<<" samples="<<samples<<" layer="<<layer<<" position="<<position<<std::endl;}
  std::cout<<"IO_CHECK OK record="<<record<<" layer="<<layer<<" position="<<position<<" endpoints="<<outputs.size()<<" epochs="<<epoch<<std::endl;
  HIP(hipEventDestroy(begin));HIP(hipEventDestroy(end));HIP(hipStreamDestroy(stream));HIP(hipModuleUnload(mod));for(auto& b:buffers)HIP(hipFree(b.device));HIP(hipFree(ab));HIP(hipFree(table));return 0;
}
