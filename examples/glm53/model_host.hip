// Actual multi-GPU GLM launcher (PP / FFN TP / EP), pre-staged native FP8 only.
#include "plan_data.h"
#include "glm53_contract.h"
#include "etx/host.h"
#include "examples/glm53/native.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "glm53_kernels.h"   // etx_megakernel_d<k> for every participating rank
constexpr int NDEV=ETX_N_DEVICES;

// Layers of the compiled model: the l<i>_out activations (grid types differ between the phased and per-layer graphs).
static int model_layers(){int n=0;for(;;++n){std::string nm="l"+std::to_string(n)+"_out";bool found=false;
  for(int a=0;a<ETX_N_ARGS;++a)if(nm==etx_arg_names[a])found=true;if(!found)return n;}}

int main(int argc,char** argv){
  setvbuf(stdout,nullptr,_IOLBF,0);
  if(argc!=5){fprintf(stderr,"usage: run STAGE LOAD_TSV PROMPTS OUTPUT_PREFIX\n");return 2;}
  int count=0;ETX_CHECK(hipGetDeviceCount(&count));if(count!=NDEV){fprintf(stderr,"exactly %d GPUs required (HIP_VISIBLE_DEVICES), %d visible\n",NDEV,count);return 2;}
  int batch=etx_shape[ETX_SYM_B];
  printf("GLM-VARIANT devices=%d parallel=%d attention_tp=%d up_tail_balance=%d up_pairs=%d\n",NDEV,ETX_GLM_PARALLEL,ETX_GLM_ATTENTION_TP,ETX_GLM_UP_TAIL_BALANCE,ETX_GLM_UP_PAIRS);
  std::vector<int> owner(ETX_N_ARGS);std::vector<size_t> needed(NDEV,0);std::vector<void*> buffers(ETX_N_ARGS,nullptr);
  for(int a=0;a<ETX_N_ARGS;++a)owner[a]=glm_arg_owner[a]>=0?glm_arg_owner[a]:std::max(0,etx_arg_writer_device[a]);
  const std::string load_path=argv[2];const auto slash=load_path.find_last_of('/');
  const std::string build_dir=slash==std::string::npos?".":load_path.substr(0,slash);
  auto safe_file=[](const std::string& f){return !f.empty()&&f!="."&&f!=".."&&f.find('/')==std::string::npos&&f.find('\\')==std::string::npos;};
  auto valid_weight=[&](int a,int rank){return a>=0&&rank>=0&&rank<NDEV&&glm_arg_owner[a]==rank;};
  std::ifstream load(argv[2]);std::string line;
  struct Entry{int arg,rank;std::string file;size_t bytes;bool build;};std::vector<Entry> entries;
  std::vector<uint64_t> loaded(ETX_N_ARGS,0);std::vector<std::vector<std::pair<uint64_t,uint64_t>>> spans(ETX_N_ARGS);
  while(std::getline(load,line)){
    std::istringstream row(line);std::string name,file,origin,extra;int rank;size_t bytes;
    if(!(row>>name>>file>>rank>>bytes)){fprintf(stderr,"invalid load row\n");return 2;}
    if(row>>origin){if(origin!="build"||(row>>extra)){fprintf(stderr,"invalid load source\n");return 2;}}
    int a=etx_host::arg_index(name.c_str());
    if(!valid_weight(a,rank)||bytes!=etx_arg_bytes[a]||!safe_file(file)||loaded[a]){fprintf(stderr,"invalid/duplicate load contract\n");return 2;}
    entries.push_back({a,rank,file,bytes,origin=="build"});loaded[a]=bytes;spans[a].push_back({0,bytes});
  }
  if(!load.eof()||entries.empty()){fprintf(stderr,"invalid/empty load contract\n");return 2;}
  struct Range{int arg,rank;std::string file;uint64_t src,dst,rows,ss,width,ds;};std::vector<Range> ranges;
  if(ETX_GLM_PARALLEL){
    std::ifstream input(build_dir+"/ranges.tsv");
    while(std::getline(input,line)){
      std::istringstream row(line);std::string name,file,extra;int rank;uint64_t src,dst,rows,ss,width,ds;
      if(!(row>>name>>rank>>file>>src>>dst>>rows>>ss>>width>>ds)||(row>>extra)){fprintf(stderr,"invalid shard range row\n");return 2;}
      int a=etx_host::arg_index(name.c_str());
      // Destinations are compact; only source columns can be strided. Division checks avoid overflow.
      if(!valid_weight(a,rank)||!safe_file(file)||!rows||!width||ss<width||ds!=width||
         dst>etx_arg_bytes[a]||rows>(etx_arg_bytes[a]-dst)/width||
         src>UINT64_MAX-width||(rows-1)>(UINT64_MAX-src-width)/ss){fprintf(stderr,"invalid shard range bounds\n");return 2;}
      ranges.push_back({a,rank,file,src,dst,rows,ss,width,ds});
      spans[a].push_back({dst,dst+rows*width});loaded[a]+=rows*width;
    }
    if(!input.eof()){fprintf(stderr,"missing/invalid ranges.tsv\n");return 2;}
  }
  for(int a=0;a<ETX_N_ARGS;++a){
    if(glm_arg_owner[a]>=0){
      if(loaded[a]!=glm_arg_load_bytes[a]){fprintf(stderr,"incomplete weight %s: %llu/%llu bytes\n",etx_arg_names[a],(unsigned long long)loaded[a],(unsigned long long)glm_arg_load_bytes[a]);return 2;}
      auto& s=spans[a];std::sort(s.begin(),s.end());for(size_t i=1;i<s.size();++i)if(s[i-1].second>s[i].first){fprintf(stderr,"overlapping shard writes\n");return 2;}
    }
    needed[owner[a]]+=etx_arg_bytes[a];
  }
  std::ifstream prompt(argv[3]);int pb=0,context=0;prompt>>pb>>context;
  if(pb!=batch||context<1||context+32>GLM_MAXLEN){fprintf(stderr,"prompt shape differs from compiled model\n");return 2;}
  std::vector<int> inputs(batch*context);for(auto& x:inputs){if(!(prompt>>x)||x<0||x>=GLM_VOCAB){fprintf(stderr,"invalid prompt token\n");return 2;}}
  int extra;if(prompt>>extra){fprintf(stderr,"trailing prompt input\n");return 2;}
  for(int d=0;d<NDEV;++d){ETX_CHECK(hipSetDevice(d));hipDeviceProp_t p;ETX_CHECK(hipGetDeviceProperties(&p,d));
    if(strncmp(p.gcnArchName,"gfx942",6)){fprintf(stderr,"binary requires gfx942\n");return 2;}
    size_t free=0,total=0;ETX_CHECK(hipMemGetInfo(&free,&total));
    printf("GLM-DEVICE rank=%d name=%s arch=%s free=%zu payload_and_state=%zu\n",d,p.name,p.gcnArchName,free,needed[d]);
    if(needed[d]>free){fprintf(stderr,"model allocation does not fit rank %d\n",d);return 2;}
    // Every stage reads step/events on device 0 and its neighbours' activations: peer access to all.
    for(int o=0;o<NDEV;++o){if(o==d)continue;
      int access=0,atomics=0;ETX_CHECK(hipDeviceCanAccessPeer(&access,d,o));ETX_CHECK(hipDeviceGetP2PAttribute(&atomics,hipDevP2PAttrNativeAtomicSupported,d,o));
      if(!access||!atomics){fprintf(stderr,"peer access + integer atomics required\n");return 2;}
      auto e=hipDeviceEnablePeerAccess(o,0);if(e==hipErrorPeerAccessAlreadyEnabled)(void)hipGetLastError();else ETX_CHECK(e);}
  }
  for(int a=0;a<ETX_N_ARGS;++a){ETX_CHECK(hipSetDevice(owner[a]));
    bool shared=!strcmp(etx_arg_names[a],"step") || glm_arg_shared[a];
    if(shared)ETX_CHECK(hipExtMallocWithFlags(&buffers[a],etx_arg_bytes[a],hipDeviceMallocFinegrained));
    else ETX_CHECK(hipMalloc(&buffers[a],etx_arg_bytes[a]));
    if(ETX_GLM_PARALLEL||(!strstr(etx_arg_names[a],"_weights")&&!strstr(etx_arg_names[a],"_meta")))ETX_CHECK(hipMemset(buffers[a],0,etx_arg_bytes[a]));
  }
  std::vector<char> chunk(4*1024*1024);
  for(auto& e:entries){ETX_CHECK(hipSetDevice(e.rank));std::ifstream f((e.build?build_dir:std::string(argv[1]))+"/"+e.file,std::ios::binary);size_t offset=0;
    while(offset<e.bytes){size_t n=std::min(chunk.size(),e.bytes-offset);if(!f.read(chunk.data(),n)){fprintf(stderr,"truncated tensor %s\n",e.file.c_str());return 2;}
      ETX_CHECK(hipMemcpy((char*)buffers[e.arg]+offset,chunk.data(),n,hipMemcpyHostToDevice));offset+=n;}
    char x;if(f.get(x)){fprintf(stderr,"oversized tensor\n");return 2;}
    printf("GLM-LOAD rank=%d file=%s bytes=%zu\n",e.rank,e.file.c_str(),e.bytes);
  }
  // Map one immutable source blob at a time; pack strided columns into bounded
  // host chunks before H2D. One HIP copy per row would create millions of launches.
  struct Mapping{int fd=-1;const char* p=nullptr;uint64_t bytes=0;std::string file;
    void close(){if(p)munmap((void*)p,bytes);if(fd>=0)::close(fd);p=nullptr;fd=-1;}
    ~Mapping(){close();}
    bool open(const std::string& path){close();fd=::open(path.c_str(),O_RDONLY);struct stat s;
      if(fd<0||fstat(fd,&s)||s.st_size<=0)return false;bytes=s.st_size;
      void* v=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);if(v==MAP_FAILED)return false;p=(const char*)v;return true;}
  } mapped;
  size_t range_index=0;
  for(const auto& r:ranges){
    if(mapped.file!=r.file){if(!mapped.open(std::string(argv[1])+"/"+r.file)){fprintf(stderr,"cannot map shard source %s\n",r.file.c_str());return 2;}
      mapped.file=r.file;printf("GLM-SHARD-SOURCE file=%s bytes=%llu\n",r.file.c_str(),(unsigned long long)mapped.bytes);}
    if(r.src+(r.rows-1)*r.ss+r.width>mapped.bytes){fprintf(stderr,"truncated shard source\n");return 2;}
    ETX_CHECK(hipSetDevice(r.rank));uint64_t total=r.rows*r.width;
    for(uint64_t off=0;off<total;){size_t n=std::min<uint64_t>(chunk.size(),total-off),packed=0;
      while(packed<n){uint64_t q=off+packed,row=q/r.width,col=q%r.width;
        size_t part=std::min<uint64_t>(n-packed,r.width-col);memcpy(chunk.data()+packed,mapped.p+r.src+row*r.ss+col,part);packed+=part;}
      ETX_CHECK(hipMemcpy((char*)buffers[r.arg]+r.dst+off,chunk.data(),n,hipMemcpyHostToDevice));off+=n;
    }
    if((++range_index%256)==0)printf("GLM-SHARD-PROGRESS ranges=%zu/%zu rank=%d\n",range_index,ranges.size(),r.rank);
  }
  mapped.close();
  printf("GLM-LOAD-CONTRACT OK parallel=%d ranges=%zu\n",ETX_GLM_PARALLEL,ranges.size());
  // Each device polls its own event buffer; only SYSTEM-scope (cross-device) words share one buffer on device 0.
  // GLM_SHARED_EVENTS=1 restores the single shared buffer (remote polling made remote stages ~10x slower, measured).
  const bool shared_events=getenv("GLM_SHARED_EVENTS")!=nullptr;
  int32_t* events=etx_host::alloc_events(0);std::vector<etx_host> hosts;hosts.reserve(NDEV);
  for(int d=0;d<NDEV;++d){hosts.emplace_back(d);hosts.back().kernel=glm_kernels[d];hosts.back().args=buffers;
    if(shared_events){hosts.back().setup(events);hosts.back().owns_events=d==0;}
    else{hosts.back().sys_events=events;hosts.back().owns_sys=d==0;hosts.back().setup();}if(!hosts.back().check_residency())return 2;
    size_t free=0,total=0;ETX_CHECK(hipSetDevice(d));ETX_CHECK(hipMemGetInfo(&free,&total));printf("GLM-ALLOCATED rank=%d used=%zu free=%zu\n",d,total-free,free);
  }
  std::ofstream logits(std::string(argv[4])+".logits",std::ios::binary);
  std::vector<float> head(batch*(2*GLM_H+GLM_VOCAB));std::vector<int> returned(batch),step(2*batch),generated(32*batch);
  for(int pos=0;pos<context+31;++pos){
    for(int r=0;r<batch;++r){step[2*r]=pos;step[2*r+1]=pos<context?inputs[r*context+pos]:returned[r];}
    ETX_CHECK(hipSetDevice(0));ETX_CHECK(hipMemcpy(buffers[etx_host::arg_index("step")],step.data(),step.size()*4,hipMemcpyHostToDevice));
    for(auto& h:hosts)h.reset_step();for(auto& h:hosts){ETX_CHECK(hipSetDevice(h.device));ETX_CHECK(hipStreamSynchronize(h.stream));}
    auto start=std::chrono::steady_clock::now();for(auto& h:hosts)h.launch();
    // All streams retire first; the timestamp is taken before any completion check, because
    // after_wait() does synchronous 8-byte (and, under ETX_EXEC_COUNT, trace) D2H copies.
    for(auto& h:hosts)if(!h.wait_stream(120.0,start))return 3;
    double step_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    for(auto& h:hosts){ETX_CHECK(hipSetDevice(h.device));if(!h.after_wait())return 3;}
    // Opt-in device trace of one step (ETX_TRACE=1 GLM_TRACE_POS=<pos> GLM_TRACE_OUT=<prefix>): per-type table and raw tasks per rank.
    if(getenv("ETX_TRACE")&&getenv("GLM_TRACE_POS")&&atoi(getenv("GLM_TRACE_POS"))==pos)for(auto& h:hosts){
      std::string raw=std::string(getenv("GLM_TRACE_OUT")?getenv("GLM_TRACE_OUT"):"glm_trace")+".d"+std::to_string(h.device)+"."+std::to_string(pos)+".bin";
      setenv("ETX_TRACE_RAW",raw.c_str(),1);printf("GLM-TRACE rank=%d pos=%d raw=%s\n",h.device,pos,raw.c_str());h.report_phases(etx_type_names,ETX_N_TYPES);}
    ETX_CHECK(hipSetDevice(0));ETX_CHECK(hipMemcpy(returned.data(),buffers[etx_host::arg_index("returned")],batch*4,hipMemcpyDeviceToHost));
    if(pos>=context-1){int gen=pos-context+1;for(int r=0;r<batch;++r)generated[r*32+gen]=returned[r];
      ETX_CHECK(hipSetDevice(owner[etx_host::arg_index("head_scratch")]));ETX_CHECK(hipMemcpy(head.data(),buffers[etx_host::arg_index("head_scratch")],head.size()*4,hipMemcpyDeviceToHost));
      for(int r=0;r<batch;++r)logits.write((const char*)(head.data()+r*(2*GLM_H+GLM_VOCAB)+2*GLM_H),GLM_VOCAB*4);
    }
    printf("GLM-MODEL-STEP pos=%d batch=%d ms=%.6f\n",pos,batch,step_ms);
    if(getenv("ETX_TRACE") && pos==1){for(size_t h=0;h<hosts.size();++h){
      ETX_CHECK(hipSetDevice(hosts[h].device));
      fprintf(stderr,"GLM-PHASES rank=%zu\n",h);
      hosts[h].report_phases(nullptr,ETX_N_TYPES,10.0,stderr);}}
    // Opt-in debug: every layer output after the step, outside the timed span (GLMDBG_DUMP_LAYERS=<file>).
    if(const char* dump=getenv("GLMDBG_DUMP_LAYERS")){static std::ofstream df(dump,std::ios::binary);
      for(int i=0;i<model_layers();++i){std::string nm="l"+std::to_string(i)+"_out";int a=etx_host::arg_index(nm.c_str());
        ETX_CHECK(hipSetDevice(owner[a]));std::vector<char> tmp(etx_arg_bytes[a]);ETX_CHECK(hipMemcpy(tmp.data(),buffers[a],tmp.size(),hipMemcpyDeviceToHost));df.write(tmp.data(),tmp.size());}
      df.flush();}
  }
  logits.close();std::ofstream tokens(std::string(argv[4])+".tokens");
  for(int r=0;r<batch;++r){for(int k=0;k<32;++k)tokens<<generated[r*32+k]<<" ";tokens<<"\n";}
  tokens.close();if(!logits||!tokens){fprintf(stderr,"output write failed\n");return 4;}
  printf("GLM-MODEL-EXECUTED batch=%d layers=%d generated_each=32; reference comparison required\n",batch,model_layers());
  return 0;
}
