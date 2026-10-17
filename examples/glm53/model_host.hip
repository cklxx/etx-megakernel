// Actual full GLM PP2 model launcher, pre-staged native FP8 tensors only.
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
extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_megakernel_d1(etx_params);

int main(int argc,char** argv){
  setvbuf(stdout,nullptr,_IOLBF,0);
  if(argc!=5){fprintf(stderr,"usage: run STAGE LOAD_TSV PROMPTS OUTPUT_PREFIX\n");return 2;}
  int count=0;ETX_CHECK(hipGetDeviceCount(&count));if(count!=2){fprintf(stderr,"exactly two GPUs required\n");return 2;}
  int batch=etx_shape[ETX_SYM_B];
  std::vector<int> owner(ETX_N_ARGS);std::vector<size_t> needed(2,0);std::vector<void*> buffers(ETX_N_ARGS,nullptr);
  for(int a=0;a<ETX_N_ARGS;++a){owner[a]=etx_arg_writer_device[a];if(owner[a]<0){
    // Weight grids have no writer. The generated load contract records owners.
    owner[a]=0;
  }}
  std::ifstream load(argv[2]);std::string name,file;int rank;size_t bytes;
  struct Entry{int arg,rank;std::string file;size_t bytes;};std::vector<Entry> entries;
  while(load>>name>>file>>rank>>bytes){int a=etx_host::arg_index(name.c_str());
    if(a<0 || rank<0 || rank>1 || bytes!=etx_arg_bytes[a] || file.find('/')!=std::string::npos || file.find("..")!=std::string::npos){fprintf(stderr,"invalid load contract\n");return 2;}
    for(auto& e:entries)if(e.arg==a){fprintf(stderr,"duplicate weight entry\n");return 2;}
    owner[a]=rank;entries.push_back({a,rank,file,bytes});
  }
  if(!load.eof()||entries.empty()){fprintf(stderr,"invalid/empty load contract\n");return 2;}
  for(int a=0;a<ETX_N_ARGS;++a){bool weight=strstr(etx_arg_names[a],"_weights")||strstr(etx_arg_names[a],"_meta");
    if(weight){bool found=false;for(auto& e:entries)if(e.arg==a)found=true;if(!found){fprintf(stderr,"missing weight %s\n",etx_arg_names[a]);return 2;}}
    needed[owner[a]]+=etx_arg_bytes[a];
  }
  std::ifstream prompt(argv[3]);int pb=0,context=0;prompt>>pb>>context;
  if(pb!=batch||context<1||context+32>GLM_MAXLEN){fprintf(stderr,"prompt shape differs from compiled model\n");return 2;}
  std::vector<int> inputs(batch*context);for(auto& x:inputs){if(!(prompt>>x)||x<0||x>=GLM_VOCAB){fprintf(stderr,"invalid prompt token\n");return 2;}}
  int extra;if(prompt>>extra){fprintf(stderr,"trailing prompt input\n");return 2;}
  for(int d=0;d<2;++d){ETX_CHECK(hipSetDevice(d));hipDeviceProp_t p;ETX_CHECK(hipGetDeviceProperties(&p,d));
    if(strncmp(p.gcnArchName,"gfx942",6)){fprintf(stderr,"binary requires gfx942\n");return 2;}
    size_t free=0,total=0;ETX_CHECK(hipMemGetInfo(&free,&total));
    printf("GLM-DEVICE rank=%d name=%s arch=%s free=%zu payload_and_state=%zu\n",d,p.name,p.gcnArchName,free,needed[d]);
    if(needed[d]>free){fprintf(stderr,"model allocation does not fit rank %d\n",d);return 2;}
    int access=0,atomics=0;ETX_CHECK(hipDeviceCanAccessPeer(&access,d,1-d));ETX_CHECK(hipDeviceGetP2PAttribute(&atomics,hipDevP2PAttrNativeAtomicSupported,d,1-d));
    if(!access||!atomics){fprintf(stderr,"peer access + integer atomics required\n");return 2;}
    auto e=hipDeviceEnablePeerAccess(1-d,0);if(e==hipErrorPeerAccessAlreadyEnabled)(void)hipGetLastError();else ETX_CHECK(e);
  }
  for(int a=0;a<ETX_N_ARGS;++a){ETX_CHECK(hipSetDevice(owner[a]));
    bool shared=!strcmp(etx_arg_names[a],"step") || glm_arg_shared[a];
    if(shared)ETX_CHECK(hipExtMallocWithFlags(&buffers[a],etx_arg_bytes[a],hipDeviceMallocFinegrained));
    else ETX_CHECK(hipMalloc(&buffers[a],etx_arg_bytes[a]));
    if(!strstr(etx_arg_names[a],"_weights")&&!strstr(etx_arg_names[a],"_meta"))ETX_CHECK(hipMemset(buffers[a],0,etx_arg_bytes[a]));
  }
  std::vector<char> chunk(4*1024*1024);
  for(auto& e:entries){ETX_CHECK(hipSetDevice(e.rank));std::ifstream f(std::string(argv[1])+"/"+e.file,std::ios::binary);size_t offset=0;
    while(offset<e.bytes){size_t n=std::min(chunk.size(),e.bytes-offset);if(!f.read(chunk.data(),n)){fprintf(stderr,"truncated tensor %s\n",e.file.c_str());return 2;}
      ETX_CHECK(hipMemcpy((char*)buffers[e.arg]+offset,chunk.data(),n,hipMemcpyHostToDevice));offset+=n;}
    char x;if(f.get(x)){fprintf(stderr,"oversized tensor\n");return 2;}
    printf("GLM-LOAD rank=%d file=%s bytes=%zu\n",e.rank,e.file.c_str(),e.bytes);
  }
  int32_t* events=etx_host::alloc_events(0);std::vector<etx_host> hosts;hosts.reserve(2);
  for(int d=0;d<2;++d){hosts.emplace_back(d);hosts.back().kernel=d==0?(const void*)etx_megakernel_d0:(const void*)etx_megakernel_d1;
    hosts.back().args=buffers;hosts.back().setup(events);hosts.back().owns_events=d==0;if(!hosts.back().check_residency())return 2;
    size_t free=0,total=0;ETX_CHECK(hipSetDevice(d));ETX_CHECK(hipMemGetInfo(&free,&total));printf("GLM-ALLOCATED rank=%d used=%zu free=%zu\n",d,total-free,free);
  }
  std::ofstream logits(std::string(argv[4])+".logits",std::ios::binary);
  std::vector<float> head(batch*(2*GLM_H+GLM_VOCAB));std::vector<int> returned(batch),step(2*batch),generated(32*batch);
  for(int pos=0;pos<context+31;++pos){
    for(int r=0;r<batch;++r){step[2*r]=pos;step[2*r+1]=pos<context?inputs[r*context+pos]:returned[r];}
    ETX_CHECK(hipSetDevice(0));ETX_CHECK(hipMemcpy(buffers[etx_host::arg_index("step")],step.data(),step.size()*4,hipMemcpyHostToDevice));
    for(auto& h:hosts)h.reset_step();for(auto& h:hosts){ETX_CHECK(hipSetDevice(h.device));ETX_CHECK(hipStreamSynchronize(h.stream));}
    auto start=std::chrono::steady_clock::now();for(auto& h:hosts)h.launch();for(auto& h:hosts)if(!h.wait(120.0,start))return 3;
    ETX_CHECK(hipSetDevice(0));ETX_CHECK(hipMemcpy(returned.data(),buffers[etx_host::arg_index("returned")],batch*4,hipMemcpyDeviceToHost));
    if(pos>=context-1){int gen=pos-context+1;for(int r=0;r<batch;++r)generated[r*32+gen]=returned[r];
      ETX_CHECK(hipSetDevice(1));ETX_CHECK(hipMemcpy(head.data(),buffers[etx_host::arg_index("head_scratch")],head.size()*4,hipMemcpyDeviceToHost));
      for(int r=0;r<batch;++r)logits.write((const char*)(head.data()+r*(2*GLM_H+GLM_VOCAB)+2*GLM_H),GLM_VOCAB*4);
    }
    printf("GLM-MODEL-STEP pos=%d batch=%d\n",pos,batch);
  }
  logits.close();std::ofstream tokens(std::string(argv[4])+".tokens");
  for(int r=0;r<batch;++r){for(int k=0;k<32;++k)tokens<<generated[r*32+k]<<" ";tokens<<"\n";}
  tokens.close();if(!logits||!tokens){fprintf(stderr,"output write failed\n");return 4;}
  printf("GLM-MODEL-EXECUTED batch=%d layers=%d generated_each=32; reference comparison required\n",batch,ETX_N_TYPES-3);
  return 0;
}
