// Resident-input, single-process native replay. This program requires --execute
// via the Python fail-closed capture gate. It never loads/downloads a model.
#include <hip/hip_runtime_api.h>
#include "examples/experiment_program/hip_dynamic.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(ETX_G13_DRIVER) || defined(ETX_G03_DRIVER) || defined(ETX_COLUMN_DRIVER)
#include "plan_data.h"
#include "etx/abi.h"
#endif
#define HIP(call) do { auto etx_hip_status=(call); if(etx_hip_status!=hipSuccess) throw std::runtime_error(std::string(#call)+" failed "+std::to_string((int)etx_hip_status)); } while(0)
struct ColumnsArgs { const void *weights; const float* inputs; float* outputs; const float* scales; unsigned *completions; unsigned long long *stamps; int N,K,ld,workers,xcds,epoch,role,probe; float* output_columns[4]; void* output_bf16_columns[4]; int explicit_geometry,task_worker,task_xcd,coherent_output; };
#include "micro_state.h"
struct HistoricalArgs { const void *weights,*inputs; void *outputs; const int *ids; const float *mix; unsigned *completions; unsigned long long *stamps; int N,K,ld,workers,xcds,epoch,role,probe; long long expert_stride; int slots,rpt,tpe; };
struct Config {
 std::string module,payload,out,record;
 size_t weight_off,input_off,scale_off,ids_off,down_off,golden_off,golden_up_off;
 int N,K,ld,B,role,columns,slots,experts; long long stride;
 long long down_stride; int warmup,samples,seed,probe; std::vector<unsigned long long> arrivals;
 std::vector<int> ids; int g13_order=0,native_xcd=6,physical[8]={0,1,2,3,4,5,6,7};
};
static void launch(hipFunction_t fn,int grid,int threads,size_t bytes,hipStream_t stream,void* arg) {
 void* args[]={arg}; HIP(hipModuleLaunchKernel(fn,grid,1,1,threads,1,1,(unsigned)bytes,stream,args,nullptr));
}
struct Resident {
 Config c; hipModule_t mod,delay_mod; hipFunction_t original,shared[5],delay,gather,scatter,up,down;
 void* payload=nullptr; float *control=nullptr,*candidate=nullptr,*temp=nullptr,*gathered=nullptr,*scales=nullptr;
 int* slot_map=nullptr; unsigned* done=nullptr; unsigned long long* stamps=nullptr;
 std::vector<hipStream_t> streams; std::vector<hipEvent_t> ready,finish,up_done; hipEvent_t start,stop,reset_done;
 #include "g13_runtime.h"
 #include "micro_runtime.h"
 size_t element; int epoch=0; std::vector<float> host_scales; std::vector<char> host_golden,host_up_golden;
 Resident(Config cfg):c(cfg) {
  HIP(hipModuleLoad(&mod,c.module.c_str()));
  const bool hist=c.record=="G03"||c.record=="G13"; element=hist||c.role==2?2:4;
  if(hist) {
   HIP(hipModuleGetFunction(&original,mod,"historical_nb1"));
   HIP(hipModuleGetFunction(&shared[2],mod,"historical_nb2"));
   if(c.record=="G13") {HIP(hipModuleGetFunction(&up,mod,"historical_expert_up"));HIP(hipModuleGetFunction(&down,mod,"historical_expert_down"));}
   // Delay shares the Fleet module; its arithmetic does not enter the dot.
   auto directory=c.module.substr(0,c.module.find_last_of('/'));
   #if defined(ETX_G13_DRIVER) || defined(ETX_G03_DRIVER)
   HIP(hipModuleLoad(&delay_mod,(directory+"/../columns.hsaco").c_str()));
#else
   HIP(hipModuleLoad(&delay_mod,(directory+"/columns.hsaco").c_str()));
#endif
   HIP(hipModuleGetFunction(&delay,delay_mod,"readiness_delay"));
  } else {
   HIP(hipModuleGetFunction(&original,mod,"fleet_original"));
   for(int b=1;b<=4;++b) HIP(hipModuleGetFunction(&shared[b],mod,("fleet_columns_"+std::to_string(b)).c_str()));
   HIP(hipModuleGetFunction(&delay,mod,"readiness_delay"));
   HIP(hipModuleGetFunction(&gather,mod,"gather_columns"));HIP(hipModuleGetFunction(&scatter,mod,c.role==2?"scatter_columns16":"scatter_columns"));
  }
  std::ifstream f(c.payload,std::ios::binary|std::ios::ate); if(!f) throw std::runtime_error("missing native payload");
  size_t bytes=f.tellg();f.seekg(0);std::vector<char> host(bytes);f.read(host.data(),bytes);if(!f) throw std::runtime_error("truncated payload");
  host_golden.assign(host.begin()+c.golden_off,host.begin()+c.golden_off+(size_t)c.columns*c.N*element);
  if(c.record=="G13") host_up_golden.assign(host.begin()+c.golden_up_off,host.begin()+c.golden_up_off+(size_t)c.B*8*768*2);
  host_scales.resize(c.columns);std::memcpy(host_scales.data(),host.data()+c.scale_off,c.columns*4);
  HIP(hipMalloc(&payload,bytes));HIP(hipMemcpy(payload,host.data(),bytes,hipMemcpyHostToDevice));
  const size_t output=(size_t)c.columns*c.N*element;
  HIP(hipMalloc((void**)&control,output));HIP(hipMalloc((void**)&candidate,output));
  HIP(hipMalloc((void**)&temp,output));HIP(hipMalloc((void**)&gathered,(size_t)c.columns*c.K*4));
  HIP(hipMalloc((void**)&scales,c.columns*4));HIP(hipMalloc((void**)&slot_map,c.columns*4));
  HIP(hipMalloc((void**)&done,(size_t)c.columns*512*4));HIP(hipMalloc((void**)&stamps,(size_t)c.columns*512*8));
  HIP(hipEventCreate(&start));HIP(hipEventCreate(&stop));HIP(hipEventCreate(&reset_done));
  for(int b=0;b<c.B;++b) {hipStream_t s;hipEvent_t r,e;HIP(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));HIP(hipEventCreate(&r));HIP(hipEventCreate(&e));streams.push_back(s);ready.push_back(r);finish.push_back(e);hipEvent_t u;HIP(hipEventCreate(&u));up_done.push_back(u);}
#if defined(ETX_G13_DRIVER) || defined(ETX_G03_DRIVER)
  setup_g13();
#endif
#ifdef ETX_COLUMN_DRIVER
  setup_micro();
#endif
 }
 ColumnsArgs args(float* output,int col=0) {
  ColumnsArgs a{static_cast<char*>(payload)+c.weight_off,
          reinterpret_cast<float*>(static_cast<char*>(payload)+c.input_off)+(size_t)col*c.K,
          output+(size_t)col*c.N,
          reinterpret_cast<float*>(static_cast<char*>(payload)+c.scale_off)+col,
          done+(size_t)col*512,stamps+(size_t)col*512,c.N,c.K,c.ld,37,c.role==0?8:1,epoch,c.role,c.probe};
  if(c.role==2){a.outputs=nullptr;for(int b=0;b<4;++b)a.output_bf16_columns[b]=reinterpret_cast<unsigned short*>(output)+(size_t)(col+b)*c.N;}
  return a;
 }
 HistoricalArgs historical(void* output,int col=0) {
  auto p=static_cast<char*>(payload);
  return {p+c.weight_off,p+c.input_off+(size_t)col*c.K*2,
          static_cast<char*>(output)+(size_t)col*c.N*2,
          reinterpret_cast<int*>(p+c.ids_off)+col*c.slots,
          reinterpret_cast<float*>(p+c.scale_off)+col*c.slots,
          done+(size_t)col*512,stamps+(size_t)col*512,c.N,c.K,c.ld,38,8,epoch,c.role,c.probe,c.stride,c.slots,32,24};
 }
 void poison(void* output) {
  // NaN poison is outside the decisive bracket; admitted goldens are finite.
  HIP(hipMemset(output,255,(size_t)c.columns*c.N*element));
  if(c.record=="G13")HIP(hipMemset(temp,255,(size_t)c.columns*c.N*element));
 }
 void run(bool fused,void* output) {
#ifdef ETX_COLUMN_DRIVER
  run_micro(fused,output);return;
#endif
#if defined(ETX_G13_DRIVER) || defined(ETX_G03_DRIVER)
  run_g13(fused,output);return;
#else
  if(c.record=="G13")throw std::runtime_error("G13 requires the actual generated ETX common-queue driver");
#endif
  if(epoch==2147483647) throw std::runtime_error("epoch exhausted; quiescent reallocation required");
  ++epoch; HIP(hipEventRecord(start,nullptr));
  HIP(hipMemsetAsync(done,0,(size_t)c.columns*512*4,nullptr));
  HIP(hipEventRecord(reset_done,nullptr));
  for(int b=0;b<c.B;++b) {
   HIP(hipStreamWaitEvent(streams[b],reset_done,0));auto ticks=c.arrivals[b];
   launch(delay,1,64,0,streams[b],&ticks);HIP(hipEventRecord(ready[b],streams[b]));
  }
  if(c.record=="G13") {
   // Identical task bodies, 192 up + 256 down per request; intervention
   // removes only the cross-request up join. No device-side spin waits.
   for(int b=0;b<2;++b) {
    auto a=historical(temp,b); a.outputs=reinterpret_cast<unsigned short*>(temp)+(size_t)b*8*768; a.completions=done+(size_t)b*512+256; a.N=768; a.K=2048;a.rpt=32;a.tpe=24;
    launch(up,192,1024,0,streams[b],&a);HIP(hipEventRecord(up_done[b],streams[b]));
   }
   for(int b=0;b<2;++b) {
    if(!fused) for(int j=0;j<2;++j) HIP(hipStreamWaitEvent(streams[b],up_done[j],0));
    auto a=historical(output,b);a.outputs=static_cast<unsigned short*>(output)+(size_t)b*8*2048;a.weights=static_cast<char*>(payload)+c.down_off;
    a.inputs=reinterpret_cast<unsigned short*>(temp)+(size_t)b*8*768;
    a.N=2048;a.K=768;a.expert_stride=c.down_stride;a.rpt=64;a.tpe=32;
    launch(down,256,1024,0,streams[b],&a);HIP(hipEventRecord(finish[b],streams[b]));
   }
  } else if(c.record=="G05") {
   // Rebuild stable metadata inside each host-wall sample. No arithmetic
   // combines request columns and no reduction changes the original slot order.
   std::vector<std::pair<int,std::vector<int>>> groups;
   for(int col=0;col<c.columns;++col) {
    int e=c.ids[col];auto i=std::find_if(groups.begin(),groups.end(),[&](auto& g){return g.first==e;});
    if(i==groups.end()) groups.push_back({e,{col}}); else i->second.push_back(col);
   }
   if(!fused) {
    for(int col=0;col<c.columns;++col) {
     auto a=args(static_cast<float*>(output),col);
     a.weights=static_cast<char*>(payload)+c.weight_off+c.ids[col]*c.stride*2;
     launch(original,c.role==0?296:37,256,0,streams[col/c.slots],&a);
    }
    for(int b=0;b<c.B;++b) HIP(hipEventRecord(finish[b],streams[b]));
   } else {
   std::vector<std::pair<hipEvent_t,std::vector<int>>> group_done;
   for(auto& group:groups) {
    int owner=group.second.front()/c.slots;
    hipStream_t gs=streams[owner];
    for(int col:group.second) HIP(hipStreamWaitEvent(gs,ready[col/c.slots],0));
    const int width=(int)group.second.size();
    if(fused && width>1) {
     HIP(hipMemcpyAsync(slot_map,group.second.data(),width*4,hipMemcpyHostToDevice,gs));
     const float *src=reinterpret_cast<float*>(static_cast<char*>(payload)+c.input_off);float *dst=gathered;
     int *map=slot_map,K=c.K,B=width;void* ga[]={&src,&dst,&map,&K,&B};
     HIP(hipModuleLaunchKernel(gather,(K*B+255)/256,1,1,256,1,1,0,gs,ga,nullptr));
     std::vector<float> hscale(width);for(int j=0;j<width;++j) {
      hscale[j]=host_scales[group.second[j]];
     }
     HIP(hipMemcpyAsync(scales,hscale.data(),width*4,hipMemcpyHostToDevice,gs));
     auto a=args(temp);a.inputs=gathered;a.scales=scales;a.weights=static_cast<char*>(payload)+c.weight_off+group.first*c.stride*2;
     launch(shared[width],37,256,0,gs,&a);
     src=temp;dst=static_cast<float*>(output);int N=c.N;void* sa[]={&src,&dst,&map,&N,&B};
     HIP(hipModuleLaunchKernel(scatter,(N*B+255)/256,1,1,256,1,1,0,gs,sa,nullptr));
     HIP(hipStreamSynchronize(gs)); // host metadata lifetime; charge it
    } else for(int col:group.second) {
     auto a=args(static_cast<float*>(output),col);a.weights=static_cast<char*>(payload)+c.weight_off+group.first*c.stride*2;
     launch(original,c.role==0?296:37,256,0,gs,&a);
    }
    hipEvent_t ge;HIP(hipEventCreate(&ge));HIP(hipEventRecord(ge,gs));group_done.push_back({ge,group.second});
   }
   for(int b=0;b<c.B;++b) {
    for(auto& g:group_done) if(std::any_of(g.second.begin(),g.second.end(),[&](int col){return col/c.slots==b;})) HIP(hipStreamWaitEvent(streams[b],g.first,0));
    HIP(hipEventRecord(finish[b],streams[b]));
   }
   for(auto& g:group_done) {HIP(hipEventSynchronize(g.first));HIP(hipEventDestroy(g.first));}
   }
  } else {
   if(fused) {
    for(int b=0;b<c.B;++b) HIP(hipStreamWaitEvent(streams[0],ready[b],0));
    if(c.record=="G03") {auto a=historical(output);launch(shared[2],256,1024,0,streams[0],&a);}
    else {auto a=args(static_cast<float*>(output));launch(shared[c.B],c.role==0?296:37,256,0,streams[0],&a);}
    for(int b=0;b<c.B;++b) HIP(hipEventRecord(finish[b],streams[0]));
   } else for(int b=0;b<c.B;++b) {
    if(c.record=="G03") {auto a=historical(output,b);launch(original,256,1024,0,streams[b],&a);}
    else {auto a=args(static_cast<float*>(output),b);launch(original,c.role==0?296:37,256,0,streams[b],&a);}
    HIP(hipEventRecord(finish[b],streams[b]));
   }
  }
  for(auto e:finish) HIP(hipStreamWaitEvent(nullptr,e,0));HIP(hipEventRecord(stop,nullptr));HIP(hipEventSynchronize(stop));

 }
 void check_completion(bool fused) {
  if(c.record=="G13") {std::vector<char> up_bytes((size_t)c.B*8*768*2);HIP(hipMemcpy(up_bytes.data(),temp,up_bytes.size(),hipMemcpyDeviceToHost));if(up_bytes!=host_up_golden)throw std::runtime_error("per-arm native expert-up/sealed-golden mismatch");}
#ifdef ETX_COLUMN_DRIVER
  return; // run_micro validated each actual native producing-worker count.
#endif
  if(c.record!="G05") {
   const int populations=(fused && c.record!="G13" && c.record!="G03")?1:c.B;
   const int tasks=c.record=="G03"||c.record=="G13"?256:(c.role==0?296:37);
   for(int col=0;col<populations;++col) {
    std::vector<unsigned> observed(tasks);HIP(hipMemcpy(observed.data(),done+(size_t)col*512,tasks*4,hipMemcpyDeviceToHost));
    for(unsigned value:observed) if(value!=(unsigned)epoch) throw std::runtime_error("stale/missing native producer completion");
    if(c.record=="G13") {std::vector<unsigned> u(192);HIP(hipMemcpy(u.data(),done+(size_t)col*512+256,192*4,hipMemcpyDeviceToHost));for(unsigned value:u) if(value!=(unsigned)epoch) throw std::runtime_error("missing expert up completion");}
   }
  }
 }
 void compare() {
  const size_t bytes=(size_t)c.columns*c.N*element;
  std::vector<char> a(bytes),b(bytes);HIP(hipMemcpy(a.data(),control,bytes,hipMemcpyDeviceToHost));HIP(hipMemcpy(b.data(),candidate,bytes,hipMemcpyDeviceToHost));
  if(a!=b || a!=host_golden) throw std::runtime_error("native original/candidate/sealed-golden byte mismatch; no timing admitted");
  if(c.record=="G13") {std::vector<char> intermediate((size_t)c.B*8*768*2);HIP(hipMemcpy(intermediate.data(),temp,intermediate.size(),hipMemcpyDeviceToHost));if(intermediate!=host_up_golden)throw std::runtime_error("native expert-up/sealed-golden mismatch");std::ofstream(c.out+"/up.bin",std::ios::binary).write(intermediate.data(),intermediate.size());}
  std::ofstream(c.out+"/control.bin",std::ios::binary).write(a.data(),bytes);
  std::ofstream(c.out+"/candidate.bin",std::ios::binary).write(b.data(),bytes);
 }
 void sample(const char* arm,int pair,int ordinal) {
  poison(std::string(arm)=="candidate"?candidate:control);auto begin=std::chrono::steady_clock::now();run(std::string(arm)=="candidate",std::string(arm)=="candidate"?candidate:control);
  auto end=std::chrono::steady_clock::now();check_completion(std::string(arm)=="candidate");float ms=0;HIP(hipEventElapsedTime(&ms,start,stop));
  std::cout<<"{\"pair\":"<<pair<<",\"ordinal\":"<<ordinal<<",\"arm\":\""<<arm<<"\",\"device_ms\":"<<ms<<",\"host_ms\":"<<std::chrono::duration<double,std::milli>(end-begin).count()<<",\"request_ms\":[";
  for(int b=0;b<c.B;++b) {float ready_ms=0,out_ms=0;
#if defined(ETX_G13_DRIVER) || defined(ETX_G03_DRIVER)
   unsigned long long marks[5];HIP(hipMemcpy(marks,g_marks,sizeof(marks),hipMemcpyDeviceToHost));
   if(marks[0]==~0ull || !marks[1+b] || marks[3+b]<marks[1+b])throw std::runtime_error("G13 missing/stale request endpoint");
   ready_ms=(marks[1+b]-marks[0])*g_tick_ms;out_ms=(marks[3+b]-marks[0])*g_tick_ms;
#else
#ifdef ETX_COLUMN_DRIVER
   unsigned long long mm[9];HIP(hipMemcpy(mm,m_marks,sizeof(mm),hipMemcpyDeviceToHost));ready_ms=(mm[1+b]-mm[0])*m_tick_ms;out_ms=(mm[5+b]-mm[0])*m_tick_ms;
#else
HIP(hipEventElapsedTime(&ready_ms,start,ready[b]));HIP(hipEventElapsedTime(&out_ms,start,finish[b]));
#endif
#endif
if(b)std::cout<<",";std::cout<<"{\"request\":"<<b<<",\"ready\":"<<ready_ms<<",\"done\":"<<out_ms<<",\"latency\":"<<out_ms-ready_ms<<"}";}
  std::cout<<"]}"<<std::endl;
 }
};
int main(int argc,char** argv) {try {
 if(argc!=3 || std::string(argv[1])!="--execute") throw std::runtime_error("explicit --execute launch.txt required");
 Config c;std::ifstream f(argv[2]);f>>c.module>>c.payload>>c.out>>c.record>>c.N>>c.K>>c.ld>>c.B>>c.role>>c.columns>>c.slots>>c.experts>>c.stride>>c.weight_off>>c.input_off>>c.scale_off>>c.ids_off>>c.down_off>>c.golden_off>>c.golden_up_off>>c.down_stride>>c.warmup>>c.samples>>c.seed>>c.probe;
 for(int b=0;b<c.B;++b){unsigned long long t;f>>t;c.arrivals.push_back(t);}if(c.record=="G05")for(int j=0;j<c.columns;++j){int id;f>>id;c.ids.push_back(id);}f>>c.native_xcd;for(int x=0;x<8;++x)f>>c.physical[x];f>>c.g13_order;if(!f)throw std::runtime_error("invalid launch file");
 if(c.B<2||c.B>4||c.K>2048||c.N<1||c.warmup<1||c.samples!=5||c.probe)throw std::runtime_error("invalid uninstrumented decisive population");
 load_hip();HIP(hipInit(0)); Resident r(c);
#if defined(ETX_G13_DRIVER) || defined(ETX_G03_DRIVER)
 r.c.probe=1;r.gp.trace_exec=r.g_exec;r.gp.trace_time=r.g_trace;
 r.poison(r.control);r.run(false,r.control);r.check_completion(false);r.poison(r.candidate);r.run(true,r.candidate);r.check_completion(true);r.compare();
 r.c.probe=0;r.gp.trace_exec=nullptr;r.gp.trace_time=nullptr;
 std::cerr<<"HISTORICAL_ETX_PLACEMENT_OK original and candidate: all"<<ETX_N_TASKS<<" actual ETX tasks once; raw traces retained; decisive pairs uninstrumented\n";
#endif

#ifdef ETX_COLUMN_DRIVER
 r.c.probe=1;r.poison(r.control);r.run(false,r.control);r.poison(r.candidate);r.run(true,r.candidate);r.compare();r.c.probe=0;
 std::cerr<<"NATIVE_ETX_PLACEMENT_OK all actual ETX tasks once, native worker/XCC pins checked, raw trace/placement retained; decisive pairs uninstrumented\n";
#endif
 for(int i=0;i<c.warmup;++i){r.poison(r.control);r.run(false,r.control);r.check_completion(false);r.poison(r.candidate);r.run(true,r.candidate);r.check_completion(true);r.compare();std::cerr<<"warmup "<<i+1<<"/"<<c.warmup<<" byte-exact\n";}
 std::vector<int> order{0,1,2,3,4};std::mt19937 rng(c.seed);std::shuffle(order.begin(),order.end(),rng);
 for(int ordinal=0;ordinal<5;++ordinal){int pair=order[ordinal];if(ordinal%2==0){r.sample("control",pair,ordinal);r.sample("candidate",pair,ordinal);}else{r.sample("candidate",pair,ordinal);r.sample("control",pair,ordinal);}r.compare();}
 return 0;
 }catch(const std::exception& e){std::cerr<<"REFUSED: "<<e.what()<<"\n";return 2;}}
