// Resident captured-input replay. No runtime or device work during compilation.
#include <hip/hip_runtime_api.h>
#include "examples/experiment_program/hip_dynamic.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>
#define CK(x) do { auto rc=(x); if(rc!=hipSuccess) { fprintf(stderr,"HIP failure %s rc=%d\n",#x,(int)rc); exit(2); }} while(0)
#include "row_export_contract.h"
using Clock=std::chrono::steady_clock;
static auto deadline=Clock::now();
static void wait(hipStream_t stream) {
  const auto started=Clock::now(); auto next=started+std::chrono::seconds(10);
  for (;;) { auto rc=hipStreamQuery(stream); if(rc==hipSuccess) return;
    if(rc!=hipErrorNotReady) { fprintf(stderr,"stream rc=%d\n",(int)rc); exit(2); }
    if(Clock::now()>deadline) { fprintf(stderr,"NOT-COMPLETE stream deadline\n"); _Exit(3); }
    if(Clock::now()>next) { puts("PROGRESS resident replay waiting"); next=Clock::now()+std::chrono::seconds(10); }
    if(Clock::now()-started>std::chrono::milliseconds(1)) std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}
static std::vector<char> read(const std::string& p,size_t offset,size_t bytes) {
  std::ifstream f(p,std::ios::binary); f.seekg(offset); std::vector<char> v(bytes); f.read(v.data(),bytes);
  if(!f || size_t(f.gcount())!=bytes) { fprintf(stderr,"short file %s\n",p.c_str()); exit(1); } return v;
}
struct Buffer { void *live,*pristine; size_t bytes; bool mutable_=false; };
struct Reference { int buffer; size_t offset; std::vector<char> bytes; void* poison=nullptr; };
int main(int argc,char**argv) {
  if(argc==2 && std::string(argv[1])=="--help") {puts("resident captured row replay; no device calls for help");return 0;}
  if(argc!=2) return 1; setvbuf(stdout,nullptr,_IOLBF,0);
  std::ifstream f(argv[1]); std::string path,output,args_path,export_name; int arms,first,count,warmup,samples,nb,np,nr; double seconds;
  f>>std::quoted(output)>>std::quoted(args_path)>>std::quoted(export_name)>>arms>>first>>count>>warmup>>samples>>seconds>>nb>>np>>nr;
  if(!f || arms<2 || arms>3 || first<0 || count<1 || warmup<0 || samples<1 || seconds<=0 || nb<1 || nr<1) return 1;
  deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
  load_hip(); CK(hipInit(0)); hipStream_t stream; CK(hipStreamCreate(&stream));
  std::vector<hipModule_t> modules(arms); std::vector<hipFunction_t> kernels(arms),checks(arms);
  for(int a=0;a<arms;++a) { f>>std::quoted(path); CK(hipModuleLoad(&modules[a],path.c_str())); CK(hipModuleGetFunction(&kernels[a],modules[a],export_name.c_str())); CK(hipModuleGetFunction(&checks[a],modules[a],(export_name+"_check").c_str())); }
  std::vector<Buffer> buffers;
  for(int b=0;b<nb;++b) { size_t off,n; f>>std::quoted(path)>>off>>n; auto h=read(path,off,n); Buffer x; x.bytes=n;
    CK(hipMalloc(&x.live,n)); CK(hipMalloc(&x.pristine,n)); CK(hipMemcpy(x.pristine,h.data(),n,hipMemcpyHostToDevice)); CK(hipMemcpy(x.live,x.pristine,n,hipMemcpyDeviceToDevice)); buffers.push_back(x); }
  auto arguments=read(args_path,0,160);
  for(int p=0;p<np;++p) { int offset,buffer; size_t add; f>>offset>>buffer>>add; if(offset<0 || offset+8>160 || buffer<0 || buffer>=nb || add>=buffers[buffer].bytes) return 1;
    auto pointer=(char*)buffers[buffer].live+add; memcpy(arguments.data()+offset,&pointer,8); }
  std::vector<Reference> refs;
  for(int r=0;r<nr;++r) { Reference x; size_t n; int output_only; f>>x.buffer>>x.offset>>std::quoted(path)>>n>>output_only; if(x.buffer<0 || x.buffer>=nb || x.offset+n>buffers[x.buffer].bytes) return 1; x.bytes=read(path,0,n); buffers[x.buffer].mutable_=true;
    if(output_only) { auto poison=x.bytes; for(auto&c:poison)c=(char)~(unsigned char)c; CK(hipMalloc(&x.poison,n)); CK(hipMemcpy(x.poison,poison.data(),n,hipMemcpyHostToDevice)); }
    printf("REFERENCE buffer=%d offset=%zu bytes=%zu poison=%s\n",x.buffer,x.offset,n,output_only?"bitwise-complement-of-reference":"LIVE_INPUT_ALIAS_PRESERVED");
    refs.push_back(x); }
  std::vector<int> orders(samples*arms);
  for(auto &v:orders) { f>>v; if(v<0 || v>=arms) return 1; } if(!f) return 1;
  void* args; CK(hipMalloc(&args,160)); CK(hipMemcpy(args,arguments.data(),160,hipMemcpyHostToDevice));
  const size_t record_bytes=size_t(count)*16*72; void* records; const size_t check_bytes=size_t(count)*16384; CK(hipMalloc(&records,check_bytes));
  std::ofstream timings(output+".timing.jsonl"); hipEvent_t begin,end; CK(hipEventCreate(&begin)); CK(hipEventCreate(&end));
  auto invoke=[&](int arm,int sample) {
    for(auto&b:buffers) if(b.mutable_) CK(hipMemcpyAsync(b.live,b.pristine,b.bytes,hipMemcpyDeviceToDevice,stream));
    for(auto&ref:refs) if(ref.poison) CK(hipMemcpyAsync((char*)buffers[ref.buffer].live+ref.offset,ref.poison,ref.bytes.size(),hipMemcpyDeviceToDevice,stream));
    CK(hipMemsetAsync(records,0,record_bytes,stream)); wait(stream);
    int selection=sample<0?0:sample%(count*16);
#ifdef ROW_PHASE_DIAGNOSTIC
    selection=sample<0?0:sample;
#endif
 void* av[]={&args,&first,&records,&selection}; auto wall=Clock::now(); CK(hipEventRecord(begin,stream));
    CK(hipModuleLaunchKernel(kernels[arm],count,1,1,1024,1,1,0,stream,av,nullptr)); CK(hipEventRecord(end,stream)); wait(stream);
    double wall_us=std::chrono::duration<double,std::micro>(Clock::now()-wall).count(); float ms; CK(hipEventElapsedTime(&ms,begin,end));
    for(size_t r=0;r<refs.size();++r) { auto&ref=refs[r]; std::vector<char> got(ref.bytes.size()); CK(hipMemcpy(got.data(),(char*)buffers[ref.buffer].live+ref.offset,got.size(),hipMemcpyDeviceToHost));
      if(got!=ref.bytes) { fprintf(stderr,"CHECK FAILED arm=%d sample=%d reference=%zu\n",arm,sample,r); exit(4); }
      if(sample>=0) { std::ofstream raw(output+".s"+std::to_string(sample)+".a"+std::to_string(arm)+".r"+std::to_string(r)+".bin",std::ios::binary); raw.write(got.data(),got.size()); }
    }
    if(sample>=0) { std::vector<char> h(record_bytes); CK(hipMemcpy(h.data(),records,record_bytes,hipMemcpyDeviceToHost)); std::ofstream raw(output+".s"+std::to_string(sample)+".a"+std::to_string(arm)+".records",std::ios::binary); raw.write(h.data(),h.size());
      timings<<"{\"sample\":"<<sample<<",\"arm\":"<<arm<<",\"device_us\":"<<ms*1000<<",\"host_wall_us\":"<<wall_us<<",\"exact\":true}\n"; timings.flush();
      printf("SAMPLE sample=%d arm=%d device_us=%.3f wall_us=%.3f CHECK=exact\n",sample,arm,ms*1000,wall_us); }
  };
  // Separate untimed intermediate comparisons, never present in decisive kernels.
  std::vector<char> original_intermediate;
  for(int arm=0;arm<arms;++arm) {
    for(auto&b:buffers) if(b.mutable_) CK(hipMemcpyAsync(b.live,b.pristine,b.bytes,hipMemcpyDeviceToDevice,stream));
    for(auto&ref:refs) if(ref.poison) CK(hipMemcpyAsync((char*)buffers[ref.buffer].live+ref.offset,ref.poison,ref.bytes.size(),hipMemcpyDeviceToDevice,stream));
    CK(hipMemsetAsync(records,0,check_bytes,stream));int selection=0;void* av[]={&args,&first,&records,&selection};
    CK(hipModuleLaunchKernel(checks[arm],count,1,1,1024,1,1,0,stream,av,nullptr));wait(stream);
    std::vector<char> raw(check_bytes);CK(hipMemcpy(raw.data(),records,check_bytes,hipMemcpyDeviceToHost));
    // Only the explicit own-arena capture region is a numerical comparison.
    const size_t values=row_intermediate_values(export_name,arguments);
    if(values*4>16384) return 1;
    std::vector<char> intermediate;for(int b=0;b<count;++b)intermediate.insert(intermediate.end(),raw.begin()+b*16384,raw.begin()+b*16384+values*4);
    if(arm==0)original_intermediate=intermediate;
    else if(original_intermediate!=intermediate){fprintf(stderr,"CHECK FAILED exact native intermediates arm=%d\n",arm);return 4;}
    std::ofstream saved(output+".a"+std::to_string(arm)+".intermediates",std::ios::binary);saved.write(intermediate.data(),intermediate.size());
    printf("INTERMEDIATE arm=%d values=%zu exact-original=YES\n",arm,intermediate.size()/4);
  }
  for(int i=0;i<warmup;++i) for(int a=0;a<arms;++a) invoke(a,-1);
  for(int s=0;s<samples;++s) for(int j=0;j<arms;++j) invoke(orders[s*arms+j],s);
  puts("COMPLETE all declared invocations and byte references");
}
