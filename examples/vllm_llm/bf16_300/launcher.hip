// Minimal HIP module replay. Python validates the pack/reference/identity;
// this host relocates original blocks, restores all mutable inputs each run,
// alternates the unchanged original with the timestamp clone, and saves EVERY
// sample and EVERY task/wave record. No GPU use during source/CPU checks.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess) { fprintf(stderr,"HIP %s: %s\n",#x,hipGetErrorString(e)); exit(2); } } while(0)
struct Record { unsigned long long begin,end; unsigned task,wave,xcd,row; };
static_assert(sizeof(Record)==32, "raw record ABI");
using Clock=std::chrono::steady_clock;
static Clock::time_point deadline;
static void wait(hipStream_t s, const char* state) {
  auto progress=Clock::now();
  for (;;) {
    auto e=hipStreamQuery(s);
    if(e==hipSuccess) return;
    if(e!=hipErrorNotReady) { fprintf(stderr,"stream %s: %s\n",state,hipGetErrorString(e)); exit(2); }
    if(Clock::now()>=deadline) { fprintf(stderr,"NOT-COMPLETE deadline state=%s\n",state); fflush(stderr); _Exit(3); }
    if(Clock::now()-progress>std::chrono::seconds(10)) { printf("PROGRESS waiting %s\n",state); fflush(stdout); progress=Clock::now(); }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
static std::vector<unsigned char> read(const std::string& name,size_t offset,size_t size) {
  std::ifstream f(name,std::ios::binary); f.seekg(offset); std::vector<unsigned char> h(size);
  f.read((char*)h.data(),size); if(!f || (size_t)f.gcount()!=size) { fprintf(stderr,"short input %s\n",name.c_str()); exit(1); } return h;
}
int main(int argc,char** argv) {
  if(argc!=2) { fprintf(stderr,"usage: bf16-launch config.txt\n"); return 1; }
  setvbuf(stdout,nullptr,_IOLBF,0);
  std::ifstream f(argv[1]); std::string original,candidate,export_name,output,reference,args_path;
  int first,count,iterations,select; double seconds; size_t ref_offset,ref_size;
  f>>std::quoted(original)>>std::quoted(candidate)>>std::quoted(export_name)>>std::quoted(output)
   >>std::quoted(reference)>>std::quoted(args_path)>>first>>count>>iterations>>seconds>>select>>ref_offset>>ref_size;
  if(!f || count<=0 || first<0 || iterations<=0 || seconds<=0) return 1;
  deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
  hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop,0));
  int runtime,driver; CHECK(hipRuntimeGetVersion(&runtime)); CHECK(hipDriverGetVersion(&driver));
  printf("DEVICE name=%s arch=%s runtime=%d driver=%d\n",prop.name,prop.gcnArchName,runtime,driver);
  hipStream_t stream; CHECK(hipStreamCreate(&stream));
  auto arguments=read(args_path,0,160);
  std::vector<void*> buffers; std::vector<std::vector<unsigned char>> initial;
  for(int i=0;i<5;++i) {
    std::string path; size_t offset,size; f>>std::quoted(path)>>offset>>size;
    if(!f || !size) return 1;
    auto h=read(path,offset,size); void* d; CHECK(hipMalloc(&d,size));
    CHECK(hipMemcpy(d,h.data(),size,hipMemcpyHostToDevice)); buffers.push_back(d); initial.push_back(std::move(h));
    unsigned long long original_pointer; memcpy(&original_pointer,arguments.data()+i*8,8);
    if(original_pointer) memcpy(arguments.data()+i*8,&d,8); // retain original null pointers
    printf("INPUT slot=%d bytes=%zu\n",i,size);
  }
  if(ref_offset+ref_size>initial[2].size()) return 1;
  auto independent=read(reference,0,ref_size);
  void* a; CHECK(hipMalloc(&a,160)); CHECK(hipMemcpy(a,arguments.data(),160,hipMemcpyHostToDevice));
  hipModule_t mo,mc; hipFunction_t ko,kc;
  CHECK(hipModuleLoad(&mo,original.c_str())); CHECK(hipModuleLoad(&mc,candidate.c_str()));
  CHECK(hipModuleGetFunction(&ko,mo,("bf16_300_"+export_name).c_str()));
  CHECK(hipModuleGetFunction(&kc,mc,("bf16_300_probe_"+export_name).c_str()));
  int ov,cv; CHECK(hipFuncGetAttribute(&ov,HIP_FUNC_ATTRIBUTE_NUM_REGS,ko)); CHECK(hipFuncGetAttribute(&cv,HIP_FUNC_ATTRIBUTE_NUM_REGS,kc));
  printf("RESOURCE original_num_regs=%d candidate_num_regs=%d\n",ov,cv);
  const size_t nr=1+(size_t)count*16; Record* records; CHECK(hipMalloc(&records,nr*sizeof(Record)));
  std::ofstream raw(output+".bin",std::ios::binary), timing(output+".timing.jsonl");
  std::vector<Record> saved(nr); hipEvent_t begin,end; CHECK(hipEventCreate(&begin)); CHECK(hipEventCreate(&end));
  for(int it=0;it<iterations;++it) {
    double us[2]={};
    // Reverse pair order each iteration. Restore OUT as well as inputs, so
    // unwritten output holes cannot pass using the previous arm's values.
    for(int k=0;k<2;++k) {
      int arm=(it%2) ? 1-k : k;
      for(int b=1;b<5;++b) CHECK(hipMemcpyAsync(buffers[b],initial[b].data(),initial[b].size(),hipMemcpyHostToDevice,stream));
      CHECK(hipMemsetAsync(records,0,nr*sizeof(Record),stream));
      void* av[]={&a,&first,&records}; CHECK(hipEventRecord(begin,stream));
      CHECK(hipModuleLaunchKernel(arm?kc:ko,count,1,1,1024,1,1,0,stream,av,nullptr));
      CHECK(hipEventRecord(end,stream)); wait(stream,arm?"candidate":"original");
      float ms; CHECK(hipEventElapsedTime(&ms,begin,end)); us[arm]=ms*1000.;
      std::vector<unsigned char> got(ref_size); CHECK(hipMemcpy(got.data(),(char*)buffers[2]+ref_offset,ref_size,hipMemcpyDeviceToHost));
      if(got!=independent) { fprintf(stderr,"CHECK FAILED iteration=%d arm=%d\n",it,arm); return 4; }
      if(arm) { CHECK(hipMemcpy(saved.data(),records,nr*sizeof(Record),hipMemcpyDeviceToHost));
        // Return the complete companion diagnostic records for either
        // selector; the unstamped control has no fabricated timestamps.
        raw.write((char*)saved.data(),nr*sizeof(Record));
      }
    }
    timing<<"{\"iteration\":"<<it<<",\"original_us\":"<<us[0]<<",\"candidate_us\":"<<us[1]
          <<",\"body_perturbation_us\":"<<us[1]-us[0]<<"}\n";
    printf("PAIR iteration=%d original_us=%.3f candidate_us=%.3f CHECK=byte-identical\n",it,us[0],us[1]);
  }
  if(!raw || !timing) return 1;
  printf("COMPLETE iterations=%d records_per_iteration=%zu selector=%s\n",iterations,nr,select?"candidate":"original");
  return 0;
}
