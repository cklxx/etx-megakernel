// Generated-host correctness path. This is a serial full-model replay with
// C6 replacing each attention segment. Its wall time is NOT a C6 result.
#pragma once
#include "examples/vllm_llm/attention_early_merge/probe.h"
extern "C" __global__ void c6_probe(C6Args);
struct C6Hybrid {
  C6Args a{};
  unsigned long long epoch=0;
  bool enabled=false;
  template<class T> static T* alloc(size_t n) { T* p; ETX_CHECK(hipMalloc(&p,n*sizeof(T))); return p; }
  void setup(etx_host& hst) {
    const char* flag=getenv("C6_HYBRID");
    if (!flag) return;
    if (std::string(flag)!="1") die("C6_HYBRID must be 1 or unset");
    enabled=true;
    hipDeviceProp_t prop; ETX_CHECK(hipGetDeviceProperties(&prop,0));
    if (std::string(prop.gcnArchName).find("gfx942")!=0 || prop.multiProcessorCount!=304 || !prop.cooperativeLaunch)
      die("C6 hybrid needs one full MI300X");
    int resident=0; ETX_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,c6_probe,1024,0));
    if (resident<1) die("C6 hybrid residency");
    void* args=hst.args[etx_host::arg_index("impargs")];
    auto table=alloc<void*>(1); ETX_CHECK(hipMemcpy(table,&args,8,hipMemcpyHostToDevice));
    a.prep.args=table; a.attention.args=table; a.reduce.args=table;
    a.state=alloc<C6State>(4); a.tickets=alloc<unsigned>(8); a.abort=alloc<int>(1); a.outputs=alloc<unsigned>(1); a.observed=alloc<unsigned short>(32*128);
    a.mode=1; a.spin_limit=1000000;
    printf("C6-HYBRID enabled=1 mode=combined purpose=correctness-only\n");
  }
  bool step(etx_host& hst,const std::vector<Inst>& insts) {
    for (int li=etx_launch_begin[0];li<etx_launch_begin[1];++li) {
      const Inst* prep=nullptr;
      if (li>0 && (size_t)(li-1)<insts.size() && insts[li-1].role=="prep") prep=&insts[li-1];
      if (!prep) {
        int32_t first=etx_launch_first[li]; void* args[]={&hst.p,&first};
        ETX_CHECK(hipLaunchKernel(hst.unfused_kernel,dim3(etx_launch_count[li]),dim3(ETX_THREADS),args,0,hst.stream));
        continue;
      }
      const auto& at=insts.at(prep->n+1); const auto& rd=insts.at(prep->n+2);
      if (prep->exp!="dsl_attnprep" || at.exp!="attn_mfma16_g8" || rd.exp!="attn_reduce" ||
          prep->tasks!=4 || at.tasks!=20 || rd.tasks!=4) die("C6 hybrid plan mapping");
      if (!wait_stream(hst.stream,5)) return false;
      if (++epoch==~0ull) die("C6 epoch rollover requires quiescence and clear");
      a.epoch=epoch; a.prep.cst[0]=prep->off; a.attention.cst[0]=at.off; a.reduce.cst[0]=rd.off;
      C6State state[4]{}; for (auto& s:state) { s.epoch=epoch; s.elected=0xffffffff; }
      ETX_CHECK(hipMemcpyAsync(a.state,state,sizeof(state),hipMemcpyHostToDevice,hst.stream));
      ETX_CHECK(hipMemsetAsync(a.tickets,0,32,hst.stream)); ETX_CHECK(hipMemsetAsync(a.abort,0,4,hst.stream));
      unsigned four=4; ETX_CHECK(hipMemcpyAsync(a.outputs,&four,4,hipMemcpyHostToDevice,hst.stream));
      void* args[]={&a}; ETX_CHECK(hipLaunchCooperativeKernel((const void*)c6_probe,dim3(304),dim3(1024),args,0,hst.stream));
      if (!wait_stream(hst.stream,5)) return false;
      int aborted=0; unsigned outputs=4,tickets[8];
      ETX_CHECK(hipMemcpy(&aborted,a.abort,4,hipMemcpyDeviceToHost));
      ETX_CHECK(hipMemcpy(&outputs,a.outputs,4,hipMemcpyDeviceToHost));
      ETX_CHECK(hipMemcpy(state,a.state,sizeof(state),hipMemcpyDeviceToHost));
      ETX_CHECK(hipMemcpy(tickets,a.tickets,sizeof(tickets),hipMemcpyDeviceToHost));
      if (aborted || outputs || tickets[7]<1) return false;
      for (int h=0;h<4;++h) if (tickets[h]<7 || state[h].mask!=31 || state[h].arrived!=5 || state[h].output!=epoch) return false;
      li+=2;
    }
    return wait_stream(hst.stream,5);
  }
};
