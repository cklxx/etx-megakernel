// Standalone real-tile probe. Compile/link/load/execute are separate gates.
#include "plan_data.h"
#include "probe_config.h"
#include "etx/host.h"
#include <algorithm>
#include <fstream>
#include <string>

extern "C" __global__ void etx_megakernel_d0(etx_params);
extern "C" __global__ void etx_unfused_d0(etx_params, int32_t);
using Clock = std::chrono::steady_clock;
static void die(const std::string& s) { fprintf(stderr, "BLOCKED %s\n", s.c_str()); exit(2); }
static std::vector<unsigned char> read(const std::string& p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  std::vector<unsigned char> b(n);
  f.read((char*)b.data(), n);
  if ((size_t)f.gcount() != n || f.peek() != EOF) die("input size: " + p);
  return b;
}
static void* load(const std::string& p, size_t n) {
  const auto b = read(p, n); return etx_host::upload(b.data(), n);
}
static void save(const std::string& p, const void* d, size_t n) {
  std::vector<unsigned char> b(n);
  ETX_CHECK(hipMemcpy(b.data(), d, n, hipMemcpyDeviceToHost));
  std::ofstream f(p, std::ios::binary); f.write((char*)b.data(), n);
  if (!f) die("output write: " + p);
}
static void wait(hipStream_t stream, double cap) {
  const auto start = Clock::now();
  for (;;) {
    const auto e = hipStreamQuery(stream);
    if (e == hipSuccess) return;
    if (e != hipErrorNotReady) die(hipGetErrorString(e));
    if (std::chrono::duration<double>(Clock::now()-start).count() > cap)
      die("device deadline: stop this process; retain logs; do not time or reuse its state");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
struct Block {
  unsigned char b[160]{};
  template<class T> void at(int off, T v) { memcpy(b+off, &v, sizeof(v)); }
  void fill(void* w, void* x, void* out, void* ids, void* wts,
            int M, int K, int rpt, int tpe, int tasks) {
    at(0,w); at(8,x); at(16,out); at(24,ids); at(32,wts);
    at(40,(int64_t)M*K); at(48,M); at(52,K); at(56,rpt); at(60,tpe);
    at(100,1.e-6f); at(120,1);
    at(144,tasks); at(148,1); at(152,1); at(156,tasks);
  }
};
int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 4) die("usage: run INPUT OUT check|time [fused|graph] [iterations]");
  const std::string in=argv[1], out=argv[2], mode=argv[3];
  const bool graph = argc > 4 && std::string(argv[4]) == "graph";
  const int repeat = argc > 5 ? atoi(argv[5]) : 200;
  if ((mode != "check" && mode != "time") || repeat < 1 || repeat > 2000)
    die("invalid mode or repeat (1..2000)");
  if (mode == "time" && EEM_DIAGNOSTIC) die("diagnostic executable cannot time");
  if (getenv("ETX_TRACE") || getenv("ETX_INBODY_RAW") || getenv("ETX_EXEC_COUNT"))
    die("external instrumentation is not permitted in this probe");
  // The Python runner checks the input and executable identities and same-build
  // byte certificate before it invokes time. Direct time is also fail-closed.
  if (mode == "time" && !getenv("EEM_CHECKED_IDENTITY")) die("missing runner correctness gate");
  hipDeviceProp_t prop; ETX_CHECK(hipGetDeviceProperties(&prop,0));
  printf("EEM device=%s arch=%s arm=%s diagnostic=%d consumer=%s identity=%s\n",
         prop.name, prop.gcnArchName, EEM_CANDIDATE ? "candidate" : "control",
         EEM_DIAGNOSTIC, EEM_HEAD ? "head" : "qkv", EEM_IDENTITY);
  if (std::string(prop.gcnArchName).find("gfx942") != 0) die("requires gfx942");
  if (read(in+"/role.bin",1)[0] != EEM_HEAD) die("input consumer role differs from build");
  auto x=load(in+"/x.bin",4096), residual=load(in+"/residual.bin",4096);
  auto norm=load(in+"/norm.bin",4096), ids=load(in+"/ids.bin",32), wts=load(in+"/wts.bin",32);
  auto w13=load(in+"/w13.bin",(size_t)8*2*768*2048*2);
  auto w2=load(in+"/w2.bin",(size_t)8*2048*768*2);
  auto wc=load(in+"/consumer.bin",(size_t)EEM_M*2048*2);
  etx_host h;
  h.kernel=(const void*)etx_megakernel_d0; h.unfused_kernel=(const void*)etx_unfused_d0;
  h.alloc_args();
  auto alloc=[](size_t n) { void* p; ETX_CHECK(hipMalloc(&p,n)); ETX_CHECK(hipMemset(p,0xA5,n)); return p; };
  auto act=alloc(8*768*2), contrib=alloc(8*2048*2), rout=alloc(4096);
  auto normalized=alloc(4096), result=alloc((size_t)EEM_M*2);
  Block blocks[3];
  blocks[0].fill(w13,x,act,ids,nullptr,768,2048,32,24,192);
  blocks[0].at(40,(int64_t)2*768*2048);
  blocks[1].fill(w2,act,contrib,ids,wts,2048,768,64,32,256);
  blocks[2].fill(wc,normalized,result,nullptr,nullptr,EEM_M,2048,(EEM_M+255)/256,0,256);
  blocks[2].at(64,contrib); blocks[2].at(72,residual);
  blocks[2].at(80,EEM_HEAD ? nullptr : rout); blocks[2].at(88,norm);
  blocks[2].at(96,8); blocks[2].at(104,1);
  ETX_CHECK(hipMemcpy(h.args[etx_host::arg_index("impargs")],blocks,480,hipMemcpyHostToDevice));
  h.setup();
  if (!h.check_residency()) die("persistent workers are not resident");
  const int init[33]={8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,32};
  const auto counts=h.args[etx_host::arg_index("counts")];
  auto reset=[&]() {
    h.reset_step();
    if (EEM_CANDIDATE) ETX_CHECK(hipMemcpyAsync(counts,init,sizeof(init),hipMemcpyHostToDevice,h.stream));
  };
  if (mode == "check") {
    // Several changed payloads reuse exactly the same cross-XCD addresses.
    // Row permutations change the physical producer order while the arithmetic
    // always remains in original slot order. Restore input on each replay.
    const auto original_x=read(in+"/x.bin",4096);
    for (int round=0; round<12; ++round) {
      auto changed=original_x;
      for (size_t i=0; i<changed.size(); i+=2) if (round&1) changed[i+1]^=0x80;
      ETX_CHECK(hipMemcpyAsync(x,changed.data(),4096,hipMemcpyHostToDevice,h.stream));
      ETX_CHECK(hipMemcpyAsync(h.args[etx_host::arg_index("order")],&round,4,hipMemcpyHostToDevice,h.stream));
      reset();
      if (graph) {
        // enqueue_unfused is also captured in the timed same-kernel HIP graph.
        h.enqueue_unfused(); wait(h.stream,5);
      } else {
        h.launch(); if (!h.wait(5,Clock::now())) die("incomplete fused check");
      }
      const std::string pre=out+"/r"+std::to_string(round)+"_";
      save(pre+"act.bin",act,8*768*2); save(pre+"contrib.bin",contrib,8*2048*2);
      save(pre+"output.bin",result,(size_t)EEM_M*2);
      save(pre+"normalized.bin",normalized,4096);
      if (EEM_DIAGNOSTIC) {
        save(pre+"merged.bin",h.args[etx_host::arg_index("merged")],4096);
        save(pre+"partials.bin",h.args[etx_host::arg_index("partials")],1024);
        save(pre+"inverse.bin",h.args[etx_host::arg_index("inverse")],4);
        save(pre+"staged.bin",h.args[etx_host::arg_index("staged")],256*2048*2);
      }
      save(pre+"residual.bin",rout,4096);
      if (EEM_CANDIDATE) {
        if (EEM_DIAGNOSTIC) save(pre+"owners.bin",h.args[etx_host::arg_index("owners")],128);
        int c[33]; ETX_CHECK(hipMemcpy(c,counts,sizeof(c),hipMemcpyDeviceToHost));
        for (int v:c) if (v != 0) die("counter did not reach zero");
      }
      printf("EEM CHECK round=%d mode=%s complete\n",round,graph ? "graph" : "fused");
    }
  } else {
    // Common event window includes reset, original up, down, added merge,
    // retained full completion, consumer staging, GEMV and runtime retirement.
    // Capture once to remove host enqueue gaps. No diagnostic stamps are built.
    hipGraph_t cap; hipGraphExec_t exec;
    ETX_CHECK(hipStreamBeginCapture(h.stream,hipStreamCaptureModeGlobal));
    reset(); if (graph) h.enqueue_unfused(); else h.launch(false);
    ETX_CHECK(hipStreamEndCapture(h.stream,&cap));
    ETX_CHECK(hipGraphInstantiate(&exec,cap,nullptr,nullptr,0));
    for (int i=0;i<20;++i) ETX_CHECK(hipGraphLaunch(exec,h.stream));
    wait(h.stream,5);
    hipEvent_t start,stop; ETX_CHECK(hipEventCreate(&start)); ETX_CHECK(hipEventCreate(&stop));
    std::vector<float> us;
    const auto t0=Clock::now();
    for (int i=0;i<repeat;++i) {
      ETX_CHECK(hipEventRecord(start,h.stream)); ETX_CHECK(hipGraphLaunch(exec,h.stream));
      ETX_CHECK(hipEventRecord(stop,h.stream)); wait(h.stream,5);
      float ms; ETX_CHECK(hipEventElapsedTime(&ms,start,stop)); us.push_back(ms*1000);
      if (i%25 == 0) printf("EEM TIME progress=%d/%d\n",i,repeat);
    }
    const auto ordered=us; std::sort(us.begin(),us.end());
    printf("EEM-TIME {\"identity\":\"%s\",\"arm\":\"%s\",\"mode\":\"%s\",\"n\":%d,\"us\":%.9f,\"wall_us\":%.9f}\n",
           EEM_IDENTITY,EEM_CANDIDATE ? "candidate" : "control",graph ? "graph" : "fused",repeat,
           (us[(repeat-1)/2]+us[repeat/2])*0.5,
           std::chrono::duration<double,std::micro>(Clock::now()-t0).count()/repeat);
    std::ofstream raw(out+"/samples.bin",std::ios::binary);
    raw.write((const char*)ordered.data(),ordered.size()*sizeof(float)); if (!raw) die("sample write");
  }
  return 0;
}
