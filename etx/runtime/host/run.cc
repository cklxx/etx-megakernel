// Compile alongside generated megakernel_d*.hip and link libetx_rt.
// ETX_RUN_KERNELS may name a generated header declaring additional kernels and
// defining ETX_RUN_KERNEL_TABLE. ETX_RUN_ADAPTER may name model hook definitions.
#include "plan_data.h"
#include "etx/host.h"
#include "loader.h"
#include "step.h"
#include <climits>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
extern "C" __global__ void etx_megakernel_d0(etx_params);
#ifdef ETX_RUN_KERNELS
#include ETX_RUN_KERNELS
#else
static_assert(ETX_N_DEVICES==1, "provide ETX_RUN_KERNELS for a multi-device plan");
#define ETX_RUN_KERNEL_TABLE {(const void*)&etx_megakernel_d0}
#endif
struct etx_run_context {
  etx::loaded_plan& plan;
  std::vector<std::unique_ptr<etx_host>> hosts;
  std::vector<std::chrono::steady_clock::time_point> started;
};
#ifdef ETX_RUN_ADAPTER
// Header defines int etx_run_prepare/finish(void*, etx_step_descriptor*).
// Both can access etx_run_context::plan.argument(name) and hosts' streams.
#include ETX_RUN_ADAPTER
#else
static int etx_run_prepare(void*,etx_step_descriptor*) {return 0;}
static int etx_run_finish(void*,etx_step_descriptor*) {return 0;}
#endif
static int reset(void* p,uint32_t r,double) {
  auto& h=*static_cast<etx_run_context*>(p)->hosts[r];
  h.reset_step(); return h.check_residency()?0:1;
}
static int launch(void* p,uint32_t r,double) {
  auto& c=*static_cast<etx_run_context*>(p);
  c.started[r]=std::chrono::steady_clock::now();c.hosts[r]->launch();return 0;
}
static int wait(void* p,uint32_t r,double timeout) {
  auto& c=*static_cast<etx_run_context*>(p);
  return c.hosts[r]->wait(timeout,c.started[r])?0:1;
}
static uint64_t number(const char* s) {
  if(!*s || std::string(s).find_first_not_of("0123456789")!=std::string::npos)
    throw std::runtime_error("expected unsigned integer");
  size_t end=0;auto n=std::stoull(s,&end);
  if(s[end]) throw std::runtime_error("invalid integer");return n;
}
int main(int argc,char** argv) {
  if(argc<4 || argc>6) {
    fprintf(stderr,"usage: run PLAN MANIFEST WEIGHT_MAP [STEPS [TOKEN]]\n");return 2;
  }
  try {
    auto steps=argc>4?number(argv[4]):1;
    auto token=argc>5?number(argv[5]):0;
    if(token>INT_MAX) throw std::runtime_error("token exceeds int32");
    auto transport=etx::make_loader_transport();
    etx::loaded_plan plan(argv[1],argv[2],argv[3],*transport);
    etx_run_context c{plan,{}, {}};
    const void* kernels[]=ETX_RUN_KERNEL_TABLE;
    static_assert(sizeof(kernels)/sizeof(*kernels)==ETX_N_DEVICES,"kernel table size");
    int32_t* shared=ETX_N_DEVICES>1?etx_host::alloc_events(0):nullptr;
    for(int r=0;r<ETX_N_DEVICES;++r) {
      auto h=std::make_unique<etx_host>(r);h->kernel=kernels[r];
      h->args=plan.arguments();h->setup(shared);c.hosts.push_back(std::move(h));
    }
    c.started.resize(ETX_N_DEVICES);
    etx_step_backend backend{reset,launch,wait};
    etx_step_descriptor d{ETX_STEP_ABI_VERSION,sizeof(d),&backend,&c,
      ETX_N_DEVICES,0,5.0,0,int32_t(token),int32_t(token),etx_run_prepare,etx_run_finish};
    for(uint64_t i=0;i<steps;++i) {
      int status=etx_step(&d);
      if(status) {fprintf(stderr,"step %llu: %s\n",(unsigned long long)i,etx_status_string(status));return status;}
      printf("step=%llu position=%lld token=%d\n",(unsigned long long)i,(long long)d.position,d.token);
      fflush(stdout);
    }
    // Retire streams before the loader releases arguments. Runtime metadata is
    // process-owned; run exits immediately, while embedded users own teardown.
    for(auto& h:c.hosts) ETX_CHECK(etx::host_backend::stream_synchronize(h->stream));
    return 0;
  } catch(const std::exception& e) {fprintf(stderr,"run: %s\n",e.what());return 2;}
}
