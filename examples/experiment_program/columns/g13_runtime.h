// Included inside Resident by the separately linked G13 driver. This uses the
// actual generated ETX plan/ABI; there is one kernel, not two mock schedules.
#if defined(ETX_G13_DRIVER) || defined(ETX_G03_DRIVER)
struct G13State {
 HistoricalArgs up,down;
 unsigned long long readiness_ticks,spin_ticks;
 int combined;int* abort;unsigned long long* marks;int* remaining;int request;unsigned long long* placement;
};
etx_params gp{};hipFunction_t gkernel;hipStream_t gs;
G13State gst[2];G13State* d_gst[2];
int32_t *g_events,*g_slots,*g_abort,*g_exec,*g_remaining;uint64_t* g_trace;unsigned long long *g_marks,*g_placement;
double g_tick_ms;
unsigned long long mark_init[5]={~0ull,0,0,0,0};int remaining_init[2]={256,256};
std::vector<etx_task> combined_desc,independent_desc;
std::vector<int32_t> combined_ids,independent_ids;
etx_task *d_combined_desc,*d_independent_desc;int32_t *d_combined_ids,*d_independent_ids;
template<class T> T* gupload(const T* host,size_t n) {
 T* ptr;HIP(hipMalloc((void**)&ptr,std::max((size_t)1,n)*sizeof(T)));if(n)HIP(hipMemcpy(ptr,host,n*sizeof(T),hipMemcpyHostToDevice));return ptr;
}
void setup_g13() {
 int rate=0;HIP(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0));if(rate<=0)throw std::runtime_error("missing actual native timer frequency");g_tick_ms=1.0/rate;
 HIP(hipMalloc((void**)&g_placement,898*8));HIP(hipMalloc((void**)&g_marks,5*8));HIP(hipMalloc((void**)&g_remaining,2*4));
 HIP(hipModuleGetFunction(&gkernel,mod,"etx_megakernel_d0"));HIP(hipStreamCreateWithFlags(&gs,hipStreamNonBlocking));
 std::vector<void*> ga(ETX_N_ARGS,nullptr);
 for(int r=0;r<2;++r) {
  HIP(hipMalloc((void**)&d_gst[r],sizeof(G13State)));
  for(int j=0;j<ETX_N_ARGS;++j)if(std::string(etx_arg_names[j])=="r"+std::to_string(r))ga[j]=d_gst[r];
 }
 for(auto v:ga)if(!v)throw std::runtime_error("G13 unexpected generated tensor");
 gp.args=gupload(ga.data(),ga.size());
 std::vector<void*> ta(ETX_N_TYPES*ETX_MAX_ARGS,nullptr);
 for(int t=0;t<ETX_N_TYPES;++t)for(int k=0;k<ETX_MAX_ARGS;++k)if(etx_type_arg_index[t][k]>=0)ta[t*ETX_MAX_ARGS+k]=ga[etx_type_arg_index[t][k]];
 gp.type_args=gupload(ta.data(),ta.size());gp.max_args=ETX_MAX_ARGS;
 gp.shape=gupload(etx_shape,ETX_N_SYMBOLS?ETX_N_SYMBOLS:1);gp.descs=gupload(etx_descs,ETX_N_TASKS);
 combined_ids.assign(etx_static_queue,etx_static_queue+ETX_STATIC_LEN);independent_ids=combined_ids;
 auto priority=[](int id) {if(id<0)return 999;auto name=std::string(etx_type_names[etx_descs[id].type]);int r=name.find("r1")!=std::string::npos;return r*3+(name.find("ready")!=std::string::npos?0:name.find("up")!=std::string::npos?1:2);};
 for(int w=0;w<ETX_N_WORKERS;++w) {
  auto begin=independent_ids.begin()+etx_static_begin[w],end=independent_ids.begin()+etx_static_end[w];
  std::stable_sort(begin,end,[&](int a,int b){return priority(a)<priority(b);});
 }
 for(int id:combined_ids)combined_desc.push_back(etx_descs[id>=0?id:0]);
 for(int id:independent_ids)independent_desc.push_back(etx_descs[id>=0?id:0]);
 d_combined_ids=gupload(combined_ids.data(),combined_ids.size());d_independent_ids=gupload(independent_ids.data(),independent_ids.size());
 d_combined_desc=gupload(combined_desc.data(),combined_desc.size());d_independent_desc=gupload(independent_desc.data(),independent_desc.size());
 gp.static_begin=gupload(etx_static_begin,ETX_N_WORKERS);gp.static_end=gupload(etx_static_end,ETX_N_WORKERS);
 gp.ev_offset=gupload(etx_ev_offset,ETX_N_EVENTS);gp.ev_shape=(const int32_t(*)[4])gupload(&etx_ev_shape[0][0],ETX_N_EVENTS*4);
 HIP(hipMalloc((void**)&g_events,ETX_EVENT_WORDS*sizeof(int32_t)));gp.events=g_events;
 HIP(hipMalloc((void**)&g_slots,(ETX_N_DOMAINS*2+ETX_N_DOMAINS*512)*sizeof(int32_t)));gp.domain_slots=g_slots;
 HIP(hipMalloc((void**)&g_abort,64));gp.ctrl_abort=g_abort;gp.ctrl_done=g_abort+1;
 gp.push_index=gupload(etx_push_index,sizeof(etx_push_index)/4);gp.push_offsets=gupload(etx_push_offsets,sizeof(etx_push_offsets)/4);gp.push_lists=gupload(etx_push_lists,sizeof(etx_push_lists)/4);
 HIP(hipMalloc((void**)&g_exec,ETX_N_TASKS*4));HIP(hipMalloc((void**)&g_trace,(size_t)ETX_N_TASKS*4*8));
 gp.trace_exec=c.probe?g_exec:nullptr;gp.trace_time=c.probe?g_trace:nullptr;
 gp.ev_share=nullptr;gp.ev_sub=nullptr;gp.event_words=ETX_EVENT_WORDS;
 gp.workers_per_domain=ETX_WORKERS_PER_DOMAIN;gp.n_workers=ETX_N_WORKERS;gp.n_tasks=ETX_N_TASKS;
 gp.n_dynamic=0;gp.n_events=ETX_N_EVENTS;gp.spin_limit=20000000u;
}
void run_g13(bool independent,void* output) {
 if(epoch==2147483647)throw std::runtime_error("G13 epoch exhausted");++epoch;
 HIP(hipEventRecord(start,gs));
 HIP(hipMemcpyAsync(g_marks,mark_init,5*8,hipMemcpyHostToDevice,gs));HIP(hipMemcpyAsync(g_remaining,remaining_init,2*4,hipMemcpyHostToDevice,gs));
 HIP(hipMemcpyAsync(g_events,etx_ev_counts,ETX_EVENT_WORDS*4,hipMemcpyHostToDevice,gs));
 HIP(hipMemsetAsync(g_abort,0,64,gs));HIP(hipMemsetAsync(g_slots,0,(ETX_N_DOMAINS*2+ETX_N_DOMAINS*512)*4,gs));
 HIP(hipMemsetAsync(done,0,(size_t)c.columns*512*4,gs));
 if(c.probe){HIP(hipMemsetAsync(g_placement,255,898*8,gs));HIP(hipMemsetAsync(g_exec,0,ETX_N_TASKS*4,gs));HIP(hipMemsetAsync(g_trace,0,(size_t)ETX_N_TASKS*4*8,gs));}
 for(int r=0;r<2;++r) {
  auto& state=gst[r];state={};state.combined=!independent;state.readiness_ticks=c.arrivals[r];state.spin_ticks=(unsigned long long)(1000./g_tick_ms);state.abort=g_abort;state.marks=g_marks;state.remaining=g_remaining+r;state.request=r;state.placement=g_placement;
#ifdef ETX_G03_DRIVER
  state.up=historical(output,r);state.down=historical(output,0);
#else
  state.up=historical(temp,r);state.up.outputs=reinterpret_cast<unsigned short*>(temp)+(size_t)r*8*768;
  state.up.N=768;state.up.K=2048;state.up.rpt=32;state.up.tpe=24;state.up.completions=done+(size_t)r*512+256;
  state.down=historical(output,r);state.down.weights=static_cast<char*>(payload)+c.down_off;
  state.down.inputs=reinterpret_cast<unsigned short*>(temp)+(size_t)r*8*768;
  state.down.outputs=static_cast<unsigned short*>(output)+(size_t)r*8*2048;
  state.down.N=2048;state.down.K=768;state.down.rpt=64;state.down.tpe=32;state.down.expert_stride=c.down_stride;
#endif
  HIP(hipMemcpyAsync(d_gst[r],&state,sizeof(state),hipMemcpyHostToDevice,gs));
 }
 gp.static_queue=independent && c.g13_order==0?d_independent_ids:d_combined_ids;gp.static_descs=independent && c.g13_order==0?d_independent_desc:d_combined_desc;
 void* arguments[]={&gp};HIP(hipModuleLaunchKernel(gkernel,ETX_N_WORKERS,1,1,ETX_THREADS,1,1,0,gs,arguments,nullptr));
 HIP(hipEventRecord(stop,gs));HIP(hipEventSynchronize(stop));
 int error=0;HIP(hipMemcpy(&error,g_abort,4,hipMemcpyDeviceToHost));if(error)throw std::runtime_error("G13 ETX dependency/epoch deadline failure");
 if(c.probe) {
  std::vector<int> counts(ETX_N_TASKS);HIP(hipMemcpy(counts.data(),g_exec,counts.size()*4,hipMemcpyDeviceToHost));
  for(int count:counts)if(count!=1)throw std::runtime_error("G13 actual ETX task coverage not exactly once");
  std::vector<uint64_t> trace((size_t)ETX_N_TASKS*4);HIP(hipMemcpy(trace.data(),g_trace,trace.size()*8,hipMemcpyDeviceToHost));
  std::vector<unsigned long long> placement(898);HIP(hipMemcpy(placement.data(),g_placement,placement.size()*8,hipMemcpyDeviceToHost));
  for(int worker=0;worker<ETX_N_WORKERS;++worker)for(int q=etx_static_begin[worker];q<etx_static_end[worker];++q){int id=etx_static_queue[q];if(id<0)continue;const auto& task=etx_descs[id];const std::string type=etx_type_names[task.type];
   const int request=type.find("r1")!=std::string::npos;int index;
   if(type.find("ready")!=std::string::npos)index=896+request;else {
#ifdef ETX_G03_DRIVER
   const int x=std::stoi(type.substr(type.find("_x")+2));index=request*256+x*32+task.coord[0];
#else
   const bool up=type.find("up")!=std::string::npos;index=request*448+(up?0:192)+task.coord[0]*(up?24:32)+task.coord[1];
#endif
   }
   const auto word=placement[index];if((word>>32)!=(unsigned)c.physical[worker/ETX_WORKERS_PER_DOMAIN] || (unsigned)word!=(unsigned)worker)throw std::runtime_error("actual historical ETX worker/XCC pin differs from source-bound input");
  }
  auto prefix=c.out+(independent?"/candidate":"/control");std::ofstream(prefix+".placement.bin",std::ios::binary).write((const char*)placement.data(),placement.size()*8);std::ofstream(prefix+".counts.bin",std::ios::binary).write((const char*)counts.data(),counts.size()*4);
  auto name=c.out+(independent?"/candidate.trace.bin":"/control.trace.bin");std::ofstream(name,std::ios::binary).write((const char*)trace.data(),trace.size()*8);
 }
}
#endif
