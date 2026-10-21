// Real generated ETX fixed queues and native placement; included inside Resident.
#ifdef ETX_COLUMN_DRIVER
etx_params mp{};hipFunction_t mkernel;hipStream_t ms;
ColumnMicroState mh{};ColumnMicroState* md;
int32_t *m_events,*m_slots,*m_abort,*m_exec,*m_remaining;
unsigned *m_ready,*m_owners;uint64_t* m_trace;
unsigned long long *m_marks,*m_placement;
double m_tick_ms;
template<class T> T* mupload(const T* host,size_t n) {
 T* ptr;HIP(hipMalloc((void**)&ptr,std::max((size_t)1,n)*sizeof(T)));if(n)HIP(hipMemcpy(ptr,host,n*sizeof(T),hipMemcpyHostToDevice));return ptr;
}
void setup_micro() {
 int rate=0;HIP(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0));if(rate<=0)throw std::runtime_error("native wallclock rate missing");m_tick_ms=1./rate;
 HIP(hipModuleGetFunction(&mkernel,mod,"etx_megakernel_d0"));HIP(hipStreamCreateWithFlags(&ms,hipStreamNonBlocking));
 HIP(hipMalloc((void**)&md,sizeof(ColumnMicroState)));std::vector<void*> ga(ETX_N_ARGS,nullptr);
 for(int j=0;j<ETX_N_ARGS;++j)if(std::string(etx_arg_names[j])=="state")ga[j]=md;
 for(auto ptr:ga)if(!ptr)throw std::runtime_error("unexpected native columns tensor");
 mp.args=mupload(ga.data(),ga.size());std::vector<void*> ta(ETX_N_TYPES*ETX_MAX_ARGS,nullptr);
 for(int t=0;t<ETX_N_TYPES;++t)for(int k=0;k<ETX_MAX_ARGS;++k)if(etx_type_arg_index[t][k]>=0)ta[t*ETX_MAX_ARGS+k]=ga[etx_type_arg_index[t][k]];
 mp.type_args=mupload(ta.data(),ta.size());mp.max_args=ETX_MAX_ARGS;
 mp.shape=mupload(etx_shape,ETX_N_SYMBOLS?ETX_N_SYMBOLS:1);mp.descs=mupload(etx_descs,ETX_N_TASKS);
 mp.static_queue=mupload(etx_static_queue,ETX_STATIC_LEN);
 std::vector<etx_task> desc;for(int id:std::vector<int>(etx_static_queue,etx_static_queue+ETX_STATIC_LEN))desc.push_back(etx_descs[id>=0?id:0]);
 mp.static_descs=mupload(desc.data(),desc.size());mp.static_begin=mupload(etx_static_begin,ETX_N_WORKERS);mp.static_end=mupload(etx_static_end,ETX_N_WORKERS);
 mp.ev_offset=mupload(etx_ev_offset,ETX_N_EVENTS);mp.ev_shape=(const int32_t(*)[4])mupload(&etx_ev_shape[0][0],ETX_N_EVENTS*4);
 HIP(hipMalloc((void**)&m_events,std::max(1,ETX_EVENT_WORDS)*4));mp.events=m_events;
 HIP(hipMalloc((void**)&m_slots,(ETX_N_DOMAINS*2+ETX_N_DOMAINS*512)*4));mp.domain_slots=m_slots;
 HIP(hipMalloc((void**)&m_abort,64));mp.ctrl_abort=m_abort;mp.ctrl_done=m_abort+1;
 mp.push_index=mupload(etx_push_index,sizeof(etx_push_index)/4);mp.push_offsets=mupload(etx_push_offsets,sizeof(etx_push_offsets)/4);mp.push_lists=mupload(etx_push_lists,sizeof(etx_push_lists)/4);
 HIP(hipMalloc((void**)&m_exec,ETX_N_TASKS*4));HIP(hipMalloc((void**)&m_trace,(size_t)ETX_N_TASKS*4*8));
 HIP(hipMalloc((void**)&m_ready,4*4));HIP(hipMalloc((void**)&m_owners,24*296*4));HIP(hipMalloc((void**)&m_remaining,4*4));
 HIP(hipMalloc((void**)&m_marks,9*8));HIP(hipMalloc((void**)&m_placement,(size_t)c.B*297*8));
 mp.trace_exec=nullptr;mp.trace_time=nullptr;mp.ev_share=nullptr;mp.ev_sub=nullptr;mp.event_words=ETX_EVENT_WORDS;
 mp.workers_per_domain=ETX_WORKERS_PER_DOMAIN;mp.n_workers=ETX_N_WORKERS;mp.n_tasks=ETX_N_TASKS;mp.n_dynamic=0;mp.n_events=ETX_N_EVENTS;mp.spin_limit=20000000u;
}
void run_micro(bool fused,void* output) {
 if(epoch==2147483647)throw std::runtime_error("native columns epoch exhausted; quiescent reset required");++epoch;
 HIP(hipEventRecord(start,ms));
 unsigned long long marks[9]={~0ull,0,0,0,0,0,0,0,0};int remaining[4]={};
 const int per_request=c.record=="G05"?222:c.role==0?296:37;
 for(int r=0;r<c.B;++r)remaining[r]=per_request;
 HIP(hipMemcpyAsync(m_marks,marks,sizeof(marks),hipMemcpyHostToDevice,ms));HIP(hipMemcpyAsync(m_remaining,remaining,sizeof(remaining),hipMemcpyHostToDevice,ms));
 HIP(hipMemsetAsync(m_ready,0,16,ms));HIP(hipMemsetAsync(m_owners,0,24*296*4,ms));HIP(hipMemsetAsync(m_abort,0,64,ms));
 HIP(hipMemsetAsync(m_slots,0,(ETX_N_DOMAINS*2+ETX_N_DOMAINS*512)*4,ms));HIP(hipMemsetAsync(m_slots+ETX_N_DOMAINS*2+ETX_N_DOMAINS*256,255,ETX_N_DOMAINS*256*4,ms));
 if(ETX_EVENT_WORDS)HIP(hipMemcpyAsync(m_events,etx_ev_counts,ETX_EVENT_WORDS*4,hipMemcpyHostToDevice,ms));
 if(c.probe){HIP(hipMemsetAsync(m_exec,0,ETX_N_TASKS*4,ms));HIP(hipMemsetAsync(m_trace,0,(size_t)ETX_N_TASKS*4*8,ms));HIP(hipMemsetAsync(m_placement,255,(size_t)c.B*297*8,ms));}
 std::memset(&mh,0,offsetof(ColumnMicroState,groups));mh.batch=c.B;mh.slots=c.record=="G05"?6:1;mh.role=c.role;mh.fused=fused;mh.epoch=epoch;mh.probe=c.probe;
 mh.ready=m_ready;mh.owner_done=m_owners;mh.remaining=m_remaining;mh.abort=m_abort;mh.marks=m_marks;mh.placement=m_placement;
 mh.deadline=(unsigned long long)(1000./m_tick_ms);
 for(int r=0;r<c.B;++r)mh.arrival[r]=c.arrivals[r];
 // Original per-column arguments/weight pointers are common to both arms.
 for(int col=0;col<c.columns;++col){mh.columns[col]=args((float*)output,col);
  if(c.record=="G05")mh.columns[col].weights=(char*)payload+c.weight_off+(long long)c.ids[col]*c.stride*2;
 }
 size_t group_metadata_bytes=0;
 if(c.record=="G05" && fused){
  std::memset((char*)&mh+offsetof(ColumnMicroState,groups),0,sizeof(mh)-offsetof(ColumnMicroState,groups));
  // Stable grouping, CPU packing and incremental upload are charged only to
  // the candidate, inside the decisive host-wall sample.
  int ng=0;std::vector<int> experts;
  for(int col=0;col<c.columns;++col){const int expert=c.ids[col];auto it=std::find(experts.begin(),experts.end(),expert);
   const int group=it==experts.end()?ng++:(int)(it-experts.begin());if(it==experts.end())experts.push_back(expert);
   auto& g=mh.groups[group];if(g.width>=4)throw std::runtime_error("native group width exceeded4");g.members[g.width++]=col;mh.group_for[col]=group;
  }
  group_metadata_bytes=sizeof(mh)-offsetof(ColumnMicroState,groups);
 }
 HIP(hipMemcpyAsync(md,&mh,offsetof(ColumnMicroState,groups)+group_metadata_bytes,hipMemcpyHostToDevice,ms));
 mp.trace_exec=c.probe?m_exec:nullptr;mp.trace_time=c.probe?m_trace:nullptr;
 void* arguments[]={&mp};HIP(hipModuleLaunchKernel(mkernel,ETX_N_WORKERS,1,1,ETX_THREADS,1,1,0,ms,arguments,nullptr));
 HIP(hipEventRecord(stop,ms));HIP(hipEventSynchronize(stop));int error=0;HIP(hipMemcpy(&error,m_abort,4,hipMemcpyDeviceToHost));if(error)throw std::runtime_error("native columns ETX publication/deadline failure");
 HIP(hipMemcpy(remaining,m_remaining,sizeof(remaining),hipMemcpyDeviceToHost));for(int r=0;r<c.B;++r)if(remaining[r])throw std::runtime_error("native writing-worker completion count incomplete");
 if(c.probe){
  std::vector<int> counts(ETX_N_TASKS);HIP(hipMemcpy(counts.data(),m_exec,counts.size()*4,hipMemcpyDeviceToHost));for(int count:counts)if(count!=1)throw std::runtime_error("native ETX task not executed exactly once");
  std::vector<uint64_t> trace((size_t)ETX_N_TASKS*4);HIP(hipMemcpy(trace.data(),m_trace,trace.size()*8,hipMemcpyDeviceToHost));
  std::vector<unsigned long long> placement((size_t)c.B*297);HIP(hipMemcpy(placement.data(),m_placement,placement.size()*8,hipMemcpyDeviceToHost));
  for(int r=0;r<c.B;++r)for(int x=0;x<(c.record=="G05"?6:c.role==0?8:1);++x)for(int w=0;w<37;++w){
   const int index=c.role==0?r*296+x*37+w:(r*mh.slots+x)*37+w,domain=c.role==0||c.record=="G05"?x:c.native_xcd;
   auto value=placement[index];if((value>>32)!=(unsigned)c.physical[domain] || (unsigned)value%ETX_WORKERS_PER_DOMAIN!=(unsigned)(38*(r%2)+w))throw std::runtime_error("actual native ETX worker/XCC placement differs from source-bound input");
  }
  for(int r=0;r<c.B;++r){const auto word=placement[c.B*296+r];if((word>>32)!=(unsigned)c.physical[r] || (unsigned)word!=(unsigned)(r*ETX_WORKERS_PER_DOMAIN+37))throw std::runtime_error("actual independent readiness producer worker/XCC differs");}
  const auto stem=c.out+(fused?"/candidate":"/control");std::ofstream(stem+".counts.bin",std::ios::binary).write((const char*)counts.data(),counts.size()*4);std::ofstream(stem+".trace.bin",std::ios::binary).write((const char*)trace.data(),trace.size()*8);std::ofstream(stem+".placement.bin",std::ios::binary).write((const char*)placement.data(),placement.size()*8);
 }
}
#endif
