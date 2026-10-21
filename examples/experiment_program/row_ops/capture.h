// Checking-host-only native capture for G11. No consumer scratch is sampled.
#pragma once
#include <iomanip>
struct RowCapture {
  struct B { const void* pointer; size_t bytes; std::string file; bool mutable_; };
  std::string directory; std::vector<B> buffers; std::vector<std::pair<int,int>> mapping;
  static void save(const std::string& file,const void* p,size_t n,bool device) {
    std::vector<char> h(n); if(device) ETX_CHECK(hipMemcpy(h.data(),p,n,hipMemcpyDeviceToHost)); else memcpy(h.data(),p,n);
    std::ofstream f(file,std::ios::binary); f.write(h.data(),n); if(!f)die("row capture write failed");
  }
  template<class T> static T field(const unsigned char*a,int off) { T x;memcpy(&x,a+off,sizeof(x));return x; }
  void before(const std::string& dir,const unsigned char*a,const std::string& weight,size_t wo,size_t wn,int layer,int position,int first,int count,int total,const std::string& role) {
    const bool fold=role=="dsl_gemv_fold";
    if(field<int>(a,120)!=1 || field<int>(a,96)!=8) die("row capture fixed B1/top8 only");
    if(fold && (field<int>(a,52)!=2048 || field<int>(a,104)!=1))die("G11 fixed current qkv shape only");
    if(!fold && role!="dsl_moe_up" && role!="dsl_moe_down")die("unsupported row role");
    directory=dir; save(dir+"/arguments.bin",a,160,false);
    std::ofstream meta(dir+"/capture.json");
    meta<<"{\"schema\":\"etx-row-replay-v1\",\"baseline_revision\":\"b573664faa4e7988dff353bc03aca4ce8561a729\",\"position\":"<<position<<",\"layer\":"<<layer<<",\"role\":"<<std::quoted(role)<<",\"first\":"<<first<<",\"count\":"<<count<<",\"total_tasks\":"<<total<<",\"reference_provenance\":\"unchanged-original-host\",\"arguments\":{\"file\":"<<std::quoted(dir+"/arguments.bin")<<"},\"buffers\":[{\"file\":"<<std::quoted(weight)<<",\"offset\":"<<wo<<",\"bytes\":"<<wn<<"}";
    const int K=field<int>(a,52),M=field<int>(a,48),R=field<int>(a,96),E=field<int>(a,108);
    const std::vector<int> offsets=fold?std::vector<int>{8,16,64,72,80,88}:std::vector<int>{8,16,24,32,112};
    const std::vector<size_t> sizes=fold?std::vector<size_t>{size_t(K)*2,size_t(M)*2,size_t(R)*K*2,size_t(K)*2,size_t(K)*2,size_t(K)*2}:std::vector<size_t>{size_t(role=="dsl_moe_down"?R:1)*K*2,size_t(R)*M*2,size_t(R)*4,size_t(R)*4,size_t(E)*2};
    for(size_t i=0;i<offsets.size();++i) {
      const void*p=field<void*>(a,offsets[i]); if(!p)continue;
      int existing=-1; for(size_t j=0;j<buffers.size();++j) if(buffers[j].pointer==p) { if(buffers[j].bytes!=sizes[i])die("row alias size mismatch");existing=j; }
      if(existing>=0) { if((fold && offsets[i]==8) || offsets[i]==16 || offsets[i]==80) buffers[existing].mutable_=true; mapping.push_back({offsets[i],existing+1}); continue; }
      std::string file=dir+"/buffer"+std::to_string(buffers.size()+1)+".bin";
      bool mut=(fold && offsets[i]==8) || offsets[i]==16 || offsets[i]==80;
      buffers.push_back({p,sizes[i],file,mut});mapping.push_back({offsets[i],(int)buffers.size()});save(file,p,sizes[i],true);
      meta<<",{\"file\":"<<std::quoted(file)<<",\"offset\":0,\"bytes\":"<<sizes[i]<<"}";
    }
    meta<<"],\"pointers\":[{\"offset\":0,\"buffer\":0,\"add\":0}";
    for(auto m:mapping)meta<<",{\"offset\":"<<m.first<<",\"buffer\":"<<m.second<<",\"add\":0}";
    meta<<"],\"references\":[";bool comma=false;
    for(size_t i=0;i<buffers.size();++i) if(buffers[i].mutable_) { if(comma)meta<<',';comma=true;meta<<"{\"buffer\":"<<i+1<<",\"offset\":0,\"file\":"<<std::quoted(dir+"/reference"+std::to_string(i+1)+".bin")<<",\"bytes\":"<<buffers[i].bytes<<"}"; }
    meta<<"],\"dependency_edges\":{\"native_predecessor\":\"expert-down publication -> current qkv fold\",\"first_task\":"<<first<<",\"count\":"<<count<<"}}\n";
  }
  void after() {
    if(directory.empty())return;
    for(size_t i=0;i<buffers.size();++i)if(buffers[i].mutable_)save(directory+"/reference"+std::to_string(i+1)+".bin",buffers[i].pointer,buffers[i].bytes,true);
    printf("ROW-CAPTURE original folded native snapshots=%s\n",directory.c_str());
  }
};
