// Only included by a generated, checking-only copy of the kept full-model host.
// The production host and all timed model builds stay byte-identical.
#pragma once
struct EemCapture {
  template<class T> static T field(const unsigned char* a,int off) { T v; memcpy(&v,a+off,sizeof(v)); return v; }
  static void bytes(const std::string& name,const void* p,size_t n) {
    if (!p) die("null expert merge capture input: "+name);
    std::vector<unsigned char> b(n); ETX_CHECK(hipMemcpy(b.data(),p,n,hipMemcpyDeviceToHost));
    std::ofstream f(name,std::ios::binary); f.write((const char*)b.data(),n);
    if (!f) die("expert merge capture write: "+name);
  }
  static void before(const std::string& base,const std::string& role,int layer,int position,
                     int target,const unsigned char* a,const void* residual,const float* weights) {
    if (role=="moe_up" && layer==target) {
      if (field<int>(a,104)!=0 || field<int>(a,120)!=1 || field<int>(a,48)!=768 || field<int>(a,52)!=2048)
        die("expert merge capture requires kept unfolded top-k, batch-one expert geometry");
      int ids[8]; ETX_CHECK(hipMemcpy(ids,field<void*>(a,24),32,hipMemcpyDeviceToHost));
      bytes(base+"/x.bin",field<void*>(a,8),4096);
      bytes(base+"/residual.bin",residual,4096); bytes(base+"/wts.bin",weights,32);
      const unsigned char* up=(const unsigned char*)field<void*>(a,0);
      for (int s=0;s<8;++s) if (ids[s]<0 || ids[s]>=128) die("invalid captured expert ID");
      std::ofstream f(base+"/w13.bin",std::ios::binary);
      std::vector<unsigned char> b(2*768*2048*2);
      for (int s=0;s<8;++s) {
        ETX_CHECK(hipMemcpy(b.data(),up+(size_t)ids[s]*b.size(),b.size(),hipMemcpyDeviceToHost));
        f.write((const char*)b.data(),b.size());
      }
      if (!f) die("expert merge w13 capture write");
      std::ofstream idfile(base+"/selected_ids.bin",std::ios::binary); idfile.write((char*)ids,32);
      for (int s=0;s<8;++s) ids[s]=s;
      std::ofstream packed(base+"/ids.bin",std::ios::binary); packed.write((char*)ids,32);
      std::ofstream meta(base+"/capture.txt"); meta<<target<<' '<<position<<'\n';
      if (!idfile || !packed || !meta) die("expert merge capture metadata write");
      printf("EEM-CAPTURE up layer=%d position=%d\n",layer,position); fflush(stdout);
    }
    if (role=="moe_down" && layer==target) {
      int ids[8]; ETX_CHECK(hipMemcpy(ids,field<void*>(a,24),32,hipMemcpyDeviceToHost));
      const unsigned char* down=(const unsigned char*)field<void*>(a,0);
      std::vector<unsigned char> b(2048*768*2);
      std::ofstream f(base+"/w2.bin",std::ios::binary);
      for (int s=0;s<8;++s) {
        if (ids[s]<0 || ids[s]>=128) die("invalid captured down expert ID");
        ETX_CHECK(hipMemcpy(b.data(),down+(size_t)ids[s]*b.size(),b.size(),hipMemcpyDeviceToHost));
        f.write((const char*)b.data(),b.size());
      }
      if (!f) die("expert merge w2 capture write");
    }
    if ((role=="qkv" && layer==target+1) || (role=="lm_head" && target==47)) {
      const int head=target==47, M=head?151936:5120;
      if (field<int>(a,48)!=M || field<int>(a,52)!=2048 || field<int>(a,96)!=8 || field<int>(a,104)!=1)
        die("expert merge capture requires original folded consumer geometry");
      bytes(base+"/norm.bin",field<void*>(a,88),4096);
      bytes(base+"/consumer.bin",field<void*>(a,0),(size_t)M*2048*2);
      std::ofstream f(base+"/role.bin",std::ios::binary); const char rolebyte=head; f.write(&rolebyte,1);
      if (!f) die("expert merge role capture write");
      printf("EEM-CAPTURE consumer=%s layer=%d position=%d\n",head?"head":"qkv",layer,position); fflush(stdout);
    }
  }
};
