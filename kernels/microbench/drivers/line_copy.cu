#include "common.cuh"
#include "../kern/src/line_copy.cu"
int main() {
  constexpr int seq=4,nb=65536,off=16,lb=nb+32;
  Buffer<int> idx(seq); idx.put({0,3,1,2});
  Buffer<unsigned char> state(4*lb),dst(seq*nb),target(4*lb);
  std::vector<unsigned char> st(4*lb); for(size_t i=0;i<st.size();++i) st[i]=(i*17+i/lb)%251; state.put(st);
  auto launch=[&] {
    kern_line_gather<<<dim3(16,seq),256>>>(idx.p,state.p,dst.p,lb,off,nb);
    kern_line_scatter<<<dim3(16,seq),256>>>(idx.p,target.p,dst.p,lb,off,nb);
  };
  launch(); CUDA(cudaDeviceSynchronize()); auto d=dst.get(),o=target.get(); int ids[]={0,3,1,2};
  for(int q=0;q<seq;++q) for(int j=0;j<nb;++j) require(d[q*nb+j]==(q?st[ids[q]*lb+off+j]:0));
  for(int l=0;l<4;++l) for(int j=0;j<lb;++j) require(o[l*lb+j]==(l&&j>=off&&j<off+nb?st[l*lb+j]:0));
  measure("line_copy","seqs=4 active=3 nbytes=65536 offset=16 grid=(16,4)x2 block=256x2",launch,4ull*3*nb+2*seq*4);
}
