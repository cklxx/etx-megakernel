#include "common.cuh"
#include "../kern/src/gdn_advance.cu"
int main() {
  constexpr int seq=4,rps=8,rows=seq*rps,cols=257,tb=4096,lb=8*tb;
  Buffer<uint16_t> src(rows*cols),dst(rows*cols);
  Buffer<int> idx(seq*rps),nacc(seq); Buffer<unsigned char> state((seq+1)*lb);
  std::vector<uint16_t> x(rows*cols); for(int i=0;i<rows*cols;++i) x[i]=i%65536; src.put(x);
  std::vector<int> ids(seq*rps,0),acc={1,2,3,4};
  for(int q=0;q<seq;++q) ids[q*rps+acc[q]-1]=q==3?0:q+1;
  idx.put(ids); nacc.put(acc);
  std::vector<unsigned char> st((seq+1)*lb);
  for(size_t i=0;i<st.size();++i) st[i]=(i/tb*13+i%251)%256; state.put(st);
  auto launch=[&] {
    kern_mask_row0<<<rows,256>>>(dst.p,src.p,rows,rps,cols);
    kern_conv_shift<<<dim3(seq,1),256>>>(idx.p,rps,state.p,nacc.p,lb,tb);
  };
  launch(); CUDA(cudaDeviceSynchronize()); auto d=dst.get(); auto o=state.get(); auto want=st;
  for(int r=0;r<rows;++r) for(int j=0;j<cols;++j) require(d[r*cols+j]==(r%rps?x[r*cols+j]:0xff80));
  for(int q=0;q<seq;++q) { int a=acc[q]-1,line=ids[q*rps+a]; if(a>0&&line>0)
    for(int j=0;j<3*tb;++j) want[line*lb+j]=st[line*lb+a*tb+j]; }
  require(o==want); state.put(st);
  measure("gdn_advance","seqs=4 rows=32 cols=257 tok_bytes=4096 grid=(32)+(4,1) block=256x2 in_place_repeated=1",launch);
}
