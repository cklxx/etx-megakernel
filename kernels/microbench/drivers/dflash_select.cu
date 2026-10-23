#include "common.cuh"
#include "../kern/src/dflash_select.cu"
int main() {
  constexpr int seq=4,steps=7,rank=256,rows=seq*8;
  Buffer<long long> cand(rows*16),out(seq*steps);
  Buffer<float> unary(rows*16);
  Buffer<__nv_bfloat16> hidden(rows*rank),succ(rows*16*rank),pred(rows*16*rank),anchor(seq*rank);
  std::vector<long long> c(rows*16); std::vector<float> u(rows*16,0);
  std::vector<__nv_bfloat16> h(rows*rank,__float2bfloat16(1)),s(rows*16*rank),p(rows*16*rank),a(seq*rank,__float2bfloat16(1));
  for(int row=0;row<rows;++row) for(int k=0;k<16;++k) {
    c[row*16+k]=row*100+k;
    // Alternating predecessor sign makes the greedy path depend on prior choices.
    for(int j=0;j<rank;++j) {
      s[(row*16+k)*rank+j]=__float2bfloat16(float(k-8)/16);
      p[(row*16+k)*rank+j]=__float2bfloat16(k==15?-1.f:1.f);
    }
  }
  cand.put(c); unary.put(u); hidden.put(h); succ.put(s); pred.put(p); anchor.put(a);
  auto launch=[&] { kern_dflash_select<<<seq,rank,16*rank*sizeof(float)>>>(cand.p,unary.p,hidden.p,succ.p,pred.p,anchor.p,out.p,steps,rank,rank,8); };
  launch(); CUDA(cudaDeviceSynchronize()); auto o=out.get();
  for(int q=0;q<seq;++q) for(int l=0;l<steps;++l) require(o[q*steps+l]==c[(q*8+1+l)*16+(l%2?0:15)]);
  measure("dflash_select","seqs=4 steps=7 rank=256 candidates=16 grid=(4) block=256 shared_bytes=16384",launch);
}
