#include "common.cuh"
#include "../kern/src/markov_rows.cu"
int main() {
  constexpr int rows=32,n=151936,nb=64,stride=7,hidden=4096;
  Buffer<long long> ids(rows*stride),out(rows*stride);
  Buffer<__nv_bfloat16> table(hidden),embedding(rows*hidden),logits((size_t)rows*stride*n);
  Buffer<float> pm(rows*nb); Buffer<int> pi(rows*nb);
  std::vector<__nv_bfloat16> t(hidden,__float2bfloat16(1)); table.put(t);
  out.put(std::vector<long long>(rows*stride,-1));
  std::vector<__nv_bfloat16> input(logits.n);
  for(int r=0;r<rows;++r) {
    size_t base=(size_t)r*stride*n;
    for(int j=0;j<n;++j) input[base+j]=__float2bfloat16(float((j*73)%1024)/32);
    input[base+17]=input[base+n-1]=__float2bfloat16(64);
  }
  logits.put(input);
  auto launch=[&] {
    kern_embedding_rows_i64_bf16<<<rows,256>>>(ids.p,stride,table.p,embedding.p,hidden);
    kern_argmax_rows_partial_bf16<<<dim3(rows,nb),1024>>>(logits.p,(long long)stride*n,pm.p,pi.p,n);
    kern_argmax_rows_final_i64<<<rows,64>>>(pm.p,pi.p,out.p,stride,nb);
  };
  launch(); CUDA(cudaDeviceSynchronize()); auto o=out.get(); auto e=embedding.get();
  for(int i=0;i<rows*stride;++i) require(o[i]==(i%stride==0?17:-1));
  for(auto x:e) require(__bfloat162float(x)==1);
  measure("markov_rows","rows=32 n=151936 hidden=4096 stride=7 grid=(32)+(32,64)+(32) block=256+1024+64",launch, size_t(rows)*(n*2+hidden*4+nb*16+16));
}
