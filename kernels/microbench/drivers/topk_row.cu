#include "common.cuh"
#include "../kern/src/topk_row.cu"
int main() {
  constexpr int rows=1,n=151936;
  Buffer<__nv_bfloat16> logits(rows*n); Buffer<long long> ids(rows*K); Buffer<float> vals(rows*K);
  std::vector<__nv_bfloat16> input(n); std::vector<int> order(n);
  for(int i=0;i<n;++i) { input[i]=__float2bfloat16(float((i*73)%1024)/32); order[i]=i; }
  logits.put(input);
  std::sort(order.begin(),order.end(),[&](int a,int b) {
    float av=__bfloat162float(input[a]),bv=__bfloat162float(input[b]);
    return av>bv || (av==bv && a<b);
  });
  auto launch=[&] { kern_topk16_bf16<<<rows,1024>>>(logits.p,ids.p,vals.p,n,n); };
  launch(); CUDA(cudaDeviceSynchronize()); auto id=ids.get(); auto v=vals.get();
  for(int i=0;i<K;++i) require(id[i]==order[i] && v[i]==__bfloat162float(input[order[i]]));
  measure("topk_row","rows=1 n=151936 k=16 grid=(1) block=1024",launch);
}
