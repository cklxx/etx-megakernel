#include "common.cuh"
#include "../kern/src/argmax.cu"
int main() {
  constexpr int rows=1,n=151936,nb=64;
  Buffer<__nv_bfloat16> logits(rows*n); Buffer<float> pm(rows*nb);
  Buffer<int> pi(rows*nb); Buffer<long long> out(rows);
  std::vector<__nv_bfloat16> input(n);
  for(int i=0;i<n;++i) input[i]=__float2bfloat16(float((i*73)%1024)/32);
  input[17]=input[n-1]=__float2bfloat16(64); logits.put(input);
  auto launch=[&] {
    kern_argmax_partial_bf16<<<dim3(rows,nb),1024>>>(logits.p,pm.p,pi.p,n);
    kern_argmax_final_i64<<<rows,64>>>(pm.p,pi.p,out.p,nb);
  };
  launch(); CUDA(cudaDeviceSynchronize()); require(out.get()[0]==17);
  measure("argmax","rows=1 n=151936 grid=(1,64)+(1) block=1024+64",launch, size_t(n)*2+nb*16+8);
}
