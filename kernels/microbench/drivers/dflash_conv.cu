#include "common.cuh"
#include "../kern/src/dflash_conv.cu"
int main() {
  constexpr int t=16,d=5120,g=320,side=1;
  Buffer<__nv_bfloat16> x(t*d),delta(t*4*g),base(4*d);
  std::vector<__nv_bfloat16> h(t*d),b(4*d),dl(t*4*g);
  for(int i=0;i<t*d;++i) h[i]=__float2bfloat16(float((i%7)-3)/8);
  // Nonzero second tap and delta exercise the history and 8-token reset.
  for(int i=0;i<4*d;++i) b[i]=__float2bfloat16(.25f);
  for(int i=0;i<t*4*g;++i) dl[i]=__float2bfloat16(.25f);
  x.put(h); base.put(b); delta.put(dl);
  auto launch=[&] { kern_dflash_conv_bf16<<<20,256>>>(x.p,delta.p,base.p,t,d,g,4*g,side); };
  launch(); CUDA(cudaDeviceSynchronize()); auto out=x.get();
  for(int r=0;r<t;++r) for(int j=0;j<d;++j) {
    float want=.5f*__bfloat162float(h[r*d+j]);
    if(r%8) want+=.5f*__bfloat162float(h[(r-1)*d+j]);
    require(__bfloat162float(out[r*d+j])==__bfloat162float(__float2bfloat16(want)));
  }
  x.put(h);
  measure("dflash_conv","tokens=16 dim=5120 groups=320 side=1 grid=(20) block=256 in_place_repeated=1",launch);
}
