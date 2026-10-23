#include "common.cuh"
#include "../kern/src/weight_prep.cu"
#include <cmath>
int main() {
  constexpr int n=4099,rows=32,half=64,grid=(n+255)/256;
  Buffer<__nv_bfloat16> w(n),cs(rows*half),sn(rows*half);
  Buffer<float> plus(n),cast(n),ff(n); Buffer<int> fi(n),iota(n*2);
  Buffer<long long> fl(n); Buffer<unsigned char> fb(n);
  std::vector<__nv_bfloat16> h(n); for(int i=0;i<n;++i) h[i]=__float2bfloat16(float(i%31)/8); w.put(h);
  auto launch=[&] {
    kern_weight_p1_bf16_f32<<<grid,256>>>(w.p,plus.p,n);
    kern_cast_bf16_f32<<<grid,256>>>(w.p,cast.p,n);
    kern_rope_table_bf16<<<rows,64>>>(cs.p,sn.p,rows,half,half,10000.f);
    kern_fill_f32<<<grid,256>>>(ff.p,n,1.5f);
    kern_fill_i32<<<grid,256>>>(fi.p,n,-7);
    kern_fill_i64<<<grid,256>>>(fl.p,n,12345678901ll);
    kern_fill_u8<<<grid,256>>>(fb.p,n,171);
    kern_iota_i32<<<grid,256>>>(iota.p,n,3,2,2);
  };
  launch(); CUDA(cudaDeviceSynchronize()); auto a=plus.get(),b=cast.get(),f=ff.get(); auto i=fi.get(),it=iota.get(); auto l=fl.get(); auto u=fb.get();
  for(int j=0;j<n;++j) require(a[j]==__bfloat162float(h[j])+1&&b[j]==__bfloat162float(h[j])&&f[j]==1.5f&&i[j]==-7&&l[j]==12345678901ll&&u[j]==171&&it[2*j]==3+2*j&&it[2*j+1]==0);
  auto c=cs.get(),s=sn.get();
  for(int r=0;r<rows;++r) for(int j=0;j<half;++j) {
    double angle=r/std::pow(10000.,double(j)/half);
    require(std::abs(__bfloat162float(c[r*half+j])-std::cos(angle))<.004);
    require(std::abs(__bfloat162float(s[r*half+j])-std::sin(angle))<.004);
  }
  measure("weight_prep","n=4099 rope_rows=32 half=64 grid=(17)x2+(32)+(17)x5 block=256x2+64+256x5",launch);
}
