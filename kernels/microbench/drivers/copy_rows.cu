#include "common.cuh"
#include "../kern/src/copy_rows.cu"
int main() {
  constexpr int rows=32,w=4096,ss=w+16,ds=w+8,own=4;
  Buffer<__nv_bfloat16> src(rows*ss),dst(rows*ds),last(w),slice(own*ss);
  Buffer<int> blocks(1); blocks.put({rows-2});
  std::vector<__nv_bfloat16> x(rows*ss);
  for(int i=0;i<rows*ss;++i) x[i]=__float2bfloat16(float(i%127)); src.put(x);
  auto launch=[&] {
    kern_copy_rows_bf16<<<rows,256>>>(dst.p,src.p,ds,ss,w);
    kern_last_row_bf16<<<1,256>>>(last.p,src.p,ss,w,rows);
    kern_own_rows<<<own,256>>>(slice.p,src.p,blocks.p,0,ss*2,rows);
  };
  launch(); CUDA(cudaDeviceSynchronize()); auto d=dst.get(),l=last.get(),o=slice.get();
  for(int r=0;r<rows;++r) for(int j=0;j<ds;++j)
    require(__bfloat162float(d[r*ds+j])==(j<w?__bfloat162float(x[r*ss+j]):0));
  for(int j=0;j<w;++j) require(__bfloat162float(l[j])==__bfloat162float(x[(rows-1)*ss+j]));
  for(int r=0;r<own;++r) for(int j=0;j<ss;++j)
    require(__bfloat162float(o[r*ss+j])==__bfloat162float(x[std::min(rows-2+r,rows-1)*ss+j]));
  measure("copy_rows","rows=32 width=4096 src_stride=4112 dst_stride=4104 own=4 grid=(32)+(1)+(4) block=256x3",launch,4ull*(rows*w+w+own*ss)+4*own);
}
