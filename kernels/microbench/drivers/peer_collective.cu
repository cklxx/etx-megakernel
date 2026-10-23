#include "common.cuh"
#include "../kern/src/peer_collective.cu"
int main() {
  // nranks=1 is local-copy control, not a measurement of a fabric collective.
  constexpr int rows=16,rb=8192,n=rows*rb/8,grid=16;
  Buffer<uint2> x(n),y(n); Buffer<uint4> sym(2*n);
  Buffer<unsigned long long> peers(1); peers.put({reinterpret_cast<unsigned long long>(sym.p)});
  Buffer<unsigned> epochs(grid); Buffer<int> err(1),blocks(2); blocks.put({0,rows});
  std::vector<uint2> h(n); for(int i=0;i<n;++i) h[i]=make_uint2(i,~unsigned(i)); x.put(h);
  auto launch=[&] { kern_peer_allgather<<<grid,256>>>(x.p,y.p,sym.p,peers.p,epochs.p,err.p,blocks.p,0,1,rb,n,1000000); };
  launch(); CUDA(cudaDeviceSynchronize()); auto o=y.get();
  for(int i=0;i<n;++i) require(o[i].x==h[i].x&&o[i].y==h[i].y); require(err.get()[0]==0);
  measure("peer_collective","nranks=1 local_copy_only=1 rows=16 row_bytes=8192 grid=(16) block=256",launch,2ull*rows*rb);
  require(err.get()[0]==0); for(auto e:epochs.get()) require(e==221);
}
