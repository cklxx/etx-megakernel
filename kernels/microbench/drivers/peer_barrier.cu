#include "common.cuh"
#include "../kern/src/peer_barrier.cu"
int main() {
  // A single-rank self peer exercises the barrier without claiming inter-GPU timing.
  Buffer<unsigned> flags(1),epoch(1); Buffer<int> err(1); Buffer<unsigned long long> peers(1);
  peers.put({reinterpret_cast<unsigned long long>(flags.p)});
  auto launch=[&] { kern_peer_barrier<<<1,32>>>(flags.p,peers.p,epoch.p,err.p,0,1,1000000); };
  launch(); CUDA(cudaDeviceSynchronize()); require(err.get()[0]==0&&epoch.get()[0]==1&&flags.get()[0]==1);
  measure("peer_barrier","nranks=1 self_peer_only=1 grid=(1) block=32",launch);
  require(err.get()[0]==0&&epoch.get()[0]==221&&flags.get()[0]==221);
}
