#include "common.cuh"
#include "../kern/src/spec_round.cu"
int main() {
  constexpr int seqs=32,block=7,rows=64,ds=8;
  Buffer<long long> anchor(seqs),drafts(seqs*ds),verify(seqs*block),ids(seqs*block),draft_ids(seqs*block);
  Buffer<int> count(seqs),lines(rows*seqs*block),advanced(rows*seqs*block),ones(seqs);
  anchor.put(std::vector<long long>(seqs,42)); drafts.put(std::vector<long long>(seqs*ds,17));
  verify.put(std::vector<long long>(seqs*block,17)); lines.put(std::vector<int>(rows*seqs*block,123));
  auto launch=[&] {
    kern_splice_draft<<<seqs,64>>>(anchor.p,draft_ids.p,block,99);
    kern_splice_verify<<<seqs,64>>>(anchor.p,drafts.p,ids.p,block,ds);
    kern_spec_count<<<seqs,64>>>(drafts.p,verify.p,count.p,block,ds);
    kern_spec_lines<<<seqs,64>>>(lines.p,count.p,advanced.p,block,rows,seqs);
    kern_ones_i32<<<1,64>>>(ones.p,seqs);
  };
  launch(); CUDA(cudaDeviceSynchronize());
  auto d=draft_ids.get(); auto v=ids.get(); auto c=count.get(); auto a=advanced.get(); auto o=ones.get();
  for(int i=0;i<seqs;++i) { require(c[i]==7 && o[i]==1);
    for(int e=0;e<block;++e) { require(d[i*block+e]==(e?99:42)); require(v[i*block+e]==(e?17:42)); }
  }
  for(size_t i=0;i<a.size();++i) require(a[i]==(i%block==6?123:0));
  measure("spec_round","seqs=32 block_len=7 rows=64 dstride=8 grid=(32)x4+(1) block=64x5",launch);
}
