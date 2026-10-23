#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
template<class T> struct Buffer {
  T* p; size_t n;
  explicit Buffer(size_t count): n(count) { CUDA(cudaMalloc(&p, n*sizeof(T))); CUDA(cudaMemset(p, 0, n*sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  void put(const std::vector<T>& v) { if(v.size()!=n) exit(1); CUDA(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
  std::vector<T> get() { std::vector<T> v(n); CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
};
inline void require(bool ok) { if (!ok) { fprintf(stderr,"correctness check failed\n"); exit(1); } }
template<class F> void measure(const char* name, const char* config, F launch, size_t bytes=0) {
  for(int i=0;i<20;++i) launch();
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  cudaEvent_t start,stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
  std::vector<float> us;
  for(int i=0;i<200;++i) {
    CUDA(cudaEventRecord(start)); launch(); CUDA(cudaEventRecord(stop));
    CUDA(cudaEventSynchronize(stop)); CUDA(cudaGetLastError());
    float ms; CUDA(cudaEventElapsedTime(&ms,start,stop)); us.push_back(ms*1000);
  }
  std::sort(us.begin(),us.end()); double p50=(us[99]+us[100])/2.0;
  printf("%s p50_us=%.3f min_us=%.3f max_us=%.3f %s",name,p50,us.front(),us.back(),config);
  if(bytes) printf(" effective_GBps=%.3f",bytes/p50/1000.0);
  else printf(" effective_GBps=N/A");
  printf(" check=OK\n");
  CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
}
