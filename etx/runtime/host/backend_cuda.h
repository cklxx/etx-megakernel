// Vendor implementation of the ETX host runtime interface.
// Always inline adapters to preserve the direct-call host object ABI at -O3.
#pragma once
#include <cuda_runtime.h>

namespace etx { namespace host_backend {
using dim3 = ::dim3;
using error = cudaError_t;
using stream = cudaStream_t;
using event = cudaEvent_t;
using graph = cudaGraph_t;
using graph_exec = cudaGraphExec_t;
using device_properties = cudaDeviceProp;
using copy_kind = cudaMemcpyKind;
using capture_mode = cudaStreamCaptureMode;
inline constexpr auto success = cudaSuccess;
inline constexpr auto not_ready = cudaErrorNotReady;
inline constexpr auto host_to_device = cudaMemcpyHostToDevice;
inline constexpr auto device_to_host = cudaMemcpyDeviceToHost;
inline constexpr auto device_to_device = cudaMemcpyDeviceToDevice;
inline constexpr auto stream_non_blocking = cudaStreamNonBlocking;
inline constexpr auto capture_global = cudaStreamCaptureModeGlobal;
inline __attribute__((always_inline)) const char* error_string(error e) { return cudaGetErrorString(e); }
inline __attribute__((always_inline)) error set_device(int device) { return cudaSetDevice(device); }
inline __attribute__((always_inline)) error get_device_count(int* count) { return cudaGetDeviceCount(count); }
inline __attribute__((always_inline)) error get_device_properties(device_properties* p, int device) { return cudaGetDeviceProperties(p, device); }
inline __attribute__((always_inline)) error device_can_access_peer(int* can, int device, int peer) { return cudaDeviceCanAccessPeer(can, device, peer); }
inline __attribute__((always_inline)) error enable_peer_access(int peer, unsigned flags = 0) { return cudaDeviceEnablePeerAccess(peer, flags); }
inline __attribute__((always_inline)) error disable_peer_access(int peer) { return cudaDeviceDisablePeerAccess(peer); }
inline __attribute__((always_inline)) error free(void* p) { return cudaFree(p); }
inline __attribute__((always_inline)) error free_host(void* p) { return cudaFreeHost(p); }
inline __attribute__((always_inline)) error memcpy(void* dst, const void* src, size_t n, copy_kind kind) { return cudaMemcpy(dst, src, n, kind); }
inline __attribute__((always_inline)) error memcpy_async(void* dst, const void* src, size_t n, copy_kind kind, stream s) { return cudaMemcpyAsync(dst, src, n, kind, s); }
inline __attribute__((always_inline)) error memset(void* dst, int value, size_t n) { return cudaMemset(dst, value, n); }
inline __attribute__((always_inline)) error memset_async(void* dst, int value, size_t n, stream s) { return cudaMemsetAsync(dst, value, n, s); }
inline __attribute__((always_inline)) error stream_create(stream* s, unsigned flags) { return cudaStreamCreateWithFlags(s, flags); }
inline __attribute__((always_inline)) error stream_destroy(stream s) { return cudaStreamDestroy(s); }
inline __attribute__((always_inline)) error stream_query(stream s) { return cudaStreamQuery(s); }
inline __attribute__((always_inline)) error stream_synchronize(stream s) { return cudaStreamSynchronize(s); }
inline __attribute__((always_inline)) error event_create(event* e) { return cudaEventCreate(e); }
inline __attribute__((always_inline)) error event_destroy(event e) { return cudaEventDestroy(e); }
inline __attribute__((always_inline)) error event_record(event e, stream s) { return cudaEventRecord(e, s); }
inline __attribute__((always_inline)) error event_elapsed_time(float* ms, event start, event stop) { return cudaEventElapsedTime(ms, start, stop); }
inline __attribute__((always_inline)) error stream_begin_capture(stream s, capture_mode mode) { return cudaStreamBeginCapture(s, mode); }
inline __attribute__((always_inline)) error stream_end_capture(stream s, graph* g) { return cudaStreamEndCapture(s, g); }
inline __attribute__((always_inline)) error graph_instantiate(graph_exec* exec, graph g) { return cudaGraphInstantiate(exec, g, nullptr, nullptr, 0); }
inline __attribute__((always_inline)) error graph_launch(graph_exec exec, stream s) { return cudaGraphLaunch(exec, s); }
inline __attribute__((always_inline)) error graph_destroy(graph g) { return cudaGraphDestroy(g); }
inline __attribute__((always_inline)) error graph_exec_destroy(graph_exec exec) { return cudaGraphExecDestroy(exec); }
inline __attribute__((always_inline)) error launch_kernel(const void* kernel, dim3 grid, dim3 block, void** args, size_t shared, stream s) { return cudaLaunchKernel(kernel, grid, block, args, shared, s); }
inline __attribute__((always_inline)) error launch_cooperative_kernel(const void* kernel, dim3 grid, dim3 block, void** args, size_t shared, stream s) { return cudaLaunchCooperativeKernel(kernel, grid, block, args, shared, s); }
inline __attribute__((always_inline)) error occupancy(int* blocks, const void* kernel, int threads, size_t shared) { return cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks, kernel, threads, shared); }
template<class T> inline __attribute__((always_inline)) error malloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
// Managed memory is the CUDA system-coherent allocation. Require concurrent
// access: silently substituting ordinary device memory would break polling.
inline __attribute__((always_inline)) error malloc_fine_grained(void** p, size_t n) {
  *p = nullptr;
  int device = 0, concurrent = 0;
  error e = cudaGetDevice(&device);
  if (e != success) return e;
  e = cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device);
  if (e != success) return e;
  if (!concurrent) return cudaErrorNotSupported;
  return cudaMallocManaged(p, n, cudaMemAttachGlobal);
}
inline __attribute__((always_inline)) error malloc_host_coherent(void** p, size_t n) {
  return cudaHostAlloc(p, n, cudaHostAllocMapped | cudaHostAllocPortable);
}
}} // namespace etx::host_backend
