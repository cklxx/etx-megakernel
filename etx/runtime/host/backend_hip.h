// Vendor implementation of the ETX host runtime interface.
// Always inline adapters to preserve the direct-call host object ABI at -O3.
#pragma once
#include <hip/hip_runtime.h>

namespace etx { namespace host_backend {
using dim3 = ::dim3;
using error = hipError_t;
using stream = hipStream_t;
using event = hipEvent_t;
using graph = hipGraph_t;
using graph_exec = hipGraphExec_t;
using device_properties = hipDeviceProp_t;
using copy_kind = hipMemcpyKind;
using capture_mode = hipStreamCaptureMode;
inline constexpr auto success = hipSuccess;
inline constexpr auto not_ready = hipErrorNotReady;
inline constexpr auto host_to_device = hipMemcpyHostToDevice;
inline constexpr auto device_to_host = hipMemcpyDeviceToHost;
inline constexpr auto device_to_device = hipMemcpyDeviceToDevice;
inline constexpr auto stream_non_blocking = hipStreamNonBlocking;
inline constexpr auto capture_global = hipStreamCaptureModeGlobal;
inline __attribute__((always_inline)) const char* error_string(error e) { return hipGetErrorString(e); }
inline __attribute__((always_inline)) error set_device(int device) { return hipSetDevice(device); }
inline __attribute__((always_inline)) error get_device_count(int* count) { return hipGetDeviceCount(count); }
inline __attribute__((always_inline)) error get_device_properties(device_properties* p, int device) { return hipGetDeviceProperties(p, device); }
inline __attribute__((always_inline)) error device_can_access_peer(int* can, int device, int peer) { return hipDeviceCanAccessPeer(can, device, peer); }
inline __attribute__((always_inline)) error enable_peer_access(int peer, unsigned flags = 0) { return hipDeviceEnablePeerAccess(peer, flags); }
inline __attribute__((always_inline)) error disable_peer_access(int peer) { return hipDeviceDisablePeerAccess(peer); }
inline __attribute__((always_inline)) error free(void* p) { return hipFree(p); }
inline __attribute__((always_inline)) error free_host(void* p) { return hipHostFree(p); }
inline __attribute__((always_inline)) error memcpy(void* dst, const void* src, size_t n, copy_kind kind) { return hipMemcpy(dst, src, n, kind); }
inline __attribute__((always_inline)) error memcpy_async(void* dst, const void* src, size_t n, copy_kind kind, stream s) { return hipMemcpyAsync(dst, src, n, kind, s); }
inline __attribute__((always_inline)) error memset(void* dst, int value, size_t n) { return hipMemset(dst, value, n); }
inline __attribute__((always_inline)) error memset_async(void* dst, int value, size_t n, stream s) { return hipMemsetAsync(dst, value, n, s); }
inline __attribute__((always_inline)) error stream_create(stream* s, unsigned flags) { return hipStreamCreateWithFlags(s, flags); }
inline __attribute__((always_inline)) error stream_destroy(stream s) { return hipStreamDestroy(s); }
inline __attribute__((always_inline)) error stream_query(stream s) { return hipStreamQuery(s); }
inline __attribute__((always_inline)) error stream_synchronize(stream s) { return hipStreamSynchronize(s); }
inline __attribute__((always_inline)) error event_create(event* e) { return hipEventCreate(e); }
inline __attribute__((always_inline)) error event_destroy(event e) { return hipEventDestroy(e); }
inline __attribute__((always_inline)) error event_record(event e, stream s) { return hipEventRecord(e, s); }
inline __attribute__((always_inline)) error event_elapsed_time(float* ms, event start, event stop) { return hipEventElapsedTime(ms, start, stop); }
inline __attribute__((always_inline)) error stream_begin_capture(stream s, capture_mode mode) { return hipStreamBeginCapture(s, mode); }
inline __attribute__((always_inline)) error stream_end_capture(stream s, graph* g) { return hipStreamEndCapture(s, g); }
inline __attribute__((always_inline)) error graph_instantiate(graph_exec* exec, graph g) { return hipGraphInstantiate(exec, g, nullptr, nullptr, 0); }
inline __attribute__((always_inline)) error graph_launch(graph_exec exec, stream s) { return hipGraphLaunch(exec, s); }
inline __attribute__((always_inline)) error graph_destroy(graph g) { return hipGraphDestroy(g); }
inline __attribute__((always_inline)) error graph_exec_destroy(graph_exec exec) { return hipGraphExecDestroy(exec); }
inline __attribute__((always_inline)) error launch_kernel(const void* kernel, dim3 grid, dim3 block, void** args, size_t shared, stream s) { return hipLaunchKernel(kernel, grid, block, args, shared, s); }
inline __attribute__((always_inline)) error launch_cooperative_kernel(const void* kernel, dim3 grid, dim3 block, void** args, size_t shared, stream s) { return hipLaunchCooperativeKernel(kernel, grid, block, args, shared, s); }
inline __attribute__((always_inline)) error occupancy(int* blocks, const void* kernel, int threads, size_t shared) { return hipOccupancyMaxActiveBlocksPerMultiprocessor(blocks, kernel, threads, shared); }
template<class T> inline __attribute__((always_inline)) error malloc(T** p, size_t n) { return hipMalloc((void**)p, n); }
// Fine-grained device memory supports uncached cross-device atomic communication.
inline __attribute__((always_inline)) error malloc_fine_grained(void** p, size_t n) { return hipExtMallocWithFlags(p, n, hipDeviceMallocFinegrained); }
inline __attribute__((always_inline)) error malloc_host_coherent(void** p, size_t n) { return hipHostMalloc(p, n, hipHostMallocCoherent | hipHostMallocMapped); }
}} // namespace etx::host_backend
