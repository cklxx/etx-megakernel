// Real HIP module API loaded on the future device host, without a fake local runtime.
// Include after hip/hip_runtime_api.h. Local --help/preflight never calls load_hip().
#pragma once
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#define ETX_DYNAMIC_HIP_APIS(X) \
 X(hipInit) X(hipGetDeviceCount) X(hipDeviceGetAttribute) X(hipGetDeviceProperties) X(hipSetDevice) \
 X(hipFree) X(hipMemcpy) X(hipMemcpyAsync) X(hipMemset) X(hipMemsetAsync) \
 X(hipStreamCreate) X(hipStreamCreateWithFlags) X(hipStreamQuery) X(hipStreamSynchronize) X(hipStreamDestroy) X(hipStreamWaitEvent) \
 X(hipEventCreate) X(hipEventCreateWithFlags) X(hipEventRecord) X(hipEventSynchronize) X(hipEventQuery) \
 X(hipEventElapsedTime) X(hipEventDestroy) X(hipDeviceSynchronize) \
 X(hipModuleLoad) X(hipModuleGetFunction) X(hipModuleLaunchKernel) X(hipModuleUnload) X(hipGetErrorString)
namespace etx_dynamic {
inline hipError_t (*p_hipMalloc)(void**,size_t)=nullptr;
#define ETX_HIP_FIELD(name) inline decltype(&::name) p_##name=nullptr;
ETX_DYNAMIC_HIP_APIS(ETX_HIP_FIELD)
#undef ETX_HIP_FIELD
inline void load_hip() {
  static void* library=nullptr;
  if(library)return;
  library=dlopen("libamdhip64.so",RTLD_NOW|RTLD_LOCAL);
  if(!library)throw std::runtime_error(std::string("HIP runtime unavailable: ")+dlerror());
  p_hipMalloc=reinterpret_cast<decltype(p_hipMalloc)>(dlsym(library,"hipMalloc"));
  if(!p_hipMalloc)throw std::runtime_error("missing HIP symbol: hipMalloc");
#define ETX_HIP_STRING_INNER(name) #name
#define ETX_HIP_STRING(name) ETX_HIP_STRING_INNER(name)
#define ETX_HIP_LOAD(name) p_##name=reinterpret_cast<decltype(p_##name)>(dlsym(library,ETX_HIP_STRING(name))); if(!p_##name)throw std::runtime_error("missing HIP symbol: " #name);
  ETX_DYNAMIC_HIP_APIS(ETX_HIP_LOAD)
#undef ETX_HIP_LOAD
}
}
inline void load_hip(){etx_dynamic::load_hip();}
#define hipInit etx_dynamic::p_hipInit
#define hipGetDeviceCount etx_dynamic::p_hipGetDeviceCount
#define hipDeviceGetAttribute etx_dynamic::p_hipDeviceGetAttribute
#undef hipGetDeviceProperties
#define hipGetDeviceProperties etx_dynamic::p_hipGetDeviceProperties
#define hipSetDevice etx_dynamic::p_hipSetDevice
#define hipMalloc etx_dynamic::p_hipMalloc
#define hipFree etx_dynamic::p_hipFree
#define hipMemcpy etx_dynamic::p_hipMemcpy
#define hipMemcpyAsync etx_dynamic::p_hipMemcpyAsync
#define hipMemset etx_dynamic::p_hipMemset
#define hipMemsetAsync etx_dynamic::p_hipMemsetAsync
#define hipStreamCreate etx_dynamic::p_hipStreamCreate
#define hipStreamCreateWithFlags etx_dynamic::p_hipStreamCreateWithFlags
#define hipStreamQuery etx_dynamic::p_hipStreamQuery
#define hipStreamSynchronize etx_dynamic::p_hipStreamSynchronize
#define hipStreamDestroy etx_dynamic::p_hipStreamDestroy
#define hipStreamWaitEvent etx_dynamic::p_hipStreamWaitEvent
#define hipEventCreate etx_dynamic::p_hipEventCreate
#define hipEventCreateWithFlags etx_dynamic::p_hipEventCreateWithFlags
#define hipEventRecord etx_dynamic::p_hipEventRecord
#define hipEventSynchronize etx_dynamic::p_hipEventSynchronize
#define hipEventQuery etx_dynamic::p_hipEventQuery
#define hipEventElapsedTime etx_dynamic::p_hipEventElapsedTime
#define hipEventDestroy etx_dynamic::p_hipEventDestroy
#define hipDeviceSynchronize etx_dynamic::p_hipDeviceSynchronize
#define hipModuleLoad etx_dynamic::p_hipModuleLoad
#define hipModuleGetFunction etx_dynamic::p_hipModuleGetFunction
#define hipModuleLaunchKernel etx_dynamic::p_hipModuleLaunchKernel
#define hipModuleUnload etx_dynamic::p_hipModuleUnload
#define hipGetErrorString etx_dynamic::p_hipGetErrorString
