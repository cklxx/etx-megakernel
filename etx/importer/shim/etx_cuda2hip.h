// The renames hipify applies to vLLM's CUDA-named ROCm sources when vLLM is built for ROCm (only the ones
// its kernels and headers use); included first by hip_ir_local.sh so the sources compile unmodified.
#pragma once
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaStream_t hipStream_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
// torch host-side checks that survive in shared headers (host code only; never reached by the kernels)
#ifndef TORCH_CHECK
#define TORCH_CHECK(...) ((void)0)
#endif
// torch scalar tags used as template keys by the kernels' type maps (never instantiated on the device)
namespace c10 { struct Half; struct BFloat16; }
