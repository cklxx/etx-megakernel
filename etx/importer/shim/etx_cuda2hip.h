// The renames hipify applies to vLLM's CUDA-named ROCm sources when vLLM is built for ROCm (only the ones
// its kernels and headers use); included first by hip_ir_local.sh so the sources compile unmodified.
#pragma once
#include <cfloat>   // torch headers bring these in for the real build
#include <climits>
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
// torch's scalar types (header-only), which the libtorch_stable kernels are instantiated with
#include <hip/hip_runtime.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>
