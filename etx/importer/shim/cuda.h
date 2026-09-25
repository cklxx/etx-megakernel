#pragma once
#include <hip/hip_runtime.h>
#ifndef CUDA_VERSION
#define CUDA_VERSION 0   /* hipify leaves CUDA_VERSION undefined on ROCm: 0 keeps the CUDA-12.9-only paths off */
#endif
