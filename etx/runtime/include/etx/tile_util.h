// Small helpers for hand-written tile bodies (link mode). Nothing here touches
// events or queues; tiles only compute. Wave size comes from the lowering header.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "etx/abi.h"

#ifndef ETX_WAVE_SIZE
#define ETX_WAVE_SIZE 64
#endif

typedef __hip_bfloat16 etx_bf16;

static __device__ __forceinline__ float etx_b2f(etx_bf16 v) { return __bfloat162float(v); }
static __device__ __forceinline__ int etx_lane() { return threadIdx.x % ETX_WAVE_SIZE; }
static __device__ __forceinline__ int etx_wave() { return threadIdx.x / ETX_WAVE_SIZE; }
static __device__ __forceinline__ int etx_nwaves() { return blockDim.x / ETX_WAVE_SIZE; }

static __device__ __forceinline__ float etx_wave_sum(float v) {
  for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, ETX_WAVE_SIZE);
  return v;
}
static __device__ __forceinline__ float etx_wave_max(float v) {
  for (int off = ETX_WAVE_SIZE / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, ETX_WAVE_SIZE));
  return v;
}
// block-wide reductions; `red` is a scratch array of >= nwaves floats in LDS; result in every thread
static __device__ __forceinline__ float etx_block_sum(float v, float* red) {
  v = etx_wave_sum(v);
  if (etx_lane() == 0) red[etx_wave()] = v;
  __syncthreads();
  float t = 0.f;
  for (int w = 0; w < etx_nwaves(); ++w) t += red[w];
  __syncthreads();
  return t;
}
static __device__ __forceinline__ float etx_block_max(float v, float* red) {
  v = etx_wave_max(v);
  if (etx_lane() == 0) red[etx_wave()] = v;
  __syncthreads();
  float t = red[0];
  for (int w = 1; w < etx_nwaves(); ++w) t = fmaxf(t, red[w]);
  __syncthreads();
  return t;
}
// wave-cooperative dot product: bf16 weight row (n elements) with an fp32 vector
static __device__ __forceinline__ float etx_wave_dot(const etx_bf16* __restrict__ w, const float* __restrict__ x, int n) {
  float acc = 0.f;
  for (int i = etx_lane(); i < n; i += ETX_WAVE_SIZE) acc += etx_b2f(w[i]) * x[i];
  return etx_wave_sum(acc);
}
// Pass 7 hook helper: touch `bytes` of memory so the lines land in L2 (l2_warm method).
// One 4-byte load per 128-byte line, spread over the workgroup; the asm keeps the loads alive.
static __device__ __forceinline__ void etx_touch(const void* p, size_t bytes) {
  const char* base = (const char*)p;
  const size_t lines = bytes / 128;
  int acc = 0;
  for (size_t l = threadIdx.x; l < lines; l += blockDim.x) acc += *(const volatile int*)(base + l * 128);
  asm volatile("" :: "v"(acc));
}
