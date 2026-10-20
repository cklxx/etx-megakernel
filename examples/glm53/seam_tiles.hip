// Production-shaped synthetic seam only: no GLM layer arithmetic.
#pragma once
#include <stdint.h>
#include "etx/abi.h"

extern "C" __device__ __attribute__((always_inline)) void glm53_seam_send(const etx_ctx* c) {
  const int epoch = *(const int32_t*)c->args[0];
  uint16_t* dst = (uint16_t*)c->args[1];
  const int control = *(const int32_t*)c->args[2];
  const int base = c->coord[0] * 16384;
  // Direct producer output, like the layer: no host payload or staging memcpy.
  // One whole 256-thread workgroup per request, not B*64 tasks.
  for (int j = threadIdx.x; j < 16384; j += 256)
    if (!control || j == 0 || j == 16383) dst[base+j] = (base+j+17*epoch) % 0x7F00;
}

extern "C" __device__ __attribute__((always_inline)) void glm53_seam_consume(const etx_ctx* c) {
  const uint16_t* src = (const uint16_t*)c->args[0];
  uint16_t* dst = (uint16_t*)c->args[1];
  int32_t* token = (int32_t*)c->args[2];
  const int control = *(const int32_t*)c->args[3];
  const int row = c->coord[0], base = row * 16384;
  for (int j = threadIdx.x; j < 16384; j += 256)
    if (!control || j == 0 || j == 16383) dst[base+j] = src[base+j];
  __syncthreads();
  // Returned integer is computed only after this row's ready event and copy.
  if (!threadIdx.x) token[row] = (int32_t)dst[base] + (int32_t)dst[base+16383];
}

extern "C" __device__ __attribute__((always_inline)) void glm53_seam_feedback(const etx_ctx* c) {
  if (!threadIdx.x) ((int32_t*)c->args[1])[c->coord[0]] = ((const int32_t*)c->args[0])[c->coord[0]];
}
