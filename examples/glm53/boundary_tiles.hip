#pragma once
#include <stdint.h>
#include "etx/abi.h"

extern "C" __device__ __attribute__((always_inline)) void glm53_send_state(const etx_ctx* c) {
  const uint16_t* src = (const uint16_t*)c->args[0];
  uint16_t* dst = (uint16_t*)c->args[1];
  const int i = c->coord[0] * 256 + threadIdx.x;
  dst[i] = src[i];
}
extern "C" __device__ __attribute__((always_inline)) void glm53_receive_state(const etx_ctx* c) {
  const uint16_t* src = (const uint16_t*)c->args[0];
  uint16_t* dst = (uint16_t*)c->args[1];
  const int i = c->coord[0] * 256 + threadIdx.x;
  dst[i] = src[i];
}
extern "C" __device__ __attribute__((always_inline)) void glm53_send_token(const etx_ctx* c) {
  if (threadIdx.x) return;
  const uint16_t* state = (const uint16_t*)c->args[0];
  int32_t* token = (int32_t*)c->args[1];
  const int r = c->coord[0];
  token[r] = (int32_t)state[r * 16384] + (int32_t)state[(r+1) * 16384 - 1];
}
extern "C" __device__ __attribute__((always_inline)) void glm53_receive_token(const etx_ctx* c) {
  if (threadIdx.x) return;
  ((int32_t*)c->args[1])[c->coord[0]] = ((const int32_t*)c->args[0])[c->coord[0]];
}
