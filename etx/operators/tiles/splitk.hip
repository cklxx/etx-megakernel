// Tile bodies for examples/splitk_sum.py (link mode). Reference implementation,
// not tuned: the point is the ABI, not the throughput.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "etx/abi.h"

extern "C" __device__ void etx_tile_partial_sum(const etx_ctx* c) {
  const int i = c->coord[0], j = c->coord[1];
  const __hip_bfloat16* A = (const __hip_bfloat16*)c->args[0];
  float* B = (float*)c->args[1];
  const int rows = c->shape[0] * 32;          // n*32
  for (int r = threadIdx.x; r < 32; r += blockDim.x) {
    const int row = i * 32 + r;
    if (row >= rows) break;
    float acc = 0.f;
    for (int k = j * 32; k < j * 32 + 32; ++k) acc += (float)A[row * 128 + k];
    B[row * 4 + j] = acc;
  }
}

extern "C" __device__ void etx_tile_final_sum(const etx_ctx* c) {
  const int i = c->coord[0];
  const float* B = (const float*)c->args[0];
  float* C = (float*)c->args[1];
  for (int r = threadIdx.x; r < 32; r += blockDim.x) {
    const int row = i * 32 + r;
    C[row] = B[row * 4 + 0] + B[row * 4 + 1] + B[row * 4 + 2] + B[row * 4 + 3];
  }
}
