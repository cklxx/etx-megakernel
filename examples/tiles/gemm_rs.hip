// Tile bodies for examples/gemm_rs.py. Stubs; see moe.hip.
#include <hip/hip_runtime.h>
#include "etx/abi.h"

extern "C" __device__ void etx_tile_matmul(const etx_ctx* c) { (void)c; }
extern "C" __device__ void etx_tile_reduce_scatter(const etx_ctx* c) { (void)c; }
