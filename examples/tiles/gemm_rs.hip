// Tile bodies for examples/gemm_rs.py. Reference implementations (correct,
// not tuned): a 128x128 bf16 GEMM tile with fp32 accumulation staged through
// LDS, and owner-side reduce-scatter tiles that read the peers' C over xGMI.
#include "etx/tile_util.h"

#define BM 128
#define BN 128
#define BK 32
#define WORLD 2

// C[i*BM.., j*BN..] = A[i*BM.., :] * Bw[:, j*BN..]   (A: M x K row-major bf16, Bw: K x N row-major bf16, C fp32)
extern "C" __device__ void etx_tile_matmul(const etx_ctx* c) {
  const int i = c->coord[0], j = c->coord[1];
  const int M = c->shape[0], K = c->shape[1], N = c->shape[2];   // symbols in graph order: M, K, N
  (void)M;
  const etx_bf16* A = (const etx_bf16*)c->args[0];
  const etx_bf16* B = (const etx_bf16*)c->args[1];
  float* C = (float*)c->args[2];
  float* As = (float*)c->lds;                 // BM x BK
  float* Bs = As + BM * BK;                   // BK x BN
  // each thread owns an 8x8 sub-tile: 256 threads = 16 x 16 thread grid over 128 x 128
  const int tr = (threadIdx.x / 16) * 8, tc = (threadIdx.x % 16) * 8;
  float acc[8][8];
  for (int a = 0; a < 8; ++a) for (int b = 0; b < 8; ++b) acc[a][b] = 0.f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
      const int r = idx / BK, kk = idx % BK;
      As[idx] = etx_b2f(A[(size_t)(i * BM + r) * K + k0 + kk]);
    }
    for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
      const int kk = idx / BN, cc = idx % BN;
      Bs[idx] = etx_b2f(B[(size_t)(k0 + kk) * N + j * BN + cc]);
    }
    __syncthreads();
    for (int kk = 0; kk < BK; ++kk) {
      float av[8], bv[8];
      for (int a = 0; a < 8; ++a) av[a] = As[(tr + a) * BK + kk];
      for (int b = 0; b < 8; ++b) bv[b] = Bs[kk * BN + tc + b];
      for (int a = 0; a < 8; ++a) for (int b = 0; b < 8; ++b) acc[a][b] += av[a] * bv[b];
    }
    __syncthreads();
  }
  for (int a = 0; a < 8; ++a) for (int b = 0; b < 8; ++b) C[(size_t)(i * BM + tr + a) * N + j * BN + tc + b] = acc[a][b];
}

// D_r[i*BM.., j*BN..] = sum_rank C_rank[(i + r*rows_local)*BM.., j*BN..]; C of other ranks is read over peer access
static __device__ __forceinline__ void reduce_scatter(const etx_ctx* c, int rank) {
  const int i = c->coord[0], j = c->coord[1];
  const int M = c->shape[0], N = c->shape[2];
  const int rows_local = (M / BM) / WORLD;
  const float* C0 = (const float*)c->args[0];
  const float* C1 = (const float*)c->args[1];
  float* D = (float*)c->args[2];
  const int gi = i + rank * rows_local;                 // global tile row
  for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
    const int r = idx / BN, cc = idx % BN;
    const size_t src = (size_t)(gi * BM + r) * N + j * BN + cc;
    D[(size_t)(i * BM + r) * N + j * BN + cc] = C0[src] + C1[src];
  }
}
extern "C" __device__ void etx_tile_reduce_scatter_r0(const etx_ctx* c) { reduce_scatter(c, 0); }
extern "C" __device__ void etx_tile_reduce_scatter_r1(const etx_ctx* c) { reduce_scatter(c, 1); }
