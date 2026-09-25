// The ETX tiles of examples/vllm_llm that are not vLLM kernels: what vLLM does with torch ops or on the host.
//   vl_prologue  one step's inputs: the token's embedding row (= the layer-0 residual) and the scalars
//                vLLM's attention metadata carries (positions, seq_lens, slot_mapping), from the step's pos
//   vl_argmax    greedy sampling over the bf16 logits (first maximum, like torch.argmax on float32 logits)
// Both take the step from __constant__ VlParams (host.hip updates it before each step).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "etx/abi.h"

struct VlParams {
  int32_t tok, pos, write_next, H, V, pad;
  const __hip_bfloat16* embed;         // [V][H]
  __hip_bfloat16* resid;               // [H] embedding row -> residual stream
  int64_t* positions;                  // [1]
  int32_t* seq_lens;                   // [1]
  int64_t* slot_mapping;               // [1]
  const __hip_bfloat16* logits;        // [V]
  int32_t* token_in;                   // [steps + 2]
  int32_t* token_out;                  // [steps + 1]
  uint64_t* token_time;                // [2 * (steps + 1)] start / end, 100 MHz ticks
};
__constant__ VlParams VP;
extern "C" int vl_params_bytes() { return (int)sizeof(VlParams); }

extern "C" __device__ __attribute__((always_inline)) void vl_prologue(const etx_ctx* c) {
  (void)c;
  const int t = VP.tok;
  if (threadIdx.x == 0) {
    VP.token_time[2 * t] = wall_clock64();
    VP.positions[0] = VP.pos; VP.seq_lens[0] = VP.pos + 1; VP.slot_mapping[0] = VP.pos;   // block table is the identity
  }
  const int id = VP.token_in[t];
  const uint4* src = (const uint4*)(VP.embed + (size_t)id * VP.H);
  uint4* dst = (uint4*)VP.resid;
  for (int i = threadIdx.x; i < VP.H / 8; i += blockDim.x) dst[i] = src[i];
}

extern "C" __device__ __attribute__((always_inline)) void vl_argmax(const etx_ctx* c) {
  float* sv = (float*)c->lds; int* si = (int*)(sv + blockDim.x);
  float bv = -INFINITY; int bi = 0x7fffffff;
  const unsigned short* lg = (const unsigned short*)VP.logits;
  for (int i = threadIdx.x; i < VP.V; i += blockDim.x) {       // ascending per thread: the first maximum wins
    const float v = __uint_as_float((unsigned)lg[i] << 16);
    if (v > bv) { bv = v; bi = i; }
  }
  sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) {
      const float ov = sv[threadIdx.x + s]; const int oi = si[threadIdx.x + s];
      if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int t = VP.tok;
    VP.token_out[t] = si[0];
    if (VP.write_next) VP.token_in[t + 1] = si[0];
    VP.token_time[2 * t + 1] = wall_clock64();
  }
}

// host side (compiled in the megakernel's translation unit, called from host.hip)
void vl_set_params(const VlParams* p, hipStream_t s) {
  if (hipMemcpyToSymbolAsync(HIP_SYMBOL(VP), p, sizeof(VlParams), 0, hipMemcpyHostToDevice, s) != hipSuccess) abort();
}
