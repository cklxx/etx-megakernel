// Tile bodies for examples/moe_layer.py. STUBS with the right ABI so the link
// step and the persistent loop can be exercised; replace with CK / hand-written
// tiles (link mode) when bringing up on hardware.
#include <hip/hip_runtime.h>
#include "etx/abi.h"

#define ETX_STUB(name) extern "C" __device__ void name(const etx_ctx* c) { (void)c; }

ETX_STUB(etx_tile_rmsnorm)
ETX_STUB(etx_tile_gemv_qkv)
ETX_STUB(etx_tile_attention)
ETX_STUB(etx_tile_gemv_o)
ETX_STUB(etx_tile_router)
ETX_STUB(etx_tile_gather)
ETX_STUB(etx_tile_group_gemm_up)
ETX_STUB(etx_tile_group_gemm_down)
ETX_STUB(etx_tile_combine)

// grouping is the runtime initialiser of E_expert / E_down: it must write the
// counters *before* its arrive on E_group (release ordering is added by the
// generated code; the writes themselves are the tile's job).
extern "C" __device__ void etx_tile_grouping(const etx_ctx* c) {
  // args: topk (B,K) int32, indptr (NE+1) int32, tile_expert (n_gg,1) int32
  // events: E_expert at ev_offset[6], E_down at ev_offset[8] (see plan.json)
  (void)c;
}
