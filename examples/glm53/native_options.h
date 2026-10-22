// Compile-time native experiments. Defaults retain the original arithmetic and
// ordinary loads. These switches never change activation/state cache policy.
#pragma once
#ifndef ETX_GLM_PARALLEL
#define ETX_GLM_PARALLEL 0
#endif
#ifndef ETX_GLM_ATTENTION_TP
#define ETX_GLM_ATTENTION_TP 0
#endif
#ifndef ETX_GLM_UP_TAIL_BALANCE
#define ETX_GLM_UP_TAIL_BALANCE 0
#endif
#ifndef ETX_GLM_UP_PAIRS
#define ETX_GLM_UP_PAIRS 36
#endif
#ifndef ETX_GLM_ROUTED_SLC
#define ETX_GLM_ROUTED_SLC 0
#endif
#ifndef ETX_GLM_SHARED_SLC
#define ETX_GLM_SHARED_SLC 0
#endif
#ifndef ETX_GLM_CACHE_AB
#define ETX_GLM_CACHE_AB 0
#endif
#ifndef ETX_GLM_ROW_TILE
#define ETX_GLM_ROW_TILE 1
#endif
#ifndef ETX_GLM_K_TILE
#define ETX_GLM_K_TILE 0
#endif
#ifndef ETX_GLM_LDS_BYTES
#define ETX_GLM_LDS_BYTES 0
#endif
static_assert(ETX_GLM_ROUTED_SLC == 0 || ETX_GLM_ROUTED_SLC == 1, "ETX_GLM_ROUTED_SLC must be 0 or 1");
static_assert(ETX_GLM_SHARED_SLC == 0 || ETX_GLM_SHARED_SLC == 1, "ETX_GLM_SHARED_SLC must be 0 or 1");
static_assert(ETX_GLM_CACHE_AB == 0 || ETX_GLM_CACHE_AB == 1, "ETX_GLM_CACHE_AB must be 0 or 1");
static_assert(ETX_GLM_ROW_TILE >= 1 && ETX_GLM_ROW_TILE <= 32, "ETX_GLM_ROW_TILE must be in [1,32]");
static_assert(ETX_GLM_K_TILE >= 0 && ETX_GLM_K_TILE <= 2147483647, "ETX_GLM_K_TILE must be in [0,2147483647]");
static_assert(ETX_GLM_LDS_BYTES >= 0 && ETX_GLM_LDS_BYTES <= 65536 && ETX_GLM_LDS_BYTES % 4 == 0,
              "ETX_GLM_LDS_BYTES must be a multiple of 4 in [0,65536]");
#if defined(__HIP_DEVICE_COMPILE__) && !defined(GLM53_CPU) && (ETX_GLM_ROUTED_SLC || ETX_GLM_SHARED_SLC || ETX_GLM_CACHE_AB) && !defined(__gfx942__)
#error "GLM explicit payload loads require gfx942 (or GLM53_CPU fallback)"
#endif
// ROUTED_SLC covers only selected routed-expert gate/up/down payloads;
// SHARED_SLC covers only shared-expert SG/SU/SD payloads. FP8 scales, router,
// dense, attention, norm, head and embedding weights always use ordinary loads.
// CACHE_AB=1 makes both expert categories use matched explicit asm+wait loads:
// cached controls differ from SLC treatments ONLY by the SLC bit. CACHE_AB=0
// retains compiler-managed cached loads; SLC-only experiments without CACHE_AB
// also change load scheduling and are NOT isolated cache-bit comparisons.
// ROW_TILE is adjacent output rows per lane, not a tensor-core M/N dimension.
// K_TILE=0 retains the full K traversal; positive chunks preserve c=0..K-1
// summation order (no partial sums/reduction). LDS_BYTES is a single FP32 input
// slab per workgroup, NOT a token slab or an occupancy reservation. If input K
// exceeds capacity the call falls back to direct input reads. Account the
// compiler's TOTAL LDS including ETX's other shared storage before launch.
