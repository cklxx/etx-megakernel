# Bare-metal validation of the 2026-10-01 work

Hot Aisle `enc1-gpuvm005`, bare metal MI300X (2 GPUs, GPU 0 used), ROCm 7.2.4, 26 cores. Qwen3-30B-A3B bf16,
greedy, batch 1, context 1024, the same protocol as every other number in this session. Run 2026-09-30 23:28
to 23:55 UTC. Every ETX run matched 32/32 tokens against HF greedy; the first also passed `--check-dsl`
(0 values beyond 2 bf16 ulps at layer 47). Both vLLM runs reported `first_divergence=32/32`.

Purpose: today's entire result was measured on a *virtualised* MI300X VF with ROCm 7.14, where our build
compiles differently. This box is the machine type every historical number in the project came from.

## Build

Best configuration: `ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head
ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_COH=1 ETX_VL_FOLD_TOPK=0 ETX_VL_CU=256` (`ETX_VL_NT` default-on,
import prune in build.sh).

| | bare metal (ROCm 7.2.4) | VF box (ROCm 7.14) |
|---|---|---|
| tasks / events | 60362 / 436 | 60362 / 436 |
| megakernel VGPR / spill / scratch | 128 / **9** / 32 B | 128 / **15** / — |
| megakernel LDS | 29736 | 29736 |
| unfused VGPR / spill | 110 / 0 | 104 / 0 |

The plan is identical; only the register allocator differs. The import prune (round 16) works here too: LDS
29736, not 41448.

## Measured

ETX fused and unfused interleaved, three iterations of each, then one fused drift check at the end. vLLM via
`rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0` (vLLM 0.27.1.dev, the same 0.27.x line the
VF box used; `rocm/vllm:latest` no longer exists on Docker Hub), `examples/llm/vllm_bench.py`, three internal
repeats per invocation.

| | runs (ms/token) | median | spread |
|---|---|---|---|
| ETX fused | 5.039, 5.045, 5.039, 5.068 (drift check) | **5.042** | 0.029 |
| ETX unfused | 5.268, 5.262, 5.266 | **5.266** | 0.006 |
| vLLM default | 4.893, 4.887, 4.877 | **4.887** | 0.016 |
| vLLM aiter | 4.934, 4.928, 4.922 | 4.928 | 0.012 |

- **Fusion wins by 0.224 ms (4.3%)** — the same margin as the VF box's +0.219.
- **ETX is 3.1% behind vLLM default** (5.042 vs 4.887). aiter is slower than default here, as on every box.
- The first three fused runs spread 0.006; the drift check 12 minutes later is 0.029 higher, so a session-long
  drift of that order exists on this box too. Comparisons within a few minutes are good to ~0.01.

## Against the historical bare-metal baseline (2026-09-30, same machine type)

| | then | now | delta |
|---|---|---|---|
| ETX fused | 5.895 | 5.042 | **−0.853 (−14.5%)** |
| ETX unfused | 5.660 | 5.266 | −0.394 |
| fusion margin | **−0.235** (fusing was slower) | **+0.224** | flipped |
| vLLM default | 4.836 | 4.887 | +0.051 (different vLLM build) |

Today's work reproduces on bare metal at full size. The fusion-penalty flip — the claim ETX exists to prove —
holds on the machine type where it was originally measured as negative.

## Cross-box comparison

| | bare metal | VF box |
|---|---|---|
| ETX fused | 5.042 | 4.885–4.917 |
| vLLM default | 4.887 | 4.861 |
| ratio ETX/vLLM | 1.032 | 1.005–1.012 |

Both are faster on the VF box, ETX by more than vLLM, so our relative position is slightly worse here. The
absolute numbers are not interchangeable between the boxes; the ratio and the fusion margin are.

## Microbenchmarks reproduce (the extra measurement)

Chosen because the whole budget in `docs/BUDGET-2026-10-01.md` uses the VF box's achievable-rate table, and if
bare metal differed, every residual in it would move. It does not:

| | bare metal | VF box |
|---|---|---|
| `restart` 28 KB/WG, no inv (warm span) | 0.82 us | 0.76 |
| `restart` 28 KB/WG, `buffer_inv sc1` (warm) | 3.16 us | 3.14 |
| `restart` 312 KB/WG, no inv vs inv sc1 | 30.36 / 30.44 | 29.25 / 30.48 |
| `stream` o_proj 28 KB x 296 | 7.07 us | 7.14 |
| `stream` gate_up 312 KB x 296 | 34.57 us | 38.21 |
| `stream` gate_up 312 KB x 192 | 19.87 us | 19.03 |
| `stream` 1 MB x 296 contig / rows | 92.87 / 90.15 | 91.32 / 87.93 |

The `buffer_inv sc1` penalty on an L2-resident phase — 4.1x, the finding the whole COH redesign rests on — is
identical on both machines. The achievable-rate table transfers, so the budget's residuals stand.

## Note

`rocm/vllm:latest` has been removed from Docker Hub; the session scripts that reference it
(`examples/sessions/moe_fix.py`, and the pattern copied into later sessions) will fail on a fresh box. The tag
used here is the current 0.27.x CDNA image.

## The full machine-parameter suite on bare metal (added after the validation run)

`hw_bench` was run end to end on the bare-metal box before it was released — `lat`, `launch`, `evx`, `lfx`,
`cohload`, `moegather`, `sdat`, `lf`, 156 RESULT lines in `results_2026-10-01/baremetal/hw_all.log` — so the
machine parameters the whole budget rests on now exist on both machines. They agree:

| parameter | bare metal (ROCm 7.2.4) | VF box (ROCm 7.14) |
|---|---|---|
| dependent hop, L1, loaded | 104.3 ns | 103.3 ns |
| **dependent hop, L1, after `buffer_inv sc1`, loaded** | **1106.9 ns** | **1174.8 ns** |
| kernel boundary (`launch`, empty) | 1.61 us | 1.61 us (2026-09-28) |
| activation read: inv + plain | 2.91 us | 2.89 us |
| activation read: coherent | 0.55 us | 0.50 us |
| `moegather` sweep / gather / gather+ (WG median) | 11.04 / 14.20 / 11.88 us | 10.96 / 14.08 / 11.96 us |
| `lfx` 8 XCDs (WG median) | 19.76 us | 19.80 us |
| `lfx` 1 XCD, per XCD | 0.623 TB/s | 0.687 TB/s |
| `evx` device scope, P=8 C=8 | 2.61 us | — |

**The strongest evidence for the COH redesign is this suite's `lat` row, and it was not measured before today.**
A dependent load chain that crosses a `buffer_inv sc1` pays **1107 ns per hop against 103 ns without it — 10.6x on
bare metal, 11.4x on the VF box.** The 4.1x figure the redesign was argued from (`restart`, an L2-resident phase
losing its reuse) is the bandwidth side of the same mechanism; this is the latency side, it is larger, and it bears
directly on the phases that are dependent chains rather than streams (`prep` 8.5 hops, `reduce` 8.1, `topk` 11.8).
Both numbers are machine-independent, so neither is an artefact of the virtualised box the day's work was done on.

`rocprofv3` could not be used here: it aborted at startup (SIGABRT, `results_2026-10-01/baremetal/prof_bm.log`),
a different failure from the VF box where it writes its database and then segfaults at teardown. The occupancy
figures in `docs/BUDGET-2026-10-01.md` therefore remain VF-only; the VGPR 128 that sets them is 128 on both boxes,
so the 4 waves/SIMD conclusion carries, but it is derived here rather than measured.

## Timing basis (added after the vLLM ladder)

The ETX and vLLM figures in this document are **not the same basis**, the same mismatch
`docs/VLLM-LADDER-2026-10-01.md` found on the VF box: ETX's medians are the device span, vLLM's come from
`vllm_bench.py`'s host wall clock. On the VF box the megakernel carried 0.197 ms/step of host work the device
timer does not see, which turned "0.8% behind vLLM" into "5.0% behind" and the fusion margin from 3.9% into 0.8%.

**The bare-metal run did not record ETX's wall per step**, and the box was released before this was understood, so
the correction cannot be applied here. Treat the bare-metal "3.1% behind vLLM" as a device-basis figure that is
optimistic by an unknown amount of the same kind, and the "+0.224 fusion margin" likewise. What does **not** depend
on the basis, because both sides are ETX device spans measured the same way: **5.895 -> 5.042 (-14.5%) against the
2026-09-30 build on this machine type, and the fusion margin changing sign** (-0.235 -> +0.224). Those are the
claims this validation supports.
