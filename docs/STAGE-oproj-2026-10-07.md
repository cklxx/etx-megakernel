# Staging o_proj weights into LDS across the attention chain, 2026-10-07

## Scope and result

This report prices one lever on the CPU. No GPU ran for this report. No kernel code changes.
The tool is `etx/tools/stage_sim.py`. Its tests are `tests/test_stage_sim.py`.

Conditions: Qwen3-30B-A3B bf16 decode, batch one, one MI300X, `ETX_VL_CU=256`, DSL tiles.

| Path | Staged bytes per task | Extra VGPR | Predicted gain, ms per token | Verdict |
|---|---:|---:|---:|---|
| `buffer_load ... lds` into free LDS | 32,768 of 65,536 | 0 | 0.117 to 0.171 (2.4% to 3.5% of 4.823) | above the 0.030 KEEP threshold |
| Hold the rows in VGPRs | 32,768 of 65,536 | +8 | 0 | infeasible: the megakernel is at the 128-VGPR limit |

These values are predictions, not measurements.
The low end is above the 0.028 ms same-binary spread.

## Why this lever

The [latency report](LATENCY-2026-10-07.md) shows prep, attn and reduce on the critical path for 866.64 us.
During this chain only 3.45 to 17.48 tasks run at a time. The 256 o_proj workers wait.
The o_proj weights do not depend on the activation. Each worker knows its slice before the chain starts.

The lever: before a worker polls its o_proj event, it issues `buffer_load ... lds` for the first part of its slice.
gfx942 supports this load at 32 bits per lane with no VGPR destination (`etx/machine/arch/gfx942.yaml`).
The o_proj body then reads the staged part from LDS and streams only the rest.

This is a schedule change that only a megakernel can make. A kernel boundary releases the LDS.

## Prior results that constrain it

1. fleet-mi300x `docs/STATUS.md` line 30 (commit 85b087e): prefetch of the next task's rows across the event wait,
   held in VGPRs (8 x 16 B per thread), was 4% slower (3.812 against 3.673 ms). Register pressure caused the loss.
   The sim reproduces this blocker for the VGPR path. The LDS path needs no VGPRs.
2. fleet-mi300x `docs/STATUS.md` line 72: idle-worker prefetch into the Infinity Cache gave no gain.
   This lever does not warm a cache level. It removes the staged loads from the o_proj body.
3. [TILES](TILES-2026-10-07.md): L2 prefetch is a measured loss on this path. This work does not repeat it.

## Model

Per layer, for the critical o_proj task:

1. Slice: 805,306,368 B / 12,288 tasks = 65,536 B (8 rows x 4096 x 2 B).
2. Free LDS: 65,536 - 29,736 = 35,800 B. The 29,736 B group segment is the dsl control build after the import prune,
   on the VF and on bare metal ([VALIDATION](VALIDATION-baremetal-2026-10-01.md)). The 41,448 B value belongs to the
   all-imported build. Staged: 8 whole workgroup steps of 4,096 B = 32,768 B (50%). The rest streams as today.
   At `ETX_VL_CU=304` the slice is 57,344 B (7 rows) and the same 32,768 B is 57%; the timing inputs exist only for 256.
3. Streaming part of the body: critical body 5.613 us minus the 0.506 us empty-task body = 5.107 us.
4. Hidden time: scenario A scales the streaming part by the staged share (2.554 us).
   Scenario B uses the measured 8.9 KB/us per-workgroup rate at 296 workgroups (3.682 us).
5. Added cost: LDS reads of the staged bytes at 128 B per clock and 2.1 GHz (0.122 us, assumed peak values).
   The visibility wait before the first LDS read is omitted.
6. Window check: staging takes 3.68 us; the prep + attn + reduce critical body is 18.05 us per layer.

Gain = 48 x (hidden - added cost) = 0.117 to 0.171 ms per token.

## Sensitivity to the group segment

The group segment of today's HEAD build is not verified. The gain depends on it:

| LDS used, B | Staged, B | Gain, ms per token | Verdict |
|---:|---:|---:|---|
| 29,736 (dsl build, measured 2026-10-01) | 32,768 | 0.117 to 0.171 | above threshold |
| 33,792 | 28,672 | 0.102 to 0.150 | above threshold |
| 41,448 (all-imported build) | 20,480 | 0.073 to 0.107 | above threshold |
| 49,152 | 16,384 | 0.058 to 0.085 | above threshold |
| 57,344 | 8,192 | 0.029 to 0.043 | marginal |
| 61,440 | 4,096 | 0.015 to 0.021 | below threshold |

## Unknowns, in the order to resolve them

1. **Local, no GPU:** the current `.amdhsa_group_segment_fixed_size` in `mk.s` from `rocm_build_check.sh`.
   If it exceeds about 57 KB, stop here.
2. **Microbenchmark:** the `buffer_load ... lds` rate per workgroup with 256 workgroups and a co-running attention phase.
3. **Trace:** the split of the o_proj body between weight streaming, activation wait, reduction and publication.
   The sim assumes that every non-fixed microsecond is streaming. The 2026-10-01 counters show 86% of wave cycles in a wait,
   so part of the body can be latency. Staging removes that latency for the staged rows too, but the split is not measured.
4. KV interference: staging loads run during attn. Use non-temporal loads so that the MALL-resident KV cache stays.
5. Critical path: whether a shorter o_proj body moves router's first ready time one for one.

## Go / no-go

Go to a flag-gated implementation (`ETX_DSL_OSTAGE`, off by default) only if unknown 1 leaves at least 16 KB free.
The GPU A/B then uses the [TILES](TILES-2026-10-07.md) verdict rules: KEEP at a mean gain at or below -0.030 ms,
bit-identical output, 32/32 tokens, `CHECK-DSL OK`.
If the A/B is NEUTRAL, record the trace split from unknown 3 and close the lever.

## Next candidate

Router tasks read 4 KB each and the router body is 6.7 us. The whole router slice fits in free LDS.
The router body is dominated by the folded norm, which `ETX_DSL_RBATCH` targets. Price it after that A/B.

## Reproduction

    .venv/bin/python -m etx.tools.stage_sim --json build/stage_sim.json
    .venv/bin/python -m pytest -q tests/test_stage_sim.py
