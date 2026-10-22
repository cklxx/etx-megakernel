# GLM phased execution: event locality, 2026-10-22

## Status

**Measured candidate execution, not a correctness-qualified model baseline.**
Full-model independent HF correctness is not established. No vLLM comparison or
same-kernel HIP-graph comparison was measured in this work.

The owner has **stopped the box**; it is unreachable. No SSH or GPU work is
permitted. Current work is code editing and analysis of archived logs only.
The measurements below precede the owner's repeat runs. The owner separately
reported PP2 **28.54 ms** versus PP8 **28.05 ms**; those raw repeat logs, timing
conditions and spread are not in this archive. Do not mix those numbers with the
instrumented single-run table below or claim statistical scaling from them.
See [FFN TP/EP implementation and log analysis](FFN-PARALLEL-2026-10-22.md).

## Conditions and timing boundary

- Host `162.243.244.52`, MI325X, gfx942; same box throughout these measurements.
- Full 45-layer GLM model, native block-scaled FP8 weights, batch 1.
- 1024 prompt tokens processed one position at a time, then generation: 1055
  executed model steps produce 32 greedy outputs. This is not a chunked-prefill
  throughput benchmark.
- PP2 preserves the staged 24/21-layer split. PP4 and PP8 assign contiguous
  whole-layer blocks; embedding is on rank 0 and the head on the last rank.
- Each arm ran alone; `ETX_TRACE=1`, task-execution checks enabled, a detailed
  task trace exported at position 4. These are instrumented measurements.
- Reported milliseconds are **host wall time from launch submission through all
  streams retiring**, before completion checks and output D2H. Event resets,
  step upload, logits copies and file output are excluded. Therefore this is not
  total application request latency.
- The decode window below is positions **1023–1054**, all 32 generated logits.

## Measured results: separate local and SYSTEM event buffers

| GPUs / pipeline | All 1055 steps, median ms | Early positions 1–100, median ms | Last 100, median ms | Decode 32, median ms | Decode 32, mean ms |
|---|---:|---:|---:|---:|---:|
| PP2 | 29.455 | 28.376 | 30.567 | 30.690 | 30.999 |
| PP4 | 30.547 | 29.463 | 31.705 | 31.653 | 31.686 |
| PP8 | 28.955 | 27.934 | 29.947 | 30.030 | 30.134 |

Every arm completed all 1055 steps. This is **one run per width**, not a paired
statistical scaling claim. More pipeline stages do not execute one batch-1 token's
layers concurrently; extra cards primarily reduce the memory footprint per card.
Small differences between widths need the owner's repeat measurements.

Logs and receipts: [`results/o55-events-20261022/`](../../results/o55-events-20261022/).
The machine-readable table is `split-events-summary.json`; full run logs are
`le2run.log`, `le4run.log`, `le8run.log`. Recomputed completion checks, timing
windows, distributions and trace estimates are in `log-analysis.json`; all earlier
archived logs are also covered by `archived-log-analysis.json`. Both are generated
by `examples/glm53/analyze_logs.py`. The historical remote binary paths were
`/ram/o55/le{2,4,8}/run`, with adjacent `load.tsv`; these are not locally retained.

## What changed and why

The earlier phased candidate used one event allocation on GPU 0 for **all** event
scopes. Idle workers on peer devices repeatedly polled remote counters, including
counters used only within their own stage.

The isolating experiment moved the entire event allocation to GPU 1. The slow
stage flipped: GPU 0 KDA tiles became approximately 450–700 us while GPU 1 tiles
became approximately 37–39 us. This rules out layer arithmetic or a consistently
slow physical device as the primary cause of those stage-specific tails. A prior
standalone same-device poller experiment had not reproduced this remote traffic
and did not rule it out.

The implementation now gives each device its own DOMAIN/DEVICE event buffer and
retains a single shared SYSTEM buffer on GPU 0. Generated SYSTEM waits, arrivals
and readiness probes use the shared pointer. Release/acquire operations are
unchanged. All reset streams finish before any device launches.

`GLM_SHARED_EVENTS=1` restores the old shared-buffer behavior for an A/B run.
The check currently uses environment-variable presence: even a value of `0`
enables that diagnostic mode; **unset** it for the local-buffer candidate.

In the new PP2 position-4 trace, KDA tile medians on the second stage are about
37–40 us rather than the earlier hundreds of microseconds. The earlier **28.196
ms** reconstruction used per-rank relative trace tables; **it is not a globally
aligned device span**. Each table has its own earliest-taken time origin. The
position-4 host wall is **28.353256 ms**; summed grid spans are **25.4984 ms**.
Their **2.854856 ms** residual includes host launch/retirement and scheduling:
it is **not a measurement of pure inter-kernel gaps**. Reconstructing a global
critical path requires the raw trace timestamps, which are not locally archived.
Summed phase spans (rounded):

| Kind | Total us across the model |
|---|---:|
| up | 7423 |
| p1 | 5304 |
| down | 4538 |
| o | 2710 |
| KDA | 1484 |
| p2 | 1461 |
| router | 995 |

These phase costs are from one traced early-context step, not the decode window.
Earlier v7's 71.3 ms all-step median and the new 29.455 ms median were measured in
this same ongoing box session, but not as an interleaved paired experiment.

## Correctness evidence and limits

The remote byte comparison recorded all 32 generated-logit arrays (19,824,640
bytes per arm) as byte-identical between PP2, PP4, PP8 and v7. The recorded digest
is retained in `split-events-summary.json`; the raw logit files were not downloaded
and this equality was **not rechecked locally**:

```text
SHA256 b818cc906d87e59db26552f08c097006774bd43ce5a184e222844f1f188aebd7
```

This establishes output preservation and partition-width consistency for this
candidate and prompt. It does **not** establish independent HF model correctness
for the full model. No HF reference job was restarted.

Runs exited successfully but printed ROCm `SharedSignalPool` resource-leak
warnings. Those warnings are retained in the complete logs; lifecycle cleanup is
not claimed complete.

The initial le2 receipt omitted `phases.h`; le4/le8 include it. Subsequent build
identity code also includes the changed code generators. Do not retroactively
claim complete source coverage for the original le2 receipt.

## Prepared, not measured: up-tile tail balancing

At production shape, up has 18,432 gate/up pairs. The measured default uses 512
tiles (36 pairs/tile). Four waves process eight pairs per round, so the tail
requires five rounds. A 576-tile grid (32 pairs/tile) should need four rounds.
This is a prediction, not a measured improvement.

Go/no-go: preserve all 32 output-logit arrays and reduce end-to-end step latency
by at least about 0.5 ms under matched instrumentation and conditions. Otherwise
retain the measured default.

Three diagnostic binaries `/ram/o55/bal{2,4,8}/run` compiled successfully before
the reservation, but were **never run**. Their build wrapper failed only while
printing its final logs (shell quoting produced a Python SyntaxError); each
per-width log and linked receipt says `BUILD OK`. These diagnostic builds also
raised the head tile cap, so they do not isolate the up-only hypothesis.

The local working tree instead preserves the measured default and exposes
`--up-pairs 32` as an explicit build option affecting up grids only. That revised
option has not yet been compiled or measured on the box. No local tests were run,
per owner instruction.

## Runtime compatibility hardening (CPU-compiled; broader modes untested)

The diagnostic bal builds also include:

- SYSTEM pointer fallback for manual ABI initializers that leave it null.
- Runtime-count initialization routed to the same event buffer as its consumers.
- Device-tail reset disabled with an explicitly separate SYSTEM buffer: a local
  tail cannot prove peer consumers are finished.
- Scope metadata and merged event-counter snapshots for watchdog/ledger reports.

These changes compiled on the box. Runtime-count and watchdog error paths were
not exercised; do not treat successful static-count GLM runs as coverage for them.
Tile contexts still expose only the ordinary event pointer, so custom tile-side
SYSTEM event manipulation is not covered by the split-buffer contract.

## Cost and next condition

No new VM was provisioned. The box's quoted rate is $30.4/hour; no provider billing
or account-balance measurement was made here, so no exact dollar cost is claimed.
The three completed split-event runs and CPU builds are logged; the owner's
repeated-run raw logs were not fetched. The box is now stopped. No further
GPU work, SSH or provisioning is authorized; there is no timed automatic resumption.
