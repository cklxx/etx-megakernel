# Measured GPU work, 2026-10-15 UTC

Both completed local scopes are published here: the [five-experiment queue](results/2026-10-15/result.md)
and [multibatch attribution](results/2026-10-16-multibatch/result.md). The later shared-weight prototype
was cancelled before any edit, build, deployment or measurement. GPU destruction and billing cessation
were not verified. This publication does not resume GPU work.

## Corrected application baselines

One MI300X VF, gfx942, ROCm 7.14; DeepSeek-Coder-V2-Lite-Base, bf16, context 1024, 32 decode tokens.
Five interleaved same-box pairs; every request passed 32/32 tokens, 32/32 warm-up tokens and the
27 hidden boundaries on the first decode step. All selected vLLM full/short timing-call gates passed.
AITER was strongest only among the tested default/AITER choices, selected separately for each batch.

| Batch | ETX step ms | AITER step ms | ETX total tok/s | AITER total tok/s | Paired throughput gain | Spread |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3.222786 | 4.097652 | 310.290537 | 244.042198 | +27.110935% | 1.043288 pp |
| 2 | 4.532310 | 4.732610 | 441.276082 | 422.599792 | +4.419380% | 0.527359 pp |
| 4 | 8.921324 | 6.165512 | 448.363942 | 648.770126 | −30.930675% | 9.314794 pp |

Steps are medians: ETX uses direct host bracketing; vLLM uses the registered host full-minus-short
estimator. Gain is the median of paired `vLLM/ETX−1`; spread is max−min across five paired gains.
The [raw baseline arrays](results/2026-10-15/b4/batch_report.json) also retain request-latency verdicts.
B4 is a **LOSS**. These baseline numbers are reused by the subsequent trace study, not measured again there.
The same-kernel graph baseline remains **UNAVAILABLE** because grid-by-grid replay ends the required LDS
lifetime. No complete DeepSeek fusion certification is claimed.

## Other queue outcomes

- [Expert row merge](results/2026-10-15/expert/): synthetic screen, all five pairs slower; median
  gain −6.675003052 µs, spread 1.264495850 µs. STOP-NO-BENEFIT; no model speedup.
- [Qwen producer/router scoring](results/2026-10-15/producer/): actual-model mixed-weight output mismatch
  at position 0, `L0_mix.bin`. Qualification refused; later full-model gates and timing were not reached.
- [DeepSeek x8](results/2026-10-15/x8/): all six strict diagnostics fail hidden boundaries, including
  matching single-request controls. Concurrency is not necessary for the failure; its underlying cause is
  unproven. Exit 0 records successful diagnostic collection, not semantic correctness.
- [Attention C6](results/2026-10-15/c6/): required correctness checks passed, but all four checked positions
  had mixed timing signs and large spreads. The registered local target closed; no stable model/host gain.
- [B4 profiler failure](results/2026-10-15/b4/attribution.json): the first traffic invocation exited −11
  during HSA profiler finalization. Partial CSV is rejected traffic evidence. No unchanged profiler retry.

The queue parent records retain their actual incomplete/failed status. Completed timing subjobs do not
turn those parents into successful complete experiments; see the [export explanation](results/2026-10-15/README.md).

## What the nine new traces establish

The serialized forward/reverse/forward B1/B2/B4 study passed every strict per-request gate and exact
task-row validation. It captures the final decode token with instrumentation, not candidate performance.
B1 uses D4, so B2→B4 is the cleaner recipe-matched scaling comparison.

| Batch | Complete task rows | Median traced span ms | Min–max ms |
|---|---:|---:|---:|
| 1 | 53,022 | 3.094430 | 3.093460–3.100780 |
| 2 | 109,892 | 4.339200 | 4.324760–4.347170 |
| 4 | 219,784 | 8.670160 | 8.664580–8.696920 |

B4/B2 task count is 2×, summed body work is 1.982145× and span is 1.998101×. The B4 median-span run
contains 7.340840 ms critical-path bodies, 0.963120 ms dependency gaps and 0.366200 ms queue gaps.
Bodies include internal waits; queue serialization can appear as predecessor bodies, not just queue gaps.

First-request QKV `dep_gap_p95_us`, rounds 1/2/3, is **3.78/3.60/3.60 at B2** versus
**151.76/149.98/151.38 at B4**, directly from the [per-request phase rows](results/2026-10-16-multibatch/trace_report.json).
These are recorded gaps, not proof of HBM saturation or a demonstrated root cause. No alternative schedule,
shared-weight tile, traffic reduction or paired candidate gain was measured.

The registered B4 host target requires 3.3723632 ms reduction; observed critical-path gaps are about
1.33 ms. This scale comparison supports investigating repeated tile-body work, but is not a host speedup
prediction. The prior 5.130547 ms batching projection uses HF picks and byte/rate assumptions; native ETX
pick equality remains unmeasured. No estimate is promoted to implementation or measurement.

## Provenance, privacy and reuse

Accepted executable source remains `554133d744b4591c9cf87c68b6817df781137c9b`.
The original queue branch tip is `1606dcd6c10d20d68b7635bddc50845e5fe872a7`; the original trace/analysis
branch tip is `7a9eff2499b2c852ff41eec1ac23093be3ad83b8`, including executed driver commit
`edd7457a333fd25dcf90402c91b8af24cb563cfe`. The two completed scopes were integrated as sanitized
squashes so private connection details do not enter published history. The original branches and full
raw exports remain local and unchanged. Model weights and giant executable/capture binaries are omitted.

The [queue redaction map](publication-queue.json) and [trace redaction map](publication-multibatch.json)
identify each derivative by original and public SHA256. A listed log/result is a **redacted publication
derivative**, even if its historical filename says `raw.log`. Original MANIFEST/SEAL files are preserved
byte-for-byte and bind original raw bytes, not changed derivatives or these intentionally partial snapshots.
`ORIGINAL_SHA256SUMS` retains prior curated checksums; current `SHA256SUMS` binds the publication view.
Do not run stock sealed-record verification on a partial/redacted directory and call it complete.

The full 32-file multibatch raw export passed stock verification with zero mismatches and no omitted
files. Its seal and manifest are [retained here](results/2026-10-16-multibatch/SEAL). The deterministic
[summary command](results/2026-10-16-multibatch/README.md) reproduces the published derived numbers.
Focused integrated local checks and publication audits are recorded in [publication-validation.json](publication-validation.json).
Valid prior 2190-pass and unchanged compiler evidence is reused; no full suite or compiler matrix was rerun.
