# Qwen3-30B-A3B experiment log: four columns

This log covers two recent measured experiments, four design-only proposals, and the CPU-only producer-router study.
Updated from the captured results on 2026-10-15; no new GPU run was performed.
Earlier measured history is in [the gains ledger](../../../docs/GAINS-LEDGER.md) and
[the baseline record](../../../docs/BASELINE-STATE-2026-10-12.md).
Closed routes are in [OPTIMISATIONS](../../../docs/OPTIMISATIONS.md).
Proposal classification is in [PROPOSALS-2026-10-14](../../../docs/PROPOSALS-2026-10-14.md);
experiment rules are in [EXPERIMENTS-2026-10-14](../../../docs/EXPERIMENTS-2026-10-14.md).

## Original GPU records

The original captures are already versioned under the joint DeepSeek/Qwen session directories.
The links below point to those originals; this update does not rewrite or duplicate the raw logs.
The cache directory includes `kept_l1_r0_*` through `kept_l1_r4_*` raw invocation logs,
build logs, input metadata and both import certificates. The JSON is the derived pair summary,
not a replacement for those raw invocation logs.

[cache-log]: ../../dsv2lite/results/2026-10-14-batch-20261014T183414Z/20261014T183414Z-68d6f6/qwen_pairs_kept_l1.log
[cache-raw]: ../../dsv2lite/results/2026-10-14-batch-20261014T183414Z/20261014T183414Z-68d6f6/qwen/
[cache-pairs]: ../../dsv2lite/results/2026-10-14-batch-20261014T183414Z/20261014T183414Z-68d6f6/qwen/pairs_kept_l1.json
[moerows-first]: ../../dsv2lite/results/2026-10-14-batch-20261014T134920Z/20261014T134920Z-324307/qwen/moerows.txt
[moerows-second]: ../../dsv2lite/results/2026-10-14-batch-20261014T142757Z/20261014T142757Z-03ff3e/qwen/moerows.txt

| Experiment | Why we think it is a good experiment | Run on GPU? | Result |
|---|---|---|---|
| Common cache mode `kept_l1` (`ETX_COHERENCE=l1` on the kept build) | Test whether the shared compiler cache switch benefits Qwen. Qwen's DSL tiles are already coherent; a DeepSeek gain does not establish a Qwen gain | Yes — session `20261014T183414Z-68d6f6`, `qwen_pairs_kept_l1`; [raw job log][cache-log], [per-invocation logs and certificates][cache-raw], [pair data][cache-pairs] | **NO-GO.** Device-time gains 9/10/-5/14/0 us; median +9 us, spread 19 us. Three pairs faster, one slower, one equal. Does not meet every-pair-positive, median >=30 us and median >2x spread. Both builds passed import certificates. Conditions: one MI300X, Qwen3-30B-A3B, bf16, batch 1, context 1024, 32 tokens |
| MoE row ownership between workgroups (`hw_bench moerows`) | Decide, before any hot-loop rewrite, whether interleaving each expert's rows between workgroups improves the weight-read microbenchmark. The old 0.50 ms was a cache-contaminated model bound | Yes — sessions `20261014T134920Z-324307` and `20261014T142757Z-03ff3e`; [first raw log][moerows-first], [second raw log][moerows-second] | **NO-GO in both runs.** Median gains -6.01% / +3.42%; spreads 44.68% / 44.38%, respectively. Checksums passed. Neither run meets every-pair-positive, median >=3% and median >2x spread. No stable benefit established; these are microbenchmark results, not model throughput |
| Router grid of 128 tasks, 16 per XCD (P1, `router128`) | Team proposal: the 128 empty router tasks and the 4-XCD row placement are literal today. The narrowing itself is kept (C1, +0.0626 ms) | not yet — design only; not worth a box: empty tasks 0.28 us off the path, the stage is one task's serial chain | Estimate about 0; reopens only if a trace shows the path task's body >= 0.625 us/layer above the median |
| Epoch-tagged events, static-only reset (P3, `epoch_reset`) | Team proposal: reset submission is 0.0386 ms per step, measured in two sessions. The device loop is closed by the bar and the device reset lost (-0.008 ms wall) | not yet — design only; not worth a box now: best case 0.022 ms by the device-reset decomposition | Reopens on a host-only diagnostic showing >= 0.030 ms of wall behind the reset submissions |
| Producer-side combine, row-partitioned moe_down (P4, `moe_rowsum`) | Team proposal: QK2 refolds residual + 8 slot rows in all 256 qkv tasks; ceiling 0.073 ms. The combine phase (+4.10 us/layer) and Fleet's one-writer fold already lost | not yet — design only; gated on X6 stamps (QK2 exposed >= 0.625 us/layer) before any code | None |
| Last-arriver top-k (P5, `topk_lastarr`) | Team proposal (jyoti): removes the 74.0 us/step router -> top-k handoff; Qwen's top-k is already serial, unlike Fleet's lost `--topk-published` | not yet — design only; a sync_bench mode decides first (GO if >= 0.625 us/layer saved) | Estimate 0.040-0.052 ms if visibility is not paid, -0.003 to +0.017 ms if it is |

## Producer-side router fusion study (2026-10-15)

[Executed study, plan and conditional GPU experiment](../../../docs/QWEN-ROUTER-FUSION-STUDY-2026-10-15.md).
The exact mechanism is o_proj slice scoring before normalization, with delayed RMS scale
and last-arriver reduction/top-k; it is distinct from P5 above.

| Experiment | Why we think it is a good experiment | Run on GPU? | Result |
|---|---|---|---|
| o_proj producer scoring + delayed RMS scale + last-arriver top-k | Test whether normalization's global dependency can move after router dot products | **No — CPU simulation/tests executed; conditional successor hardware design only; $0 GPU spend** | **Not yet supported; NO-GO as an exact bf16 rewrite.** Synthetic counterexample changes eighth expert 7→8. 64 synthetic cases: ordered IDs differ 6/64, sets 1/64, mixing-weight bits 64/64; not actual-model correctness. Current gain unknown. Historical trace sensitivity is not transferable to 4.051 ms kept wall time. No implementation/run of this exact mechanism found in searched history. |
