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

## Original producer-scoring implementation preparation (2026-10-15)

The user requested implementation of the original hypothesis after the study.
This work does not substitute P5 or re-label KEY32 as producer scoring.
`ETX_VL_PRODUCER_SCORE=1` selects an explicit, default-off o_proj producer candidate.
Each row owner scores its completed residual slice before the RMS scale.
The last producer folds the partials, writes the normalized expert input, and calls the existing top-k.

[Design and limits](../../../docs/QWEN-PRODUCER-PREP-2026-10-15.md).
[Bounded session runbook](../../../docs/QWEN-PRODUCER-RUNBOOK.md).
[Local execution record](2026-10-15-producer-prep/).

CPU simulation executes the shared production scoring header.
It retains the sparse eighth-expert change and tests reordered completion and reuse.
The synthetic 32-case run has 30/32 ordered-ID matches, 32/32 expert-set matches,
and 0/32 bit-exact mixing-weight matches.
These are synthetic counts. They are not actual Qwen pass rates.
Real-model capture and replay are prepared. No complete retained input set was found.

Status: implementation and local preparation only. No GPU execution. GPU spend $0.
No real-model correctness pass, imported-launch certificate, token pass, KEEP,
ESTABLISHED fusion, system LEAD, or speed gain is claimed for this candidate.
The GPU script blocks timing on any required routing or expert-input byte, import, DSL, or token gate failure.
Raw logit-bit changes remain diagnostic under the existing numerical gate.
The registered incremental wall rule remains five positive pairs, median >=30 us,
and median greater than twice the spread.
The candidate and kept recipe remain separate. The default is unchanged.


Final local preparation acceptance: **2,052 passed, 17 skipped, zero failures**.
The focused suite passed **165**, with one platform skip.
All **16/16** local matrix builds passed.
Both arms passed ROCm 7.2.4 device and host-device compilation, LLVM 23 IR and assembly,
and native host object compilation.
The source-bound receipt covers **324 source files** and **17 successful commands**.
These are offline checks. Linux HIP executable linking, loading and GPU execution remain pending.
See [the exact commands, skip reasons and compiler limits](2026-10-15-producer-prep/validation.md).

## Multi-request correctness and the batch benchmark (2026-10-17/18)

| Experiment | Why we think it is a good experiment | Run on GPU? | Result |
|---|---|---|---|
| B2/B4/B8 correctness, replicated requests (`examples/sessions/qwen_batch_correctness.py`, `plain` recipe) | A batch throughput number is meaningless until every request is shown to compute the same thing as B1: imported bytes per request, full coverage, HF greedy tokens | **Yes** — one MI300X, ROCm 7.14.0, session `20261017T140801Z-988ffe` (qualify 283 / 558 / 1572 s) | **PASS for all three.** Imported instances byte-identical at positions 0 and 1040: 194/388/776 (97 per request). Coverage 1055 positions per request with 0 wrong beyond ties (ties per request: B4 0,0,1,1; B8 0,0,1,1,3,0,0,0). 32/32 tokens per request in both repetitions of fused, unfused and coverage runs (128/256/512 per mode). The record seal verifies (147 files, 0 bad), but it is sealed INCOMPLETE: the in-session verifier wrongly required 20 asset files for every batch. With the fix (`f2d29d1`) the same receipts verify. No timing claim |
| Imported vs original launch time under `--check` (same record) | Shows which imported kernel grows with replication before any throughput run | **Yes** — same record, GPU events per launch, one launch each | Summed over the checked launches, at position 0: B2 −10.5%, B4 +17.7%, B8 +35.0%; at 1040: −17.0% / +4.8% / +26.7%. At B8 position 0, `attn_reduce` accounts for +2473.3 of +2483.7 µs (2455.9 → 4929.2 µs), with a per-launch median of 6.2 → 12.7 µs. Isolated launches, not decode time |
| `bench.py --batch 1/2/4/8`, `kept` and `plain` recipes, against vLLM batch mode | The owner's baseline: tokens/s and step wall time per batch, ETX and vLLM on one box in one session | **Partly** — one MI300X, ROCm 7.14.0, source commit `380ed02`, one runner at a time under `flock`. Kept B1/B2 completed in the clean serial run; kept B4/B8 produced no result (the remote shell exited at B4). Plain B4 completed in a separate runner (the correctness-qualified recipe); plain B8 was running. vLLM comparison not run. Each batch has its own build and certificate; kept B2 compiles locally with ROCm 7.2.4: 120,724 tasks, 128 VGPR, 9 VGPR spills, 40 B/lane scratch, occupancy 4 | **B1/B2 kept PASS, B4 plain PASS.** Host step median / tokens/s: kept B1 4.087415 ms / 244.65 (32/32, 194/194 imports); kept B2 5.701229 ms / 350.80 (64/64, 388/388, coverage 2x1055, ties 0,0); plain B4 11.929137 ms / 335.31 (128/128, 776/776, coverage 4x1055, ties 0,0,1,1). Kept B1 to B2: step x1.3948, tokens/s x1.4339, per-request cost -30.26%. Kept B4/B8 and plain B8 [U]. No vLLM numbers. See `docs/QWEN-BATCH-BASELINE-2026-10-18.md` |
