# Qwen3-30B-A3B experiment log: four columns

Only the experiments queued or designed now (the team proposals of 2026-10-14 are classified in
docs/PROPOSALS-2026-10-14.md). Measured history is in docs/GAINS-LEDGER.md and docs/BASELINE-STATE-2026-10-12.md,
and the closed routes are in docs/OPTIMISATIONS.md. Budgets and rules live in the queue (docs/EXPERIMENTS-2026-10-14.md).

| Experiment | Why we think it is a good experiment | Run on GPU? | Result |
|---|---|---|---|
| Common cache mode `kept_l1` (`ETX_COHERENCE=l1` on the kept build) | One compiler switch for both models. DeepSeek's half measured +330 us (KEEP); proving it on Qwen is the reuse criterion. Qwen's DSL tiles are already coherent, so expect much less | not yet — queued (`qwen_pairs_kept_l1`) | Locally: both builds identical in resources; 82 system-scope loads move to agent scope, 0 acquires removed |
| MoE row ownership between workgroups (`hw_bench moerows`) | Decide, before any hot-loop rewrite, whether interleaving each expert's rows between workgroups moves more HBM traffic. The old 0.50 ms was a cache-contaminated model bound | not yet — queued (`qwen_moerows`) | Locally: compiles, one kernel for both modes, CPU coverage and coldness oracle pass |
| Router grid of 128 tasks, 16 per XCD (P1, `router128`) | Team proposal: the 128 empty router tasks and the 4-XCD row placement are literal today. The narrowing itself is kept (C1, +0.0626 ms) | not yet — design only; not worth a box: empty tasks 0.28 us off the path, the stage is one task's serial chain | Estimate about 0; reopens only if a trace shows the path task's body >= 0.625 us/layer above the median |
| Epoch-tagged events, static-only reset (P3, `epoch_reset`) | Team proposal: reset submission is 0.0386 ms per step, measured in two sessions. The device loop is closed by the bar and the device reset lost (-0.008 ms wall) | not yet — design only; not worth a box now: best case 0.022 ms by the device-reset decomposition | Reopens on a host-only diagnostic showing >= 0.030 ms of wall behind the reset submissions |
| Producer-side combine, row-partitioned moe_down (P4, `moe_rowsum`) | Team proposal: QK2 refolds residual + 8 slot rows in all 256 qkv tasks; ceiling 0.073 ms. The combine phase (+4.10 us/layer) and Fleet's one-writer fold already lost | not yet — design only; gated on X6 stamps (QK2 exposed >= 0.625 us/layer) before any code | None |
| Last-arriver top-k (P5, `topk_lastarr`) | Team proposal (jyoti): removes the 74.0 us/step router -> top-k handoff; Qwen's top-k is already serial, unlike Fleet's lost `--topk-published` | not yet — design only; a sync_bench mode decides first (GO if >= 0.625 us/layer saved) | Estimate 0.040-0.052 ms if visibility is not paid, -0.003 to +0.017 ms if it is |
