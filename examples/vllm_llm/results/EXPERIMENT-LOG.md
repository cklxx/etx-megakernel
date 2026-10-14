# Qwen3-30B-A3B experiment log: four columns

Only the experiments queued now. Measured history is in docs/GAINS-LEDGER.md and docs/BASELINE-STATE-2026-10-12.md,
and the closed routes are in docs/OPTIMISATIONS.md. Budgets and rules live in the queue (docs/EXPERIMENTS-2026-10-14.md).

| Experiment | Why we think it is a good experiment | Run on GPU? | Result |
|---|---|---|---|
| Common cache mode `kept_l1` (`ETX_COHERENCE=l1` on the kept build) | One compiler switch for both models. DeepSeek's half measured +330 us (KEEP); proving it on Qwen is the reuse criterion. Qwen's DSL tiles are already coherent, so expect much less | not yet — queued (`qwen_pairs_kept_l1`) | Locally: both builds identical in resources; 82 system-scope loads move to agent scope, 0 acquires removed |
| MoE row ownership between workgroups (`hw_bench moerows`) | Decide, before any hot-loop rewrite, whether interleaving each expert's rows between workgroups moves more HBM traffic. The old 0.50 ms was a cache-contaminated model bound | not yet — queued (`qwen_moerows`) | Locally: compiles, one kernel for both modes, CPU coverage and coldness oracle pass |
