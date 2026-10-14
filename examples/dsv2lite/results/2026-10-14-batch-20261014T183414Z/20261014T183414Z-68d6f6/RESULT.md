# Qwen's cache-mode verdict, 2026-10-14 — box session 20261014T183414Z-68d6f6

Condition: one MI300X VF (192 GB, ROCm 7.14), bf16, batch 1, context 1024, 32 greedy tokens per request, five
interleaved pairs; both arms from the same tree, gated by the import certificate and 32/32 before timing.

What ran and what it said (raw lines):

    BUILD kept OK /root/sess/20261014T183414Z-68d6f6/qwen/b_kept
    BUILD kept_l1 OK /root/sess/20261014T183414Z-68d6f6/qwen/b_kept_l1
    CERTIFIED kept: import certificate at positions 0 and 1040, 32/32, CHECK-DSL OK
    CERTIFIED kept_l1: import certificate at positions 0 and 1040, 32/32, CHECK-DSL OK
    VERDICT arm=kept_l1 NO-GO rule='G6 (docs/INVENTORY-SYSTEM-2026-10-13.md 4.1): KEEP if every pair is faster than
      kept, median >= 30 us and median > 2 x spread; otherwise NO-GO'
    PAIRS arm=kept_l1 vs kept median 9.0 us spread 19.0 us every_positive=0

Reading: the same `ETX_COHERENCE=l1` switch that gave DeepSeek **+330 us (KEEP)** gives Qwen **+9.0 us with no pair
faster** and a spread twice the median. The switch is correct on Qwen — both arms built from one tree and both passed
the certificate — so this is a gain-portability result, not a correctness one. The reuse criterion ("one mode, two
models, one rule") is therefore answered: the rule applies, the gain does not transfer.

This session also carried the batch-8 diagnostic (`bgate_x8p/x8m/x8k/x8`), which did not run: the `source` job refused
the tree (`fleet src is not the exact pinned 3fb67a6 export`) because earlier overlapping start attempts had let an arm
build patch `/root/fleet-3fb67a6` in place. Re-staging restores it; the diagnostic was re-run in a later session.

Seal: `VERIFIED: session 20261014T183414Z-68d6f6, 129 files match, 0 bad, 0 left on the box by design`.
