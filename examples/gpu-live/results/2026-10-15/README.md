# Reviewed native queue, 2026-10-15 — terminal evidence

Executable source remains `554133d744b4591c9cf87c68b6817df781137c9b`. All five entries are terminal; no executable source changed and no push occurred. See [result.md](result.md) for outcomes, conditions, exact bindings, export extent and remaining gates; [status.md](status.md) is the completion handoff.

This directory contains curated text/JSON/raw-log receipts and original manifests, not complete sealed result copies. Complete raw binaries, captures, per-layer checks and samples remain in `/LOCAL_WORKSPACE/.hermes/cache/scratch/etx-gpu-live-20261015-evidence`, verified as documented. DeepSeek's two external weight inputs are deliberately not exported; their exact hashes and remote seal verification are retained. B4's reused-reference materialization records original symlink targets and verified bytes. C6's original failed runner record remains failed; its direct unchanged retry has a separately verified terminal work snapshot and true command/exit receipts, not an invented sealed runner success.

Historical interruption records remain under `interruption-1600Z`; those UNKNOWN snapshots were reconciled before continuation. `SHA256SUMS` binds the current curated receipts. No invalid graph, incomplete counter coverage, failed correctness gate or nonzero profiler exit is promoted to success.
