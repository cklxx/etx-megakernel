# MoE body anatomy — 2026-10-11

Scope: Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X (gfx942), ROCm 7.14.0. CPU-only source and evidence audit; no GPU run or provisioning. Work is on `wf/moeanat`.

Evidence convention: `[M]` is a measurement with a source location; `[E]` is an estimate with arithmetic; `[U]` is unknown and contributes nothing to a numeric total. Instruction counts are not phase latency measurements. Overlapping costs must not be added as if serial.

## 1. Evidence, identity, and reconciliation

Pending: trace identity, task/body versus critical-path totals, ablation semantics, and measured hardware parameters.

## 2. Execution-order anatomy and phase ledger

Pending: one moe_up and one moe_down body, source and emitted ISA, phase costs, classifications, and explicit unattributed balance.

## 3. Removable pool and smallest change

Pending: measured versus conditional removable pool, comparison with the requested 0.324 ms, largest item and correctness invariants.

## 4. Probe coverage and decisive GPU measurement

Pending: existing isolated timestamps, blind spots, exact commands, time and dollar budget. Commands are a future protocol, not executed here.

## 5. Counter-argument and unresolved evidence

Pending: strongest alternative explanation and falsification criteria.
