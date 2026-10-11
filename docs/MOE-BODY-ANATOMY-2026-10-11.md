# MoE body anatomy — 2026-10-11

Scope: Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X (gfx942), ROCm 7.14.0. CPU-only source and evidence audit; no GPU run or provisioning. Work is on `wf/moeanat`.

Evidence convention: `[M]` is a measurement with a source location; `[E]` is an estimate with arithmetic; `[U]` is unknown and contributes nothing to a numeric total. Instruction counts are not phase latency measurements. Overlapping costs must not be added as if serial.

## 1. Evidence, identity, and reconciliation

### 1.1 Which measurement is being anatomised

All external `results_*` paths below are relative to `/Users/ckl/code/etx`, not this worktree. The supplied numbers belong to **`results_qwen300_2026-10-11/attempt1/tr_off.log`**, not the current top-level file. The archive reports span **4295.4 us [M]** (`:26`), **32/32 [M]** (`:25`), device median **4.297 ms**, wall **4.474 ms**, **233 tok/s [M]** (`:464-466`). Hardware is MI300X VF, gfx942, ROCm 7.14 (`results_qwen300_2026-10-11/record.log:7,15,18`). The current top-level log reports **4.299/4.471 ms [M]** (`:464-466`); do not silently substitute it. Both logs bracket the same build-identity hash (`:3,467`), but are different observations.

Summing the printed mean-body column of the archived trace gives **761.52 us up**, **444.99 us down**, and **3086.99 us for the nine layer roles [E]** (recomputed, with every input line and hashes, in [trace-sums.json](moe-body-anatomy-2026-10-11/trace-sums.json)). Therefore the exact rounded-input targets here are `761.52/48 = 15.865 us up`, `444.99/48 = 9.270625 us down`, total `1206.51/48 = 25.135625 us/layer [E]`. The brief's **15.86/9.27** are rounded descriptions. **1067 us is not this trace's MoE body sum**: `761.52+444.99 = 1206.51 us [E]`. Its provenance is [U].

The body is an **all-task mean for a layer**, not a measured particular task or the last task's critical body. Tasks run concurrently: do not multiply by the task count. Body timing includes `etx_call_body`, **all-wave store drain and its final barrier**, between `tr[1]` and `tr[2]` (`etx/codegen/kernel.py:314,323-333`). Event arrival, next-task fetch and dependency wait lie outside it. Source-tile end is thus not trace-body end.

`3086.99/4295.4 = 71.867% [E]` is a sum of mean layer bodies divided by the span, **not a disjoint critical-path partition**. It excludes prologue, input norm, lm_head and argmax (`attempt1/tr_off.log:28-30,462-463`). The complement is not simply gap/wait. A critical-path partition needs the raw task trace and matching plan.

### 1.2 The two older analyses cannot be transplanted

The XCD report explicitly uses `results_tiles/tr_all.bin`, a **4412.1 us [M]** span, with **1785.1 us machinery and 1551.3 us body residual [E]** (`docs/XCD-TIMELINE-2026-10-11.md:12,19,88-102`). Its “machinery” is measured body **minus a byte/hop model**, not intra-body measurement. Neither those totals nor their split is a phase price for this trace.

The ladder is older again: **18.37/11.11 us [M]** controls versus **11.55/7.64 us [M]** NOVMEM (`etx/ops/gemv.py:17-29`; raw finding `results_2026-10-01/round37/FINDING.md:3-17`). NOVMEM **still loads one weight vector per wave**, reusing it; it is not literally zero weight reads (`gemv.py:237-245`). It retains dot, activation LDS reads and loop arithmetic, but changes dependencies and compiler scheduling and gives wrong output. Its marginal differences cannot be summed as exclusive phases.

For example the old up residual is `18.37 - 1.50 - (18.37-11.55) - (18.37-15.45) - (18.37-17.81) = 6.57 us [E]`; down gives `11.11 - 1.43 - 3.47 - 2.47 - 0.43 = 3.31 us [E]`. These are **subtraction residuals of separate experiments**, not measured loop slices. In particular **6.5 us each for up and down is unsupported**. `[U]` exclusive current-loop time contributes no saving.

### 1.3 Parameters and ceiling before a new experiment

One up task owns **32 gate plus 32 up rows**, K=2048; one down task owns **64 rows**, K=768 (`examples/vllm_llm/model.py:358-360`). bf16 weight bytes are `64×2048×2 = 262144 B up` and `64×768×2 = 98304 B down [E]`. With **192/256 tasks [M]** (`attempt1/tr_off.log:37-38`), the phase bytes are `50.331648/25.165824 MB [E]`. At the repository's **4.3 TB/s achievable-rate convention** (`AGENTS.md`, §2), the bandwidth service floors are **11.704/5.852 us/layer [E]**. They are phase-wide floors, not each CU's bandwidth, and overlap instruction work.

The corresponding residuals are `15.865-11.704=4.161 us up`, `9.270625-5.852=3.418625 us down`, `48×7.579625/1000 = 0.363822 ms/step [E]`. **This is diagnostic headroom above an ideal rate, not a removable pool.** It includes essential dots, staging, wave sum, serial latency, rounding, epilogues and publication. Removing **0.324 ms** requires `324/48=6.75 us/layer [E]`, about `324/363.822=89.1% [E]` of that ideal-rate headroom. The more exact plan gap is **0.323667 ms [E]**, `6.743 us/layer` (`docs/PLAN-300TOKS-2026-10-11.md:136,146`).

Measured older hardware parameters: scalar hit **0.113 us**, loaded vector L2/MALL/HBM **0.73/0.71/0.69 us**, bare round barrier **0.106 us** (`docs/MODEL-vllm-kernels.md:320-329`). These are other-session microbenchmarks, not measured current phase times. `bench/hw/hw_bench.hip:16-18,23-28` describes the primitive/clock/latency tests. No qualified current shader-clock, SALU issue rate, LDS latency, or exp/reciprocal throughput measurement was found [U]. Nominal **2.1 GHz** is a reported device property (`results_2026-10-01/round21/streamw.log:1`), not a loaded-clock measurement. Cycle-to-us prices using it must be `[E]`.

The shipped build's full-model allocation is **128 VGPR**, **44 B/lane scratch**, **10 VGPR and 265 SGPR spills**, **29736 B LDS [M compiler metadata]** (`results_qwen300_2026-10-11/off_hipcc.log:4-12`). Spill counts are static allocation metadata, not dynamic MoE reload counts; their MoE latency share is [U]. An isolated no-spill compile cannot explain away full-model spills.


## 2. Execution-order anatomy and phase ledger

Pending: one moe_up and one moe_down body, source and emitted ISA, phase costs, classifications, and explicit unattributed balance.

## 3. Removable pool and smallest change

Pending: measured versus conditional removable pool, comparison with the requested 0.324 ms, largest item and correctness invariants.

## 4. Probe coverage and decisive GPU measurement

Pending: existing isolated timestamps, blind spots, exact commands, time and dollar budget. Commands are a future protocol, not executed here.

## 5. Counter-argument and unresolved evidence

Pending: strongest alternative explanation and falsification criteria.
