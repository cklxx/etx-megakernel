# Audit of the four-layer synchronisation taxonomy against the repository's evidence, 2026-10-11

Qwen3-30B-A3B, bf16, batch 1, context 1024, one MI300X VF (gfx942), ROCm 7.14.0, the kept KEY32-on build
`3288aa5c` (`results_qwen300_2026-10-11/r1_on_graph.log:3`). CPU only: no GPU ran, and no number below is a new
device measurement. Labels as in `docs/MASTER-TABLE-2026-10-11.md:10-12`: **[M]** measured, **[E]** estimate with
its arithmetic, **[R]** residual of an [M] interval after an [E] model, **[U]** unknown.

## 0. Basis

Paths starting `results_` are under `~/code/etx/`, outside git; `build/` is an untracked symlink. Short names:

| Name | File | What it is |
|---|---|---|
| J | `results_qwen300_2026-10-11/job2_run4.log` (sha256 `ac88d81e…`) | kept build, five untraced rounds of fused and HIP-graph arms |
| G | `results_qwen300_2026-10-11/r1_on_graph.log` | graph arm, round 1: launch count and build identity |
| MT | `docs/master-table-2026-10-11/master.txt` | generator output over `tr_on.bin` (sha256 `aa684682…`), the kept build's trace |
| MTD | `docs/MASTER-TABLE-2026-10-11.md` | the document built on MT |
| ACC | `docs/review-pool-analysis-2026-10-11/account_on.txt` | `xcd_account` on `tr_on.bin`; lines 40-80 re-run here, byte-identical |
| DEP | `docs/DEP-GAP-LEDGER-2026-10-11.md` | the same account on `tr_off.bin`, the KEY32-**off** build: not the kept build |
| HW | `results_2026-10-01/baremetal/hw_all.log` | `hw_bench` suite on a bare-metal MI300X: a different machine |
| FS | `~/code/fleet-mi300x/docs/STATUS.md` (`85b087e`) | fleet's DeepSeek megakernel on MI300X: evidence for that kernel only |
| Y, K, H, PR | `etx/machine/arch/gfx942.yaml`, `etx/codegen/kernel.py`, `etx/runtime/include/etx/host.h`, `…/primitives.h` | the lowering and the runtime |
| R1, TD, FL | `docs/RESULTS-2026-10-01.md`, `docs/ETX_Technical_Design.md`, `docs/FLOORS-2026-10-08.md` | earlier results and design |

**Primary rule.** A log or trace beats the generator output over it, which beats a document. Where a document
quotes a number, the number is checked at its source and the source is cited.

**Anchor [M], kept build, medians of five untraced rounds (J:144-147).** Fused: wall 4.180 ms, device 4.045 ms.
HIP graph of the same tiles: wall 4.357 ms, device 4.323 ms. Fusion gain: **0.177 ms wall (4.1%)**, **0.278 ms
device**. Host excess (wall − device): fused **0.135 ms**, graph **0.034 ms**. The need for 300 tok/s is
4.180 − 3.333 = **0.847 ms** wall (MTD:111). A device cut transferred to wall 1:1 once (MTD:112).

## 1. Verdicts: eight mechanisms and three limits

| # | Mechanism | Status | Decisive evidence | The number, with its basis | What it would need |
|---|---|---|---|---|---|
| 1 | Megakernel: one launch per step | **DONE** | H:301-307 (one launch); G:5 (graph arm: 434 grid instances, one launch each, H:375-378) | Gain 0.177 ms wall / 0.278 ms device (J:144-147). The single launch costs 11.994 us host per step [M, older build] (`results_nvref_2026-10-09/hostprof.log:2120`); host excess rose 0.034 → 0.135 ms (J) | nothing at the launch (§2a) |
| 2 | Event tensor: counter in memory, atomic arrive, spin | **DONE** | K:82-122; PR:19-28 (`fetch_sub`, poll until 0) | Idle one-way 0.642 us same-XCD, 0.742 cross-XCD, 0.916-0.951 DEVICE flag with payload [M, microbenchmark] (Y:69-72). On the kept path: 435 edges, **875.5 us** [M] (MT:18) | a lower per-edge price (§5) |
| 3.1 | Fold | **DONE** (norm into qkv, router, lm_head; in_norm, post_norm, moe_sum deleted) | `examples/vllm_llm/model.py:119-124`; MTD:178,195 | VECNORM −0.105 ms (`examples/vllm_llm/build.sh:68`); fold + touch together −0.039 ms on an old build (`docs/MODEL-vllm-kernels.md:554-555`). The wait does not disappear: qkv now waits on moe_down's 256-producer event, **183.2 us** on the path [M] (MT:25) | nothing; a fold *is* 3.2 |
| 3.2 | Recompute instead of wait | **MEASURED** | `etx/passes/p3_event_elim.py:1-5`; R1:71,82,86 | Pass 3: +16% on a synthetic MoE (hybrid), none static (TD:1297). Top-k recomputed in 448 expert tasks: **0.231 ms loss** [M, device] (R1:71) | no candidate above the bar (§3) |
| 3.3 | Ordering as synchronisation | **ABSENT** | K:82-93 (every in-edge emits a wait); `etx/passes/plan.py:152-164` (queue order checked only for deadlock) | 0 of 98 single-producer path edges have the consumer on the producer's worker (ACC:70-71,76-77): ceiling **0** on this schedule [E] | a schedule that co-locates every producer of an event with its consumer, serialising the consumers |
| 3.4 | TOUCH | **DONE**, value unvalidated | K:283-289; MTD:168 | Never isolated after COH (`docs/OPTIMISATIONS.md:27`). Free gaps into touch grids 2.10 us/edge vs 1.34 into the rest [M, confounded by event scope] (MTD:150-151); stake **285.4 us** [R] (MTD:397) | X1, NO_TOUCH traced A/B, $0.66 [E] (MTD:380) |
| 4.1 | Epoch stamp | **ABSENT** | `etx/ir/types.py:58` (field never set); H:237-268 (memsets every step); `build/qwen300-rocm-on/plan_data.h:48` (device reset off) | Host reset 49.466 us/step [M, older build] (`hostprof.log:2120`); the device-side reset saved 0.008 ms wall (R1:107). Fleet has epochs (FS:320,374) | epoch targets in every wait; ≤ 0.049 ms [E], inside the host row (`docs/PLAN-300TOKS-2026-10-11.md:46`) |
| 4.2 | Sentinel | **MEASURED** (microbenchmark only), rejected | HW:111-115 vs HW:82-86; `docs/RESULTS-2026-09-30.md:83-93`; Y:64 (capability bit, no lowering) | Coherent-store sentinel 2.98-3.75 us median vs event 2.61-4.12 us [M, bare metal] (HW). Ceiling on the kept path **27.9 us** [E] (§2b) | nothing worth it (§2b) |
| L1 | The data dependency | Right in kind, wrong in size | MT:18,36; ACC:80 | 9 edges per layer agrees. Gaps **875.5 us** [M], of which 231.3 us is the consumer worker's own previous task, not data (MT:36); floor 435 × 1.00 = **0.435 ms** [E] (MTD:403) | §2c |
| L2 | Cross-XCD write-back and invalidate | Physics right; flush avoided (**DONE**: COH) | K:109 (`_nf`: coherent producers skip the L2 write-back); K:296-299 (no acquire); R1:70 | COH −0.317 ms device (R1:70). Same event, same-XCD vs cross-XCD medians differ ≤ 0.04 us (§4) | — |
| L3 | One host launch per step | **ABSENT** | TD:81,148; `docs/ETX_Brief.md:120` | Ceiling = host excess **0.135 ms** [M] (J:144-145); fleet's 32-token resident launch vs one per token: **0.005 ms** (FS:175-176) | a resident loop with epoch, reset and abort proofs (`docs/PLAN-300TOKS-2026-10-11.md:117`) |

## 2. Numbers in the taxonomy that the artifacts contradict

## 3. Mechanisms without a counterpart in the repository

## 4. What the critical path spends that the taxonomy omits

## 5. The single most valuable item

## 6. Errors in project documents and in the brief

## 7. Reproduce
