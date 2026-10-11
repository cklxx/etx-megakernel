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

### 2a. "500 launches × 5-10 us = 2.5-5 ms, the bulk of the +4%": refuted

The number is the project's own: `docs/ETX_Brief.md:29` and TD:50 say "launch gaps are 5-10 us". No artifact of
this build supports it.

1. **The baseline does not launch from the host per kernel.** The graph arm replays 434 kernels (G:5) from one
   captured graph (H:393-401). Its whole host excess is 4.357 − 4.323 = **0.034 ms** [M] (J:146-147), 0.08 us per
   kernel [E]. At 5-10 us per launch the 434 launches alone would be 2.2-4.3 ms [E], against a 4.357 ms wall that
   also holds 3.19 ms of task bodies (MT:18). vLLM runs 843 kernels per token (`results_2026-09-27/next.log:313`);
   at 5-10 us each that is 4.2-8.4 ms [E], against S3's whole 4.749 ms step
   (`results_vlbest2_2026-10-10/report.log:9`, a different session).
2. **What a kernel boundary costs on the device**, measured inside a 200-kernel HIP graph: **1.60-1.62 us** empty,
   **1.78-2.39 us** with 304 workgroups doing 0-4 dependent loads [M, bare metal] (HW:41-56). For 434 boundaries:
   0.69-1.04 ms of device time [E].
3. **What ETX pays instead.** 435 event edges on the path, **875.5 us** [M], mean 2.01 us/edge; free edges alone
   1.51 us/edge (470.1 us / 312, MT:36; MTD:150-151). Per edge this is not clearly cheaper than a graph boundary.
4. **Where the +4.1% comes from.** Device 0.278 ms minus a host penalty of 0.135 − 0.034 = **0.101 ms** (J:144-147).
   The launch layer's contribution to the fusion gain is therefore **−0.101 ms**: the one megakernel launch costs
   more host time than the graph's 434. The 0.278 ms device gain cannot be split between boundary cost and body
   speed [U]: no trace of the graph arm exists. The one measured switch that turned fusion from a loss into a
   gain on one box and day is COH, a cache-continuity change: fused 5.224 → 4.907 ms against the unfused 5.107
   [M, device, older build] (R1:31,70). That points at caches surviving the phase boundary, not at launches or
   barriers [E].

**Corrected number:** launch removal is worth **−0.101 ms** wall to the kept build's fusion gain; the gain is
**0.278 ms of device time**, of unknown split.

### 2b. "Sentinel 0.8 us vs 7.6 us for counter plus fence, about 10x": refuted for ETX

The pair is Kog's: TD:490 records it as a **whole-card phase switch** on 256 CUs. The repository reproduces the
large side: a whole-GPU counter barrier of 304 workgroups costs **8.6 us** [M] (Y:75). ETX never pays that per
edge. Its edges are per-word counters with a per-domain last-arriver flush (PR:42-58), and their measured cost is:

| Quantity | us | Basis | Source |
|---|---:|---|---|
| Same-XCD one-way, DOMAIN counter | 0.642 | [M] microbenchmark, idle | Y:69 |
| Cross-XCD one-way | 0.742 | [M] microbenchmark, idle | Y:70 |
| DEVICE flag with payload, one-way | 0.916-0.951 | [M] microbenchmark, 0 stale in 10k | Y:71-72 |
| Kog's sentinel, point-to-point | 0.80-0.93 | public, not reproduced | TD:490; Y:77 |
| Repository's sentinel test (coherent store + poll), median | 2.98-3.75 | [M] bare metal | HW:111-115 |
| Repository's event, same test harness, DEVICE, median | 2.61-4.12 | [M] bare metal | HW:82-86 |
| Kept path, single-producer same-XCD edges, median | 1.08-1.12 | [M] trace | ACC:70-71 |
| Kept path, cross-XCD DEVICE free edges, median | 1.56-1.97 | [M] trace | ACC:65-67 |

Against the counter ETX actually uses, the sentinel's 0.80-0.93 us is 0.84-1.45x of 0.642-0.951 us, not 10x. The
repository's own sentinel test found no gain; it scanned a whole 4 KB vector per round, which
`docs/RESULTS-2026-09-30.md:91-92` names as unfair, so the test does not prove the sentinel is slower either. The
sentinel only applies point-to-point: the 256-producer events would need one sentinel per producer, which is the
shape the test measured. On the kept path the single-producer edges (prep → attn, topk → moe_up) carry 104.7 us on
96 edges (ACC:70-71); at 0.80 us each the ceiling is 104.7 − 76.8 = **27.9 us** [E], below the 30 us bar
(MTD:369-371). **Corrected number:** at most ~1.4x per point-to-point edge and **≤ 0.028 ms** per step.

### 2c. "Nine handoffs per layer, the source of the 2.84 ms serial wall": qualified

**Nine agrees.** Events `E_{9L+1}..E_{9L+9}` per layer (DEP:90-91); 435 path edges = 48 × 9 + 3 (ACC:80).

**2.84 ms is not the cost of the handoffs.** It appears in neither MTD nor DEP. Its source is FL:212, the serial
floor at 4.3 TB/s and 0.50 us per round trip, computed on the round-11 trace with a 5.08 ms span (FL:21,94,211),
not on the kept build. Its formula is `48 × (Σ phases max(bytes/BW, chain floor) + handoffs) + tails`
(`etx/tools/floors.py:16`); `python -m etx.tools.floors --table` prints 2837.8 us. Split [E]:

| Part of the 2.84 ms | us | Source |
|---|---:|---|
| 48 × phase floors (bytes at 4.3 TB/s or dependent-load chains, whichever is larger) | 2255.0 | MT:51 |
| 48 × 9 handoffs at 1.00 us | 433.0 | FL:78; `docs/BUDGET-2026-10-01.md:161` |
| Tails (prologue, in_norm, lm_head, argmax) | 149.8 | difference |

Handoffs are 15% of it. Of the rest, FL:213,299 put 1.40 ms in weight streams that depend on no event, so the
floor model itself does not call it a dependency wall; the measured attempts to overlap it gave nothing
(FL:303-304,310).

**On the kept build** (MT:18,36): the dependency gaps are **875.5 us** [M] (DEP's KEY32-off trace: 871.0 us,
DEP:122-124). Of that, 231.3 us [M] is the consumer worker still finishing its previous task's arrive and queue
fetch, 470.1 us [M] free notification, 171.4 us [R] the rest of the late edges, 2.6 us the window start. The
protocol floor at the measured 1.00 us/edge is **0.435 ms** [E] (MTD:403). **Corrected number:** the
nine-per-layer data dependency costs **0.876 ms** of device time on the kept path, of which about **0.435 ms** is
the floor; the 2.84 ms is a different quantity.

## 3. Mechanisms without a counterpart in the repository

Three taxonomy entries have no implementation: 3.3 ordering (ceiling 0 on this schedule), 4.1 epoch (≤ 0.049 ms,
inside the host row) and L3 resident loop (≤ 0.135 ms). The rows in §1 give their evidence. This section asks the
question for 3.2: does any edge on the kept path have a recompute counterpart beyond the folds and Pass 3?

Recompute replaces a wait with a second copy of the producer's work. It can pay only where the producer's input is
small and every consumer can repeat it. On the nine per-layer edges (MT:5-17 for bytes; ACC:80 for edges):

| Edge (consumer ← producer) | Producer MB/step | Why recompute fails here | Basis |
|---|---:|---|---|
| prep ← qkv, router ← o_proj, moe_down ← moe_up, next qkv ← moe_down | 805-2416 | each consumer would re-stream the producer's weights, which are the bandwidth floor | MT:7,11,14-15 [M] |
| reduce ← attn | 104 | a merge of 20 partitions: one copy per consumer is the whole attention | MT:9 |
| topk ← router | 25.4 | 8 top-k tasks/layer would each stream the router's 0.53 MB alone [E] | MT:12-13 |
| o_proj ← reduce | 2.4 | each of 256 o_proj tasks would add the 5.3 us merge (MT:45) to save a 2.0 us/layer gap (96.7/48, MT:29) [E] | MT:10 |
| attn ← prep | 0.12 | prep rewrites qkv in place, so it cannot run twice; idempotent prep costs 5 × 6.06 = 30 us/layer to delete a ~1.2 us edge | model.py:316-327 |
| moe_up ← topk | 0.14 | **measured**: 448 expert tasks each recomputing top-k lost **0.231 ms** device | R1:71,82 [M] |

**Result:** no 3.2 counterpart clears the 0.625 us/layer bar (MTD:369-370). The two small-input edges were tried
or blocked, and the rest would re-stream weights. Pass 3 needs `grid.pure` (`etx/passes/p3_event_elim.py:51`), which
`examples/vllm_llm/` sets nowhere, so P3 never fires on this model (model.py:320-322; the kept build's
`decisions.log` has no P3 entry).

## 4. What the critical path spends that the taxonomy omits

The taxonomy prices only the 875.5 us of gaps (21.5% of the 4067.3 us span). The path spends the other 78.5% in
task bodies (MT:18). Split as in MT, with the source rows:

| Item | us on the path | Basis | Source | Taxonomy layer that covers it |
|---|---:|---|---|---|
| Weight and activation bytes at 4.3 TB/s | 1439.6 | [E] | MT:18 | none (the floor) |
| Exposed load round trips (hop model, 0.50 us each) | 534.4 | [E] | MT:18; `xcd_account.py:36-56` | none |
| Body machinery: instructions, barriers, reductions, spills, everything not bytes or hops | 1217.8 | [R] | MT:18 | none |
| … of which the five small roles prep, attn, reduce, router, topk | 697.9 | [R] | MT:8-10,12-13 | none |
| Critical body above the round-11 floor, 48 layers (B) | 782.9 | [E] | MT:51 | none |
| … of which the five small roles | 647.0 | [E] | MT:43-45,47-48 | none |
| Before-taken: consumer's worker still on its previous task (arrive, queue fetch) | 231.3 | [M] | MT:36 | none (scheduling, not data) |
| Late after-take and free gaps into touch grids (qkv 102.0 + router 38.7 + free 144.7) | 285.4 | [R] | MT:25,30,37 | 3.4 (TOUCH) |
| Tails: prologue, in_norm, lm_head, argmax bodies | 169.1 | [M] | MT:52 | none |
| Host excess of the fused arm (wall − device) | 135 | [M] | J:144-145 | L3 in part; event reset 49.466 [M, older build] (`hostprof.log:2120`) |
| Register spills of `etx_megakernel_d0`: 11 VGPR, 265 SGPR | [U] | [M] count, location unknown | `results_qwen300_2026-10-11/on_hipcc.log:1,10-11` | none |

**The largest omission is latency inside the small roles.** The path bodies of prep, attn, reduce, router and topk
sum to **1120.5 us** [M], of which bytes at 4.3 TB/s are **30.6 us** [E] (MT:8-10,12-13). They are chains of
dependent loads, block reductions and barriers in single tasks, not data movement and not synchronisation between
tasks. The before-taken 231.3 us is also not a data dependency: the consumer's worker was busy with its own previous
task (MT:21-38), which the taxonomy's L1 would book as a dependency wait.

**Coherence on the critical path is mostly paid already.** COH removed the L2 write-back on coherent producers
(K:109, −0.317 ms, R1:70). The residual coherence cost is the acquire on the imported grids, 44-181 ns × 3 roles
[E] (MTD:399), and the per-domain last-arriver flush, whose duration is unstamped [U] (MTD:402). On free edges
(no late consumer), the same event's same-XCD and cross-XCD medians differ by at most 0.04 us: qkv → prep 1.72 vs
1.68, reduce → o_proj 1.96 vs 1.97, router → topk 1.60 vs 1.56 us (ACC:65-67,73-75). Crossing an XCD is not
a large term on this path [E].

## 5. The single most valuable item

## 6. Errors in project documents and in the brief

## 7. Reproduce
