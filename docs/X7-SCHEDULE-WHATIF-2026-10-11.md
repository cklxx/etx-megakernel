# X7: can a legal layout shorten the layer period? (2026-10-11)

**Verdict: WEAK.** A legal frozen layout predicts **0.820 us/layer saved [E]**, above the registered
0.625 bar, without deleting work or dependencies. The **same layout** saves only **0.53125 us/layer [E]**
under another timing decomposition that reproduces every baseline stamp. A policy using median profiles
saves nothing. Thus DEP §6's universal period-conservation inference is unsupported; a practical KEEP
layout is also unestablished. **Close the tested profile-policy search below KEEP; do not declare rows
9–10 proved structural.** The nominal oracle passes the original threshold; the sensitivity is a limit
on that prediction, not a newly imposed KEEP rule.

## 1. Registered rule, measured inputs, ceiling

The skeleton registered **>= 0.625 us/layer = 0.030 ms/48-layer step** before results (commit `ed4a839`).
Worktree `wf/x7whatif`, base `e31bf04`; CPU only, $0 GPU/network spend, no provisioning or push.
All predictions below are **[E]**, not new GPU measurements. **[M]** means existing trace/log evidence;
**[U]** contributes no saving. A measured correctness gate is not supplied by CPU simulation.

[M] Qwen3-30B-A3B, bf16, batch 1, context 1024, MI300X VF, ROCm 7.14.0, kept KEY32-on build:
4.180 ms untraced wall, 4.045 ms device; analysed trace span 4067.290 us
(`docs/MASTER-TABLE-2026-10-11.md:104,105,109`; `results_qwen300_2026-10-11/job2_run4.log:144,145`).
The 0.847 ms target need is wall time. Predicted wall = **4.180 + simulated span delta/1000**, transferring
cuts at 1:1, following the one KEY32 precedent (`docs/MASTER-TABLE-2026-10-11.md:112`), not adding host excess to a traced span.
The byte spec's 6.1904 GB byte floor is about 1.440 ms at 4.3 TB/s [E]; gaps above the proposed floors offer
0.425 ms on paper, not a measured removable pool (`docs/MASTER-TABLE-2026-10-11.md:82,83`; byte floor: `docs/master-table-2026-10-11/master.txt:18`).

Input identity: `tr_on.bin`, sha256 `aa68468246acde1a8d0e19ad0fbc6d5913bc37c7a3f352f44cecec8c0bdef320`;
checked plan `results_vlbest2_2026-10-10/b_all/plan.json`, sha256
`b32c3299d4165ba8d68c7b04461d9ed050818d397c7d99082928922836f9f315`.
[M] 60,362 tasks, **zero type/coordinate/worker mismatches**, 6,794,632 dependency pairs, 436 event names,
1,403 instances (`x7-schedule-whatif-2026-10-11/accounting.txt:1-3`, abbreviated **A** below).
The borrowed plan's rows match; this is not independent certification of its event graph against device code.
Compressed copies of both inputs, individual timing arrays and byte-spec provenance are retained beside this file.

## 2. Replay and bounded search

`search.py` uses **`etx.sim.xcd.simulate`**, with each task's exact recorded `done-ready` as its fixed body.
There are no invented operator costs, replacement kernels, precision changes, or bandwidth-derived body cuts.
Worker epilogue/fetch residual **q[t] = next_taken - done[t]** follows its original producer task.
Consumer residual **r[t] = ready[t] - max(taken[t], latest dependency body_done)** follows the consumer.
The additive replay is `taken = previous_done + q[previous]`, `ready = max(taken, dep_done) + r[t]`.
Initial worker takes remain fixed. It reproduces **all 181,086 stamps exactly** (A:4).
The 304 terminal q values have no measured successor: type-median imputation is [E], not zero tax.
Setting them to 0 or the largest observed q, 4.32 us, leaves the winner's layer period unchanged and gives
37.05–39.36 us step saving (A:28). This does not constrain costs under new memory contention.

Both branch models instead place free-edge notification before `max(event, worker)`; they retain local
setup floors of 0 or 0.52 us. Busy-edge local setup stays measured; a hidden notification <= min(1 us,
original busy interval) remains on the event branch. **Both reproduce the identical baseline stamps.**
The trace cannot identify this split; these are alternative estimates, not independent hardware fits.
Work conservation alone does not imply a conserved makespan: a fetch can finish while another worker runs.

Search definitions were committed in stages before each added tranche's results. The retained JSON spaces
name every candidate: **145** remap/order/scope cases, **32** queue/width/composite cases, **12** branch cases,
**6** exact-trace oracle cases, **6** frozen-finalist/profile cases, plus two terminal-tax sensitivities.
Five explicit illegal scope/dependency deletions are retained, not timed. All same-task cases check coverage,
38-worker/XCD capacity, original coordinates/types, scope legality, task-body equality, producer completion,
and completion of every task. Queue order stays topological; no event counts or producers are silently dropped.

- **Task-to-XCD maps:** original; rotate the intact prep/attention/reduce head groups through eight XCDs;
  balance dense tiles by role-coordinate median body cost (equal task count, greedy XCD work sums);
  spread the 20 attention partitions across eight XCDs. The wide map keeps head dependencies and promotes
  both head events to DEVICE. Promotion price sensitivities are 0, 0.3, 0.6 us [E]; no unknown scope cut is free.
- **Task waves/queue order:** coordinates, reversed, longest-first, shortest-first; per-role worker strides
  0, 6, 14, 19, 26, 32. A second family preserves the baseline worker-slot progression before applying offsets.
  Earliest-free-worker policies use median profiles; the oracle separately uses exact task costs from this trace.
- **Actual wave order:** moving phase-1 poll/acquire to wave 15 and permuting touch waves preserves the body
  and ready barrier in principle. Fixed residuals predict zero change; its actual delta is [U].
  `wave-scope.json` records the required runtime edits. There is no wave-level timing shape in these stamps,
  so in-body instruction-order savings cannot be simulated honestly without finer measurements.
- **Scope:** forced DOMAIN on the original map is illegal for **242 event names** with cross-XCD parties.
  Co-locating everything on one XCD makes DOMAIN legal, but serializes the same bodies onto 38 workers.
  Moving intact heads preserves existing DOMAIN scopes; spreading partitions requires promotion.
- **Boundary count:** simply deleting prep, reduce, MoE-fold or layer events deletes required dependencies.
  A whole-head composite is representable: unchanged prep + five attention + reduce calls, serially on one
  worker. An owner prep+attention composite can publish only after its whole body, delaying sibling partitions.
  The optimistic bounds below give away their internal ready/fetch costs and still lose. They are bounds,
  not certified same-cost winners. A parallel owner with mid-body publication needs a different protocol that
  the current plan's whole-body event emission does not express (`model.py:309-322`; `kernel.py:335-358`).

## 3. Results: period and reconciled step

Period is `(last moe_down.done - last in_norm.done)/48`, an end-to-end layer average, not steady-state
single-layer throughput. [M] baseline = **80.995 us**, with fixed prefix 14.760 us and tail 164.770 us:
**14.760 + 48*80.995 + 164.770 = 4067.290 us** (A:6-7). Positive deltas below are slower.
All absolute predictions are conditional on preserving recorded task durations and residual costs.

| Candidate / best of family | Period delta us/layer [E] | Step delta us [E] | Wall ms [E] |
|---|---:|---:|---:|
| Current layout | 0 | 0 | 4.18000 |
| Reset slots, reverse, stride 6 | +1.38667 | +71.23 | 4.25123 |
| Rotate heads, reverse, stride 6 | +1.37750 | +70.79 | 4.25079 |
| Balance dense XCD work, reverse, stride 6 | +1.80521 | +91.14 | 4.27114 |
| Spread attention, reverse, stride 6, +0.3 scope price | +1.98667 | +100.03 | 4.28003 |
| Median-profile earliest-free, coordinate order | +1.19063 | +57.15 | 4.23715 |
| Median-profile body+ready priority | +0.03125 | +1.50 | 4.18150 |
| **Exact body+ready priority, additive replay** | **−0.82000** | **−39.36** | **4.14064** |
| **Same frozen table, branch replay (both floors)** | **−0.53125** | **−25.50** | **4.15450** |
| Restrict to 6 / 14 workers per XCD | +265.067 / +100.865 | +13491.54 / +5148.25 | 17.67154 / 9.32825 |
| One XCD, all DOMAIN, optimistic scope rebate | +300.264 | +15320.96 | 19.50096 |
| Serial whole-head composite, optimistic bound | +20.85792 | +1001.18 | 5.18118 |
| Prep+owner attention, optimistic bound | +5.36604 | +257.57 | 4.43757 |

Sources A:8-9,25-39; every candidate is in the result JSONs and `candidate-ledgers.csv`.
`build-requirements.json` records the required compiler, placement or wrapper changes per timed case.
Reoptimizing the oracle for each branch model yields 0.686875 and 0.631667 us/layer saving (A:27), but those
are **different worker tables**. They are not evidence that the frozen nominal winner clears KEEP robustly.
The median policy has no foreknowledge of per-layer stamp jitter and fails the bar under every tested split.

## 4. Move versus remove: no free lunch

The winner retains all task work: **515432.80 worker-us** of bodies, **95322.92 worker-us** of additive ready
residuals, and **81246.40 worker-us** of queue residuals including imputation (A:5). These worker sums are
not added to the step. Some queue work becomes hidden behind another worker's body; nothing is erased.
Each candidate has matched consumer/event edge deltas for the union of old and new critical-path tasks.
Noncritical edges do not belong in a step sum. `edge-deltas.csv.gz` also retains every consumer edge for
one representative remap. Critical-path ledgers separately telescope to **every** candidate's whole step.

| Role on path | Gap before → after us [M→E] | Gap delta [E] | Body-path delta [E] | Net [E] |
|---|---:|---:|---:|---:|
| qkv | 183.15 → 157.15 | −26.00 | **+7.48** | −18.52 |
| router | 129.22 → 102.24 | −26.98 | **+23.32** | −3.66 |
| moe_down | 139.08 → 122.48 | −16.60 | −0.24 | −16.84 |
| moe_up | 52.20 → 51.68 | −0.52 | **+1.40** | **+0.88** |
| topk | 74.64 → 74.42 | −0.22 | −1.00 | −1.22 |
| Other roles | unchanged | 0 | 0 | 0 |
| **Total** | **875.49 → 805.17** | **−70.32** | **+30.96** | **−39.36** |

A:13-24. No individual body lengthens: remapping changes **which unchanged task** sits on the path.
The destinations of the debit are qkv/router/MoE-up body exposure; MoE-up's whole phase costs +0.88 us.
Opposite-signed matched edges are explicit: qkv_11 task 14026, E_99, **4.38 → 2.42 us (−1.96)**;
moe_down_5 task 7307, E_53, **1.36 → 2.32 us (+0.96)** (A:41-42). They illustrate moves, not a summable pair.
The total is **3222.76 body + 805.17 gap = 4027.93 us**, exactly current 4067.29 minus 39.36.
This is a **move with a conditional net period reduction**, not destruction of 0.425 ms of protocol work.
Rows 9–10 therefore have **25.50–39.36 us conditional net exposure recovery** in this frozen layout,
not a demonstrated 425 us removal. Only 3.0–4.6% of the 847 us target need is addressed.

## 5. Best legal layout, build, limits, GPU discriminator

The nominal best is `oracle-best-layout.json`: original XCDs, scopes, counts and dependency graph;
within each grid/XCD, sort by this trace's **body + ready residual**, descending, and assign each task to
that XCD's earliest-free worker. The output is a fixed, legal per-layer worker/queue table, not a dynamic
scheduler requiring future knowledge on the GPU. Its **selection** is clairvoyant and may fit one trace's noise.
Exact changes to build it: load this table after `compile_graph` and before kernel/plan-header emission
(`etx/tools/cli.py:42-55`), replace `TaskInst.worker` and `Plan.static_queues`, run `verify_plan`.
`apply_layout.py` supplies that opt-in hook. It is not wired into production; no tile, argument, dependency,
precision, worker count, or boundary-count edit is needed. Regenerate `plan_data.h`, plan and megakernel,
then compile with the kept build's flags. Changed identity needs fresh import checks and token correctness.

[U] The trace cannot decide notification/setup overlap, placement-induced HBM/L2/MALL contention,
poll jitter, physical CU-slot variation, touch survival, or generalization over 32 decode positions.
Fixed-duration replay deliberately cannot reprice those bodies. BAR2's +0.095 ms is one measured protocol
failure, not a proof over worker permutations (`docs/BUDGET-2026-10-01.md:257`).
A future GPU session must first stamp arrive/fetch, notification and touch separately; then compare the fixed
oracle table and median-profile table against current queues, with traces at multiple positions and paired
untraced wall measurements. KEEP remains >=0.030 ms, every gain positive, median >= twice spread,
32/32 HF greedy tokens, and same-build `host --check` for every imported launch. No session ran here.

## 6. Corrections and reproduction

DEP §6 prices KEY32-off; this analysis rebases to KEY32-on. **XCD-PREFETCH contains no explicit list of
impossible boundary-count candidates**; its §3 states phase dependence. The actual owner/mid-body limitation
is in `examples/vllm_llm/model.py:309-322` and DEP §3/§6. The brief's row-10 expression omits the already
booked 0.013 ms fused-edge discount: `(470.1−312)/1000 = 0.1581`, not 0.145; MASTER-TABLE includes it.
The 1.00 us/edge floor is a pricing estimate: 33 measured baseline path gaps are already below 1.00 us
(minimum 0.68 us); the winner has 35, also minimum 0.68 us. No floor clipping was used.
Conservation of total work is not conservation of latency; the alternative replay splits also prevent
promoting a nominal counterexample to a measured refutation. **WEAK** is the supported verdict (A:43-45).

Run `search.py`, `secondary.py`, `branches.py`, `oracle.py`, `finalists.py`, `tails.py`, `report.py`, then
`timeline_check.py`, then `build_requirements.py` with the workspace's `.venv/bin/python`. Scripts use retained compressed
inputs, require no network; full run logs are retained. Timeline CLI outputs cover both full steps and layer 24.
Validation: **18 passed** (`tests.log:2`); every timing body/dependency/completion and candidate reconciliation
was checked. Artifact hashes are in `SHA256SUMS`; no generated output or future build hook is GPU evidence.
