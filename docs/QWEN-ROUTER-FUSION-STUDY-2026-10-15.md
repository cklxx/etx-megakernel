# Qwen producer-side router fusion: executed CPU study, 2026-10-15

## Decision

**Original proposal: not yet supported; NO-GO as an exact bf16-preserving rewrite.**
Moving the RMS scale after scoring is an identity over real numbers, with the learned
normalization weight included. It is **not** the arithmetic in the shipped bf16 path.
The executed CPU model gives a small, reproducible counterexample that changes the
selected eighth expert. This disproves the equivalence premise; it does not establish
that the actual Qwen checkpoint would fail 32-token decoding, or that the mechanism
would be slower on a GPU.

**Current kept-build speedup: unknown.** There is no matching retained current trace,
no candidate machine code, and no measured partial-scoring or publication tail. A
historical schedule sensitivity sweep crosses zero widely; it is not a gain interval
or a throughput prediction. Do not rent a GPU to rediscover the rounding counterexample.

The weakest assumption is that a single delayed scale preserves the bf16 normalized
activation seen by the router. It does not. After relaxing that premise, the weakest
performance assumption is that partial scoring plus the last-arriver tail, and the
normalized input still required by the experts, are cheaper than the removed work.

The next cheapest deciding action for the **exact** route is already complete: the CPU
counterexample. For a useful successor, implement and statically check the separate,
exact-preserving Qwen P5 last-router-arriver/top-k protocol before any hardware request.
That successor does **not** implement producer-side o_proj scoring. An approximate
producer-scoring research branch would first need real router inputs/weights and CPU
replay under the unchanged policy; no tolerance relaxation is proposed or authorized.

No production kernel, default, precision, AGENTS.md, cloud resource, or credential changed.
GPU spend for this study: **$0**. All new executions were on CPU.

## Scope and evidence identity

Started from clean main `8d4ec985df145a31e5bae96bb3ceb3eda562d6ae`; both remote main
refs matched before work. Work was isolated in `study/qwen-router-fusion`, at
`/Users/ckl/code/.wt/etx-router-study`. The separate live `/Users/ckl/code/etx` worktree
was only read for retained artifacts. Deliverables and evidence are committed together.

Scope is Qwen3-30B-A3B, bf16, batch 1, context 1024, 32 greedy tokens, one MI300X.
The [config](../examples/llm/configs/qwen3-30b-a3b.json) confirms 48 layers, H=2048,
128 experts, top-8, epsilon 1e-6, and renormalized selected weights. The router matrix
is 128 x 2048 x 2 = 524,288 bytes (0.5 MiB, 0.524 MB), per layer.

The latest retained control has five wall times 4.051, 4.052, 4.042, 4.059, 4.048 ms:
median **4.051 ms**, device median **3.917 ms**. These are the **kept** rows of
[pairs_kept_l1.json](../examples/dsv2lite/results/2026-10-14-batch-20261014T183414Z/20261014T183414Z-68d6f6/qwen/pairs_kept_l1.json),
not the rejected L1 arm. That run has no retained matching task trace in its indexed
capture. We never subtract October 11 stage costs from those October 14 times.
The October 12 4.078/3.943 numbers and C1/C5 gains have lost their original pairing
records ([baseline evidence status](BASELINE-STATE-2026-10-12.md)); they are historical
recorded measurements, not re-derived measurements in this study.

Artifact directory: [2026-10-15-router-study](../examples/vllm_llm/results/2026-10-15-router-study/).
`manifest.json` records input/source hashes. `SHA256SUMS` covers retained evidence,
including raw files, full stdout, failed execution, guard mutants, and test reports.

## Was this exact mechanism already tried?

The exact signature searched is: **o_proj output-row owners score all experts on their
own hidden slices before global normalization; the last arriver reduces scores, applies
one RMS scale, and performs top-k.** No implementation or GPU run of that signature was
found in the searched ETX reachable history, current Qwen operators/session records,
Fleet source/history, STATUS and results. It was **genuinely new in the searched record**
before this study; now **CPU-only, disproved as an exact identity; GPU untested**.
This is bounded negative evidence, not a claim about unretained or private experiments.
Search commands and reachable-history hits are in `history-search.log` and the manifest.

Related work has different dependencies:

- **Norms folded into GEMV consumers: implemented and GPU-tested.** ETX `86df586`,
  with residual-buffer fix `98a7b8d`; current `dsl_stage_norm_at` first waits for the
  full residual, then normalizes before dot products. It does not delay the scale.
- **Router narrowing and norm-weight load hoist: implemented and GPU-tested/kept.**
  `f8542bc`, `2ac1fdd`, absorbed by `45e1d36`; see `kept_build.py` and the gains ledger.
  They specialize one-row GEMV and issue gamma loads earlier. Original GPU pairing
  logs were lost, as disclosed above. Neither performs partial scoring in o_proj.
- **Qwen KEY32 selection: implemented and GPU-tested/kept.** `130bcf4`, documented
  in [TOPK-KEY32](TOPK-KEY32-2026-10-08.md). Retained October 11 off/on output files,
  copied byte-for-byte into `routing-originals.json.gz`, cover 48 layers x positions
  0/1040 x IDs/weights x two builds. Our CPU audit verifies all 384 hashes, all eight
  replicas, and **96/96 off/on ID and fp32-weight cases identical**. This validates
  reuse of those historical outputs, not the producer-scoring candidate.
- **Top-k recomputed in every MoE consumer: implemented and GPU-tested, rejected for
  Qwen's useful shape.** `b617b12` and `f439088`; the 448-task fold was costly.
  Current shape-derived `fold_topk` retains a standalone replicated top-k.
- **Qwen P5, last router task performs top-k: design-only before and after this study.**
  [PROPOSALS section 5](PROPOSALS-2026-10-14.md) is not a run. It retains the original
  normalized router scores, changes their publication, and never moves scaling past
  the dot. Its old 0.040–0.052 ms upper estimate and -0.003…+0.017 ms visibility case
  cannot be added to this proposal's allowance or transferred to the current build.
- **Fleet/DeepSeek last-arriver published top-k: implemented and GPU-tested, lost.**
  Fleet `72aa666` and `docs/STATUS.md:29,78`. Preserved original JSONs in `fleet-raw/`
  give device medians 3.66168/3.65732/3.66012 ms per-task versus
  3.67756/3.67016/3.67544 ms published; corresponding wall medians per invocation
  are also retained. The exact raw medians are used instead of smoothing STATUS's
  rounded range. Fleet reduced repeated consumer routing but added a serial tail.
  Different model, 64 experts, different graph: **not Qwen evidence against this mechanism**.
- **ETX Fleet-port local fold F: implemented, CPU/compiler-only, GPU unmeasured**
  in `results/2026-10-12/fleet-port.md`; it assembles eight expert rows before qkv
  and final lm_head. Its source explicitly excludes router R=1 and o_proj.
- **Fleet last-writer residual fold and row-split o_proj: different mechanisms.**
  STATUS v0.11/v0.12 (`fc47acf`) records a serial fold loss; `261f8c9` row-split is
  implemented/prepared but unmeasured in the consulted record. Qwen already owns
  complete output rows. Neither proves the delayed-scale identity.

Also consulted: `docs/DATA.md`, `ETX_Technical_Design.md` section 15,
`REPORT-STATUS-2026-10-15.md`, `GAINS-LEDGER.md`, `XCD-TIMELINE-2026-10-11.md`,
`MASTER-TABLE-2026-10-11.md`, `ISA-FIXES-2026-10-11.md`, `TILES-2026-10-07.md`,
and Qwen's `results/2026-10-12/fleet-port.md`.

## Actual dependencies and numerics

Read together: `model.py:296–370`, `host.hip:385–445`, `gemv.py:547–697,779–806`,
`topk.py`, `tile_util.h:73–130`, and `gfx942.yaml:116–131`.

1. o_proj is H=2048 by 4096, 256 tasks, **8 complete output rows per task**, not
   split-K partial output elements. Each fp32 GEMV result is stored as bf16 `o`.
2. Post-attention norm is **already folded into each useful router task**, not a
   standalone phase. There are still separate o_proj, folded-router and top-k phases.
   Router's 256-task grid has 128 useful rows and 128 early returns. The useful tasks
   use four 512-element segments; the persistent block is 1024 threads/16 waves.
3. Let Q be round-to-nearest-even bf16, stored then expanded exactly to fp32.
   `z[i] = Q(float(o[i]) + float(resid_X[i]))`. R=1 here; there is no eight-expert
   sum at this boundary. The previous qkv fold created residual X; router task 0
   writes residual Y. Using o_proj's unrounded accumulator would already be a change.
4. The vectorized norm sums eight adjacent squares per active thread, wave-reduces
   by XOR distances 32,16,8,4,2,1, then adds wave partials in increasing wave order.
   `s = rsqrt_fp32(sum(z*z)/2048 + 1e-6)`. Gamma is learned bf16 `ln2`, not one constant.
   The normalized vector is `x[i] = Q(fp32(fp32(z[i]*s)*gamma[i]))`.
5. Router scores are `l[e] = Q(dot_fp32(W[e], x))`. Dot8 accumulates in element
   order; the 64-lane butterfly feeds four wave partials added in wave order. The
   source's FMA contraction, order, and final bf16 store all matter. Norm hoist
   changes load timing only. Packing `W*gamma` to bf16 adds another rounding;
   packing to fp32 doubles its bytes and still cannot remove Q(x).
6. Kept KEY32 ranks the **rounded logits**, lower expert ID winning finite ties
   (signed zeros equal). It computes exp(logit-max), all-128 softmax with wave
   summation, then renormalizes selected probabilities in selection order. IDs are
   ordered slots; each slot maps to an XCD/expert. Reordering the same set is not
   an interchangeable output. The experts consume fp32 mixing weights, used by
   the down path, and the bf16 normalized input `ox`, published by router task 0.
   Equal top-8 membership alone is not correctness.

Over real numbers, `W(s*z*gamma) = s*W(z*gamma)`. Even there, s can be omitted for
ranking only when finite and positive; it cannot be omitted from softmax: changing
s changes temperature and mixing weights even after top-8 renormalization. In bf16,
`Q(dot(W,Q(s*z*gamma))) != Q(s*dot(W,z*gamma))`. A different partial reduction order
adds another source of error. Atomic floating sums in arrival order are not acceptable.

The CPU model follows the source's reduction DAG and rounding points using NumPy
float32 and double-intermediate scalar FMA emulation (not a universal correctly rounded FMA implementation for arbitrary
float32 triples; the sparse witness is unaffected). Its **rsqrt and exp are not ROCm
intrinsics**, and it is not an instruction-set emulator or an imported-launch check.
The sparse witness has one nonzero product per competing expert, so its discrepancy
is not attributable to reassociating a long dot, exp accuracy, or gamma omission.
Every deferred candidate is granted the **same scale as the original** to isolate the
rounding change; moving the square reduction could introduce further differences.
The sparse witness also survives +/-16 float32 ULP perturbations of rsqrt, tested
independently with scalar arithmetic. Tests compare routing against the existing
`etx.ops.topk_key32` CPU oracle.
Nonfinite inputs are rejected rather than claiming to reproduce GPU NaN/overflow behavior.

### Reproducible counterexample (synthetic)

Use residual zero, bf16 o = (3,1,1,...,1), gamma all one. Squared sum is exactly
2056; s ≈ 0.9980521202. `Q(s*3)=3` and `Q(s*1)=1`.
For experts 0–6 put weight 2 on column 2. All other weights are zero except
`W[7,0]=0.50390625` and `W[8,1]=1.515625`.

The current path gives rounded logits 7 and 8 both **1.515625**, so expert **7** wins
place eight. Deferred scaling gives **1.5078125** and **1.515625**, so expert **8** wins.
Both inputs and weights are representable bf16. The same witness holds in the
original real-valued deferred dot: the activation rounding is the decisive difference.
This is a deliberately constructed near-tie, not a Qwen activation or a token failure.

Executed seed **20261015**, 64 full-shape independent synthetic rows: deferred8 changes
**3,455/8,192 logit values**, ordered top-8 IDs in **6/64**, membership in **1/64**, and
fp32 mixing-weight bits in **64/64**. These are descriptive synthetic counts, **not**
an estimated Qwen failure probability. Width 64, reversed producer reduction, and
keeping the original dot DAG while moving only the scale give the same ID mismatch
counts. Omitting gamma changes membership in 63/64. Per-case values are retained.
Tests cover exact ties, scale-dependent mixing, zero norm, residual RNE, reduction
order, malformed trace input, visibility/epoch requirements, and cross-build refusal.

Real artifacts were sought in the indexed records, `~/code/etx-cache`, local ETX
captures, `~/.cache/huggingface`, and `~/llm`. Router weights, o, residual and gamma
needed for an actual-model replay were not found. Retained top-8 outputs are reused
for audit but cannot reconstruct the missing 128 logits or hidden vector. No download,
checkpoint substitution, synthetic-as-real claim, or new golden generation was used.

## Ceiling and schedule simulation

The ideal bandwidth floor for the router matrix at **4.3 TB/s** is **0.122 us/layer**,
5.853 us over 48 layers. This is bytes/bandwidth, not observed latency. The matrix is
still read by producer-side scoring; moving it does not save its bytes.

**Matching historical source:** `results_qwen300_2026-10-11/tr_off.bin` and its own
`tr_off.log`, retained here losslessly. Raw trace SHA256
`5e3a64b30f5b596ddcec0a7b255e1fe8a3b3b26e5420cd64daf203ed73e7fb7e`;
60,362 tasks, 436 printed types, span 4303.54 us, 32/32 in the same log, wall median
4.471 ms, device median 4.299 ms. It predates narrowing/hoist and is KEY32 **off**.
The companion historical plan was no longer present locally. We therefore use the
trace log's own type order and validate every type count and timestamp ordering;
**we do not claim a new 60,362-row plan/dependency-identity check**. The older ledger
records such a check, but this run is a phase-window replay, not a full DAG resimulation.

For each of 48 layers, save all 256 o_proj finish offsets. The disjoint interval from
last o_proj finish to first MoE-up readiness totals **870.95 us** (median 18.13 us/layer):

- last o_proj → first router ready: 91.46 us;
- first router ready → last router done: 300.03 us;
- last router done → first top-k ready: 65.53 us;
- first top-k ready → last top-k done: 405.86 us;
- last top-k done → first MoE-up ready: 8.07 us.

These telescope exactly, including signed edges where applicable. They are **phase
envelopes**, not sums of all task durations or all idle-worker time. The 8.07 us is
an earliest-consumer window residual; it is not the device's top-k fanout latency.
In particular it is not interchangeable with the ledger's critical-path 51.2 us edge.
The median o_proj finish skew is only **1.15 us**: the final producer's scoring work
cannot hide under an earlier producer. No gain credit is assigned to preceding o_proj
work or to unrelated idle workers.

The historical critical-path ledger independently assigns router body 255.7 us,
o_proj→router gaps 135.8 us, top-k body 387.1 us, router→top-k gaps 74.0 us and
top-k→MoE gaps 51.2 us (`docs/dep-gap-ledger-2026-10-11/account.txt:45–46,64–73`).
The two candidate-deleted incoming gaps total about **209.8 us** in that old
critical path: a boundary-only ceiling before paying replacement visibility, not
an expected gain. Top-k arithmetic and payload publication remain necessary. Those ledger allowances
**are not added** to the phase-window model. Its boundary-only opportunity is not
its entire 870.95 us window; removing bodies requires replacing their useful work.

### Candidate costs that survive

The literal producer scheme partitions hidden dimensions in 256 slices of 8. Each
producer must read 128 x 8 bf16 weights (2 KiB), compute 1,024 FMAs, and publish
128 fp32 partials. Total work remains 262,144 FMAs/layer. Partials occupy **128 KiB**;
write plus read is **256 KiB/layer** (12 MiB/token), before scale partials, counters,
residual/normalized output and cache-line amplification. A last worker must reduce
**32,640 additions** in a deterministic order, compute s and top-8, and publish IDs,
weights and normalized expert inputs to eight XCDs. Parallel expert lanes reduce the
serial instruction chain, but a single last-arriver workgroup has much less throughput
than the existing router grid.

Current row-major W gives each producer 128 short 16-byte slices at a 4096-byte row
stride. Layout `[producer][expert][8]` makes one producer's 2 KiB contiguous without
changing stored weight values. It adds a packing/identity obligation; cache-line
traffic can exceed the logical minimum without it. FP32 preweighted gamma is 1 MiB,
and bf16 preweighting is numerically different. No additional packed-matrix benefit
is assumed. No independent gain is credited for already-kept norm hoisting/narrowing.

Every new scratch tensor crossing XCDs needs the runtime's proven cache/fence policy.
A release increment must follow payload visibility; the winning acquire must see all
partials; epoch/reset state and exactly-one-winner need proof. A CPU happens-before
model checks these logical preconditions in 20 shuffled orders. It is **not evidence
of MI300X coherence**, fence latency, or stale-read absence on hardware.

Residual Y and `ox` remain outputs, even after deleting router tasks. Normalization
requires the global scalar before `ox` can be materialized. It either runs in the
last-arriver tail or needs a scalar broadcast and parallel producers/consumers.
The simulation prices both serial and ideal parallel normalized-input completion.
It also omits neither the final partial's scoring nor result fanout. Extra live
accumulators, new LDS/partial buffers, instruction footprint, register pressure,
spills and persistence occupancy are **unmeasured**. A positive analytic case cannot
qualify a build without resource and correctness checks.

### Executed sensitivity, not a speedup forecast

`finish_times` replays each layer's actual producer offsets with:

```
last = max(o_proj_done + partial_scoring) + payload_visibility
ready = last + tail + fanout
tail = reduction + topk + normalized_input       # one serial worker
    or max(reduction + topk, normalized_input)  # ideal separate workers
saving = historical_first_moe_ready - ready
```

Explicit **assumptions**, not fitted measurements: partial scoring 0.25/1/4/8 us;
reduction 0.5/2/8 us; normalized-input production 1/4/8 us; top-k 2 us;
visibility/fanout pairs (0.5,1), (0.9,1.8), (1.8,3) us; both tail placements.
All **216** combinations are retained, not just favorable ones. The resulting
historical sum ranges **-607.45 to +666.95 us/token**. This is an arbitrary-parameter
stress envelope around an obsolete trace, **not a supported range for any build**.
The original numerics are invalid for exact substitution irrespective of its sign.
No current tok/s estimate is justified. The alternative P5 has a different cost model.

## Bounded hardware experiment design (design-only, no spend)

**Decision changed:** after a CPU-correct exact-preserving P5 prototype exists, does
integrating top-k with the last **router** completion remove enough end-to-end
publication latency to justify a full-model arm? This is a named revision, not a
silent replacement of the user's proposal. **Do not run the original delayed-scale
variant on a paid GPU under an identity claim.** No GPU candidate was implemented
because CPU evidence closes that premise. The following is not launch-ready code.

1. **Local gate first (not yet implemented for P5):** freeze current main plus kept
   recipe and all flags; compile an opt-in candidate retaining original norm, router
   dot DAG, rounded logits, KEY32 ordering/weights and `ox`/residual publications.
   Account for the eight top-k replicas and coherent result fanout. Run the 16-build
   local matrix, UC-plan audit, host syntax, resource/ISA inspection with local LLVM
   and ROCm device libraries, and the VM-matching ROCm 7.2.4 x86 container compiler.
   Exercise adversarial ties, zeros, replica equality, exactly-once arrival, stale
   epochs, aborted launches and reset. Require no unaccounted spill/occupancy regression.
   Commit one gpu_jobs session file, dry-run every non-GPU stage, with complete logs,
   deadlines and progress. **These candidate-specific gates are pending, not passes.**
2. **Microbenchmark first, bounded to 15 GPU minutes [estimate].** A = current
   256-task router completion (128 useful producers on four XCDs, empty tasks retained),
   event → eight KEY32 consumers → local MoE reader readiness. B = same producer
   scores and norm publications, unique last-router winner does identical KEY32,
   release/acquire fanout to one reader per XCD. Include real `ox` readiness in both,
   not just a 64-byte score record. Hash exact source, compiler, flags, ELF/HSACO,
   task placement, packed layout and captured input bytes. A and B use the same GPU,
   same allocations/cache policy and fresh state. Verify actual XCD placement.
   Synthetic adversarial scores first, then retained real 128-logit captures if
   available; current top-8-only outputs are insufficient. Perform visibility stress
   across epochs with sentinel/checksum and per-reader exact outputs. Any stale value,
   mismatch, timeout or placement failure stops before timing.
3. **Timing:** five interleaved pairs, order AB/BA/AB/BA/AB, each a predeclared equal
   warmup and iteration count (100 warmups and 1,000 timed micro-iterations, with
   progress every 100). Use GPU elapsed time from producer start through **all eight
   readers ready**, including visibility, last tail and fanout; stamp diagnostic
   subintervals separately. Do not compare unrelated timer domains. GO only if every
   pair improves, median saving **>=0.625 us/layer**, and median **>2 x (max-min)**.
   The 0.625 derives from this project's **30 us/token / 48 layers** lever KEEP bar;
   it is necessary, not sufficient, for a model gain. No shorter or trimmed sample
   substitutes for an incomplete run. Miss → NO-GO, no full model arm.
4. **Only on GO: full-model test, another 45 GPU minutes [estimate], hard overall
   GPU session cap 60 minutes including setup/checks.** Fresh `host --check` on every
   imported launch at positions 0 and 1040, same-build certificates; DSL routing
   IDs/weights, residual and normalized input byte checks against A; **32/32 HF greedy
   tokens for every run**. Full unfiltered stdout and failures retained. Use five
   interleaved A/B pairs, host `RESULT wall_ms` as the deciding basis; device span
   secondary. Incremental KEEP: all positive, median >=0.030 ms, median >2 x spread.
   Separately compare candidate fused versus its **same-kernel HIP graph** with
   `pairs.py --basis wall`: all positive, median **>2%**, median **>2 x gain spread**.
   Compare against the strongest tested legal vLLM configuration on the **same GPU,
   session, prompt and wall estimator**, at least S3 plus any proven stronger current
   arm. Report both baselines; an incremental tile win is not ESTABLISHED fusion.
5. **Evidence and stop:** record UUID, GPU model/VF, clocks, ROCm/compiler/library/model
   revisions, golden hash, source/build/plan/input hashes, all env overrides, checks,
   every warmup and pair log, no hidden retries; take post-gate trace/stamps separately
   from uninstrumented timing. Per-step 60 s microbenchmark deadline, 180 s silent
   warning/probe during known long model loads, explicit build/load/check deadlines
   bounded by the session cap. Instrumentation perturbation must be measured before
   using stamps quantitatively. On any failure report NOT-COMPLETE, not a partial
   timing verdict. Capture/fetch before immediate teardown and verify no VM remains.

For budgeting only, at the **historical, unverified $1.99/GPU-hour assumption**, the
15-minute probe is about **$0.50**, and the one-hour ceiling about **$1.99**; current
pricing and provisioning are not checked or authorized. Setup exceeding the cap means
NOT-COMPLETE. Hardware alone can settle memory visibility, register/occupancy effects,
actual partial/tail latency, current build attribution, and full-model token correctness.
This design reserves no resource and executes no cloud command.

## Reproduce and validation

From the repository root, Python with NumPy and pytest installed:

```sh
python -u -m examples.vllm_llm.router_study --out /tmp/qwen-router-study-replay
python -u -m examples.vllm_llm.router_study_guards --out /tmp/qwen-router-guards
python -m pytest -q tests/test_router_study.py
python -m pytest -q -ra
# Inside the retained artifact directory:
shasum -a 256 -c SHA256SUMS
```

Default inputs are the committed lossless raw captures; `--out` can differ so originals
are preserved. Random seed and 64-case count are fixed by default. New GPU measurements
are neither required nor silently synthesized.

The initial execution failed because the parser incorrectly expected `moe_up_0_x`;
actual trace name is `moe_up_0`. `initial-parser-failure.log` is preserved; corrected
replay completed, counts/timestamps and disjoint windows are regression-tested.
Three intentional child-process mutants fail the guards: cross-build transfer,
missing normalized-input dependency and counter-only visibility. The restored suite
passes **14/14** (`guard-execution.log`, `guards-restored.log`). These are actual pytest
failures followed by passing tests, not merely expected-exception assertions.
Full-suite results and remaining limitations are recorded in `validation.md` beside
the raw full-suite output. GPU compilation of a nonexistent candidate and paid runs
are explicitly not performed; the original path has not been declared GPU-ready.
