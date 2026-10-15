# ETX: complete measured status and defect closure, 2026-10-15

The CPU fixes and reporting corrections do **not** establish a new GPU result. Historical DeepSeek batch measurements
favoured ETX at B=1/2 and favoured vLLM at B=4; B=8 failed its hidden-state gate. The older vLLM batch logs lack the
short-run correctness gates now required by the harness, so those comparisons are historical evidence, **not a fresh
certification under the repaired protocol**. The batch-8 numerical cause and batch-4 performance cause remain open.

Clock correction from the two-gap preparation: the historical DeepSeek `BATCH` step field used a HIP-event device
span, despite its wall label. Its application pairs mixed ETX device time with vLLM's host full-minus-short estimate.
The new host uses matching host-clock brackets. Historical wall labels below are subject to this correction.

Scope: code, experiment tooling and reporting, including the earlier single-request studies, both Qwen studies,
batching, design-only proposals and interrupted experiments. No paid resource or remote experiment was used for this
closure. Operational watchdogs outside this repository were neither changed nor independently verified.

## Measurement conditions and limits

DeepSeek figures below use DeepSeek-Coder-V2-Lite-Base, bf16, one MI300X VF, context 1024 and 32 greedy tokens per
request. Single-request lever decisions use five interleaved **device-time** pairs; the application comparisons use
**host-wall step throughput** and a separate worst-request token interval. Hidden-state gates cover 27 layer boundaries
on the first decode step, not all layers on all 32 steps. Tokens are compared with HF greedy references; vLLM uses the
corresponding exported prompt/token windows. Colliding windows at requests 0/4 limit request-mixup detection at B=8.

Every gain is relative to its named reference in the same session. Absolute times from different sessions are never
combined into a performance verdict. These lever and application studies do not replace the separate Qwen fusion claim
against the **same kernels in a HIP graph**; see [the fusion evidence](VLBEST2-2026-10-10.md). No new same-kernel HIP-graph
comparison was performed in this task.

## Single-request experiments: completed and still open

The original [experiment log](../examples/dsv2lite/results/EXPERIMENT-LOG.md) and
[D1–D5 decision register](DSV2-D1-D5.md) preserve the session-specific rules and records.

- **Initial tile studies, October 13:** terminal-load removal Q / Qall measured -3 / -3 us alone; top-k rank +66 us;
  norm-tree-all +37 us (spread 7); norm-tree +30 us, attribution only; attention prologue +27 us; argmax tree +2 us
  (spread 21), inconclusive alone. These were initial attribution studies, not retrospectively subjected to the later
  30-us KEEP rule. The five-lever combination gained 92 us; its noQ follow-up gained 108 us versus that session's
  default. Removing Qall then helped by 14 us, but later kept-arm ablations reversed that interaction.
  Original raw records: `examples/dsv2lite/results/2026-10-13-ds_inplace/`, including `followup/`.
- **Local-only initial work:** `norm_gamma_early` passed local gates but was never measured alone;
  `attn_cache_double_buffer` was rejected locally in three variants by scratch/register growth. Neither is a GPU win.
  The Fleet six-switch port passed local checks but never ran on GPU. Existing Fleet local-fold, attention-variant,
  cached-working-set, prefetch, split-worker and published-top-k losses remain historical exclusions, not new ETX
  measurements. Their sources and conditions are in the experiment log and `~/code/fleet-mi300x/docs/STATUS.md`.
- **Compiler control:** ROCm 7.14 cost 62 us versus the 7.2.3 build in the same historical follow-up. That is a compiler
  comparison, not a machine calibration or permission to reuse absolute times across sessions.
- **D1 safe acquire:** default is the safe protocol. Relaxed sc0 is an explicit attribution arm, never the kept arm;
  the later full-hour session measured a 232-us safety cost (spread 13).
- **D2 system-coherent loads:** NO-GO, -9-us gain versus default and 237 us slower than relaxed. The old “+9” was a
  time increase, not a positive gain (`2026-10-13-runs/sess_run2/pairs_coh.log`). Code identifies changed cache access
  and writeback work; the isolated cost of each component remains unmeasured.
- **D2b L1 mode:** the interrupted 05:25 run remains NOT COMPLETE. It was subsequently completed in
  `2026-10-13-hour/`: +330 us versus default (spread 13), 99 us faster than relaxed. Its registered equivalence
  condition required distance from relaxed <=26 us, so the result is **NO-GO**, even though it is faster.
  Separately, `alld4r0l1` passed its own KEEP rule: +334 us versus `alld4r0` (spread 19), +498 us versus default.
  The aborted attempt is not a remaining need to repeat an already completed study.
- **D3 four K-chunks:** NO-GO, -741 us versus `all` (spread 19). **D4 routed slices:** initial arm NO-GO, -271 us;
  zero-slice `alld4r0` KEEP, +61 us versus `all` (spread 8). Optimized slices `alld4s` still NO-GO, -13 us versus
  `alld4r0` (spread 12). **D5 prefetch:** DROP on `all`, pairs 2/-1/8/0/8 us, median +2, spread 9. D5 with L1 is
  unmeasured. Queue order and source mechanisms constrain explanations but do not isolate dynamic costs.
- **K1 shared kv_a:** KEEP, gains 40/40/41/48/44 us, median +41, spread 8. **M1 four-wave merge:** KEEP,
  76/65/55/74/66 us, median +66, spread 21. **Kept-arm noQ ablation:** -29/-29/-23/-32/-22 us, lever STAYS.
  **NoA ablation:** -13/-6/-21/-6/-14 us, lever STAYS, below a resolved improvement. All 40 timed invocations passed
  32/32 and 27/27. The K1 trace span 3380.9 us is attribution only; its comparison with an older trace is cross-session.
  Source: [K1 session](../examples/dsv2lite/results/2026-10-14-k1/RESULT.md).
- **KM, K1+M1:** KEEP versus M1 alone, gains 48/52/50/58/54 us, median +52, spread 10. Against the earlier baseline,
  gains 120/125/122/120/119 us, median +120, spread 6. All 15 invocations passed. Current recipe:
  `alld4r0l1kvsmw`. The individual +41 and +66 values must not be added as its measured gain.
  Source: [KM session](../examples/dsv2lite/results/2026-10-14-km/RESULT.md).

The October 13 full hour also captured default, D4r0 and D4r0+L1 stage tables. Earlier missing traces and post-fetch
hashes remain provenance limits of those earlier attempts; later captures do not retroactively seal them.

## Both Qwen experiments

The cache-mode model study uses Qwen3-30B-A3B, bf16, B=1, context 1024, 32 tokens, one MI300X.
The row-ownership study measures a weight-read microbenchmark on gfx942 (304 CUs); it is not a model decode run.

- **MoE row ownership:** two NO-GO microbenchmarks. Original `qwen/moerows.txt` records in sessions
  `20261014T134920Z-324307` and `20261014T142757Z-03ff3e` give medians **-6.01% / +3.42%**, spreads
  **44.68% / 44.38%**. The older -0.12% / +14.29% claims were unsupported. Neither completed noisy experiment
  establishes a stable direction; no row-ownership model arm is justified by it.
- **Common cache mode:** NO-GO, gains **9/10/-5/14/0 us**, median +9, spread 19. Three pairs win, one loses,
  one ties; `every_positive=0` means “not all positive.” Both builds passed their import certificates at 0/1040 and
  32/32 tokens in session `20261014T183414Z-68d6f6`. This does not transfer DeepSeek's gain to Qwen. DeepSeek's
  +330-us D2b number has the distinct NO-GO equivalence rule above; the +334-us kept-combination result is the KEEP.

Raw directories are under `examples/dsv2lite/results/2026-10-14-batch-<timestamp>/`. Earlier Qwen router narrowing
(+62.6 us) and norm hoist (+55.6 us) are already in its kept baseline. Its rows-per-task sweep lost and the row-wave
prototype was rejected statically; see [the gains ledger](GAINS-LEDGER.md). None was rerun here.

## Batches and attribution

The historical deciding record is `2026-10-14-batch-20261014T142757Z/20261014T142757Z-03ff3e`, five rounds against
selected AITER (the faster legal configuration under the then-recorded gates, among default/AITER only):

- **B=1:** 274 versus 206 tok/s, historical throughput +32.98%, spread 1.50%; latency gain +28.80%.
- **B=2:** 403 versus 354 tok/s, +13.95%, spread 0.60%; latency gain +10.11% versus vLLM. This is **better latency
  relative to vLLM**, not a 10% increase. ETX's latency still increases relative to its own B=1 case.
- **B=4:** 410 versus 567 tok/s, -27.70%, spread 6.18%; latency gain -29.62%.
- **B=8:** NOT COMPLETE. Actual x8 logs show r5 prefix=3 and r7 prefix=7, both with 32/32 greedy and warm-up tokens.
  A prefix count identifies the first failed boundary, not every later boundary. The genuine M1-only fallback fails
  too: K1 is not necessary for failure. The four-label diagnostic ran one binary and proves nothing about x8p/x8m/x8k.

Historical B1/B2/B4 arithmetic was recomputed from raw logs, but **all fail current vLLM legality checks because the
short subtraction runs were not gated**. The old per-request cache/reference tuples also lack the new producer-binding
certificates. No corruption or short-run divergence is asserted; those missing records cannot be recreated on CPU.
The old B4 build projection used an averaged-subset expert union, not the executed prefix. Its 5.454-ms prediction
is not a validated build trigger; exact-prefix picks are missing. Current reports leave that projection PENDING.

The corrected bulk-payload model is `B * (4,902,621,184 + 31,104 * context)` bytes: 4.934/9.869/19.738 GB at
B=1/2/4 and context 1024. Dividing by historical wall medians gives 1352/1990/2022 GB/s. These are modeled useful
payload rates, **not measured HBM bandwidth**. The old 5.20-GB value belongs to Fleet's earlier counter capture.
B4's 4.590-ms floor at the assumed 4.3 TB/s is below vLLM's 7.054-ms measured step: bytes alone do not force the loss.
Replicated work on fixed request slots, load issue, spills, synchronization and HBM traffic remain competing costs.
Column batching is a design candidate; it is not a proven fix for this loss.

The reported 17.6% cross-session difference is **not a measured hardware offset**. The KM and batch manifests record
executable SHA-256 values `443e6a613cfe40b54d0bac9357e6152d431bc862d2687bc6500e3333f75f9d00` and
`7266d55116aa0afef52105e4eb95916dee44b194600ce431222b7e4d28a27527` respectively. Plans/generated kernel sources
match; executables do not. Compiler, host, runtime and machine effects are not separated. `boxcontrol` now explicitly
records a **within-session recipe difference**, one invocation per recipe, and `machine_offset_measured=false`.
It supplies neither a five-pair verdict nor a conversion between machines.

## Design-only work and unfinished questions

[Four Qwen proposals](PROPOSALS-2026-10-14.md) remain unbuilt and unmeasured: router remapping (P1), epoch/static-only
reset (P3), producer-side row-partitioned combine (P4), and last-arriver top-k (P5). Pre-shuffled weights (P2) are
already implemented; their row-ownership remainder is the noisy study above. P1/P3 lack a demonstrated ceiling above
the bar, P4 needs exposed combine-time evidence, and P5 needs its deciding synchronization microbenchmark. No proposal
is a queued or achieved optimization. Column-batched DeepSeek tiles and a Qwen batch study remain unbuilt/unmeasured.

Blocked GPU questions: x8 arithmetic/reference versus concurrency; real reference/cache identity for the old rotations;
B4 traffic and phase critical path; stable Qwen row-ownership timing; isolated cache/D3/D4/D5 costs; D5+L1 interaction.
Local schedule proofs do not establish payload visibility, and correct tokens do not waive hidden-state failures.

## Defect disposition and validation

The 14 audit fixes were already in main at `1679f72`; they were retained, not reimplemented. Their before/after logs
are preserved in `etx-close-known-issues-evidence/etx-auditfix-evidence/` under the result directory named below.

- **B1 scheduling/build identity:** fixed producer dependencies and candidate-specific binaries; local runner and gate
  tests pass. Historical four-arm results remain withdrawn; corrected GPU gates are missing.
- **B2 routing subsets:** fixed exact executed-prefix accounting; tests pass. Old prefix picks are absent, so the old
  projection is blocked, not repaired by substituting an average.
- **B3 golden binding:** fixed producer-bound reference verification; tests pass. Old tuples cannot be certified from
  hashes alone; real retained payloads or fresh certified references are missing.
- **B4/B5/B6 baseline legality:** fixed short-call gates, nonpositive timing refusal and preservation of failed/extra
  sequence gates; regression tests pass. Historical short outputs remain missing.
- **B7 verifier honesty:** fixed false success by explicitly excluding unsupported batch semantic replay and returning
  incomplete. Complete automatic raw-batch semantic replay remains unsupported; file seals alone do not prove verdicts.
- **B8 model override:** fixed refusal of unprepared models; tests pass.
- **N1–N6:** fixed checked-window collision warnings, standalone golden prep, rerun artifact closure, unsupported CLI
  bindings, ignored mode/arm arguments and Qwen config transport; regression tests pass.
- **Manifest audit:** fixed classification of two tile-internal row-wave macros. The two runtime protocol switches are
  accepted only as explicit `#error` rejection guards; removing their rejection still fails the audit.
- **Recipe control:** fixed misleading offset schema/output; both gates and positive medians remain required.
- **Combined session:** fixed omission of `boxcontrol` from `all_verify`, discovered by the complete CPU suite.
- **Layer diagnostics:** the previously unmerged instrument prints every boundary without weakening the prefix gate;
  single-request runs use the selected request's reference. Portable host-gate tests pass. No numerical correction is
  claimed, and the real r5/r7 diagnostics have not run.
- **Bulk-byte model:** fixed dense-layer router overcount (262,144 bytes/request). An independent shape-count regression
  fails before and passes after. Floors remain estimates; this is not a measured performance improvement.
- **Reporting:** corrected pair interpretation, raw moerows values, binary identity, machine-offset claims, D2b status,
  historical batch gate coverage, B2 latency sign, unsupported B4 attribution and stale “not yet measured” claims.

The complete CPU suite finished on the resumed source: **2,016 passed, 17 skipped, 0 failed in 550.81 seconds**
(`.venv/bin/python -m pytest -q -ra`). Skips require absent historical data/builds, optional toolchains/source
checkouts, or Linux `/proc`; they are not passes. The preserved focused suite passed **171 tests**. Before-fix logs
show the manifest audit, recipe schema and bulk-byte regression each failing, and the combined session refusing
to load; the full suite now includes those regressions. All five real session dry runs passed (`ds_batch`,
`all_verify`, `x8_diag`, `x8_diag_qwen`, `rerun_skips`); relevant shell syntax and `git diff --check` passed.

ROCm 7.2.4 gfx942 device assembly/ISA and complete HIP host translation-unit checks pass for kept B1 and x8;
all ten retained device artifact checksums verify. Kept B1 uses 0 scratch B/lane; x8 uses 144. Full x86 HIP host
objects exist for both. The resumed x8 object differs from the prior object only in ELF symbol/string table contents;
it is not claimed to have the same binary hash. Final HIP executable linking and GPU runtime loading remain
unverified. Compilation does not establish numerical correctness or identify the reason for x8's hidden-state failure.
Native LLVM 23 kept-arm checks pass; x8 code generation initially hit a 240-second deadline, then completed in
399 seconds under a 600-second deadline. Its gfx942 assembly and HIP host device-IR checks also pass. Native and
ROCm compiler outputs are separate builds; their resource numbers are not interchangeable.

Main's `a21613b` Qwen results/original-log update is preserved byte-for-byte. Revision IDs and complete local
build outcomes are recorded in the closure result alongside the unfiltered logs:
`/Users/ckl/.hermes/cache/scratch/codex-close-known-issues-result.md` and
`/Users/ckl/.hermes/cache/scratch/etx-close-known-issues-evidence/`. The recomputation script and JSON there retain
raw-log arithmetic and explicitly mark the historical batch certification gap. Synthetic regression logs test the
harness only; they are never substituted for GPU measurements.

## Preparation update, no new GPU evidence

The [two-gap preparation](DEEPSEEK-PREP-TWO-GAPS-2026-10-15.md) implements the missing recipe-matched B1 control,
producer-bound reference refusal/regeneration path, and a correctness-only x8 queue. It also prepares the separate
B1/B2/B4 attribution and corrected-baseline queue. The prior closure's missing-implementation statements below are
historical; the linked acceptance record gives current local checks and remaining native/device gates.
No new performance verdict or historical-reference certification is claimed. The preparation review also found that
DeepSeek's old `BATCH step_median_ms` read `hst.last_ms`, a HIP-event device duration, despite its wall label.
The old comparison mixed this ETX device field with vLLM's host full-minus-short estimate. The new host brackets
both fused and graph paths with the same host clock. Earlier numerical arithmetic remains reproducible, but a matched
host-wall verdict requires the new session. Older wall descriptions in this historical report are subject to this correction.

## Minimal next paid experiment, not authorized or executed here

The [actionable experiment plan](NEXT-GPU-2026-10-15.md) specifies the controls, exact acceptance thresholds,
decision branches, local readiness blockers and a 30-minute target / 60-minute hard cap (estimated $1–$2 at the
historical $1.99/GPU-hour rate, subject to an actual quote). It is **not ready to rent**: the matching B1 control and
correctness-only session still need implementation/local validation, certified r5/r7 payloads are unavailable locally,
and final HIP linking/runtime compatibility remain unverified. No reservation or spending was authorized here.

After local target builds and input-binding checks pass, the smallest useful GPU work is correctness-only: use the
same one-request recipe as the x8 candidate on r5 and r7, plus the current kept recipe as a secondary control; record
all 27 layer metrics and binary/input hashes. Compare with a fresh x8 gate before attributing a difference to
concurrency. A kept-only B1 pass changes both recipe and batch and cannot by itself isolate concurrency. Reuse the
retained references only if their binding can be verified. Otherwise regenerate just the needed reference set first.
No throughput run follows a failed hidden gate. Once this discriminator is understood, run the four correctly named
x8 candidates if still needed; B4 counters and corrected baseline pairs are a separate justified experiment.

Estimate: 5–10 GPU minutes for six diagnostic invocations once builds/inputs are ready, plus 10–20 minutes if references
need regeneration; these are unmeasured planning estimates. No rental is justified solely to regenerate the completed K1/M1/KM, Qwen cache or D2b results.
This task spent $0 on GPU resources. Historical rental cost/destroy-time claims are not independently re-audited here.
