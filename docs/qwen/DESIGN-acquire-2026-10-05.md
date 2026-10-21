# Best design: remove the per-task L2 invalidation from imported vLLM kernels (coherent activation loads)

Status: design with a local trace check. No GPU rental. Section 9 describes the check.
Merged 2026-10-05 from two independent agent designs of this repository.
Labels: [M] measured, [E] estimate on measured numbers, [U] unknown.

## 1. The claim this decides

**vLLM's own kernels, imported unchanged except for the memory-ordering policy of their
activation loads, run faster fused in one ETX megakernel than the same kernels launched one
by one.**

This is candidate 2 of `docs/NEXT-2026-10-03.md`, done as a compiler pass over imported IR.
It changes no tile, grid or algorithm, so it stays inside `CLAUDE.md` section 1.

Deliberately **not** candidate 1 (PhiMoE). PhiMoE runs ETX's own DSL tiles;
`docs/NEXT-2026-10-03.md` says so itself: neither outcome there proves the all-imported claim.

Decision unblocked:
- **Go:** one full-model A/B with both baselines, then the same pass on Mixtral (roadmap 2 and 4).
- **No-go:** record that imported kernels cannot be fused profitably on MI300X under ETX's
  current event protocol, and move the next lever to the host step work (section 2, last row).

## 2. Ceiling, before anything is built

| quantity | value | label | source |
|---|---|---|---|
| bytes per token, dense 1.84 + experts 3.62 + lm_head 0.62 | 6.08 GB | M | `docs/qwen/MODEL-vllm-kernels.md:481` |
| same ledger, other accounting: 5.57 + 0.62 | 6.19 GB | M | `docs/qwen/MODEL-vllm-kernels.md:37` |
| bandwidth floor, 6.08 GB at 4.3 TB/s (doc says 1.44 for 6.19) | 1.41-1.44 ms | E | both lines above |
| baseline, same kernels unfused (`qknr`, wall, VF) | 5.787 ms | M | `results_2026-10-01/round24/FINDING.md:28` |
| ETX fused, same build | 6.071 ms, penalty 0.284 | M | same line |
| vLLM default on this box | 4.862 ms | M | `docs/qwen/VLLM-LADDER-2026-10-01.md:15` |
| phases / events per token (`qknr`) | 13 / 628 | M | `round24/FINDING.md:13-14,71` |
| kernel boundary | 1.61 us | M | `docs/core/DATA.md:67` |
| ETX handoff floor per edge | 1.00 us | M | `docs/qwen/BUDGET-2026-10-01.md:267` |
| today's fused excess per edge | 0.284 / 628 = 0.45 us | E | derived |
| host work outside the device timer, fused vs unfused | 0.197 vs 0.036 ms/step | M | `docs/qwen/RESULTS-2026-10-01.md` |

Gap budget [E]:
- Against unfused, if every edge reached the 1.00 us floor: (1.61 - 1.00) x 628 = **0.38 ms/token,
  6.6% of 5.787**. The 10% target in `docs/qwen/PLAN-v3.md` section 1 is not reachable on this path.
- Against today's fused build: at most 0.67 ms/token can be recovered.
- Prediction from the DSL lever: 0.73 x 628 = **0.46 ms/token**, i.e. fused about 0.18 ms ahead.
- Against vLLM: even at the 0.38 ceiling fused `qknr` lands near 5.41 ms against 4.862. The rest
  is imported structure (one-workgroup 16.24 us chain, `docs/qwen/PHASES-2026-10-01.md:26`).
  **This experiment decides the fusion half of the claim, not the vLLM half.** Say that in the report.

## 3. The deciding unknown and the microbenchmark

**Unknown [U]:** when a real imported consumer reads its activations through coherent `sc0 sc1`
loads and skips `buffer_inv sc1`, does it read zero stale data, does it stay byte-identical to
vLLM's original, and how much does it save per edge in the imported setting?

Why this is the lever, with numbers:
- imported tasks invalidate their whole L2 per phase boundary; DSL tiles do not.
- per read: inv + plain load **2.89 us** against coherent load **0.50 us** (16 B, `docs/core/DATA.md:67`,
  `results_2026-10-01/hw_cohload.log:2-3`). Warm 28 KB phase with vs without `inv sc1`:
  3.14 vs 0.76 us span (`results_2026-10-01/hw_restart.log`).
- the same lever on DSL: `ETX_VL_COH` -0.317 ms over 436 events = 0.73 us/edge
  (`docs/qwen/RESULTS-2026-10-01.md:70`).

Why it may not transfer [U]: part of the DSL saving came from the touch hook's lines surviving
in L2 (`etx/ops/coh.py`) and imported kernels have no touch hook; coherent 16 B loads could also
slow `wvSplitK`'s own body.

Difference from fleet: `--coherent-acts` measured no gain there (3.718 vs 3.709-3.715,
`~/code/fleet-mi300x/docs/STATUS.md:32,78`), but fleet already had an L1-only acquire inside one
XCD (STATUS:66), so it had no whole-L2 invalidation to remove. This is a new condition.
Uncached (MTYPE-UC) memory is not retried: fleet bench (d'') shows it does not work (STATUS:302);
fleet bench (d''') shows agent-scope atomic payloads work without fences (STATUS:256,542).

**Pair:** `attn_reduce` (4 tasks) -> `wvsplitk_y2` o_proj 2048x4096 (304 tasks), real shapes, fixed
random weights. Largest measured excess wait in the imported build: o_proj 42.78 against 24.49 us,
reduce 35.98 against 15.78 (`docs/qwen/WHY-DSL-TILES-2026-10-01.md:104-105`).

**Variants:**
- **A** today's protocol: producer `wbl2 sc1` + event, consumer `inv sc1` + plain loads.
- **B** transformed consumer: coherent activation loads, weights plain, no invalidation. Stores stay
  plain + writeback (coherent stores cost 0.223 ms in `topk`, `docs/qwen/BUDGET-2026-10-01.md:297-301`).
- **C** the two original kernels as two launches in a HIP graph.
- **N** negative control: plain loads, no invalidation, consumer caches primed by the previous
  epoch. **It must show mismatches.** Zero mismatches is inconclusive, not a pass
  (`docs/core/EXPERIMENT-fresh-handoff.md:88-89`).

**Conditions:** idle, and busy with a co-running HBM stream (`bench/overlap/overlap_bench.hip`).
Three alternating A/B/C orders per condition. Epoch-tagged payload, byte-for-byte comparison with
the original kernel's output every round.

**Counts:** 1,000,000 rounds each for B and N per condition (the commit `4c753f9` rule), 200,000
each for A and C.

**Metric:** pair span, producer start to consumer last finish, raw p50/p95/p99.

**Cost [E]:** about 15 GPU-minutes, $0.75 at $2.99/GPU-hour.

## 4. Thresholds, fixed before the run

Let s be the median over the three busy pairs of span(A) - span(B), P = s x 628.

| outcome | condition | action |
|---|---|---|
| correctness fail | any mismatch in B, any output not byte-identical, or N inconclusive | stop. No full model. Close the transformation until the memory contract explains it; report the stale pattern |
| no-go | P < 0.284 ms (s < 0.45 us), or span(B) >= span(C) | stop, delete the VM. Imported fusion cannot be repaired by load policy on this protocol. Next lever: the ~0.16 ms host step work (0.197 vs 0.036 ms) |
| inconclusive | 0.284 <= P < 0.40 ms | no full model this session. Report the band; combine with the host lever first |
| go | P >= 0.40 ms (s >= 0.64 us), span(B) < span(C), correctness gates pass | run the full-model A/B |

0.40 = 0.284 + 0.116, where 0.116 is 2% of 5.787. Stricter than NEXT's 0.35, which leaves only a
1.1% margin.

**Full-model gates:** `host --check` byte-identical (separate `ETX_VL_CHECK=1` build, it changes
timing); 32/32 tokens equal to HF greedy; output-coverage check passes.

**Full-model win:** fused + transformation beats `qknr` unfused in every one of **5 interleaved
pairs** on the wall basis, median gain above 2% and above twice the paired relative spread.
Report device span alongside wall time. Include both baselines: `qknr` unfused, vLLM default
4.862, and vLLM with q/k-norm-rope fusion. A fusion win alone advances the imported-kernel claim;
only a same-session win against the fastest vLLM control claims system competitiveness.

## 5. Local preflight, no GPU

1. `.venv/bin/python -m pytest -q`, `bash -n` on session scripts, all 16 builds of
   `examples/vllm_llm/local_matrix.sh`, local gfx942 compile with `ETX_DEVLIBS`.
2. `examples/vllm_llm/rocm_build_check.sh qwen3-30b-a3b` for `qknr` and the transformed build.
   VGPR and spill counts must not exceed today's `qknr` (8 spills, LDS 41448 B, `round24/FINDING.md:13`).
3. **Memory-contract audit, fail-closed.** Classify every pointer argument of every imported
   kernel: read-only weight; cross-XCD activation; activation inside one XCD; KV; in-place read and
   written in the same kernel (`fused_add_rms_norm`, the q/k norm-rope chain, `reshape_and_cache`);
   scalar-loaded through the K$ (`etx/runtime/include/etx/import.h:95`). Anything in-place,
   scalar-loaded or unknown keeps its acquire. A buffer may have a transformed reader only if every
   same-step reader of that buffer is transformed.
4. **ISA diff per kernel.** Only cache-policy bits on activation loads may change; `global_load_dwordx4`
   stays x4 with `sc0 sc1`; no `buffer_inv sc1` remains in transformed tasks; weight loads unchanged.
   Anything else is a kernel rewrite, out of scope.
5. **Event-order argument, in writing:** plain store, `wbl2 sc1`, waitcnt, release event, consumer
   poll, coherent load, ordered under the gfx942 memory model (fleet STATUS:542 says the writeback
   is required).
6. Build the pair microbenchmark with ROCm 7.2.4 in the container, then dry-run the session job
   graph (deadlines, `stdbuf -oL`, progress lines).

**Reuse, do not regenerate** (`docs/core/DATA.md`): `round24/` traces and FINDING; `hw_cohload.log`,
`hw_restart.log`, `baremetal/hw_all.log` (do not re-run `lat`, `launch`, `cohload`, `restart`);
the `vprof_qknr` trace and Qwen3-30B-A3B checks in `results_2026-09-28b`; the ETX MoE unfused
rocprofv3 trace in `results_2026-09-28`; `examples/llm/results/golden_c1024_qwen3-30b-a3b.txt`;
the `vllm/vllm-openai-rocm:v0.27.1` tag; fleet STATUS lines 32, 66, 78, 302, 542.

## 6. Session

One 1x bare-metal VM, ROCm 7.2.4, `tools/hotaisle_grab.sh 1x`. **Allocated 120 minutes = $5.98**
at $2.99/GPU-hour. Check the balance first.

**One script:** `examples/sessions/imp_acquire.py`, a `gpu_jobs` job file, written and committed
before provisioning, full unfiltered logs under one directory, per-step deadlines and progress,
early exit on a failed gate.

Order: pair microbenchmark 15 min -> correctness gate 10 min (`--check`, 32/32) -> 5 interleaved
rounds of transformed-fused / fused / unfused / vLLM default / vLLM qknr, wall and device, about
30 min -> last, one decode trace of the transformed build (trace past the prompt end;
`--max-steps 8` traces only the prompt, `docs/qwen/RESULTS-2026-10-01.md`). CPU lanes from minute 0:
environment, model download, `docker pull` (5-10 min), packing, four builds.

Delete the VM when the work is done, verify "No virtual machines", record the balance.

## 7. What would make this worthless, and the cheapest check

1. **The invalidation is not on the critical path.** Check in minutes, no GPU: from
   `round24/t_r24_imp.log` and `t_r24_dsl.log`, layers 2-45, per edge compute
   next first-ready - (first-ready + last-done), plus body excess on shared phases
   (attention, reduce). First confirm whether the timestamp sits before or after the invalidation,
   and that the trace covers decode steps, not the prompt. **If the imported excess is below
   8.3 us per layer (0.40 / 48), do not rent.**
2. **The transformation is really a kernel rewrite** (16 B loads split, scalar activation loads,
   VGPR or spills rise). Check: preflight 2 and 4 in the ROCm 7.2.4 container.
3. **In-place or aliased arguments cover too many edges.** Check: audit step 3; recompute
   P_max = covered edges x 1.06 us (0.45 penalty + 0.61 to the floor). Stop if P_max < 0.40.
4. **The penalty is mostly host-side.** From 0.202 ms on the traced `imp` run against 0.036 ms
   unfused, fused and unfused are already about equal on device. Does not stop the experiment, but
   section 4's wall threshold includes the host share.

### Limits of the layer budget tool

`etx.tools.layer_budget` requires one trace row for every task in the plan.
Each row must match the task type and coordinates.
All three timestamps must be positive and satisfy `taken <= ready <= done`.
The tool rejects missing rows and zero timestamps before it reports numbers.
This check confirms trace completeness. It does not confirm model correctness.

The original `gap` field uses type-id order, as the host table does.
Type-id order is not dependency order.
The `dependency_gap` field uses a topological order of phase dependencies from `task_in` and `ev_inst`.
The lowest type id resolves each tie.
Consecutive phases in this order do not necessarily share a dependency.
If dependency lists are absent or phase dependencies have a cycle, this metric is unavailable.

Both metrics subtract the previous phase's last-done time from the next phase's first-ready time.
Each source layer owns its outgoing gap, including an edge to another layer.
Negative overlaps can cancel positive gaps in both sums.
Neither sum measures exposed device time.
Do not use either sum alone for the 8.3 us rental gate.
Check the dependency path and parallel phases before a rental decision.
The host log has no gap column. Its phase table cannot independently confirm gap arithmetic.
A hand-built trace test checks the arithmetic with known timestamps.

The round11 decode trace uses MI300X VF, batch 1, and a sequence length of 1055 tokens.
It has 436 phase rows and a device span of 5081.5 us.
For layer 2, type-id order and dependency order each give 9.95 us.
For layers 2–45, each mean is 9.003864 us per layer.
These values describe this trace only. They do not establish the round24 imported excess.

The round11 data set is external because Git ignores `results_2026-*/`.
If the data set is absent, the acceptance test fails with an explicit message.
Set `ETX_DATA_ROOT` to the directory that contains `results_2026-10-01`.
An explicit path disables the default search, so another local copy cannot hide an absent data set.

## 8. Sources

`CLAUDE.md` and `AGENTS.md` for the rules; `docs/NEXT-2026-10-03.md` candidates 1 and 2 for the
ranking and the precision rules; `docs/core/ROADMAP.md` sections 1, 2 and 4 for the claim, the stages and
the acceptance gates; `docs/qwen/MODEL-vllm-kernels.md` lines 37 and 481 for the byte ledger and the
floor, line 586 for the phase count; `results_2026-10-01/round24/FINDING.md` for the paired wall
medians and the resource table; `docs/qwen/WHY-DSL-TILES-2026-10-01.md` for the acquire mechanism and the
excess waits; `docs/qwen/RESULTS-2026-10-01.md` for the `ETX_VL_COH` lever and the host-step correction;
`docs/qwen/BUDGET-2026-10-01.md` for the per-phase budget, the 1.00 us handoff floor and the coherent
store cost; `docs/core/DATA.md` for the kernel boundary, the per-read prices and the reusable data sets;
`docs/qwen/PHASES-2026-10-01.md` for the single-workgroup chain; `docs/core/EXPERIMENT-fresh-handoff.md` for
the negative-control rule; `docs/qwen/PLAN-v3.md` section 1 for the 10% target;
`docs/core/CPU-READINESS.md` for the preflight gates; `bench/overlap/overlap_bench.hip` for the busy
condition; `~/code/fleet-mi300x/docs/STATUS.md` lines 32, 66, 78, 256, 302 and 542 for the coherent
protocol, the uncached result and the writeback requirement.

Provenance: this design merges two independent agent runs over this repository on 2026-10-05
(`droid exec` with Opus 5.5 and with GPT-6 Sol).  Both runs chose NEXT candidate 2 over candidate 1
independently.  One run reasoned from `docs/qwen/MODEL-vllm-kernels.md:481` (6.08 GB, 1.41 ms floor), the
other from line 37 (6.19 GB, 1.44 ms floor); both numbers are in that file and section 2 lists both.
The per-edge prices, the negative control and section 5 come from the longer run; the explicit
AITER control in section 4 and the one-million-round rule come from the shorter one.

## 9. Raw trace check for section 7 item 1

Use `etx.tools.layer_budget` with the plan from each build. No GPU or VM is required.
Set `PLAN_A` and `TRACE_A` to the imported build paths.
Set `PLAN_B` and `TRACE_B` to the reference build paths.
Then run this command from the repo root:

```sh
/Users/ckl/code/etx/.venv/bin/python -m etx.tools.layer_budget \
  "$PLAN_A" "$TRACE_A" --compare "$PLAN_B" "$TRACE_B" \
  --json /tmp/layer-budget.json > /tmp/layer-budget.txt
```

The report contains both phase tables, layer gaps, wait totals for each worker, and the exposed critical path by task type.
The last tables show A minus B for each layer and each worker.
The JSON file also contains each phase edge.
The tool rejects differences in task count, task type, or task coordinates between a trace and its plan.

Definitions:

- The phase table has the same columns and precision as `host.h::report_phases`.
  First-ready is relative to the earliest task-taken timestamp.
  Last-ready and last-done are relative to that phase's first-ready timestamp.
  Total span is latest done minus earliest taken.
- Phase order is type-ID order, as in the host table.
  Each gap is `next first-ready - (this first-ready + this last-done)`.
  The source layer owns the edge, including the edge to the next layer or `lm_head`.
  Negative gaps remain negative because phases can overlap.
  Names must end with `_<layer>` or `_<layer>_x` for layer assignment.
- Each worker's wait total is the sum of `ready - taken` for its tasks in that layer.
  `wait_per_worker` is the sum across workers divided by the number of workers that execute tasks in the trace.
  Workers with no task in a layer contribute zero.
  Wait totals measure worker time. Do not add them to exposed device time.
- `body` in the difference table is the sum of differences in mean body time for shared attention and reduce phases.
  The default selector is `^(attn|reduce)_`. Use `--shared-body REGEX` for other names.
  Both traces must contain the same selected phase names in each layer.
  `excess` is the gap difference plus this body difference.
  Compare layers 2–45 with the 8.3 us threshold in section 7.
  This sum is a pre-check metric. It is not a measured acquire cost.
- The critical path uses the existing timeline analysis and dependency lists from the plan.
  Its split contains body, dependency gap, and queue gap time for each type.
  The tool reports when dependency lists are absent. Such a path uses queue order only.

The code in `etx/codegen/kernel.py` places the ready timestamp after acquire and the workgroup barrier.
Thus, wait includes acquire time and event wait time.
The done timestamp follows the body, domain release, and barrier. It precedes event publication.
A raw trace alone does not identify the decode position or prove output correctness.
Check the run log and the source for each build before a rental decision.

### Round 11 correspondence check

The data resides outside git, as specified in `docs/core/DATA.md`.
The test first checks this checkout, then `~/code/etx`.
Use `ETX_DATA_ROOT` to specify another data directory.
The test skips only if the raw data is absent.

```sh
/Users/ckl/code/etx/.venv/bin/python -m pytest -q tests/test_layer_budget.py
/Users/ckl/code/etx/.venv/bin/python -m etx.tools.layer_budget \
  /Users/ckl/code/etx/results_2026-10-01/round11/plan_r8base.json \
  /Users/ckl/code/etx/results_2026-10-01/round11/trace_r11_decode.bin \
  --json /tmp/round11-budget.json > /tmp/round11-budget.txt
```

[M] The trace and plan contain 60,362 tasks and 436 grid instances.
Every raw task type and coordinate matches the plan.
All 436 printed phase rows match `trace_r11_decode.log` exactly.
The rounded device span is 5081.5 us.
The log header says 434 grid instances. Its phase table and the raw data contain 436.
The test uses the raw count, not this incorrect header count.

Conditions: MI300X VF, Qwen3-30B-A3B, batch 1, decode position 1054, sequence length 1055.
`round11/FINDING.md` identifies this final decode step.
The log reports 32/32 tokens equal to HF greedy.
These gaps come from the traced device step. They are not wall times or differences between builds.

Measured layer gaps in us, including each layer's outgoing edge:

```text
layer gap    layer gap    layer gap    layer gap
 0   10.50    12    7.05    24   10.85    36   12.30
 1    9.80    13    7.76    25    8.95    37    9.54
 2    9.95    14    9.18    26    8.26    38    9.13
 3    8.34    15    9.66    27   10.00    39    9.78
 4    8.28    16    9.11    28    9.02    40    8.18
 5    8.77    17    9.92    29    7.92    41    9.90
 6    9.15    18    7.91    30    9.00    42   12.41
 7    9.59    19    8.45    31    7.05    43    9.12
 8    6.78    20    9.13    32    9.42    44    8.96
 9    8.20    21    7.51    33   10.71    45   10.69
10    8.56    22    8.87    34    7.25    46    9.98
11    8.40    23    9.02    35    8.14    47    8.51
```
