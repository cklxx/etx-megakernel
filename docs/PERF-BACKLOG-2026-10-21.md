# Performance backlog, ordered for implementation, 2026-10-21

Source: the owner's 17-item list (`/tmp/etx-perf-backlog.md`, 2026-10-21). Every reference below was re-checked against `origin/main` at `edc2cbe`.
- Labels: [M] measured, [E] arithmetic from cited inputs, [U] not in the record.
- Thresholds marked **[plan]** are set by this document, before any run, and are not evidence.

## Rules

1. **States.** Every item has exactly one state:
   - `OPEN`;
   - `READY-FOR-GPU`: Mac work done, its acceptance `Mac` command passes;
   - `DONE`;
   - `CLOSED-BY-MEASUREMENT`.
2. **What DONE means.** An item is `DONE` only when its acceptance command has passed, with its number, under the conditions in its row. The command and its output are recorded in the item's commit message.
3. **No GPU item closes from a Mac.** A GPU item becomes `DONE` only with a log produced by `etx.tools.gpu_jobs` on the GPU box. The session's `record.log` must name the device (`gfx942`), and the log must be committed under the session's results directory.
   - A Mac run, a simulator, the numpy model, `--dry-run` or a CPU check can only move a GPU item to `READY-FOR-GPU`.
   - cc-grok rejects a `DONE` claim whose evidence path is not a GPU session log.
4. **Items 7 and 14 are `CLOSED-BY-MEASUREMENT`.** No task, branch or GPU job may be opened for them. Reopening needs a new measurement that contradicts the cited one, brought to the owner first.
5. **Workflow.**
   - One commit per item, English message, starting `perf #<n>:`.
   - cc-astra implements. cc-grok verifies the commit and the evidence before the state changes. cc-o55 orders the work and gates.
   - GPU sessions follow `CLAUDE.md` §4: everything local first, one prepared and committed job file per session, full logs, the VM deleted when done.
6. **Hard stop for cc-astra:** two failures in a row on the same cause, or 40 minutes on one task. Then stash, write the error to `/tmp/awb-questions.md`, and take the next task.
   - No defensive code that the task's test doesn't exercise.
   - Squash PRs into `main` only after the touched-area tests pass.

**Models:** Q = Qwen3-30B-A3B (`examples/vllm_llm`), D = DeepSeek-V2-Lite (`examples/dsv2lite`), G = GLM-5.3-Flash (`examples/glm53`).

## Summary table, in execution order

| Order | # | Item | Model | GPU? | Owner | Target file | State now | Phase |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | runner exclusion | all | no | astra | `etx/tools/gpu_jobs.py`, `tests/test_gpu_jobs.py` | OPEN | M0 |
| 2 | 15 | D2H outside the step timestamp | G | no | astra | `tests/test_glm53_slice.py` | fix on main (`4590434`), needs a recorded acceptance | M0 |
| 3 | 17 | floor fraction in every result | Q, G | no | astra | `examples/vllm_llm/bench.py`, `tests/test_bench_floor.py` | G done (`cea884a`), Q missing | M0 |
| 4 | 7, 14 | closure records | — | no | astra | this file, §Closed | CLOSED-BY-MEASUREMENT | M0 |
| 5 | 3 | weights shared across requests: byte account | Q | **yes** | astra | `examples/vllm_llm/dense_bytes.py`, `tests/test_dense_bytes.py` | OPEN | M1 → S1 |
| 6 | 1 | replication removed for dense nodes: task count | Q | **yes** | astra | `examples/vllm_llm/model.py:199-202`, `tests/test_batched_plan.py` | OPEN | M1 → S1 |
| 7 | 2 | column batching on, timed | Q | **yes** | astra | `examples/vllm_llm/bench.py` (`--recipe batched`) | OPEN | M1 → S1 |
| 8 | 12 | 6-token GEMV pass | G | **yes** | astra | `examples/glm53/tiles.hip`, `examples/glm53/options.py` | OPEN | M1 → S1 |
| 9 | 13 | `attn_reduce` per-launch cost at B8 | Q | **yes** | astra | `etx/importer/spec.py:82,282,350` | OPEN | M1 → S1 |
| 10 | 9 | sentinel instead of counter barrier | D | **yes** | astra | `examples/dsv2lite/` relay barrier, `etx/runtime/include/etx/primitives_amdgcn.h` | count done [E], no code | M1 → S1 |
| 11 | 11 | streaming bandwidth from occupancy and async depth | Q, G | **yes** | astra | `bench/hw/hw_bench.hip` (`lf`), `examples/sessions/perf_s1.py` | OPEN | M1 → S1 |
| 12 | 5 | in-place a4w8 dequant microbenchmark | G | **yes** | astra | `bench/dequant/a4w8_bench.hip` (new) | OPEN | M1 → S1 |
| 13 | 8 | q → attention → o_proj chain share | Q | **yes** | astra | `etx/tools/timeline.py` on an S1 trace | OPEN | M1 → S1 |
| 14 | 10 | B4 dependency and queue gaps | D | **yes** | astra | `examples/dsv2lite/` scheduler, `tools/ds_batch_attribution.py` | attributed, unfixed | M1 → S1 |
| 15 | 6 | TP4 all-reduce cost | G | **yes, 4 GPUs** | astra | `bench/sync/allreduce_bench.hip` (new), `examples/sessions/perf_s2.py` | OPEN | M2 → S2 |
| 16 | 4 | MoE routed union | Q | **yes** | astra | `examples/vllm_llm/union_bytes.py` (new); tile only if #2 passes | design only | after S1 |

**Sessions:**
- **S1:** one MI300X, items 1, 2, 3, 5, 8, 9, 10, 11, 12, 13 in one job file.
- **S2:** four GPUs, item 6.
- **S3:** item 4, only if item 2 is `DONE` with a win.
- **Session cost and budget:** [U]. The owner sets them before provisioning, as `CLAUDE.md` §4 requires.

## Phase M0: Mac only, measurement trust and records

**#16 Runner exclusion.**
- Evidence: a concurrent runner read B2 at 121 tok/s instead of 350 [owner's list; the number is not in the repo, U]. The fix lives outside the repo.
- Change: `gpu_jobs` takes an exclusive, non-blocking `flock` on `/tmp/etx-gpu-<index>.lock` for every GPU it schedules, for the whole run.
  - A second runner that needs a held GPU exits non-zero, printing the holder's PID.
  - `etx/session/lease.py:17` shows the pattern.
- Acceptance (Mac): `.venv/bin/python -m pytest -q tests/test_gpu_jobs.py -k exclusive` → `2 passed`. One test: the second runner on a held GPU refuses with a non-zero exit. The other: it runs on a different GPU.
- Done: the test passes on main. GPU flag: no.

**#15 D2H copies outside the step timestamp.**
- The fix is on main: `4590434`, `examples/glm53/model_host.hip`, test `tests/test_glm53_slice.py`.
- Acceptance (Mac): `.venv/bin/python -m pytest -q tests/test_glm53_slice.py` → `0 failed`. Plus `git grep -n "after_wait" examples/glm53/model_host.hip` showing the timestamp before `after_wait` (`docs/GLM-EXECUTION-DESIGN-2026-10-19.md:517-520`).
- Done: both outputs recorded in the commit message. GPU flag: no. The fix's effect on numbers is checked inside S1 by every timed item.

**#17 Floor fraction in every result.**
- GLM has it (`cea884a`, `examples/glm53/run_all.py`). Qwen's `bench.py` writes `median_decode_wall_ms` with no floor (`docs/GLM-EXECUTION-DESIGN-2026-10-19.md:494-497`).
- Change: Qwen `bench_result.json` gains `bandwidth_floor` (bytes per token, `floor_ms_per_token`, `fraction_of_floor`, rate used), computed as in `run_all.py`.
- Acceptance (Mac): `.venv/bin/python -m pytest -q tests/test_bench_floor.py` → `1 passed`, using a synthetic `bench_result.json` that must carry `fraction_of_floor`.
- Done: on main. GPU flag: no.

**#7, #14 closure records:** see §Closed. One commit records them; no code.

## Phase M1: Mac preparation for S1

Each item ends at `READY-FOR-GPU` with its Mac acceptance, then is validated in S1.
Before provisioning, these must pass with numbers:
- `examples/vllm_llm/local_matrix.sh` (16 builds);
- `examples/vllm_llm/rocm_build_check.sh` for every S1 build, with no spill;
- a dry run of `examples/sessions/perf_s1.py`.

**#3 Weights shared across requests (Q).**
- Mac: `examples/vllm_llm/dense_bytes.py` (`f3c7ed6`) prints dense bytes per step for `kept` and `batched`. Test: `tests/test_dense_bytes.py` asserts `kept / batched = N ± 1%` for N = 2, 4, 8 → `3 passed`.
- GPU (S1): rocprof `FETCH_SIZE` per step, B4. The measured ratio `kept / batched` of dense-weight bytes must be within 10% of the Mac prediction. The falsifier named in `docs/MEGAKERNEL-AMD-RESEARCH-2026-10-18.md:153`.
- Done: the S1 log shows the ratio within 10%.

**#1 Per-request replication of dense nodes (Q).**
- Evidence: tasks double from B2 to B4, 109,892 → 219,784 (`docs/TWO-WEEK-RESULTS-2026-10-18.md:55`). That measurement is DeepSeek data; the code that removes replication is Q's batched recipe (`dc8406c`, `examples/vllm_llm/model.py:199-202`).
- Mac: `tests/test_batched_plan.py`. The B4 `batched` plan's dense-node task count must equal the B1 dense count, and the attention and MoE counts must equal 4× B1 → `1 passed`. It prints the three counts.
- GPU (S1): `--recipe batched` at B4 passes 32/32 per request and `host --check` OK for every imported launch.
- Done: both on the S1 log.

**#2 Column batching on (Q).**
- Toggle: `ETX_VL_BATCH` (`examples/vllm_llm/model.py:199-202`; `bench.py:7,51,71`), default off.
- Mac: the B2/B4/B8 `batched` builds pass `rocm_build_check.sh` with no spill.
- GPU (S1): five interleaved pairs, `kept` vs `batched`, at B4 and B8, host wall. **Done** if every pair's gain is positive, the median gain is ≥ 2%, and it exceeds 2× the spread (the `CLAUDE.md` §1 rule) **[plan]**. Correctness as in #1 on both arms.
- Otherwise: `DONE` with a loss, and #4 is not started.

**#12 6-token GEMV pass (G).**
- Evidence: 64 KB LDS per CU; 6 tokens of bf16 K=4096 use 48 KB, 8 tokens use 64 KB (`docs/GLM-EXECUTION-DESIGN-2026-10-19.md:139-141`). Tiles are designed for 6, not 16.
- Mac: the compiler report for the 6-token option shows LDS ≤ 65,536 B and 0 spills, via `examples/glm53/local_seam_rocm.sh` or `rocm_build_check.sh`. CPU equality for B=6 passes in `tests/test_glm53_options.py`.
- GPU (S1, two GPUs if the PP2 slice is used, otherwise the layer-22-25 slice on its required GPU count):
  - 32/32 per request at B6;
  - cosine ≥ 0.999 and relative error ≤ 0.02 (`docs/GLM-EXECUTION-DESIGN-2026-10-19.md:478-479`);
  - per-token wall at B6 ≤ 0.8× per-token wall at B4 on the same build family **[plan]**.

**#13 `attn_reduce` per launch (Q).**
- Evidence: 6.2 → 12.7 µs per launch from B1 to B8 (`docs/EXPERIMENTS-2026-10-18.md:12`). Adapter: `etx/importer/spec.py:82,282,350`.
- Mac: a batched adapter that merges the B8 reduce launches. `tests/test_importer_spec.py -k attn_reduce` → `passed`, plus a local build of B8.
- GPU (S1): B8 `host --check` OK for every imported launch, and median per-launch `attn_reduce` ≤ 6.8 µs (B1 6.2 + 10%) **[plan]**.

**#9 Sentinel barrier (D).**
- Evidence:
  - counter barrier 8.61 µs at 304 workgroups [M] (`docs/ETX_Technical_Design.md:1291`);
  - sentinel 0.8 µs vs 7.6 µs counter in Kog (`docs/ETX_Technical_Design.md:162`);
  - 110 whole-group barriers per B4 path, 55 at B1 [E] (`docs/MEGAKERNEL-AMD-RESEARCH-2026-10-18.md:186-198`). The backlog's "first step, counting barriers" is therefore already done.
- Mac:
  - a sentinel variant of the relay barrier behind a default-off build flag;
  - the protocol simulator shows no lost wake-up;
  - `pytest -q -k sentinel` → `passed`.
- GPU (S1): DeepSeek B4, 32/32 and 27/27 per request. Paired A/B, five pairs: median step saving ≥ 0.43 ms, which is half of 110 × (8.6 − 0.8) µs [E] **[plan]**.

**#11 Streaming bandwidth (Q, G).**
- Evidence: 3.56 TB/s at 192 × 64 KB in flight, 3.93 maximum, out of a 5.3 peak (`docs/GLM-EXECUTION-DESIGN-2026-10-19.md:46-48`).
- Mac: an `lf` sweep job over workgroups per CU (1, 2) × in-flight depth (D = 1..16) in `examples/sessions/perf_s1.py`, with a dry run.
- GPU (S1): the sweep's best configuration reported, then the Qwen dense GEMV tile rebuilt at that occupancy and depth. **Done** if the tile microbenchmark streams ≥ 3.90 TB/s, against the measured `lf` maximum of 3.93 **[plan]**, and the Qwen B1 run stays 32/32 with `host --check` OK.

**#5 In-place a4w8 dequant (G).**
- Evidence: 1.72× weight copies; the factor's origin is [U] (`examples/glm53/run_all.py:52`).
- Mac: `bench/dequant/a4w8_bench.hip` builds locally with the device libraries.
  - Two variants: pre-expanded FP8 vs int4 plus per-group scales dequantised in registers.
  - Real per-rank expert shapes at M=1.
  - Cold cache: rotate the weights over ≥ 1 GB.
  - Outputs bit-equal to a reference dequant, checked on CPU: `pytest -q tests/test_a4w8_reference.py` → `passed`.
- GPU (S1): median of 200 device times per shape.
  - **GO** if in-place ≤ 0.70× pre-expanded on every shape;
  - **NO-GO** if ≥ 0.90× on any expert shape;
  - in between counts as NO-GO (`docs/EXPERIMENTS-GLM-2026-10-20.md`).
  - Done: the verdict line in the S1 log.

**#8 q → attention → o_proj chain (Q).**
- The backlog cites "a design section in the multi-GPU readiness doc". No such section exists on main [U].
- Mac: a `etx.tools.timeline` mode that sums the critical-path time of the q → attention → o_proj chain per layer. Tested on a synthetic trace in `tests/test_timeline.py -k chain` → `passed`.
- GPU (S1): one Qwen B1 trace from the S1 build. **Done** = the chain's share of the step's critical path, printed with the trace path. Decision **[plan]**: a share ≥ 20% opens a design item; below 20%, the chain is left alone.

**#10 B4 dependency and queue gaps (D).**
- Evidence: 0.963 ms dependency gaps + 0.366 ms queue gaps = 1.329 ms at B4 [M]. Closing the gaps alone cannot reach the 3.372 ms the B4 target needs (`docs/MEGAKERNEL-AMD-RESEARCH-2026-10-18.md:129-131`).
- Mac: a scheduling change behind a default-off flag, chosen from `docs/DS-BATCH-ATTRIBUTION-2026-10-18.md`. `pytest -q tests/test_dsv2_batch_attribution.py tests/test_ds_batch_attribution_tool.py` → `0 failed`.
- GPU (S1): a held B4 trace with 32/32 and 27/27. **Done** if dependency + queue gaps ≤ 1.00 ms (−25%) **[plan]**.

## Phase M2: Mac preparation for S2

**#6 TP4 all-reduce on the critical path (G).**
- No TP path exists: GLM runs PP2 (`docs/QWEN-BATCH-GLM53.md:107-108`). The deciding unknown is the collective's cost, so measure it before building any TP layout.
- Evidence nearby: cross-device flag 10.6 µs one-way, peer store 285 GB/s, on 2× MI300X [M] (`docs/ETX_Technical_Design.md:1293-1294`).
- Mac: `bench/sync/allreduce_bench.hip`, an in-kernel 4-rank all-reduce of 32,768 B bf16 (`[1,4,4096]`), checked against a CPU sum. It builds locally. `examples/sessions/perf_s2.py` has a dry run.
- GPU (S2, 4 GPUs): median and spread over 5 rounds × 1,000 iterations.
  - **Done** = the number.
  - Decision (`docs/GLM-MULTIGPU-2026-10-21.md` §4): ≤ 16 µs keeps an in-layer split viable; > 32 µs rules it out.

## After S1

**#4 MoE routed union (Q).**
- Evidence: ceiling −0.37 ms at B4 and −1.58 ms at B8 [E], but no shared-weight MoE tile exists, so the gain against today's code is 0 (`docs/MEGAKERNEL-AMD-RESEARCH-2026-10-18.md:315-318`). The backlog's "51.6 distinct experts" is not on main [U].
- Start only if #2 is `DONE` with a win.
- Mac: `examples/vllm_llm/union_bytes.py` prints the expected distinct experts per step at B8 from recorded routing. `pytest -q tests/test_union_bytes.py` → `passed`.
- GPU (S3): B8 paired A/B, five pairs. **Done** if the median saving is ≥ 0.79 ms (half the ceiling) **[plan]**, with correctness as #1.

## Closed

- **#7 router → top-k → first expert byte.** `CLOSED-BY-MEASUREMENT`. Forced overlap measured 314 / 341 tok/s against 366 uncoupled [M] (`docs/PLAN-v3.md:178`, recorded at `docs/MEGAKERNEL-AMD-RESEARCH-2026-10-18.md:326-327`). Not reopened.
- **#14 cold cache.** `CLOSED-BY-MEASUREMENT`. A B8 routed layer streams 1,611 MB against a 256 MB MALL, 6.3× its size (`docs/GLM-EXECUTION-DESIGN-2026-10-19.md:101-105`). L2 warming measured −3% (`:123`). The Infinity Cache is not faster than HBM for any expert-sized stream (`docs/ETX_Technical_Design.md:1042`). No design for cache reuse. Not reopened.

## First task for cc-astra

**#16 runner exclusion** (Phase M0, Mac only).
Branch `perf/16-runner-exclusion` from `origin/main`. Files: `etx/tools/gpu_jobs.py` and `tests/test_gpu_jobs.py`.
- **Behaviour:** an exclusive, non-blocking `flock` on `/tmp/etx-gpu-<index>.lock` for each scheduled GPU, held for the whole run. A second runner needing a held GPU exits non-zero, naming the holder's PID.
- **Acceptance:** `.venv/bin/python -m pytest -q tests/test_gpu_jobs.py -k exclusive` → `2 passed`, and `.venv/bin/python -m pytest -q tests/test_gpu_jobs.py` → `0 failed`.
- **Delivery:** one commit, `perf #16: ...`, squash PR into main after cc-grok's check.
