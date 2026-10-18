# Qwen batch mechanism scaffold, 2026-10-18

Status: analysis scaffold plus pointers to the measured values. This file
records which candidate mechanisms the Qwen B1/B2/B4/B8 record can decide,
which it cannot, and the decision rule for each missing measurement. The clean
run completed B1 and B2 only; kept B4/B8 and plain B8 remain [U]. Measured
values are in `docs/QWEN-BATCH-BASELINE-2026-10-18.md`.

Labels: [M] measured and present in the record. [E] estimate. [U] absent from
the record. A [U] cell has no number.

## 1. Verdict

The five candidate mechanisms come from `docs/EXPERIMENTS-2026-10-18.md:20`:
occupancy, dependency chain, launch count, HBM/L2 traffic, and serialization.
This file applies them to the Qwen replicated-request benchmark.

The current record contains one of the five, and only as a static count:

- launch count [M] as the plan's imported-instance count: 97 per request, 97
  per position at B1, and 194/388/776 instances per position at B2/B4/B8.

The record does not contain any of the five as a runtime measurement of the
batch decode step:

- occupancy [U] at runtime. Compiler resource reports exist [M].
- dependency chain [U].
- launch count [U] as dispatches per step and as dispatch cost.
- HBM and L2 traffic [U].
- serialization [U].

Two further mechanisms are partly present:

- work replication [M] as the plan design and exact counts,
- imported-kernel per-launch time [M] from `--check`, which is not decode time.

The decode wall time and throughput are now [M] for kept B1/B2 and plain B4,
and remain [U] for kept B4/B8 and plain B8. Section 6 lists the values and
points to the results file.

## 2. The records

Four records are in scope.

| Record | What it contains | Conditions | Not in it |
|---|---|---|---|
| `20261017T140801Z-988ffe` | Correctness of replicated requests, and per-launch GPU event times under `--check` | B2/B4/B8, `plain` recipe, one MI300X, ROCm 7.14.0, context 1024, 32 tokens | No decode wall time, no task trace, no counters |
| Local compile reports (`docs/QWEN-BATCH-GLM53.md:21-26`) | Static task counts and compiler resource use | ROCm 7.2.4, standard B2/B4/B8 builds | No GPU execution |
| Clean serial bench run (`/Users/ckl/.hermes/cache/scratch/bench-out9.log`) | Per-batch decode wall time and per-batch gates | B1/B2 complete, `kept` recipe, one runner with `flock` | B4/B8: the remote shell exited at B4, so no gate or timing line exists |
| Plain B4 run (`/root/bench_plain_b4.json`) | Decode wall time and gates for the correctness-qualified recipe | B4, `plain` recipe, separate runner under the same lock | B8 was still running when this file was written |

The first record is committed evidence. The session directory itself is not in
the repository. The summary is in
`examples/vllm_llm/results/EXPERIMENT-LOG.md:86-87`.

The correctness record and the clean run use different recipes. The correctness
record is `plain`. The clean run is `kept`, which replicates every request's
launches. Therefore the correctness record does not certify the timed build.
The clean run certifies each timed build itself: `bench.py` runs the byte check
at positions 0 and 1040, the per-request coverage check, and the 32-token
decode on the same binary before it prints a timing line.

## 3. Static facts

These are measured counts or compiler reports. They are not performance numbers.

| B | Requests | Imported instances per position | Instances per request | Tasks |
|---|---:|---:|---:|---:|
| 1 | 1 | 97 | 97 | [U] |
| 2 | 2 | 194 | 97 | 120724 |
| 4 | 4 | 388 | 97 | 241448 |
| 8 | 8 | 776 | 97 | 482896 |

The instance count is exactly 97 times the request count. The task count is
exactly 2.000 times the previous batch's count. Both facts are [M] from
`docs/QWEN-BATCH-GLM53.md:22-24` and `EXPERIMENT-LOG.md:86`.

B1 is not in the multi-request record. It is the kept single-request build.
The clean run checked 97 imported instances at each of positions 0 and 1040,
so the per-position count is 97 [M]. Its task count is not printed in the
committed record [U].

Compiler reports, kept separate because they describe different builds:

| Build | VGPR | Scratch (B/lane) | VGPR spills | Compiler occupancy |
|---|---:|---:|---:|---:|
| Standard B2 | 128 | 32 | 7 | [U] |
| Standard B4 | 128 | 32 | 7 | [U] |
| Standard B8 | 128 | 32 | 7 | [U] |
| Kept B2 | 128 | 40 | 9 | 4 |
| Kept B1/B4/B8 | [U] | [U] | [U] | [U] |

The standard rows are from `docs/QWEN-BATCH-GLM53.md:23-26`. The kept B2 row is
from `docs/EXPERIMENTS-2026-10-18.md:11`. The two builds are not the same. All
values are compiler reports, not measured GPU residency or timing.

## 4. Mechanism table

| Mechanism | Operational definition | Record fields | Status | Measured value |
|---|---|---|---|---|
| Work replication | Each request carries its own plan and launches | Plan recipe, instance and task counts | [M] | Instances 194/388/776 and tasks 120724/241448/482896, both exact 2.000x per batch doubling |
| Static launch count | Imported instances in the plan | `cert.expected_instances` | [M] | 97 per request, 194/388/776 per position |
| Runtime launch count | Dispatches per decode step and dispatch cost | None | [U] | No number |
| Static occupancy | Compiler resource bound | hipcc resource report | [M] | 128 VGPR, 32 B/lane, 7 spills at standard B2/B4/B8 |
| Runtime occupancy | CU, wave or SIMD busy share | None | [U] | No number |
| Dependency chain | Task start minus last producer end, and CP gap | None | [U] | No number |
| HBM/L2 traffic | Bytes, achieved bandwidth, L2 hit rate | None | [U] | No number |
| Serialization | Queue gap, start spread, finish skew | None | [U] | No number |
| Imported-kernel scaling | Imported against original event time per launch | `--check` GPU events | [M] | Position 0 sums: B2 -10.5%, B4 +17.7%, B8 +35.0%, original to imported |
| Decode wall time | Host wall of one decode step | `HOUR-WALL` | [M] for kept B1/B2 and plain B4 | 4.087415 / 5.701229 / 11.929137 ms |
| Throughput | N times 1000 divided by median step | Derived | [M] for kept B1/B2 and plain B4 | 244.65 / 350.80 / 335.31 tokens/s |

## 5. Mechanism details

### 5.1 Work replication [M]

For batch greater than one, `bench.py` sets `ETX_VL_REQUESTS=N`,
`ETX_VL_BATCH=0`, `ETX_VL_NB=1` and `ETX_VL_CHAIN=0` (`bench.py:56-58`). Each
request gets its own launches. There is no column batching and no chaining.

The counts confirm replication. Imported instances per position are 97 times
the request count. Tasks are 120724, 241448 and 482896 at B2, B4 and B8. The
per-family breakdown is not in the committed record [U]. The totals and the
plan design show repeated work, not shared work.

The record does not show the step-time consequence of replication [U]. If the
step grows linearly with the request count, throughput is flat. If it grows
faster, another mechanism adds cost. The clean run decides which case holds.

No column-batched or shared-weight tile exists in this benchmark. Its benefit
is [U]. A shared-work claim needs a new plan, correctness gates first, and a
paired host-step gain. Section 7 gives the rule.

### 5.2 Launch count: static [M], runtime [U]

The plan contains 97 imported instances per request. The correctness record
checks one launch of each instance at position 0 and one at position 1040. It
reports 194/388/776 instances per position at B2/B4/B8, with zero byte
differences (`EXPERIMENT-LOG.md:86`).

This is a static count. It is not a dispatch count for a decode step. The
record has no dispatch identity, dispatch duration or launch gap [U]. The
benchmark's timing input is the `HOUR-WALL` line per step (`bench.py:165-171`).
No profiler capture covers the batch decode path.

A task count is a work-item count, not a launch count. The record has task
counts [M] but no kernel dispatch count [U]. Launch cost is therefore [U].

### 5.3 Occupancy: compiler [M], runtime [U]

The standard B2/B4/B8 builds report 128 VGPR, 32 B/lane scratch and 7 VGPR
spills. The kept B2 build reports 128 VGPR, 40 B/lane, 9 spills and occupancy 4
(`EXPERIMENTS-2026-10-18.md:11`).

These are compiler reports. They do not say how many CUs, waves or SIMDs are
busy during a decode step at each batch. The record has no CU, wave or SIMD
counter, and no task trace. Runtime occupancy as a cause of batch scaling is
[U].

The static resource report is the same at B2, B4 and B8 for the standard
builds. Plan replication does not change the resource use of one kernel.
Whether runtime occupancy differs is [U].

### 5.4 Dependency chain [U]

The record has no task trace of a Qwen batch decode step. The per-launch event
times in the correctness record come from isolated launches in a host loop
under `--check`. They are not the ordered decode chain.

The plan has 120724, 241448 and 482896 tasks. The ordered chain and the gap
between a task start and its last producer end are not exported in the record
[U]. A trace requires `ETX_TRACE_RAW` and `etx.tools.timeline`. Neither is in
this record.

The record also has no per-task classification of a gap as dependency or queue
[U]. A start delay can come from a missing producer or from worker
unavailability. The record cannot separate them.

### 5.5 HBM and L2 traffic [U]

The record has no bytes, achieved bandwidth, L2 hit rate or hardware counter
row [U]. No profiler capture exists for this path.

The repository has `examples/vllm_llm/byte_ledger.py` and
`examples/vllm_llm/bw_spec.py`, but neither was run for this record. The
bandwidth floor, bytes divided by 4.3 TB/s, cannot be computed without bytes.
This file does not fill it with an estimate.

A traffic claim needs an accepted capture with `FETCH_SIZE` and L2 counters and
full dispatch coverage. Section 7 gives the rule.

### 5.6 Serialization [U]

The record has no queue gap, start spread or finish skew for the batch decode
path [U]. The design replicates launches and keeps `ETX_VL_CU=256`. At batch
greater than one, N requests compete for the same CUs. Whether the device or
the host serializes them is unmeasured.

The per-launch `--check` time is a host loop of isolated launches. It is not a
schedule. It cannot show queueing between requests.

A serialization claim needs an alternative slot map or queue placement and a
paired host-step gain. Section 7 gives the rule.

### 5.7 Imported-kernel scaling [M, isolated]

Under `--check`, the summed GPU event time of the imported launches against the
original launches is, at position 0:

| B | Original (us) | Imported (us) | Change |
|---|---:|---:|---:|
| 2 | 2576.2 | 2307.0 | -10.5% |
| 4 | 4053.6 | 4771.1 | +17.7% |
| 8 | 7092.9 | 9576.6 | +35.0% |

At position 1040 the changes for B2, B4 and B8 are -17.0%, +4.8% and +26.7%
(`EXPERIMENTS-2026-10-18.md:12`).

At B8 position 0, `attn_reduce` accounts for +2473.3 of the +2483.7 us total.
The pair is 2455.9 to 4929.2 us, or +101%. Its per-launch median is 6.2 to
12.7 us. At position 1040 the median is 6.2 to 13.1 us (`EXPERIMENT-LOG.md:87`).

This is one launch of each imported instance, measured once. It is not decode
time. It does not show that a decode step pays this cost. It does identify the
imported attention-reduce kernel at B8 as the first place to look if the clean
B8 step is much slower than twice the B4 step.

The committed summary breaks out only `attn_reduce` at B8. The per-kernel table
for the other batches is [U] in the repository.

### 5.8 Decode wall time and throughput [M, partial]

The clean run completed kept B1 and B2; kept B4/B8 are absent. One later plain
run completed B4. The host wall time of one decode step, with all requests
advancing one token, is 4.087415 ms at kept B1, 5.701229 ms at kept B2 and
11.929137 ms at plain B4. Throughput is 244.65, 350.80 and 335.31 tokens/s.
The values and their gates are in `docs/QWEN-BATCH-BASELINE-2026-10-18.md`.

The same-recipe pair is B1 to B2: doubling the requests costs 1.394825x the
step, so per-request cost falls 30.26%. The step does not grow linearly at B2.
The cause is [U]; no trace separates overlap from a fixed part of the step.

No B2-to-B4 scaling statement is made. The only B4 row is plain, a different
build, and the kept B4 cell is [U].

These runs are not a vLLM comparison. A comparison against vLLM needs paired
runs on one box in one session. These rows are ETX baselines.

## 6. Fill-in rule and measured values

Do not fill a cell from an incomplete run or from a run that shared the GPU.

For each batch, fill only from a completed `bench.py` run that prints all of:

- `BUILD-IDENTITY` with the expected batch and recipe,
- `CORRECTNESS PASS` for that batch,
- `TIMING batch=...` with 31 decode samples.

The runner is `/Users/ckl/.hermes/cache/scratch/bench_runner.sh`. It holds
`flock` on `/root/bench.lock` for the whole run. A second runner exits with
`ANOTHER BENCH RUNNER HOLDS THE LOCK`.

| B | Recipe | Median decode wall (ms) | Tokens/s | ms/token | Spread (ms) | Correctness of the timed build | Imports per position | Tasks |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | kept | 4.087415 | 244.65 | 4.087415 | 0.034415 | PASS 32/32 | 97 | [U] |
| 2 | kept | 5.701229 | 350.80 | 2.8506145 | 0.073494 | PASS 64/64, coverage 2x1055, ties 0,0 | 194 | 120724 |
| 4 | kept | [U] | [U] | [U] | [U] | [U] | 388 | 241448 |
| 8 | kept | [U] | [U] | [U] | [U] | [U] | 776 | 482896 |
| 4 | plain | 11.929137 | 335.31 | 2.982284 | 0.308351 | PASS 128/128, coverage 4x1055, ties 0,0,1,1 | 388 | [U] |
| 8 | plain | [U] | [U] | [U] | [U] | [U] | [U] | [U] |

The correctness column means the timed build in the same run as the timing.
The earlier `20261017T140801Z-988ffe` pass is a different recipe and does not
fill any kept cell. The plain B4 row fills only its own row.

If a batch fails a gate, its cells stay [U] and the failure is recorded. A
shorter run is not a number for a longer one. The kept B4/B8 cells are [U]
because the run produced no result there, not because a gate failed.

The plain-recipe task count is not in its result record [U].

## 7. Decision rules for the next measurement

These rules are pre-registered. They are not results. The 2% bar is the
repository's existing bar for a paired host-wall gain.

| Missing item | Next measurement | Rule |
|---|---|---|
| Runtime occupancy | CU and wave counters over one decode step at B2 and B4 | Occupancy is a cause only if the B4 CU busy share is lower than B2 at the same per-kernel work. An equal or higher share rules out an occupancy drop. |
| Dependency chain | Keep the raw trace and plan, run `etx.tools.timeline`, store the ordered path and each gap kind | Charge a gap to dependency only when the last producer ends after the previous task on the same worker. |
| Launch count and cost | Dispatch trace with markers over one decode step | Launch cost is material only if dispatch gaps or dispatch setup sum to at least 2% of the step. |
| HBM and L2 traffic | One accepted rocprof capture with `FETCH_SIZE` and L2 counters, full dispatch coverage, B1/B2/B4 | Traffic is a primary cause only if measured bytes divided by step time is at least 80% of 4.3 TB/s. Below that, rule it out. |
| Serialization | One alternative slot map or queue placement at B4 | Serialization is established only if the paired host step improves by at least 2% in every pair. |
| Shared work | A column-batched or shared-weight tile, correctness gates first | The benefit is established only if the paired host step improves by at least 2% and all imported-launch and 32/32 token gates pass. |
| vLLM comparison | Paired ETX and vLLM runs, interleaved, one box, one session | Report both the clean ETX baseline and the paired delta. No cross-session comparison. |

One accepted profiler capture can close traffic, runtime occupancy and launch
count together. The capture must pass the dispatch-coverage gate that
`examples/dsv2lite/batch_attribution.py` applies. The prior capture for the
DeepSeek path did not pass.

## 8. Provenance and contamination

The clean run is `/Users/ckl/.hermes/cache/scratch/bench-out9.log`. The GPU-side
logs are under `build/bench/<build>/` on the host. The measured values are in
`docs/QWEN-BATCH-BASELINE-2026-10-18.md`.

Two earlier B1 and B2 readings were withdrawn because more than one bench
runner shared the GPU. They are not used here and their values are not repeated.

The runner script now takes an exclusive `flock`. Two stale local runners and
one old remote B8 import check survived an earlier kill. I stopped them by PID.
The clean run's timing began after those processes were gone. The overlap was
confined to the clean run's B1 build and check.

The clean kept run completed B1 and B2 and stopped at B4: its remote shell
exited before any B4 gate or timing line. The local wrapper was left waiting
and was stopped by PID after the remote side was confirmed gone. A separate
runner then took the same lock and completed plain B4; it is a different
recipe and fills only the plain row.

No number in this file is compared across sessions. The timing values are from
one clean serial run on one MI300X, plus the separate plain B4 run on the same
GPU and commit. The static and per-launch values are from the records in
section 2.

## 9. Reproduce

The clean runner runs, for each batch:

```bash
.venv/bin/python -m examples.vllm_llm.bench --batch N \
  --json /root/bench_bN.json /root/llm/qwen3-30b-a3b
```

The result record for each batch is `/root/bench_bN.json` on the host. Its
`TIMING` line is the source for the filled cells.

The per-launch imported-against-original sums are in
`examples/vllm_llm/results/EXPERIMENT-LOG.md:87` and
`docs/EXPERIMENTS-2026-10-18.md:12`.

## 10. Limits

- This file adds no measurement of its own; the values it points to are in
  `docs/QWEN-BATCH-BASELINE-2026-10-18.md`.
- The correctness record is the `plain` recipe. The clean kept run is `kept`;
  each completed row certified its own timed build.
- The correctness record covers B2/B4/B8, not B1.
- kept B4/B8 are absent from the clean run; plain B8 is in progress.
- The per-launch timings are isolated launches, not decode time.
- There is no task trace, no hardware counter and no profiler capture for the
  Qwen batch decode path.
- Every remaining [U] cell stays [U] until a completed run passes its gates.
