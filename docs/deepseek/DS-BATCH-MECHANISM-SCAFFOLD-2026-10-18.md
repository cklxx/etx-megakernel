# DeepSeek multibatch mechanism scaffold, 2026-10-18

Status: analysis scaffold. This file adds no GPU measurement. It records which
candidate mechanisms the 2026-10-16 trace record can decide, which it cannot,
and the decision rule for each missing measurement.

Labels: [M] measured and present in the record. [E] estimate. [U] absent from
the record. A [U] cell has no number.

## 1. Verdict

The five candidate mechanisms come from `docs/EXPERIMENTS-2026-10-18.md:20`:
occupancy, dependency chain, launch count, HBM/L2 traffic, and serialization.

Of these five, the record contains direct counts for three:

- occupancy at schedule level [M],
- dependency gaps [M],
- queue serialization [M].

The record does not contain two of the five:

- launch count [U],
- HBM and L2 traffic [U].

Occupancy is [M] at schedule level and [U] at hardware level. The record has no
CU, wave or SIMD occupancy counter.

The record also lacks the ordered task-level chain [U], a counterfactual
schedule [U], a shared-work implementation [U], and a B8 trace [U]. Section 5
lists every missing item.

## 2. The record

Path: `examples/gpu-live/results/2026-10-16-multibatch/trace_report.json`.

The record holds nine final-token traces: B1, B2 and B4, three rounds each.
B1 uses arm `alld4r0l1kvsmw`, B2 uses `x2`, and B4 uses `x4`. The model is
DeepSeek-Coder-V2-Lite-Base in bf16, with context 1024 and 32 decode tokens, on
one MI300X VF with ROCm 7.14. The scope string states: "instrumented
final-token task trace; no accepted hardware traffic counters; no performance
verdict".

Fields used below:

| Field | Meaning |
|---|---|
| `trace.families[].tasks` | Executed task instances in the family. |
| `trace.families[].body_work_us` | Sum of task body times in the family. |
| `trace.families[].wait_work_us` | Sum of task event waits in the family. |
| `trace.families[].cp_*` | Critical-path body, dependency gap and queue gap. |
| `trace.latency[].active_tasks` | Mean task body work divided by the phase grid span. |
| `trace.latency[].active_xcds` | Mean number of XCDs with a body interval, divided by the phase grid span. |
| `trace.latency[].dep_gap_mean_us` | Mean of task start minus the last producer end. |
| `trace.latency[].start_spread_us` | Last start minus first start in the phase. |
| `trace.latency[].finish_skew_us` | Last XCD done minus first XCD done in the phase. |

The `grids` field counts merged task-type groups per phase, not kernel launches.
A per-request phase has 27 groups. A model-wide phase has one group. Do not read
`grids` as a launch count.

The report is the committed record. The raw traces and plans are not in the
repository. `docs/core/DATA.md` records the raw local export.

## 3. The scaling facts

Round 1 is the basis for the family and phase tables below. The three rounds
differ by 0.23% or less in step time. The complete trace summary reports medians.

| B | step (us) | tokens/s | tasks | body work (us) | CP body (us) | CP dep (us) | CP queue (us) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3254.2 | 307 | 53022 | 737261.2 | 2715.7 | 378.7 | 0.0 |
| 2 | 4576.2 | 437 | 109892 | 1943155.6 | 3735.3 | 542.2 | 69.7 |
| 4 | 9026.7 | 443 | 219784 | 3851616.3 | 7360.2 | 955.1 | 381.6 |

B2 to B4:

- tasks grow exactly 2.000 times,
- body work grows 1.982 times,
- CP body grows 1.971 times,
- step grows 1.972 times,
- throughput grows 1.4%.

The traced span grows 4349.7 us. The split is exact: body +3624.9 us (83.3%),
dependency gap +412.9 us (9.5%), and queue gap +311.9 us (7.2%). The host step
grows +4450.5 us. The remaining +100.8 us is outside the traced window.

B1 uses a different arm. Only B2 to B4 is a clean scaling comparison.

## 4. Mechanism table

| Mechanism | Operational definition | Record fields | Status | Measured value |
|---|---|---|---|---|
| Work replication | Each request carries its own task graph and reads its own weights | `tasks`, `body_work_us` | [M] | Tasks 2.000x and body work 1.982x from B2 to B4 |
| Schedule occupancy | Mean tasks live, and XCDs with work, during a phase | `active_tasks`, `active_xcds`, `body_work_us`, `span_us` | [M] | Mean live tasks 238.25 / 446.99 / 442.87 at B1/B2/B4 |
| Hardware occupancy | CU, wave or SIMD busy fraction | none | [U] | No number |
| Dependency chain | CP dependency gaps and task start delays | `cp_dep_gap_us`, `dep_gap_mean_us`, `dep_gap_p95_us`, `dep_gap_max_us` | [M] | CP dep 378.7 / 542.2 / 955.1 us |
| Launch count | Kernel dispatches per step and per launch | none | [U] | No number. Task count is [M] as a proxy |
| HBM/L2 traffic | Bytes read and written, L2 hit rate, achieved bandwidth | none | [U] | No number |
| Serialization | CP queue gaps, start spread, finish skew | `cp_queue_gap_us`, `start_spread_us`, `finish_skew_us` | [M] | CP queue 0.0 / 69.7 / 381.6 us |
| Counterfactual schedule | Step with another slot map | none | [U] | No number |
| Shared-work benefit | Step with a column-batched tile | none | [U] | No number. Candidate is "NOT IMPLEMENTED/NOT MEASURED" |
| B8 scaling | Same counts at B8 | none | [U] | No number |

## 5. Mechanism details

### 5.1 Work replication [M]

Tasks grow exactly 2.000 times from B2 to B4 in every family:

| Family | B2 tasks | B4 tasks | Ratio |
|---|---:|---:|---:|
| qkv | 15984 | 31968 | 2.000 |
| attn | 13824 | 27648 | 2.000 |
| qabs | 1728 | 3456 | 2.000 |
| router | 15392 | 30784 | 2.000 |
| oproj | 15984 | 31968 | 2.000 |
| gate_up | 15392 | 30784 | 2.000 |
| down | 15392 | 30784 | 2.000 |
| merge | 13824 | 27648 | 2.000 |

Body work grows 1.982 times. The measured record therefore shows repeated work,
not shared work. The record does not show the benefit of shared work. No batched
tile exists, and the record marks the candidate as "NOT IMPLEMENTED/NOT MEASURED"
[U].

The B2 to B4 body growth by family is: gate_up +973 us, down +579 us, attn
+511 us, qkv +418 us, router +308 us, oproj +254 us, merge +228 us, qabs
+166 us, and lm_head +103 us. Smaller families add +85 us. These are CP body
values from `tools/ds_batch_attribution.py`.

### 5.2 Schedule occupancy [M], hardware occupancy [U]

The record has no whole-window occupancy field. Two measured fields give a
schedule-level ratio: sum of `body_work_us` divided by `span_us`.

| B | Body work (us) | Span (us) | Mean tasks live |
|---|---:|---:|---:|
| 1 | 737261.2 | 3094.4 | 238.25 |
| 2 | 1943155.6 | 4347.2 | 446.99 |
| 4 | 3851616.3 | 8696.9 | 442.87 |

The ratio is [M] because both inputs are measured. The record does not store the
ratio itself.

Per-phase concurrency at B4 is close to B2. The table gives the range over the
request copies of each family.

| Family | B1 XCDs live | B2 XCDs live | B4 XCDs live | B1 tasks live | B2 tasks live | B4 tasks live |
|---|---:|---:|---:|---:|---:|---:|
| qkv | 7.83 | 7.59-7.64 | 6.51-6.80 | 280.4 | 261.1-264.1 | 201.4-217.3 |
| attn | 7.75 | 7.56-7.66 | 7.51-7.65 | 239.6 | 195.2-203.9 | 193.2-201.6 |
| qabs | 6.98 | 6.89-7.07 | 6.81-6.98 | 26.2 | 24.7-25.8 | 24.7-25.3 |
| router | 5.78 | 7.31-7.40 | 7.34-7.51 | 202.6 | 237.4-245.1 | 238.9-251.1 |
| oproj | 6.58 | 5.85-5.93 | 5.77-6.00 | 219.6 | 187.4-197.2 | 188.3-194.7 |
| gate_up | 5.80 | 7.52-7.58 | 7.52-7.61 | 205.5 | 262.2-263.1 | 259.3-268.9 |
| down | 5.48 | 6.95-7.11 | 6.90-7.14 | 190.7 | 236.0-242.0 | 235.4-245.7 |
| merge | 6.58 | 6.25-6.29 | 6.29-6.46 | 189.4 | 121.2-135.4 | 123.9-134.2 |

The measured pattern is that live-task count saturates from B2 to B4 while body
work doubles. This is a schedule metric. It is not a CU or wave occupancy
counter [U].

The record also lacks a whole-window per-XCD split of body, event wait and idle
time [U]. `etx.tools.timeline.report` computes that split, but the record does
not store it, and the raw trace is not in the repository.

### 5.3 Dependency chain [M] at family level, [U] at task level

Critical-path dependency gaps are measured: 378.7 us at B1, 542.2 us at B2 and
955.1 us at B4. The B2 to B4 increase is +412.9 us. The main family increases
are:

| Family | CP dep B2 (us) | CP dep B4 (us) | Delta (us) |
|---|---:|---:|---:|
| router | 78.0 | 183.5 | +105.5 |
| oproj | 81.7 | 181.4 | +99.7 |
| qabs | 80.5 | 172.1 | +91.6 |
| down | 71.3 | 139.1 | +67.8 |
| gate_up | 75.5 | 140.2 | +64.7 |
| merge | 55.5 | 111.7 | +56.2 |
| lm_head | 2.4 | 0.0 | -2.4 |
| qkv | 77.4 | 0.0 | -77.4 |

The record also stores the start delay after the last producer for each phase.
At B4, qkv and lm_head have the large values:

| Phase | Mean (us) | p95 (us) |
|---|---:|---:|
| r0_qkv | 140.7 | 151.8 |
| r1_qkv | 151.7 | 167.6 |
| r2_qkv | 146.7 | 152.4 |
| r3_qkv | 156.3 | 165.7 |
| r0_lm_head | 147.5 | 148.8 |
| r1_lm_head | 122.3 | 123.5 |
| r2_lm_head | 150.8 | 153.4 |
| r3_lm_head | 111.8 | 114.7 |

At B2, only `r1_dense_dn` is above 5 us, at 5.79 us mean and 7.25 us p95. At B1,
no phase is above 5 us.

This field is task start minus the last producer end. It includes delay from
worker availability. The critical-path classifier separates dependency and queue
gaps on the critical path only. The report does not store that classification
for the other tasks [U]. Therefore these large qkv and lm_head values show where
waiting occurs. They do not prove a pure dependency cause.

The ordered task-level chain is not in the record [U]. The report stores only
the aggregate by type.

### 5.4 Launch count [U]

The record has task counts, not launch or dispatch counts. Total tasks are
53022, 109892 and 219784. The task count doubles exactly from B2 to B4.

A task count is a work-item count. It is not a kernel launch count. The `grids`
field is a count of merged task-type groups, not launches. The record has no
dispatch identity, dispatch duration or launch gap [U].

The capture path exists in `examples/dsv2lite/batch_attribution.py`. Its
`traffic` function requires one counter row for every traced dispatch. The prior
profiler capture was rejected, and no profiler command was retried. No accepted
dispatch count exists in the record.

### 5.5 HBM and L2 traffic [U]

The scope string states "no accepted hardware traffic counters". The record has
no bytes, no achieved bandwidth, no L2 hit rate and no `FETCH_SIZE` row [U].

The record also has no bytes specification. Therefore it cannot supply even a
demand estimate. `etx.tools.timeline.bandwidth_report` needs a `--bytes` spec,
and the record was not produced with one.

An estimate from model shapes exists in `docs/deepseek/BATCH-2026-10-14.md` section 2.
It is [E], not a measurement, and it is not part of this record.

### 5.6 Queue serialization [M] as cost, [U] as counterfactual

Critical-path queue gaps are measured: 0.0 us at B1, 69.7 us at B2 and 381.6 us
at B4. At B4, qkv contributes 225.2 us and attn contributes 148.1 us. lm_head
contributes 4.6 us, and one embed copy contributes 3.7 us.

Start spread and finish skew also grow. At B4, the largest start spread is
255.3 us in attn and 207.0 us in qkv. The largest finish skew is 128.6 us in qkv
and 110.3 us in oproj.

The queue gap is a lower bound on scheduling loss. `result.md` records that
queue-induced serialization also appears as body time of the queue predecessor.
The recorded queue gap does not include that body time.

No alternative slot map or queue placement was measured [U]. Therefore the
record does not prove that queue placement causes the B4 loss.

### 5.7 B8 [U]

The record has no B8 trace. The previous x8 correctness failure remains a
failure. Do not extend the B1, B2 or B4 conclusions to B8.

## 6. What the record does not contain

- [U] HBM and L2 traffic counters.
- [U] CU, wave or SIMD occupancy counters.
- [U] Kernel launch or dispatch counts.
- [U] Whole-window per-XCD body, wait and idle shares.
- [U] Ordered task-level critical path and per-gap kind.
- [U] A counterfactual schedule at B4.
- [U] A column-batched or shared-weight implementation and its benefit.
- [U] Native multi-request routing or pick-union equality.
- [U] A correct B8 trace or any B8 count.

## 7. Decision rules for the next measurement

These rules are pre-registered. They are not results. The 2% bar is the
repository's existing bar for a paired host-wall gain.

| Missing item | Next measurement | Rule |
|---|---|---|
| HBM/L2 traffic | One accepted rocprof capture with `FETCH_SIZE` and L2 counters, full dispatch coverage, B1/B2/B4 on one box | Traffic is a primary cause only if measured bytes divided by step time is at least 80% of 4.3 TB/s. Below that, rule it out. |
| Hardware occupancy | CU and wave counters over the same capture | Occupancy is a cause only if B4 bodies run at a lower CU busy share than B2 bodies. An equal or higher share rules out a drop in occupancy. |
| Launch count | Kernel dispatch trace with markers | Launch cost is material only if dispatch gaps or dispatch setup sum to at least 2% of the step. Otherwise rule it out. |
| Ordered chain | Keep the raw trace and plan, run `etx.tools.timeline`, store the ordered path and each gap kind | Charge a gap to dependency only when the last producer ends after the previous task on the same worker. |
| Counterfactual schedule | One alternative slot map at B4 | Serialization is established only if the paired host step improves by at least 2% in every pair. |
| Shared work | A column-batched tile, with correctness gates first | The benefit is established only if the paired host step improves by at least 2% and all imported-launch and 32/32 token gates pass. |
| B8 | A correct B8 trace and the same gates | No B8 claim before a correct trace. |

One accepted profiler capture can close traffic and launch count together. The
same run can add hardware occupancy counters if the capture requests them. That
capture must pass the dispatch-coverage gate in
`examples/dsv2lite/batch_attribution.py`. The prior capture did not pass.

## 8. Reproduce

```bash
python tools/ds_batch_attribution.py \
  examples/gpu-live/results/2026-10-16-multibatch/trace_report.json

python -m examples.dsv2lite.multibatch_trace_summary \
  examples/gpu-live/results/2026-10-16-multibatch/trace_report.json \
  examples/gpu-live/results/2026-10-16-multibatch/baseline/batch_report.json \
  /tmp/multibatch-summary.json
```

The mean live-task ratio in section 5.2 is this calculation on round 1:

```python
sum(f["body_work_us"] for f in row["trace"]["families"]) / row["trace"]["span_us"]
```

## 9. Limits

- No new GPU run was made for this scaffold.
- Family and phase tables use round 1. The three rounds differ by 0.23% or less
  in step time.
- The trace is instrumented. It is not an unprofiled wall-time comparison.
- B1 uses a different arm. The clean scaling comparison is B2 to B4.
- B8 is absent.
- The record has no accepted hardware traffic counter, so every traffic statement
  stays [U].
