# Qwen batch baseline, 2026-10-18

Status: measured results from clean serial GPU runs. Every row below passed its
own gates. Missing rows are [U]; no number is borrowed from another row, recipe
or session.

Labels: [M] measured. [E] estimate. [U] absent. A [U] cell has no number.

Conditions common to every row: one MI300X, ROCm 7.14.0, Qwen3-30B-A3B bf16,
context 1024, 32 greedy tokens, source commit `380ed02`, host wall time of one
decode step over steps 1024-1054 (31 samples). One runner at a time under
`flock` on `/root/bench.lock`.

## 1. Verdict

Three batches completed and passed every gate:

- kept B1: **4.087415 ms** per step, **244.65 tokens/s**.
- kept B2: **5.701229 ms** per step, **350.80 tokens/s**.
- plain B4: **11.929137 ms** per step, **335.31 tokens/s**.

Two cells are absent:

- kept B4 and kept B8: the clean kept run's remote shell exited at B4 before
  any gate or timing line. There is no kept B4/B8 record [U].
- plain B8: running when this file was written [U].

There is no vLLM comparison. These numbers are ETX baselines, not a paired
speedup against anything.

## 2. Measured rows

| B | Recipe | Median host step (ms) | Tokens/s | ms/token | Spread (ms) | Spread % | Device median (ms) | Tokens | Imports checked | Coverage ties | Build SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 1 | kept | 4.087415 | 244.65 | 4.087415 | 0.034415 | 0.84% | 3.951 | 32/32 | 194/194 (97/position) | n/a (CHECK-DSL) | `788958df…219154` |
| 2 | kept | 5.701229 | 350.80 | 2.8506145 | 0.073494 | 1.29% | 5.543 | 64/64 | 388/388 (194/position) | 0,0 | `8a4acdd4…e2f4b1` |
| 4 | kept | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] |
| 8 | kept | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] |
| 4 | plain | 11.929137 | 335.31 | 2.982284 | 0.308351 | 2.58% | 11.695 | 128/128 | 776/776 (388/position) | 0,0,1,1 | `4792e8a4…6f86674` |
| 8 | plain | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] | [U] |

Sources: `/root/bench_b1.json`, `/root/bench_b2.json`, `/root/bench_plain_b4.json`
on the host, and each build's `bench_result.json`. The kept log is
`/Users/ckl/.hermes/cache/scratch/bench-out9.log`. Full build SHAs are in the
JSON records.

The device median is the decode log's "per-token latency on device
(prologue -> argmax)". The host step is the benchmark's 31-sample median.
Spread is max minus min over the same 31 samples.

B1 has no per-request coverage row because `bench.py` uses `--check-dsl` at
batch 1 and the per-request coverage check only at batch greater than one
(`bench.py:158,236-238`). Its gates are the byte-identical host `--check` at
positions 0 and 1040 and 32/32 HF greedy tokens.

## 3. What the two kept batches show

B1 and B2 use the same recipe, the same source commit, one lock and one GPU.
The pair is a controlled scaling measurement:

| Quantity | B1 | B2 | Ratio |
|---|---:|---:|---:|
| Host step (ms) | 4.087415 | 5.701229 | 1.394825 |
| Device span (ms) | 3.951 | 5.543 | 1.4029 |
| Throughput (tokens/s) | 244.65 | 350.80 | 1.433872 |
| Cost per request (ms) | 4.087415 | 2.8506145 | 0.697413 |

Doubling the request count costs 39.5% more step time, not 100%. Per-request
cost falls 30.26% and throughput rises 43.39%.

This answers the open question in the mechanism scaffold's section 5.1: at B2
the step does not grow linearly with the request count. It grows sublinearly.
The mechanism is still [U]: overlap between the two requests' launches, a
non-replicated fixed part of the step and memory-level concurrency all predict a
ratio below 2, and this record cannot separate them. No task trace, counter or
profiler capture exists for this path.

The B2 spread is 1.29% of its step and the B1 spread is 0.84%. These are
single-run spreads, not a paired comparison.

## 4. The plain B4 row

The plain recipe is the one `batch_correctness.py` qualified on hardware
(`bench.py:6,50-54`). It is a different build from kept: different compile
flags and no kept tuning. Its gates all passed:

- 128/128 tokens, 32 per request;
- 776/776 imported launches byte-identical at positions 0 and 1040, 388 per
  position, 97 per request;
- per-request coverage of 4x1055 positions, 0 wrong beyond ties, ties 0,0,1,1.

Its spread is 2.58% of the step, 2.0x the kept B2 relative spread. The two rows
are different recipes, so this difference is not a controlled comparison and is
not attributed.

The decode log reports a device median of 11.695 ms and its own host median of
11.935 ms. The benchmark's 31-sample median is 11.929137 ms. Report the
benchmark value; the 0.006 ms difference is the sample window, not a mechanism.

## 5. Host step against device span

For each completed row, the host step exceeds the device span:

| B | Recipe | Device (ms) | Host (ms) | Host minus device (ms) | Share of host step |
|---|---:|---:|---:|---:|---:|
| 1 | kept | 3.951 | 4.087415 | 0.136 | 3.34% |
| 2 | kept | 5.543 | 5.701229 | 0.158 | 2.78% |
| 4 | plain | 11.695 | 11.929137 | 0.234 | 1.96% |

This gap is an upper bound on every host-side cost outside the device span,
including enqueue, launch and synchronisation. It is [M] as a gap. It is not a
dispatch attribution: the record has no dispatch trace, so no part of the gap
is assigned to launch count [U]. The scaffold's rule still applies: launch cost
is material only if a dispatch trace shows at least 2% of the step in dispatch
gaps or setup.

## 6. Mechanism status after these rows

| Mechanism | Status before | Status now | What changed |
|---|---|---|---|
| Work replication | [M] counts | [M] counts and B1-B2 marginal cost | The second request costs +39.5% step, not +100% |
| Static launch count | [M] | [M] | Unchanged: 97 instances per request per position |
| Runtime launch count | [U] | [U] | Host-device gap measured [M]; dispatch attribution absent |
| Runtime occupancy | [U] | [U] | No counter |
| Dependency chain | [U] | [U] | No trace |
| HBM/L2 traffic | [U] | [U] | No capture |
| Serialization | [U] | [U] | No trace or alternative slot map |
| Decode wall time | [U] | [M] for kept B1/B2, plain B4 | Section 2 |
| Throughput | [U] | [M] for kept B1/B2, plain B4 | Section 2 |

The mechanism audit and its decision rules remain
`docs/QWEN-BATCH-MECHANISM-SCAFFOLD-2026-10-18.md`. That file now points here
for the measured values and keeps the [U] list.

## 7. Provenance

- Kept B1/B2: one runner, exclusive `flock`, `/root/bench.lock`. The remote
  shell exited at B4; no kept B4/B8 gate line, timing line or result JSON
  exists.
- Plain B4: a later separate runner, same lock and same source commit, after
  the kept runner's remote side was gone. It wrote
  `/root/bench_plain_b4.json` and `bench_result.json` with `status: PASS`.
- The two earlier B1/B2 readings that were measured with concurrent runners
  remain withdrawn. They are not used and their values are not repeated.
- No number here is compared across sessions or across recipes. B1 and B2 are
  the only same-recipe pair.

## 8. Limits

- kept B4/B8 are absent; plain B8 is in progress.
- There is no same-recipe B2-to-B4 pair, so no B4 scaling statement is made.
- The plain B4 row is the correctness-qualified recipe, not the kept build.
- B1 has no per-request coverage check by design.
- The per-launch `--check` timings in the correctness record are isolated
  launches, not decode time.
- There is no task trace, hardware counter or profiler capture for this path.
- These are absolute ETX values. No vLLM comparison exists.
