# Fresh-address cached handoffs on MI300X

Status: prepared; **no GPU measurements and no model validation**. This is a
mechanism experiment, not permission to change ETX's default coherence protocol.
The reported ~0.5 us bypass penalty and ~0.1–0.2 us cached-load target are hypotheses
for this experiment, not new measurements or full-handoff latency promises.

## Question and prior evidence

Can a reader safely use ordinary cached loads when its payload lines have not
previously been read since a CU-wide invalidation, with writer writeback before
cross-XCD notification? Does the complete protocol beat coherent loads?

Read `docs/RESEARCH-sync-hardware.md`, `docs/DATA.md`, and the existing
`bench/sync/sync_bench.hip` / `bench/hw/hw_bench.hip` tests first. Earlier reused-line
cached paths had stale data; those results do not establish the fresh-line case.
The external fleet STATUS checkout was unavailable in the preparation environment.
There is no reason here to revive the rejected NT-off cache-reuse proposal.

A stale value can be from a **clean, outdated cache line**; it need not be dirty.
“No read in this launch” alone is insufficient: an address may retain cache state
from a previous launch. Both epochs therefore establish the same measured-CU
invalidation precondition before any payload access.

## Frozen microbenchmark protocol

`sync_bench fresh` adds an isolated experimental mode; old modes and the model
runtime are unchanged. A word means uint32_t. Payload sizes are 64, 8192, 65536 B.
Every positive handoff has a unique 128-B-aligned slot, rounded up to whole 128-B
lines, so two payloads never share a line. An 8-KB payload occupies 64 lines; it
does not fit in a single line. No positive slot is read before its writer publishes
it, or reused within that launch. Payload data include epoch, handoff and word
indices, and every word is checked by every reader.

Each launch reserves one 256-thread block per CU with 40 KiB dynamic LDS. Host
occupancy checks and observed HW_ID/XCC identities must both agree. No block index
is assumed to be a CU identity. Every observed CU executes one `buffer_inv sc1`
in the prologue, then all blocks rendezvous. There is no payload invalidation in
the reader handoff loop. Runtime counters establish instruction execution coverage;
hardware testing is still required to establish the intended cache semantics.

The writer drains stores from every participating wave, synchronizes its block,
and for cross-XCD cases executes `buffer_wbl2 sc1` plus completion wait before
publishing the flag. Flags are isolated 128-B objects and polled using coherent
`sc0 sc1` loads. Cached payloads use plain `global_load_dword`; the paired bypass
case uses `global_load_dword ... sc0 sc1` with the same allocation and protocol.
This isolates load policy; it is not a comparison against a separately allocated
uncached-memory path or a complete existing model runtime.

Topology cases are two different CUs on XCD 0, XCD 0 to XCD 1, and one writer on
XCD 0 to seven readers, one on each other XCD. Placement is recorded and checked
for each launch. This is representative placement, not exhaustive CU-pair coverage.
Idle means no synthetic traffic; helper blocks still poll control flags. Busy means
unused CUs stream through disjoint 2-MiB regions, recording bytes processed. This
is synthetic cache/memory pressure, not Qwen or Mixtral inference.

The second epoch is a **separate cooperative kernel launch** after the host observes
completion of the first. It retains the payload allocation and virtual address,
changes the expected epoch, and invalidates once on every observed CU again.
There is no intervening host payload reset/copy. This is not a loop pretending to
be a relaunch. CSV metadata supports auditing; source/ISA inspection remains
necessary to establish that the recorded IDs describe actual launch calls.

## Counts and negative controls

There are 18 size/topology/load combinations. Each has three adjacent cached/bypass
pairs with alternating order, plus two negative controls: 144 processes, each with
two launches. **Every** variant requires at least **one million handoffs**, so the
checked-word floor follows from the payload: 16, 2048 or 16384 words per handoff, and
at least 10 million words per reader (16 million at 64 B, which is the smallest case).

That count costs address space. A positive slot is unique within its launch, so the
payload allocation is rounds x line-rounded size: 0.12 GiB at 64 B, 7.63 GiB at 8 KB,
**61 GiB at 64 KB** for the positive cases; the negative controls keep their one reused
slot. Payload traffic across the 144 cases is 7.1 TB written and 21.2 TB read, 28.3 TB
in the timed pass; the positive cases alone are 21.25 TB, and the controls' priming pass
brings the whole matrix to 35.42 TB, plus about 35 GB of raw CSV. That traffic is not the
constraint: 28.3 TB at the 4.3 TB/s ceiling is 6.6 s. The wall time comes from the
per-handoff round trips, predicted below at 6598 s, so the session budget must be
re-planned: the 2-hour cap was sized for 611 handoffs at 64 KB, not for one million.

Negative controls intentionally reuse one slot to stay within memory limits:

- `forced_old`: perform the loads but replace their observed values with the prior
  epoch's pattern before comparison. Every word must fail. This checks the checker;
  it is not evidence that hardware produced stale data.
- `primed_stale`: write an old pattern, let all readers cache it, then overwrite,
  write back, signal and read without invalidation. Read order is reversed to expose
  recently primed lines. A nonzero mismatch count is required to demonstrate the
  negative control. Zero mismatches is **inconclusive**, not a successful control.

The full matrix gate stays closed if any run is missing, incomplete, stale in a
positive case, incorrectly placed, undersized, or has an inconclusive control.
A stale result stops promotion of the protocol; do not relax the test to pass it.
A completed failed case remains in the report. Driver failures/timeouts stop the
runner rather than scheduling more kernels on a potentially unhealthy device.

## CPU cost model (pre-GPU estimate)

`etx/tools/fresh_sim.py` prices the same frozen matrix on the CPU from this repository's
measured constants. Every constant carries its `file:line` and a measured / derived /
assumed status, and the tool fails closed if the cited line disappears; `docs/MODEL-vllm-kernels.md`
section 10 supplies the L1 / L2 / MALL round trips, the `buffer_inv sc1` cost and the
loaded multiples. It is a conditional p50 scenario, not a measurement, and it cannot
decide staleness.

- Predicted per-handoff full flow (write, flush, flag, poll, read, check, ACK): 2.07 us
  at 64 B up to 65.23 us at 64 KB x fanout x busy.
- Reader first load: 0.143 us (L2, idle, same XCD) to 0.710-0.730 us under synthetic
  load, so the 0.1-0.2 us target is predicted only in the idle local cases.
- Matrix wall: 6598 s against the 28 800 s budget, conditional on the omitted overhead
  being small; the bandwidth floor of the timed pass is 6.6 s.
- Cached against bypass, same allocation and protocol: idle 65 536-fanout is 11.99 us
  cached against 40.89 us bypass, but under synthetic load the two converge (65.23
  against 63.95 us) because queueing, not the cache level, sets the loaded latency.

The model's unknown list is the GPU run's job list: real fresh / bypass single-dword
latency, bandwidth sharing, store and flag latency, loaded writeback, fanout contention,
per-word check cost, barriers and timers, setup and host I/O, the tail percentile, and
the stale rate itself.

## Timing and decision

Raw samples retain first-load ticks, bulk load + checking ticks, writer-observed
handoff + ACK ticks, and mismatch count for every handoff/reader. The writer clock
covers write, drain/writeback, notification, reads, checking and all reader ACKs.
It is a conservative, instrumented full-flow duration; it is **not** a clock-skewed
subtraction of timestamps from different CUs. Writer durations repeated in reader
rows are counted only once by the analyzer.

The first-load metric covers one instrumented 32-bit load, not all 64 KB. The HIP
wall-clock frequency and minimum back-to-back timer cost are recorded. Report raw
p50/p95/p99; do not subtract an assumed timer overhead or hide a slower tail.
The provisional target is raw median <=0.2 us (faster than 0.1 us is not a failure).
A timing target miss cannot be relabeled a correctness failure or success.

`mechanism_gate` reports evidence/correctness only. Advancement also requires the
cached load target and repeatable paired full-flow improvement without unexplained
tail regression. Three pairs are a screening experiment, not a confidence interval.
If coherent loads are already as fast, or full-flow overhead erases load savings,
stop before implementing model changes. The prior ~0.5 us claim is not subtracted
from model latency; use measured same-condition paired results.

## Prepare and execute

CPU only:

```sh
python -m etx.tools.fresh_handoff plan > build/fresh-plan.json
python -m etx.tools.fresh_sim --json build/fresh-sim.json
python -m pytest -q tests/test_fresh_handoff.py tests/test_fresh_sim.py
hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/sync/sync_bench.hip -o build/fresh-sync-bench
hipcc -O3 -std=c++17 --offload-arch=gfx942 --cuda-device-only -S bench/sync/sync_bench.hip -o build/fresh-sync-bench.s
```

Inspect the new kernel's ISA for plain/coherent payload loads, writeback before
signal publication and a single prologue invalidation site; also inspect spills.
Compilation cannot prove occupancy, placement or coherency on the device.

On an already approved dedicated MI300X, the prepared session is
`examples/sessions/fresh_handoff.py`. It builds with the device host toolchain and
runs the same CLI:

```sh
python -m etx.tools.fresh_handoff run --binary build/fresh-sync-bench --out build/fresh-results --budget-seconds 7200 --case-seconds 420
python -m etx.tools.fresh_handoff analyze build/fresh-results > build/fresh-report.json
```

Output must be new. The runner preserves full per-case logs, a frozen matrix,
binary/source hashes, Git revision and worktree state, exit statuses, evidence
checksums, placements and raw samples. Analysis independently recomputes counts,
checksums, placement, epoch pairing and timings. Missing runs fail closed. The
budget is a **cap**, not a claim that 144 processes will fit: at one million handoffs
per variant a 64 KB positive case allocates 61 GiB and moves 65 GB per epoch, so plan
about 8 h of GPU time and 10 h of wall time, and keep the per-case cap above the
slowest expected case. Earlier 7200 s / 420 s values were sized for the word-based
counts (611 handoffs at 64 KB) and will truncate the matrix. Allow several GB for raw
CSVs. Device waits have a 5-second progress timeout, a 180-second launch watchdog and
10-second host heartbeats; the outer process deadline also covers host CSV output.

No GPU has been provisioned. The runner captures uname, hipconfig and rocm-smi
identity/driver/clock/power/utilization/PID output (including missing-tool failures).
Archive ECC configuration and verify machine captures at execution time as well. Establish an
explicit rental budget before provisioning. Hardware results must be registered
in `docs/DATA.md`; no synthetic CPU fixture belongs in that measured-data catalog.

## Two-model follow-up (blocked until the mechanism gate)

The assumed models are Qwen3-30B-A3B and Mixtral-8x7B. Both are separate mandatory
validation targets, not two labels attached to the same synthetic traffic run.
For **each** model:

1. Freeze revision, weights, input tokens, context, batch/request count, dtype,
   same-operator bypass baseline and timing boundary. Run the full microbenchmark
   matrix with actual model decode producing busy load and record achieved load.
   This orchestration is not implemented by the synthetic runner yet.
2. Describe activation lifetimes, aliasing and cross-token state. Integrate fresh
   slots and a real relaunch invalidation boundary behind an opt-in feature only
   after the microbenchmark gate. The default coherent path remains the fallback.
3. Verify imported kernels independently, complete tensor writes and finite values,
   and full generation (32/32 HF greedy). Preserve negative/coverage tests. Do not
   use token equality alone to establish per-word cache correctness.
4. Repeat the complete probe under actual busy load, then compare full decode using
   identical operators and frozen measurement intervals, plus the real engine
   baseline. Record load-policy savings separately from operator/quantization gains.

Current deliverables prepare the deciding hardware experiment and its CPU evidence
checks. Neither real-model busy orchestration nor a new model coherence protocol
is claimed complete; building them before this deciding measurement would skip the
repository's microbenchmark-first gate.

## Preparation validation (CPU only)

ROCm 7.2.4 HIP/LLVM 22 compiled and linked the gfx942 binary and emitted device
assembly. The new kernel has one `buffer_inv sc1` site, plain and `sc0 sc1` load
sites, writer writebacks, 26 VGPRs and zero reported register spills. The existing
GWS mode still emits its pre-existing host target-attribute warning. Session
dependency/deadline dry-run and 39 evidence-checker unit tests passed. These are
preflight results; no occupancy, latency, stale rate, or model result was measured.
