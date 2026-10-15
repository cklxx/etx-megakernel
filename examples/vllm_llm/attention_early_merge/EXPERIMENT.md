# C6: attention early publication and elected reduce

Status: experimental, off by default. This record does not authorize GPU use.
The preparation result gives the exact source commit and local build receipts.

## Decision and hypothesis

Test the combined mechanism before any full-megakernel integration. One prep
owner publishes inside its task. The last drained attention partition continues
into the original per-head reduce. The source graph is already head-local.
No all-head barrier is removed. o_proj still needs all four head outputs.

The registered combined target is at least 50 us per 48-layer step. The separate
30 us general KEEP rule does not pass this target. A 30–50 us result is a
passenger only. A probe pass permits full-model integration. It proves no wall
gain. A correct probe below 50 us closes this target lever. Missing coverage,
incorrect payloads, timeout, stale identities or a resource failure stops timing.

Prior evidence is fixed in the attention research report under the user's
scratch directory. Its input trace SHA256 is
`aa68468246acde1a8d0e19ad0fbc6d5913bc37c7a3f352f44cecec8c0bdef320`.
The trace covers one MI300X VF, Qwen3-30B-A3B, bf16, batch 1, context 1024.
The fixed-body model gives a 101.82 us combined local advance with zero new
cost. This is not a wall forecast. The registered gross budget is 96 us. The
remaining combined cost allowance is (96−50)/48 = 0.95833 us/layer. Reduce alone
left 0.06917 us/layer in that model. Do not restart a separate reduce campaign.

## Frozen implementations and inputs

`build.py` emits one probe executable for both arm labels. Mode 0 runs separate
resident prep, five attention and one reduce tasks per head. Mode 1 runs five
resident partitions per head. Partition zero owns prep and then attention.
The elected drained partition runs reduce. The executable's `identity.json`
binds generated sources, IR, import metadata, assembly, source hashes, compiler,
commands and flags. The candidate label reaches `c6_run` with mode 1. There is
no fallback to the old model. A second binary defines C6_DIAGNOSTIC. It passes
all correctness checks but supplies no performance result or attribution.

This is a **resident separate-task protocol control**. It is not the complete
ETX queue scheduler or a same-kernel HIP graph. It keeps the constituent math
and task boundaries. It deliberately isolates publication and continuation.
Its sum of local spans is a screening result, not a measured step gain.
A timing build must use --footprint with a fresh offline.py output. Its entry
retains the original kept dispatcher body as a device function behind a
kernel-argument branch. C6Args.retained is always null in the host. Thus the
retained body is compiled into the timing entry but never executes. The
compiler reserves resources for the kept body and the C6 body in one entry.
Both arm modes use this entry and its measured resource reservation. No padding
or assumed register price replaces compiler allocation. A small build without
--footprint is correctness-only. Its launcher refuses --measure, and the
session refuses its identity before timing.

The separate full-dispatch envelope adds this actual body to the kept dispatcher. It
checks compiler resources only. It has an unbound argument and must never run.
The ordinary kept model plus a separate C6 entry supplies actual-model token
checks. C6_HYBRID=1 replaces attention segments in a serial full-model replay.
This path is for correctness. It is not timed as a megakernel candidate.

The model build uses the kept recipe from this branch. The session binds
vl.bin, vl.manifest and golden_c1024.txt by SHA256. Import IR must match between
model, performance and diagnostic probes. Cross-directory comparison removes
only LLVM's ModuleID and source_filename path lines. It also renames HIP's
path-derived unit-ID symbol after proving its only uses are its zero-initialized byte
marker definition and the compiler.used retention entry. Any other use or
marker change is refused. Every instruction, attribute, constant and remaining
metadata byte must match. Raw IR hashes remain part of each build identity. Capture frames bind every byte and
argument layout. The registration identity is completed by the commit in the
preparation result; the session adds binary and input hashes before execution.

## Ownership, ordering and reset

A cooperative launch has 304 blocks of 1024 threads. The host requires gfx942,
304 CUs and cooperative launch support. It checks at least one resident block
per CU. Each block gets an atomic ticket on its discovered XCD. Heads use XCDs
0–3. Ticket zero is the reachable prep owner. Control uses tickets 0–6. The
candidate uses 0–4. Surplus blocks exit. XCD 7 ticket zero waits for all four
output publications and performs the device acquire, then reads and copies the full output payload.
The replay checks that copy byte for byte. Post-launch checks require
at least seven discovered tickets on each active XCD and one on XCD 7. A failed discovery aborts
before timing. There is no queued owner behind resident waiters.

State is four cache-line-aligned C6State records, eight ticket counters, abort,
and the four-head output count. Each state has epoch, claim, prep-ready, mask,
arrival count, elected partition and output epoch. Agent-scope atomics protect
state. Payloads retain their original allocations and layout. Every prep writer
wave drains, then the workgroup joins, before the owner publishes ready. All
attention slot rounds finish, every writing wave drains, and all waves join
before the leader arrives. An acquire-release arrival chain joins producers.
The elected block broadcasts one decision and acquires payloads before reduce.
The original domain acquire remains. Cross-XCD output publication retains the
device flush and the other-XCD consumer retains the device acquire.

Only one prep runs. The original in-place q/k norm, RoPE and KV writes remain.
All five configured partitions arrive, including inactive imports. Reduce keeps
the original eight slots, block numbers 8*h+s, its own argument block, bf16
partials, fp32 statistics and increasing partition accumulation order. Arrival
order selects the execution block, never the math order. No online softmax or
new partial summation is used. Import barrier rounds finish before LDS reuse.
The generated manifest is authoritative: this source reports reduce as 272 B
per slot, 2176 B total, not 2176 B per slot as the older research said.

Every wait has a spin limit and observes abort. Host stream polling has a
five-second limit. Session process-tree supervision bounds failed launches.
No live state is reset after a timeout. The process must end first. Normal
reset follows complete stream quiescence. Epoch zero and u64 rollover are
refused. A new process starts with new state. Fault modes exercise duplicate,
stale, future and missing-owner timeout. Skew stresses owner/arrival order.
Repeated replay uses the same allocated cache lines across fresh epochs.

## Cost basis and gates

Measured here: CPU checks and compiler outputs only, in the preparation result.
Derived historical ceiling: the 109.1 us trace gap and the more conservative
96 us registered gross allowance. Estimated whole-step byte floor: 6.1904 GB /
4.3 TB/s = 1.440 ms. Payload traffic and the two body-hop allowances remain.
No part of the 260 us body-hop row is credited to this boundary change.

Both probe spans start after inputs and initial protocol state are ready. Both
end after all four outputs are published and acquired on XCD 7, followed by
a full output read on XCD 7 and kernel retirement. Ticket assignment, prep publication, fan-in, election,
broadcast, reduce argument setup, cache effects, output flush, waiting and
retirement are inside both spans. The candidate's owner keeps its arguments
and may keep cache state; any benefit or cost is measured, not assumed.

Copies and state reset precede both local spans and use the same state image.
This control also allocates the new state. There is no demonstrated reset
saving. The retained dispatcher branch does not execute. Its dynamic spill traffic is
not a measured full-model cost. The incremental reset cost against the actual ETX runtime, dispatcher
interference, spills and whole-step host cost remain unknown. A positive local
screen is conditional on these full-model costs. It does not satisfy the
registered net step target by itself. No GPU visibility or residency fact is
inferred from CPU simulations or static occupancy estimates.

Native whole-dispatch resource gate: at most 128 VGPR, LDS at most 29752 B and
at least one static resident 1024-thread block/CU. Preserve scratch and spill
counts as costs. Resource success is necessary, not proof of actual residency.
The session checks native resources again before any timing. A resource failure
closes this implementation. Do not relax the gate to make the preparation pass.

All 48 layers and positions 0, 1023, 1024, 1040 are mandatory. These cover one,
four and five active partitions and the 1024/1025 sequence boundary. Each frame
must byte-match all 15 before/after buffers in both arms, in both binaries.
The model must pass exhaustive same-build original/import checks at all four
positions, CHECK-DSL, and 32/32 HF greedy in both ordinary and hybrid paths.
No timing call precedes these checks. Every timed replay also byte-checks.

For each position, take five complete paired sums over all 48 layers. Reverse
invocation order on odd pairs. Use decimal log text converted to integer ns.
All gains must be positive, their median at least 50000 ns and greater than
twice their range. Require all four positions to pass this local screen. Report
the 30000 ns general decision separately. No shortened or missing run is a
number. No new S3 or same-kernel graph comparison exists here. Both baselines
are mandatory for a later full-model host-wall claim under the project rules.

## Commands and stop rules

Local commands, from the repository root (use new output paths):

```sh
.venv/bin/python -m examples.vllm_llm.attention_early_merge.simulate --out /absolute/scratch/simulation.json
.venv/bin/python -m pytest -q tests/test_attention_early_merge.py tests/test_head_local_protocol.py
.venv/bin/python -m examples.vllm_llm.attention_early_merge.offline --out /absolute/scratch/full
.venv/bin/python -m examples.vllm_llm.attention_early_merge.build --local --footprint /absolute/scratch/full --out /absolute/scratch/probe
bash examples/vllm_llm/attention_early_merge/rocm_check.sh /Users/ckl/.hermes/cache/scratch/new-c6-rocm
.venv/bin/python -m etx.tools.gpu_jobs examples/sessions/attention_early_merge.py --gpus 0 --dry-run
```

The preparation result records all 16 local matrix variants and the full suite.
Use the shared scratch advisory lock for heavy stages. Check disk first. No
local command provisions a GPU or makes a remote call. macOS checks IR, ISA and
host syntax. The offline x86 container checks ROCm 7.2.4's allocator. Neither is
a native HIP link, device load or execution certificate.

After the local gates pass, a separately authorized operator can stage the
committed repository with `tools/session.sh push`. Install the repository's
Python environment on that machine first. Set ETX_VLLM to the source
checkout and ETX_DEVLIBS to the native device-library directory when needed.
Then use the existing session runner:

```sh
ETX_VLLM=/absolute/vllm-checkout \
C6_WORK=/absolute/new/c6-work C6_MODEL=/absolute/packed/qwen \
python -m etx.tools.gpu_jobs examples/sessions/attention_early_merge.py \
  --gpus 0 --logdir /absolute/new/c6-logs
python -m examples.vllm_llm.attention_early_merge.session \
  --work /absolute/new/c6-work --model /absolute/packed/qwen status
```

These commands run on the staged GPU machine. tools/session.sh does not forward
C6_WORK or C6_MODEL from the local shell. Set them in the remote runner
environment before using its start command, or use the on-machine command
above. Keep C6_WORK stable across retries.

The job file has a CPU build lane and one pinned GPU lane. Full raw logs and
true exit codes stay under each attempt. A single nonblocking advisory lock
prevents duplicate runners. Build retry creates a new attempt. Check retry
creates a new checks directory and repeats every gate. The lease keeps its
original 2400-second deadline across retries. An expired lease needs a new,
explicitly budgeted session, not an automatic extension. stage.json prints the
stage, blocker, evidence and safe resume command. Do not use a run resume for
a failed build; rerun `build` within the same lease instead.

Proposed cap: 40 minutes including setup and CPU builds. The older 15-minute
probe estimate excluded setup. Cost estimate: (2/3 hour) times the actual
single-GPU hourly price, unknown here. No price was queried. Spend is $0.
Native link/load, payload visibility, discovered placement, actual residency,
real model data, device timing and final net gain remain device-only questions.
Delete a future rented VM as soon as its authorized plan ends, per AGENTS.md.

## Integration boundary

Only this experiment directory, its tests and its unique job file are owned.
No shared index, model.py, host.hip, lease.py or default is changed. The build
makes check-only copies of host.hip using checked insertion markers. Preserve
the router branch's host/model/lease changes during later integration and rerun
these hooks and all source identities there. The old specification-only
ETX_VL_HEAD_LOCAL_PROTOCOL opt-in still refuses a production plan. This probe
uses separate commands and C6_HYBRID. It does not relabel that hook as complete.
