# DeepSeek batch scaling: the mechanism per step (2026-10-18)

Status: offline decomposition DONE from retained traces; the on-box session is PREPARED, NOT RUN
(`examples/sessions/ds_batch_mechanism.py`). No GPU was used for this document.

Publication clarification (2026-10-21): the decomposition is an accounting identity,
not a causal isolation experiment. Residency-cap and saturation explanations below
are hypotheses pending P1–P3. N is the configured worker-slot count and u is a
trace-derived busy fraction, neither a hardware occupancy counter. Estimated bytes
are not measured HBM/L2 traffic. Keep the missing-counter caveats in
`DS-BATCH-MECHANISM-SCAFFOLD-2026-10-18.md`; unchanged task bodies do not rule out
HBM saturation or prove spare bandwidth. The compiler receipts are historical,
not a fresh build or GPU qualification of the merged revision.

## 1. The question, corrected once

The question was why B4 and B8 are slower per token than B2. Per generated token, B4 is **not** slower:

| per generated token | B2 | B4 | source |
|---|---:|---:|---|
| traced span / B (ms) | 2.1696 | 2.1675 | instrumented final-token traces, 2026-10-15 continuation |
| host step / B (ms) | 2.2662 | 2.2303 | reused unprofiled rounds: 4.532310 / 2, 8.921324 / 4 |
| throughput (tok/s) | 441.3 | 448.4 | same rounds |

What does get worse is each request's token interval: 4.571 -> 9.007 ms (x1.97). Throughput gains only 1.6%,
while the selected vLLM's step grows x1.303 (4.732610 -> 6.165512 ms) for twice the tokens. The question that the
evidence can answer is therefore: **why does going from B2 to B4 add no throughput, and so double every request's
interval?** "Twice the work" is not the answer: vLLM also does twice the work at B4.

Sources: `examples/gpu-live/results/2026-10-16-multibatch/{trace_report,summary}.json` (one MI300X VF, ROCm 7.14,
DeepSeek-Coder-V2-Lite-Base bf16, context 1024; three traced invocations per B, every request 32/32 tokens, 27/27
layers). Traces carry ETX_TRACE instrumentation and cover the last decode token; they are never a performance verdict.

## 2. Decomposition

Regenerate: `.venv/bin/python -m examples.dsv2lite.batch_mechanism offline --json ... --markdown ...` (~2 min,
no GPU; it lowers the four plans and refuses a trace whose task count differs from the lowered plan). Output kept in
[`evidence/deepseek-batch-mechanism-2026-10-18/`](../evidence/deepseek-batch-mechanism-2026-10-18/offline.md).

The identity that ties the counters to milliseconds is

    span = W / (N * u)      W summed task-body time, N resident worker slots, u = W / (N * span) slot busy fraction

It holds exactly, by the definition of u. The ratio of two batches' spans is therefore the product of the ratios of
tasks, body per task, 1/N and 1/u. Each factor below is turned into milliseconds in this fixed order:
tasks, then body, then slots, then busy fraction. The four terms add up to the measured change to within rounding.

| quantity | B1 | B2 | B4 | B8 (plan) |
|---|---:|---:|---:|---:|
| megakernel dispatches per step | 1 | 1 | 1 | 1 |
| grids per step | 271 | 438 | 876 | 1752 |
| tasks per step | 53022 | 109892 | 219784 | 439568 |
| resident worker slots N | 296 | 592 | 592 | 592 |
| requests per slot | 1 | 1 | 2 | 4 |
| tasks per slot (mean / max) | 179.1 / 186 | 185.6 / 219 | 371.3 / 438 | 742.5 / 876 |
| summed body W (ms) | 737.3 | 1934.0 | 3852.4 | — |
| body per task (us) | 13.90 | 17.60 | 17.53 | — |
| slot busy fraction u | 0.8049 | 0.7529 | 0.7506 | — |
| traced span (ms) | 3.0944 | 4.3392 | 8.6702 | 17.34 [E] |
| critical path: bodies / dependency gaps / queue gaps (ms) | 2.716 / 0.379 / 0.000 | 3.740 / 0.532 / 0.067 | 7.341 / 0.963 / 0.366 | — |
| host step of the traced run minus span (ms) | 0.160 | 0.235 | 0.341 | — |
| replicated bytes per step [E] (GB) | 4.934 | 9.869 | 19.738 | — |
| bytes / span [E] (TB/s) | 1.595 | 2.274 | 2.277 | — |

| step | span change (ms) | tasks | body per task | slots | busy fraction |
|---|---:|---:|---:|---:|---:|
| B1 -> B2 | +1.2448 | x2.0726 = +3.3190 | x1.2657 = +1.7038 | x0.5000 = -4.0586 | x1.0691 = +0.2806 |
| B2 -> B4 | +4.3310 | x2.0000 = +4.3392 | x0.9960 = -0.0348 | x1.0000 = 0 | x1.0031 = +0.0265 |

The five columns map onto the requested counters as follows:

- **launch count:** one megakernel dispatch per step at every B. Grids and tasks are device queue entries, not launches.
- **occupancy proxies (not hardware occupancy):** configured N and trace-derived u.
- **dependency chain:** the critical-path split.
- **HBM/L2 traffic:** the byte estimate per span. No counter has been accepted yet; see section 4.
- **serialization:** requests per slot, tasks per slot and the queue gaps.

Per-task body by family: B4/B2 lies in 0.974-1.066 for every family; gate_up is 0.976, down 0.974 and qkv 0.986.
B2/B1 lies in 1.19-1.45, apart from lm_head at 1.02. The full table is in the evidence file.

Relation to `docs/deepseek/DS-BATCH-ATTRIBUTION-2026-10-18.md` (2ec61c4, same traces): its numbers agree with these.
Its basis differs slightly: round 1 there, the median-span run here; the spans differ by at most 0.3%. Its verdict,
"BODY-BOUND, no amortization", restates that body time doubles. This section explains why doubled body time
turns one-for-one into span: the slot count is fixed at 592, and the busy fraction does not rise when requests share
a slot. Its queue-gap finding (qkv, attn) is the critical-path trace of that slot sharing.

## 3. Mechanism per step

**B1 -> B2: a second resident slot per CU, paid for by co-residency stretch.**
- N doubles from 296 to 592. That removes 4.059 ms of the 3.319 ms that doubled tasks would add at one slot per CU.
- Every task body stretches by a factor of 1.266, which adds 1.704 ms. Two workgroups now share each CU's SIMDs and
  memory pipes.
- The retained in-situ measurements say the same (2026-09-27, older build, attribution only; docs/deepseek/MODEL-dsv2.md section 5):
  - a stream workgroup beside a stream workgroup: x1.80;
  - a compute workgroup beside a stream workgroup: x1.74;
  - same-phase real-tile pairs: x1.35-1.55.
- Result: 3.094 -> 2.170 ms per generated token.
- Caveat: B1 is a different recipe (D4, 72 AGPR, one workgroup per CU), so this step is not a clean single-variable comparison.

**B2 -> B4 hypothesis: residency cap and queue serialization of replicated work.** The accounting chain (causality awaits P1–P3):
1. The x arms compile to 256 VGPR per lane at 2 waves per SIMD (`b_x4/resources.json`: vgpr 256, occupancy 2).
   That allows two 256-thread workgroups per CU, so N = 592 is already full at B2 (`plan.wg_per_cu = 2` for x2, x4 and x8).
2. `graph._Req` pins request r to slot r mod 2. At B4, each slot runs two requests' queues back to back: 185.6 -> 371.3
   tasks per slot.
3. The span equals the slot's serial body sum divided by u: 3.267 ms of body per slot at B2 and 6.507 ms at B4.
   With u unchanged (0.753 -> 0.751), that gives 4.339 -> 8.670 ms.
4. Body per task is unchanged (x0.996). The estimated delivered rate is 2.274 -> 2.277 TB/s,
   about 53% of 4.3 TB/s and 65% of the older 3.51 TB/s hw_bench stream
   (`results/2026-09-27d/hw_stream.log:18`). These are model-based byte rates and a cross-session
   reference, not hardware traffic counters. Unchanged bodies cannot distinguish a residency
   limit from an already-saturated memory/issue path; P2 and P3 must decide.
5. Signature of the serialization on the critical path: queue gaps rise from 0.067 to 0.366 ms. qkv's mean dependency
   gap rises from about 2.6 us at B2 to 140-156 us at B4: a request's next layer starts about 145 us after its
   producer finished, behind the other request's tasks on its slot (`trace_report.json`, r0-r3 qkv rows).
6. The host excess grows from 0.235 to 0.341 ms (+0.106 ms, 2.4% of the span change; instrumented runs).

The proposed explanation is that replicated reads (9.87 -> 19.74 GB estimated per step)
and fixed configured slots make extra requests add serial work rather than parallel capacity.
The accounting fits that explanation, but does not establish which resource is limiting or
how much physical traffic cache reuse removes. Likewise, vLLM's x1.30 step growth is a measured
comparison, not proof that weight sharing alone caused the difference.

Candidate, untested: u stays at 0.75 instead of rising when two requests share a slot. The static queues run in
creation order (layers outer, requests inner), so request r+2 cannot fill request r's dependency bubbles. Stalls at
the head of the queue stall the slot. The slot traces in section 4 count this blocking directly.

**B4 -> B8: the same mechanism, four requests per slot, plus two unknowns.**
- The plan has the same 592 slots and 742.5 tasks per slot.
- [E] Projection at B4's body per task and busy fraction: span 17.34 ms, still 2.17 ms per generated token, and a
  request interval of about 17.3 ms.
- Not included in that projection:
  - x8 spills 144 B per lane, against 64 for x4;
  - B8's hidden-state gate failure (docs/deepseek/BATCH-2026-10-14.md section 13a) blocks every B8 timing until the gate passes.

## 4. What the box must answer: pre-registered predictions

Written before any run. `batch_mechanism.evaluate` applies them as written.

- **P1, slot serialization (trace_x2 and trace_x4, three traced invocations each):**
  - Predicted: B4/B2 summed slot body in [1.94, 2.06]; busy fraction within 0.02; the share of slot time that
    is blocked behind another request's task rises by at least 0.05.
  - FAILS means the serialization reading is wrong and section 3's B2 -> B4 mechanism is withdrawn.
- **P2, traffic (rocprofv3 counters):**
  - Predicted: FETCH_SIZE per dispatch B4/B2 in [1.9, 2.1], meaning replicated reads with no cross-request L2/MALL
    reuse; DRAM read requests per dispatch-microsecond within 10%, meaning the HBM demand rate is unchanged.
  - A FETCH ratio below 1.8 means the co-queued requests do reuse weights in cache. The unchanged bodies then mean
    the tiles are latency- or issue-bound rather than byte-bound.
- **P3, queue versus co-residency (iso_x4iso, in-situ layer-5 copies, 8 repetitions):** three spans per phase
  (gate_up, qkv, attn):
  - A: r0 alone;
  - B: r0 beside r1 (the CU's other slot);
  - C: r0 with r2 queued on the same slot.

  Predicted: C/A in [1.85, 2.15] for every phase. The deciding ratio is gate_up's B/A:
  - **< 1.6, RESIDENCY-CAP:** spare memory/issue capacity exists beyond two slots per CU. Next: a 128-VGPR recipe
    allowing 4 workgroups per CU, or column batching.
  - **>= 1.85, SATURATED:** gate_up runs at its rate with two slots. Only fewer bytes help: column-batched,
    weights-once tiles (docs/deepseek/BATCH-2026-10-14.md stage B2).
  - **In between:** reported, with no branch taken.
  - Prior expectation (2026-09-27, older x2 build, cross-session): gate_up+gate_up x1.55-1.58 and qkv+qkv x1.44,
    which points to RESIDENCY-CAP. The box run validates this on the current recipe.
- **B8:** traced only if `bgate_x8` passes. Otherwise trace_x8.json and counters_x8.json record NOT-RUN and the reason.

Counters: rocprofv3 crashed with SIGSEGV on the 2026-10-15 box after writing its CSV
(`examples/gpu-live/results/2026-10-15/b4/captures.log`). Each counter group now runs as its own invocation and needs
exactly 64 megakernel dispatches. A crashed group is INCOMPLETE; nothing is salvaged from a crashed run. P1 and P3
do not depend on the profiler.

## 5. The session

`examples/sessions/ds_batch_mechanism.py`, one MI300X, GPU 0. The single command is in its docstring.

- **Job order:**
  1. CPU: environments, model, packed weights.
  2. GPU: eight producer-bound reference tuples, built in parallel with the four CPU builds (x2, x4, x8, x4iso).
  3. Gates: x2, then x4.
  4. iso_x4iso, the deciding microbenchmark.
  5. Traces: x2, then x4.
  6. B8: the x8 gate, then the x8 trace.
  7. Counters, last.
  8. `mechanism_report.json`.
- **Budget:** 60-75 min with fresh prep [E]; hard cap 7200 s; about $2.0-2.5 at $1.99/h [E].
- **Not in it:** vLLM, timed pairs, new kernels. The 2026-10-15 rounds already hold the ETX/vLLM comparison;
  this session explains it.

Local evidence:
- `tests/test_dsv2_batch_mechanism.py` (8 tests): the span identity on the retained traces, the queue-blocking
  arithmetic, the isolation-block layout, the registered rules, counter coverage, the x4iso lowering (r2 queued on
  r0's slot) and the session order.
- The gpu_jobs dry run resolves all 22 jobs.
- ROCm 7.2.4 device + host build (`rocm_build_check.sh --arm`, x86 container, the VM-compiler check of CLAUDE.md
  section 4) at this commit: x4iso, x4 and x8 all pass, each 248 VGPR / 6 AGPR / 104 SGPR / **0 scratch** / 2 waves per
  SIMD, COH-CHECK OK (`evidence/deepseek-batch-mechanism-2026-10-18/rocm724-*`). The first x4iso attempt was refused
  by COH-CHECK; it had found the compaction defect described in the experiment log, now fixed. The earlier local x8
  record was 96 B/lane of scratch; the compacted dispatch removes it on 7.2.4. The box compiler (ROCm 7.14) spilled
  x4 by 64 B/lane on 2026-10-14 and must show its own figure; the arm allowances (x4/x4iso 64, x8 192) are unchanged.
