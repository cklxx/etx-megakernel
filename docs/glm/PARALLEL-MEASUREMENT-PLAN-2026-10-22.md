# GLM parallel variants: measurement plan — 2026-10-22

**Plan only; no GPU session is authorized. All variant tests, native builds,
numerics and performance remain unqualified.** The CPU-only options emitter has
produced 30 compute options and 120 batch-bound cases, not runnable builds or
performance certificates. See [the variant runbook](PARALLEL-VARIANTS-2026-10-22.md).
Do not contact the stopped box, run HF, provision or launch this plan now.

## Case identity and comparison contract

Use `results/o55-parallel-batches-20261022/manifest.json`: worlds 2/4/8, batches
1/2/4/8, ten independently named compute arms per world. Batch is a compile
binding, not an option. A case's options+batch hash is only a planning key;
record separate staged-model, exact build/source/compiler, GPU/fabric and
runtime identities for qualification.

Freeze the GPU type, world, batch, weights/precision, independent prompts,
context/decode positions, 32 autoregressive output steps, sampling rule (greedy),
state initialization, warmup count, clock policy and environment before pairing.
Do not shorten context or change precision to make a candidate fit while retaining
its old comparator. Every batch request needs its own prompt, state and feedback.

Every surviving case has **two mandatory baseline families**:

1. **Same-kernel one-by-one HIP-graph replay**, with the same partition/layout and
   numerical order, to isolate megakernel fusion from kernel quality.
2. **Strongest legal real-system reference** with HF-correct greedy decoding,
   same world/batch/context and measurement basis. Search/configuration proof and
   version belong in the record; an untested configuration is not the strongest.

Also compare every compute variant with freshly measured **same-world,
same-batch PP**, balanced against its unbalanced parent, and head TP against its
serial-attention parent. EP/full TP are not compared with an old-session absolute
latency. A missing mandatory comparator means the corresponding claim is
**NOT-COMPLETE**, not a win.

## Order and stop gates

### 0. Before spending or contacting a future box

Obtain separate owner authorization, balance and a dollar/hour budget. Prepare
and commit one bounded job script, complete authorized local preflight, and
freeze exact case IDs and thresholds below. Current owner rules prohibit local
test runs; lifting that restriction and authorizing a future GPU session
are separate decisions. No script or job is launched by this document.

Use existing measurements rather than regenerating them. The prior B1 MI325X
PP2/4/8 numbers and optimistic FFN/EP arithmetic are context, **not predictions**
for a new build/session. The old 9.598 ms ideal8 is void for this implementation:
MLA low-rank/indexer p1 and IQ/indexer expansion are replicated.

### 1. Capacity, compiler resources and correctness first

- Native gfx942 compile all rank entrypoints and collect VGPR, spills, LDS,
  occupancy and loader ABI/source identity. A planner budget is not a resource
  report. A failed/incomplete compile is **DROP/BLOCKED**, never a timing arm.
- Establish byte identity of every shard, scale and replica against the staged
  originals; expert `id % world`, shared/root ownership and reduction ordering
  must match the declared mode. Reject any out-of-bounds, unowned or peer-weight
  access. Count peak per-rank resident and transient bytes, allocator/runtime
  reserves, not just weight payload. Require fit on **every** rank without OOM.
- At B>1, prove per-request scratch/state and events are disjoint; poison unused
  regions and exercise nonzero request IDs and empty EP ranks. Per-WG LDS does
  not scale with B, but total work and request scratch/state do. Each attention
  rank still allocates full-global state capacity per request.
- Preserve **all 32 full logits vectors for every request**, shape
  **`(32, B, VOCAB)`**, plus layer/state diagnostics required to localize failures.
  At FP32 logging precision this is `32*B*VOCAB*4` bytes per arm, not an argmax
  digest or a few selected vocabulary columns. Keep raw files and checksums.
- Require **32/32 exact HF greedy tokens for each request**, with no teacher
  forcing, early exit or speculative decoding. One request failing is a case
  failure; HF work itself requires future authorization and is not launched now.
- EP-only and PP balance target preserved per-row arithmetic/ordered combination:
  require the preserved outputs/state to match their qualified serial parent
  byte-for-byte. TP changes FP summation: do not demand or claim bit-identical
  serial logits. Compare every operator/state/logit with the independent oracle
  using a **frozen, explicitly numeric tolerance record**, and require HF32/32
  independently. If no approved tolerance record exists, the numerical gate is
  **BLOCKED**; do not choose or relax tolerances after seeing candidate outputs.
- Retain full-logit diagnostics for HF and all candidates; HF token equality alone
  is not a per-operator or source/build certificate. Imported launches additionally
  need the repository's same-build byte-identity checks where applicable.

Zero tolerated malformed shards, cross-request aliases, failed checks or incomplete
32-step requests. A correctness/capacity failure stops that arm before full timing.

### 2. Measure the deciding unknown, not the whole matrix first

For each world/batch selected for a future session, take its qualified PP/kernel
trace and count bytes, achievable bandwidth floor and gaps. Measure achievable
bandwidth on the session's MI325X; do not reuse the MI300X 4.3 TB/s figure. Separate root work,
local compute, dispatch, return/reduction and wait/skew. Gaps bound fusion's
recoverable time; replicated/root work bounds parallel scaling. Record the
model-derived ceiling and numeric time budget **before** the microbenchmark.

Run bounded, correctness-checked phase/transport microbenchmarks before full-model
campaigns. Each result needs a native resource report and real traffic/overlap,
not simply `serial_time/world` arithmetic.

| Arm | Deciding measurement, at each selected B/world |
|---|---|
| PP reference | Actual request/stage overlap and feedback serialization; root/head idle and critical path. B>1 only creates overlap opportunity. |
| Balance-only | Up duration/tail occupancy for 512→576 unsharded sparse tiles and local equivalents; confirm every non-up phase/head cap unchanged. |
| FFN TP | Local gate/up/down scaling plus ordered FP32 partial return/reduction versus serial FFN; dense/sparse and root bottlenecks separately. |
| EP | Expert placement, selected bytes, per-rank expert counts and straggler time including root shared/dense work; measure routing distribution rather than assume uniform utilization. |
| Full TP | KDA head/state work, MLA replicated p1/IQ versus sharded QB/KVB/heads/O, dispatch and output reduction; memory remains full-stride per rank. |
| Head-TP+EP | Attention gain versus EP parent, expert skew plus shared attention/root bottlenecks; do not add isolated gains without measuring their interaction. |
| Balanced compute arms | Paired up-only delta versus the exact unbalanced compute parent; no balancing-by-default substitution. |

For an illustrative uniform independent eight-selection EP8 model,
`8*(1-(7/8)^8) ≈ 5.2` of eight ranks are busy: **an estimate, not a measured
utilization or guarantee**. Record real expert collisions and rank skew.

Per-request GEMVs do not automatically become GEMM/weight reuse at B2/4/8.
Measure actual bytes and phase scaling. If measured communication/root/skew cost
already consumes all predicted compute savings, **DROP the speed candidate**;
retain the failed microbenchmark and reason, without paying for full timing.

### 3. Bounded full-model paired timing for surviving cases

Do not blindly schedule 120 full runs. Prioritize a correctness-qualified PP
reference per selected world/B, then the deciding microbenchmarks, then survivors
and their independently balanced controls within the authorized budget. Mark
unselected cases **NOT-RUN**; a promising B/world does not qualify the others.

For each surviving case/comparator pair, collect **five interleaved A/B pairs in
one session**, same GPUs pinned for both arms, counterbalance order across pairs,
apply identical warmup/state-reset policy, and keep all raw outputs. Require
complete correctness for every timed arm. A second-session repeat is needed for
a replication claim; one session remains one session even with five pairs.

Use line-buffered full logs, progress counters, and per-step deadlines derived
from expected duration plus margin; no blind long timeout. Probe a process silent
past its expected duration per repository policy. A timeout, missing logit file,
failed check or fewer than five complete pairs yields **NOT-COMPLETE / no result**,
not a median from the surviving subset. No automatic rerun or arm substitution
without a logged reason and budget check.

Report:

- **Host-wall batch step latency**, plus full 32-step wall time and actual
  per-request latency/completion times if pipeline overlap makes them differ.
- **Aggregate output throughput**: `32*B / full_32_step_seconds` tok/s. Never call
  `batch_latency/B` observed request latency; it is a throughput-derived number.
- Secondary device span, phase times, traffic, measured overlap and rank skew,
  clearly separated from wall time. Include GPU, world, B, context, compiler,
  exact build, correctness/logit hashes, every pair, median and spread.

## Frozen keep/drop rules

These are **proposed experimental decision bars**, not measured results or
predicted speedups. Register them with the future session script before any GPU
work; do not lower them after observing results.

For each pair define `gain_i = 100*(reference_wall_i - candidate_wall_i) /
reference_wall_i`; use the same wall basis in all five pairs. Let `m` be median
gain and `s = max(gain_i)-min(gain_i)` in percentage points.

| Decision | Exact gate |
|---|---|
| Correctness/capacity | Zero failures, every required native identity/shard/state check complete, all 32 logits/request retained, HF **32/32 each request**. Any failure: DROP; missing evidence: BLOCKED. |
| Microbenchmark no-go | Nonpositive modeled savings after measured communication/root/skew: stop the speed campaign for that case. No full-run prediction from ideal division alone. |
| KEEP as a measured speed improvement against a named comparator | All five `gain_i > 0`, `m > 2%`, **and** `m > 2*s`, with all correctness gates passed. |
| Inconclusive | Correct complete pairs exist but the keep bar fails: report NO-ESTABLISHED-GAIN, not a win. Any nonpositive pair fails the keep rule. |
| Default change | Never automatic. Even a KEEP leaves measured PP2/default options unchanged pending owner approval. |
| Fusion claim | Independently pass the same five-pair rule against same-kernel replay; a win over PP or a real system does not isolate fusion. |
| Real-system lead | Independently pass against the strongest tested legal same-world/B system; name untested configurations and exclusions. |
| Claimed 2× | Require a separately stated, measured **≥2.0× host-wall speedup** against the named same-world/B comparator with all gates; EP/FFN-only arithmetic is not such a claim. |

Retain negative/inconclusive measurements, correctness failures, exclusions and
costs. Record actual dollars/time; delete any future VM immediately when its
approved work ends and verify no VMs remain. None of that authorizes access now.
