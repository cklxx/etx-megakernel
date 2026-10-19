# Megakernel on AMD: research plan, 2026-10-18

Status: research plan, no GPU available (all machines destroyed). No measurement
in this document is new; every number is from our own records [M] or a labelled
estimate [E]. External claims are cited inline. Unmeasured claims are [U].

Goal: make the ETX megakernel the fastest thing on AMD for single-request and
small-batch decode, measured against vLLM+AITER on the same MI300X, and later
against vLLM on H100/B200 at matched batch and context.

The question this answers: what has to be true for a megakernel to beat a
hand-tuned batched engine on CDNA3, and what must be built in the next two weeks
to prove it.

## 1. The ceiling argument

### 1.1 What one decode step must move

Per-token weight bytes (bf16), from our configs and byte ledgers [M]:

- Qwen3-30B-A3B: dense 2.459 GB (qkv + o_proj + router) + experts 3.624 GB
  (K=8) = **6.08 GB/token** (`examples/vllm_llm/batch_pp.py:24-36`).
- DeepSeek-V2-Lite: **4.934 GB/token** (`examples/dsv2lite/batch_floor.py`).

At batch B the bytes are either **replicated** (B x the batch-1 bytes — what ETX
does today) or **batched** (dense/shared weights once, routed experts once per
distinct expert). DeepSeek floors from `batch_floor.py` [E, floor at 4.3 TB/s]:

| B | replicated GB | batched GB | batched floor ms |
|---|---:|---:|---:|
| 1 | 4.934 | 4.934 | 1.148 |
| 4 | 19.738 | 11.702 | 2.721 |
| 8 | 39.476 | 18.15 | 4.221 |

### 1.2 The hardware

MI300X (gfx942, CDNA3): 304 compute units (8 XCDs x 38), wavefront size 64,
64 KB LDS per CU, 512 KB VGPR file per CU (65,536 32-bit registers), 256 MB L3
(Infinity Cache), 32 MB L2 (4 MB per XCD), 192 GB HBM3 at 5.3 TB/s theoretical
([AMD CDNA3 white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf),
[AMD MI300X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf),
[ROCm GPU arch specs](https://rocm.docs.amd.com/en/docs-6.4.2/reference/gpu-arch-specs.html)).
128-byte cache lines, 16,384-entry TLB per XCD, measured L3 bandwidth ~11.9 TB/s
([Chips and Cheese, "Testing AMD's Giant MI300X"](https://chipsandcheese.com/p/testing-amds-giant-mi300x)).

Bandwidth tiers that matter, from our own measurements: 5.3 TB/s theoretical,
~4.3 TB/s achievable (the repo's floor constant, `CLAUDE.md`), 3.56 TB/s
measured streaming and 3.93 TB/s max (`hw_bench lf`, `batch_pp.py:9`), and a
GEMV in isolation reaches 93% of achievable (~4.0 TB/s)
(`docs/ETX_Technical_Design.md:247`).

### 1.3 Why batch 1 is latency-bound, not bandwidth-bound

At batch 1 the step is **2.8-3.5x the bandwidth floor**, and that is the whole
opening for a megakernel:

| Model | B1 floor @4.3 TB/s | ETX B1 measured | vLLM B1 measured | vLLM / floor |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 1.41 ms | 4.087 ms | 4.985 ms | 3.53x |
| DeepSeek-V2-Lite | 1.148 ms | 3.223 ms | 4.098 ms | 3.57x |

Sources: `docs/TWO-WEEK-RESULTS-2026-10-18.md`, `examples/gpu-live/results/2026-10-16-multibatch/result.md`.
The GEMV tile alone hits 93% of achievable bandwidth, so the tiles are not the
limit; the gap is the **serial phase chain** (router -> top-k -> gate_up ->
down -> ...), where one request does not issue enough concurrent loads to
saturate HBM. The megakernel's only route to the floor is to overlap the
latency-bound phases with the next phase's bandwidth-bound weight streaming,
collapsing the step from "sum of phase latencies" toward "bytes / BW + a few
uncovered hops" — the "7 x (bytes/BW + fixed) -> 3 x (bytes/BW) + covered
latency" transform already written up in `docs/ETX_Technical_Design.md:258`.

### 1.4 The ceiling, in ms and tokens/s

- **Qwen B1:** floor 1.41 ms; with a realistic uncovered-latency term of
  ~0.2-0.5 ms [E], the ceiling is **~1.6-1.9 ms**, i.e. **~530-620 tok/s**,
  against ETX's 4.087 ms (244 tok/s) and vLLM's 4.985 ms (200 tok/s). That is
  **2.1-2.5x ETX and 2.6-3.1x vLLM**.
- **DeepSeek B1:** floor 1.148 ms; ceiling **~1.3-1.5 ms** (~670-770 tok/s),
  against ETX 3.223 ms (310 tok/s) and vLLM/AITER 4.098 ms (244 tok/s) —
  **2.1-2.5x ETX and 2.7-3.1x vLLM**.
- **Qwen B8 batched:** floor 25.83 GB / 4.3 TB/s = **6.0 ms**. vLLM measures
  8.175 ms — already **1.36x the floor**. ETX replicated is 18.90 ms.

The last line is the decisive constraint: at B8 the bandwidth term dominates and
vLLM is already near the floor, so **no engine can be more than ~1.36x faster
than vLLM at B8**. The megakernel's real headroom is at B1-B2, where vLLM is
3.5x off the floor.

### 1.5 The NVIDIA-class comparison (later)

The cross-vendor bar is per-token wall time at matched batch/context. H100 SXM
has 3.35 TB/s ([Chips and Cheese](https://chipsandcheese.com/p/testing-amds-giant-mi300x)),
H200 4.8 TB/s ([NVIDIA](https://www.nvidia.com/en-us/data-center/h200/)), B200
8 TB/s HBM3e ([NVIDIA B200](https://jarvislabs.ai/ai-faqs/nvidia-b200-specs)).
MI300X's 5.3 TB/s gives it 1.6x H100's and 0.66x B200's bandwidth, so a B1 win
against vLLM-on-H100 is a bandwidth/latency-hiding story, not a raw-bandwidth
story. This is a later goal; the two-week plan below does not reach it [U].

## 2. The five gaps separating us from that ceiling

Each gap has the measured number that proves it and the code path that owns it.

**Gap 1 — Replicated weights (no shared-weight batching).** The dominant B4/B8
gap. Measured [M]: Qwen ETX B4 10.288 vs vLLM 6.099, B8 18.903 vs 8.175
(`bench_b*.json`, `vllm-bench.log`); DeepSeek B4 8.921 vs AITER 6.166
(`examples/gpu-live/results/2026-10-16-multibatch/result.md`). The marginal is a
roughly constant ~2.2 ms per added request, matching 6.08 GB/request at the
measured stream rate. Owned by `examples/vllm_llm/bench.py:58`
(`ETX_VL_BATCH='0'`); the column-batch path exists but is off
(`examples/vllm_llm/model.py:199-202`, `host.hip:192-198`). The replicated-vs-
batched byte split is `examples/dsv2lite/batch_floor.py` (39.5 vs 18.2 GB at B8).

**Gap 2 — MoE body is 44% of the critical path and scales 2.0x with requests.**
Measured [M]: DeepSeek B4 CP body 7.340840 ms of an 8.670160 ms traced span
(85%), with gate_up 2.021840 + down 1.174800 = **3.196640 ms** (44% of CP body);
body sums 1.982145x from B2 to B4 (`examples/gpu-live/results/2026-10-16-multibatch/result.md`).
Owned by the MoE tiles (`fleet_expert_gate_up`/`fleet_expert_down`) reached from
`examples/dsv2lite/graph.py:167-173`.

**Gap 3 — Bandwidth underutilization at B1 (latency-bound phase chain).**
Measured [M]: Qwen B1 4.087 vs 1.41 ms floor (2.89x); DeepSeek B1 3.223 vs
1.148 ms (2.81x), while the GEMV tile in isolation reaches 93% of achievable
bandwidth (`docs/ETX_Technical_Design.md:247`). The tiles are fine; the serial
phase chain is the gap. Owned by the schedule/phases and the event/barrier
mechanism in `examples/vllm_llm/model.py` (and the event-counter stride fix
`ETX_EV_STRIDE`, `docs/ETX_Technical_Design.md`).

**Gap 4 — Dependency + queue gaps are 1.33 ms (15% of B4), and cannot alone
supply the target.** Measured [M]: DeepSeek B4 0.963120 ms dependency gaps +
0.366200 ms queue gaps = 1.329320 ms; the registered B4 target 5.548961 ms
(0.9x vLLM's 6.165512) needs 3.372363 ms from ETX's 8.921324, so gap-only
cleanup is at most 1.33 ms — under half the required reduction
(`examples/gpu-live/results/2026-10-16-multibatch/result.md`). Owned by the
request slot map (`graph.py` `_Req`, `r%2`) and the event relay.

**Gap 5 — Imported-kernel scaling (attention).** Measured [M]: the imported
`attn_reduce` is +2473.3 of +2483.7 µs at B8 position 0 (2455.9 -> 4929.2 µs,
+101%), per-launch median 6.2 -> 12.7 µs (`examples/vllm_llm/results/EXPERIMENT-LOG.md:87`).
Owned by `dsl_attnprep` (`examples/vllm_llm/plan_exports.py:73-74`) and the
`attn_reduce` import adapter in `etx/importer/`.

## 3. Two-week plan

Order matters: Week 1 is offline (no GPU — local compile, correctness gates,
microbenchmark design), Week 2 is one MI300X box. Every experiment names the
delta it targets [E], the falsifier, and passes the repo's correctness gates
before any timing is read.

**Week 1 — offline implementation and qualification.**

1. **E1: column-batch the dense GEMVs** (`ETX_VL_BATCH=1` on qkv/o_proj/router).
   Delta B4 −2.07 ms, B8 −4.84 ms (dense read falls Nx). Falsifier: a rocprof
   `FETCH_SIZE` capture shows dense bytes/step did not fall ~Nx, or the paired
   step improves <2%.
2. **E2: column-batch the MoE to the union of routed experts** (standalone
   top-k). Delta B4 −0.37 ms, B8 −1.58 ms (64 -> 51.6 distinct experts at B8).
   Falsifier: expert bytes/step do not approach the union bound, or step <2%.
3. **E3: fix the imported `attn_reduce` scaling.** Delta B8 −1.5 to −2.0 ms,
   B4 −0.5 to −0.7 ms. Falsifier: the per-launch median does not fall from
   12.7 µs toward the 6.2 µs baseline, or step <2%.

The three targets are independent weight/activation streams; after E1+E2 the
replication marginal should fall from ~2.2 ms/request toward vLLM's ~0.5
ms/request. Full deltas and code paths are in
`docs/QWEN-BATCH-BASELINE-2026-10-18.md:5`.

**E7 — hierarchical sentinel polling for the relay (independent).**

Not sequenced with E1–E6 and a prerequisite for none of them. Replace the
relay's atomic counter barrier with hierarchical sentinel polling: one
producer-owned cache line per participant, one poller per XCD, one top level.

- **Ceiling.** The measured whole-GPU counter barrier is **8.6 µs at 304
  workgroups** (`docs/RESEARCH-sync-hardware.md:25`). DeepSeek B4's dependency
  gap is measured at **963 µs**; at roughly **2 barriers per layer over 27
  layers** that is about **464 µs**. The experiment ceiling is therefore
  **under 1 ms against a 3.37 ms target**, which is why it stays independent
  and does not preempt E1.
- **First step is not code.** Count the barriers per decode step from a trace
  and compute the ceiling from those counts. If the counted ceiling is below
  **0.9 ms, do not write code.**
- **Unverified input [U].** The external sentinel result (7.8 µs → 0.9 µs) is
  not reproduced in this repo and is marked [U] until measured here (recorded
  as "public, not reproduced" in
  `docs/SYNC-TAXONOMY-AUDIT-2026-10-11.md:89`).
- **Falsifier:** the counted ceiling is below 0.9 ms, or a paired step improves
  <2%.

**Week 2 — one MI300X box, correctness-first, then timing.**

4. **E4: measure the shared-weight B4/B8 recipe against vLLM**, interleaved
   pairs on one box. Verifies E1+E2+E3 combined; expected B4 ~7 ms, B8 ~11-12 ms
   [E]. Falsifier: any batch fails byte-identity or 32/32 tokens, or the paired
   step gain is <2%.
5. **E5: the B1 ceiling experiment — prefetch the next phase's weights during
   the latency-bound phases.** This is the actual "fastest single-request
   decode" goal. Target Qwen B1 4.087 -> ~2.0-2.5 ms [E]. Falsifier (measured
   first, before building): `bench/sync/sync_bench.hip` shows the overlap is not
   achievable — the router/top-k output must be known before the expert weights
   are read, so that serial hop cannot be prefetched. Run this microbenchmark
   before writing any code.
6. **E6: schedule/queue cleanup — alternative slot map** (replace `r%2` with a
   placement that removes the 0.366 ms queue gap). Delta B4 −0.3 ms [E].
   Falsifier: the queue gap in a fresh trace does not shrink.

## 4. What has to hold for a 2x-4x advantage over vLLM — and what makes it impossible

**Plausible, at B1-B2:**

- vLLM is 3.5x off the bandwidth floor at batch 1 (Qwen 4.985 vs 1.41 ms), and
  ETX is 2.9x off. The ceiling leaves 2.6-3.1x over vLLM if the phase chain is
  overlapped — and a batched engine cannot easily do that at batch 1, because
  there is no batch to hide latency behind. This is the one place a megakernel
  structurally beats a batched engine.
- The measured replication marginal (~2.2 ms/request) is recoverable in full by
  shared-weight batching at B4/B8, closing that gap.
- On the software side, AMD's stack is the handicap that keeps the bar low:
  upstream FlashAttention had no CDNA3 support and the Triton port was FP16/FP32
  only (no bf16) as of mid-2024 ([Chips and Cheese](https://chipsandcheese.com/p/testing-amds-giant-mi300x));
  vLLM ships seven ROCm attention backends to cover the gap
  ([vLLM blog, 2026-02-27](https://vllm.ai/blog/2026-02-27-rocm-attention-backend)),
  and AITER is AMD's hand-tuned kernel library that vLLM leans on
  ([AMD ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html)).
  A megakernel that does not depend on those kernels has room to win.

**Impossible, at B8:**

- vLLM is already at 1.36x the batched bandwidth floor at B8 (8.175 vs 6.0 ms).
  A 4x advantage at B8 would need a 2.0 ms step, below the 6.0 ms floor — not
  physically possible. At B8 the honest ceiling over vLLM is ~1.3x; the
  megakernel can match, not 4x beat, a hand-tuned batched engine there.

**Impossible, even at B1, if:**

- The router -> top-k -> gate_up chain is a true serial dependency whose latency
  cannot be prefetched (the token's expert choice must be known before those
  weights are read). If the uncovered serial latency is >= ~1 ms, the B1 ceiling
  rises to ~2.4 ms, which is only ~2.0x vLLM — the 4x case collapses, and even
  2x becomes borderline. E5's microbenchmark decides this first.
- The B4/B8 weight re-read is already L2-resident (Gap 1's falsifier fires):
  then shared-weight batching buys nothing and the B4/B8 gap is pure compute,
  which no schedule change closes.

**Bottom line.** The megakernel's defensible claim is single-request and B2
decode, where vLLM is 3.5x off the floor and a fused, prefetched megakernel can
target **2x-3x**. At B4/B8 the gap is closed by shared-weight batching, but the
hardware ceiling there is only ~1.3x over vLLM. A 2x-4x advantage over vLLM is
plausible at B1-B2 and impossible at B8; the two weeks are spent proving the B1
latency-hiding claim (E5) and shipping the shared-weight path (E1-E4).

Sources: `docs/TWO-WEEK-RESULTS-2026-10-18.md`,
`docs/QWEN-BATCH-BASELINE-2026-10-18.md`,
`docs/DS-BATCH-ATTRIBUTION-2026-10-18.md`,
`examples/gpu-live/results/2026-10-16-multibatch/result.md`,
`examples/dsv2lite/batch_floor.py`, `docs/ETX_Technical_Design.md`; external
citations inline above.

## 5. Adversarial review

Recomputed from measured bytes (dense 2.459 GB, experts 3.624 GB per request)
and the measured stream rate 3.56 TB/s (`hw_bench lf`), not the plan's framing.
The shared fatal assumption: the E1/E2 falsifiers need rocprof `FETCH_SIZE`, but
no accepted traffic capture exists on this hardware (`result.md`: "no accepted
hardware traffic counters … partial rocprof CSV remains rejected"), so "bytes
did not fall N×" cannot fire; only the wall-clock 2% bar can.

- **E1 (dense column-batch).** (a) Ceiling (N−1)×2.459 GB ÷ 3.56 TB/s = B4
  −2.07, B8 −4.84 ms; the code path exists (`model.py:199-202`) but is
  correctness-unqualified, so the realized gain is [U]. (b) rocprof falsifier
  cannot fire; the wall-clock <2% bar can. (c) Not subsumed; it gates E2/E4.
- **E2 (MoE union).** (a) Ceiling (N−union/K)×3.624 GB ÷ 3.56 = B4 −0.37, B8
  −1.58 ms, but no shared-weight MoE tile exists and it needs a standalone
  top-k, so the gain against the code as-is is 0. (b) rocprof falsifier cannot
  fire. (c) Effectively subsumed by E1 (both are "share the weights"), and its
  1.58 ms is small next to the 10.73 ms B8 gap.
- **E3 (attn_reduce).** (a) The +2473 µs is a `--check` single-launch sum the
  repo marks "not decode time", so −1.5/−2.0 ms is unbacked; [U]. (b) Falsifier
  (per-launch median back to 6.2 µs) fires via `--check`. (c) Not subsumed, but
  must measure the decode-step cost first.
- **E4 (measure vs vLLM).** (a) 0 — validation, not an optimization. (b) The
  correctness gate fires. (c) Moot unless E1 turns positive.
- **E5 (B1 prefetch).** (a) Needs new code, and the premise is measured-negative:
  forced overlap measured 314/341 tok/s vs 366 uncoupled (`docs/PLAN-v3.md:178`).
  (b) sync_bench falsifier fires. (c) Not subsumed; it is the only path to the
  2–3× B1 claim.
- **E6 (slot map).** (a) Ceiling is the 0.366 ms queue gap — 1/9 of the 3.37 ms
  target; `result.md` already concluded gap-only cleanup cannot reach it.
  (b) Needs a fresh trace. (c) Dead — delete; no replacement (E1/E2 own the
  body-reduction path).

**Run first with one GPU-hour: E1.** Flip `ETX_VL_BATCH=1`, pair kept vs
column-batched at B4/B8, read the wall clock: positive → shared weights are the
cost and E2/E4 follow; negative → the weight re-read is not the cost and effort
moves to E3. Either outcome changes the decision.

**Delete: E6.** Its 0.366 ms ceiling is measured and irrelevant to the 3.37 ms
target; the decision it would inform is already made.

**The estimate the measurements contradict.** The measured marginal is 2.154
ms/request (B4→B8, `docs/QWEN-BATCH-BASELINE-2026-10-18.md`), but E1+E2 remove
only 6.42/7 = 0.917 ms/request of it (~43%). So after E1+E2 (and even adding
E3's unbacked 2.0 ms), B8 lands near 12.5 → 10.5 ms, still 1.28× slower than
vLLM's 8.17 ms. The plan's "shared-weight batching closes the B4/B8 gap"
overstates the recoverable share: most of the marginal is per-request compute
and attention, not the weight re-read.
