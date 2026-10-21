# Qwen batch baseline, 2026-10-18

Status: measured results from clean serial GPU runs on one MI300X. Every row below
passed its own gates. No cell borrows a number from another recipe or session.

Labels: [M] measured. [E] estimate. [U] absent — no number.

Conditions common to every row: one MI300X, ROCm 7.14.0, Qwen3-30B-A3B bf16,
context 1024, 32 greedy tokens, source commit `380ed02`, host wall time of one
decode step over steps 1024-1054 (31 samples). One runner at a time under
`flock` on `/root/bench.lock`.

## 1. Verdict

All six ETX batches and the four-batch vLLM control completed and passed:

- kept B1/B2/B4/B8: **4.087415 / 5.701229 / 10.287806 / 18.903416 ms**,
  **244.65 / 350.80 / 388.81 / 423.20 tokens/s**.
- plain B4/B8: **11.929137 / 22.243487 ms**, **335.31 / 359.66 tokens/s**.
- vLLM B1/B2/B4/B8 (whole-batch step): **4.984999 / 5.821864 / 6.098564 /
  8.174588 ms**, **200.6 / 343.5 / 655.9 / 978.6 tokens/s**.

ETX beats vLLM on throughput at B1 (**1.22x**) and B2 (**1.02x**), then loses at
B4 (**0.59x**) and B8 (**0.43x**). The crossover is between B2 and B4.

The mechanism is per-request graph replication versus shared-weight batching:
ETX serves N requests by replicating the 97-instance graph per request and
re-reading each request's weights, while vLLM reads the weights once and
column-batches the N requests' activations. Section 4 ties the numbers to the
arithmetic; section 5 gives the three experiments that close the gap.

## 2. Measured rows (ETX)

| B | Recipe | Median host step (ms) | Tokens/s | ms/token | Spread (ms) | Tokens | Imports checked | Build SHA-256 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| 1 | kept | 4.087415 | 244.65 | 4.087415 | 0.034415 | 32/32 | 194/194 | `788958df…219154` |
| 2 | kept | 5.701229 | 350.80 | 2.8506145 | 0.073494 | 64/64 | 388/388 | `8a4acdd4…e2f4b1` |
| 4 | kept | 10.287806 | 388.81 | 2.5719515 | 0.169452 | 128/128 | 776/776 | `0ee656e5…9711a1` |
| 8 | kept | 18.903416 | 423.20 | 2.362927 | 0.202527 | 256/256 | 1552/1552 | `3baebd07…c8abe6` |
| 4 | plain | 11.929137 | 335.31 | 2.982284 | 0.308351 | 128/128 | 776/776 | `4792e8a4…f86674` |
| 8 | plain | 22.243487 | 359.66 | 2.7804359 | 0.266128 | 256/256 | 1552/1552 | `14f7f7b6…732d17` |

Sources: `/root/bench_b*.json` and `/root/bench_plain_b*.json` on the host;
local copies in `/Users/ckl/etx-captures/bench-20261018/`. Full build SHAs are
in the JSON records.

`kept` is the tuned recipe; `plain` is the correctness-qualified recipe. At the
same batch kept is faster: B4 10.288 vs 11.929 ms (14% faster), B8 18.903 vs
22.243 ms (15% faster). The two recipes are different builds; the plain rows are
reported for completeness, not as a controlled pair with kept.

## 3. The vLLM control and the ETX/vLLM comparison

vLLM ran on the same box, one engine measuring every batch in one process
(`--batches 1,2,4,8`, `VLLM_ROCM_USE_AITER=1`, `vllm/vllm-openai-rocm:v0.27.1`).
The `VLLMB per_token_ms` value is the whole-batch decode step (three samples per
batch, median taken). Tokens/s is `batch × 1000 / step`.

| B | vLLM step (ms) | vLLM tokens/s | ETX kept step (ms) | ETX kept tokens/s | ETX/vLLM tokens/s | ETX/vLLM step |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.984999 | 200.6 | 4.087415 | 244.65 | 1.22 | 0.82 |
| 2 | 5.821864 | 343.5 | 5.701229 | 350.80 | 1.02 | 0.98 |
| 4 | 6.098564 | 655.9 | 10.287806 | 388.81 | 0.59 | 1.69 |
| 8 | 8.174588 | 978.6 | 18.903416 | 423.20 | 0.43 | 2.31 |

ETX wins B1 (18% faster step) and B2 (2% faster), then loses B4 (1.69x slower
step) and B8 (2.31x slower step). The crossover is between B2 and B4.

## 4. The mechanism: per-request graph replication vs shared-weight batching

### 4.1 The two designs

ETX serves N requests by replication (`bench.py:58`: `ETX_VL_REQUESTS=N`,
`ETX_VL_BATCH=0`, `ETX_VL_NB=1`, `ETX_VL_CHAIN=0`). Each request carries its own
copy of the imported-instance graph: 97 instances per request per position, so
194/388/776/1552 checked instances at B1/B2/B4/B8, and 120,724 / 241,448 /
482,896 tasks at B2/B4/B8 (exact 2.000x per doubling). Each request re-reads its
own dense weights and its own K=8 experts.

vLLM serves the batch by sharing weights: the dense weights are read once per
step and the N requests' activation vectors are column-batched into the same
GEMMs; the MoE reads only the union of the N requests' routed experts.

### 4.2 The marginal cost of one more request

| Transition | ETX step ratio | ETX marginal (ms/request) | vLLM step ratio | vLLM marginal (ms/request) |
|---|---:|---:|---:|---:|
| B1 → B2 | 1.3948 | 1.614 | 1.1679 | 0.837 |
| B2 → B4 | 1.8045 | 2.293 | 1.0475 | 0.138 |
| B4 → B8 | 1.8375 | 2.154 | 1.3404 | 0.519 |

ETX pays a roughly constant **~2.2 ms per additional request** after B2. vLLM's
marginal collapses toward **~0.14–0.52 ms**. The gap is that constant per-request
tax times the request count: at B8, ETX is 18.903 vs 8.175 ms — a 10.73 ms gap,
of which ~7 requests x ~2 ms is the replication tax.

### 4.3 The arithmetic (bandwidth floor)

Per-token weight bytes (from `examples/vllm_llm/batch_pp.py:24-36` and
`examples/llm/configs/qwen3-30b-a3b.json`):

- dense (qkv + o_proj + router) = **2.459 GB/token** (L x dense 1.837 GB +
  lm_head 0.622 GB).
- experts (K=8 x gate/up/down) = **3.624 GB/token**.

Full replication means request r reads its own 2.459 + 3.624 = **6.08 GB** of
weights. At the measured streaming bandwidth 3.56 TB/s (`hw_bench lf`,
`batch_pp.py:9`), that is **1.71 ms of bandwidth floor per request** —
consistent with ETX's measured ~2.2 ms/request marginal (1.71 ms bandwidth +
~0.5 ms compute and overhead).

vLLM reads the dense weights once (2.459 GB, not N x), and reads the union of
routed experts. At B8 the union is E x (1 − (1 − K/E)^B) =
128 x (1 − 0.9375^8) = **51.6 experts**, against 64 replicated reads (replicated
expert bytes 29.0 GB vs union 23.4 GB). At B4 the union is 29.1 against 32.

This is why ETX's step grows ~1.8x per doubling after B2 while vLLM's grows
1.05–1.34x: ETX re-reads every weight per request, vLLM reuses them across the
batch. The B1/B2 win is ETX's fusion advantage — at batch 1 the megakernel's
lower kernel-boundary overhead wins (4.087 vs 4.985 ms); as the batch grows the
~2.2 ms/request replication tax swamps that one-time advantage.

### 4.4 What the arithmetic predicts

- Share the dense read: save (N−1) x 2.459 GB → **B4 −2.07 ms, B8 −4.84 ms** of
  bandwidth floor.
- Share the experts to the union: save (N − union/K) x 3.624 GB → **B4 −0.37 ms,
  B8 −1.58 ms**.
- Fix the attention prologue: the imported `attn_reduce` is measured +2.47 ms at
  B8 (single-launch sums, `EXPERIMENT-LOG.md:87`).

These are the three experiments in section 5.

## 5. How to close the gap at B4 and B8 — three experiments

Each experiment names the code path it changes, the expected delta [E], and the
metric that would falsify it. Deltas are bandwidth-floor estimates: they assume
the weight re-read sits on the critical path and is not already L2-resident,
which is exactly what the falsifying metric tests. Each must also re-pass the
replication baseline's gates (byte-identical imports at positions 0 and 1040,
32/32 tokens per request) before any timing is read.

### E1 — Column-batch the dense GEMVs (`ETX_VL_BATCH=1` on qkv, o_proj, router)

- **Code path:** `examples/vllm_llm/model.py:199-202` (emit each dense GEMV once
  with N activation columns when `ETX_VL_BATCH=1`) and `model.py:685` (request 0
  owns the shared group, the others skip), with the runtime in
  `examples/vllm_llm/host.hip:192-198,284-315` (`nb` + per-column strides). The
  mechanism is already written. `bench.py` now exposes it as `--recipe batched`
  (`bench.py:37,64,68-72`: `ETX_VL_BATCH='1'`, `ETX_VL_NB=batch`,
  `ETX_VL_ROUTER_NARROW='0'`; batch 1 is refused). The experiment is a recipe
  selection plus correctness gates, not new kernels.
- **Expected delta [E]:** dense read falls from Nx to 1x, saving
  (N−1) x 2.459 GB / 3.56 TB/s → **B4 −2.07 ms** (10.29 → ~8.2),
  **B8 −4.84 ms** (18.90 → ~14.1). Tokens/s at B8 ~565 vs vLLM 979: closes 4.84
  of the 10.73 ms gap.
- **Falsifies:** a rocprof capture with `FETCH_SIZE` shows the dense bytes per
  step did NOT fall ~Nx (the weights were already L2-resident, so the cost is
  compute, not bandwidth), OR the paired host step improves <2% in any pair.

**Local status, 2026-10-18 (offline, no GPU):**

| B | Replicated dense (GB) | Batched dense (GB) | Delta (GB) | Floor at 3.56 TB/s |
|---|---:|---:|---:|---:|
| 4 | 9.8377 | 2.4594 | 7.3783 | 2.073 ms |
| 8 | 19.6755 | 2.4594 | 17.2160 | 4.836 ms |

The replicated plan emits each dense role once per request (580 dense
instances at B4, 1160 at B8); the batched plan emits each once for all requests
(145 instances; every dense role is `nb=4` at B4 and `nb=8` at B8). Non-dense
roles stay replicated at `nb=1`. Reproduce with
`python -m examples.vllm_llm.dense_bytes --replicated vl_plan_bN_rep.txt --batched vl_plan_bN_bat.txt`.

The focused batch/byte tests pass (61) and the full suite passes (2368 passed,
24 skipped, 574.88 s).

Compile gate: `desktop` (16-core x86_64, no GPU), image
`docker.m.daocloud.io/rocm/dev-ubuntu-22.04:6.4.1`, `hipcc
--offload-arch=gfx942`. The B8 batched plan's `megakernel_d0.hip` device
codegen exits 0 (3,097,680-byte assembly); `host.hip` device codegen and host
object compile both exit 0 (690,964-byte object). No link, no GPU, no timing;
the ROCm version differs from the MI300X box, so this is a compile gate only.

The E1 GPU wall-clock delta remains [U]. A positive result requires the batched
B4/B8 run to pass the kept gates — byte-identical imports at positions 0 and
1040, per-request coverage, 32/32 tokens per request — and to record
`ETX_VL_BATCH=1` in the build identity; the new risk is a wrong column or
stride, which `host.hip:198` refuses when the plan's `nb` exceeds `ETX_VL_NB`.
E2 is a separate change and is not started here.

### E2 — Column-batch the MoE to the union of the batch's routed experts

- **Code path:** `examples/vllm_llm/host.hip:495` (the guard that a batched MoE
  needs the standalone top-k instead of the per-task fold) plus the MoE roles
  `moe_up`/`moe_down`/`moe_act` in `examples/vllm_llm/plan_exports.py:52` and
  the `dsl_topk` tile. Read each routed expert's weights once and apply to every
  column that routed to it.
- **Expected delta [E]:** expert reads fall from B x K to the union
  E x (1 − (1 − K/E)^B) — at B4 32 → 29.1 (−9%), at B8 64 → 51.6 (−19%).
  Bandwidth saving → **B4 −0.37 ms, B8 −1.58 ms**.
- **Falsifies:** expert bytes per step do not approach the union bound (routing
  is not uniform-independent, or requests rarely share an expert), OR the step
  improves <2%.

### E3 — Fix the imported `attn_reduce` scaling (attention prologue batching)

- **Code path:** the `dsl_attnprep` tile (`examples/vllm_llm/plan_exports.py:73-74`)
  and the `attn_reduce` import adapter in `etx/importer/`. This is the single
  measured imported-kernel loss: at B8 position 0 the imported `attn_reduce` is
  +2473.3 of the +2483.7 µs total (2455.9 → 4929.2 µs, +101%), per-launch median
  6.2 → 12.7 µs (`EXPERIMENT-LOG.md:87`). It launches the N requests'
  attention-reduce work as N separate kernels instead of one batched launch.
- **Expected delta [E]:** recover most of the +2.47 ms at B8 → **B8 −1.5 to
  −2.0 ms**, and the proportional B4 share (+717 µs at position 0) → **B4 −0.5 to
  −0.7 ms**.
- **Falsifies:** the imported `attn_reduce` per-launch median at B8 does not fall
  back from 12.7 µs toward the B1 6.2 µs baseline, OR the step improves <2%.

Additivity: the three targets are independent weight/activation streams, so the
bandwidth savings are roughly additive in the floor, but the actual step
improvement depends on what stays on the critical path; each experiment is
measured on its own and must pass the 2% paired-gain bar. After E1+E2 the
replication marginal should fall from ~2.2 ms/request toward vLLM's ~0.5
ms/request, putting ETX near or past vLLM at B4/B8. This is [E], not a promise.

## 6. Provenance

- kept B1–B8: `/root/bench_b*.json` on the host; copies in
  `/Users/ckl/etx-captures/bench-20261018/bench_b*.json`.
- plain B4/B8: `/root/bench_plain_b*.json` on the host.
- vLLM: `/root/vllm-bench.log` (local copy
  `/Users/ckl/.hermes/cache/scratch/vllm-bench.log`), `VLLMB per_token_ms` lines
  at batches 1/2/4/8.
- One runner at a time under `flock` on `/root/bench.lock`; the ETX runner
  waited on `pgrep -f examples/llm/vllm_bench.py` before starting, so the GPU
  was exclusive for each side.

## 7. Limits

- vLLM is one configuration (`VLLM_ROCM_USE_AITER=1`, v0.27.1), three samples
  per batch; ETX is a 31-sample median. The two sides are not interleaved paired
  rounds, so this is a baseline comparison, not a fusion-style claim.
- No vLLM sweep (backend, dispatch policy, block size) was run; vLLM's number is
  one point, not a tuned best.
- The three experiment deltas are [E] bandwidth-floor estimates; each carries a
  falsifying metric that is measured before the delta is believed.
- No task trace, hardware counter or profiler capture exists for the Qwen batch
  decode path; the mechanism's [U] list is unchanged in
  `docs/qwen/QWEN-BATCH-MECHANISM-SCAFFOLD-2026-10-18.md`.
- The per-build device median (per-token latency on device) is not recorded for
  the new kept B4/B8 and plain B8 rows; only host wall is reported here.
- kept and plain are different builds; no same-recipe kept-to-plain comparison is
  claimed.
