# Two-week results, 2026-10-18

Measured numbers only. [M] measured, [E] estimate, [U] no measurement exists.
Conditions: one MI300X, ROCm 7.14.0, bf16, context 1024, 32 greedy tokens,
unless a section says otherwise.

## 1. Qwen3-30B-A3B multi-batch baseline

**Verdict:** ETX beats vLLM at batch 1 and 2, then loses at batch 4 and 8 because it replicates each request's full graph and re-reads the weights, while vLLM shares them.

Host wall time of one decode step, median (tokens/s in parentheses):

| Batch | ETX kept (ms) | ETX kept tok/s | ETX plain (ms) | vLLM (ms) | vLLM tok/s | ETX/vLLM tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.087415 | 244.65 | [U] | 4.984999 | 200.6 | 1.22 |
| 2 | 5.701229 | 350.80 | [U] | 5.821864 | 343.5 | 1.02 |
| 4 | 10.287806 | 388.81 | 11.929137 | 6.098564 | 655.9 | 0.59 |
| 8 | 18.903416 | 423.20 | 22.243487 | 8.174588 | 978.6 | 0.43 |

Every ETX row passed its own gates: 32/32 tokens per request, 194/388/776/1552
imported instances byte-identical (97 per request per position), source commit
`380ed02`, one runner at a time under `flock`. vLLM is one engine,
`VLLM_ROCM_USE_AITER=1`, v0.27.1, three samples per batch.

Mechanism (arithmetic in `docs/QWEN-BATCH-BASELINE-2026-10-18.md`): ETX serves N
requests by replication — 97 imported instances per request, tasks 120,724 /
241,448 / 482,896 at B2/B4/B8 (exact 2.000x per doubling) — and re-reads each
request's dense weights (2.459 GB) and K=8 experts (3.624 GB), so its marginal
is a roughly constant ~2.2 ms per added request (measured 1.614 / 2.293 / 2.154
ms). vLLM reads the weights once and column-batches, so its marginal collapses
(0.837 / 0.138 / 0.519 ms). The crossover is between B2 and B4.

Next experiments [E], not run: column-batch the dense GEMVs (B4 −2.07, B8 −4.84
ms), column-batch the MoE to the union of routed experts (B4 −0.37, B8 −1.58
ms), fix the imported `attn_reduce` scaling (B8 −1.5 to −2.0 ms).

## 2. DeepSeek-V2-Lite batch scaling and attribution

**Verdict:** B2 and B4 throughput has topped out because body work scales exactly with the replicated task graph and no weight is amortized across requests; B8 has no trace.

From the historical trace (`trace_report.json`, arm `alld4r0l1kvsmw`, round 1):

| Batch | Step (us) | Tokens/s | Body (us) | Dependency gap (us) | Queue gap (us) | Tasks |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3254 | 307 | 2716 (83%) | 378.7 | 0.0 | 53022 |
| 2 | 4576 | 437 | 3735 (82%) | 542.2 | 69.7 | 109892 |
| 4 | 9027 | 443 | 7360 (82%) | 955.1 | 381.6 | 219784 |
| 8 | [U] | [U] | [U] | [U] | [U] | [U] |

Throughput flattens at B2: B1→B2 +42% (307→437), B2→B4 +1.4% (437→443).

B2→B4 step growth (+4451 us) decomposes as: body +3625 us (81%), dependency gap
+413 us (9%, from `router` 78→184, `oproj` 82→181, `qabs` 80→172), queue gap
+312 us (7%, all from `qkv` 0→225 and `attn` 70→148 us). Tasks double exactly
(109,892→219,784) and body almost exactly (3735→7360, 1.97x), so each request
replicates the whole graph with no cross-request weight amortization. `qabs`
appears only at B≥2 and adds a 172 us serial dependency at B4.

B8: no completed trace [U]. The batch session on 2026-10-18 measured B1 and B2
(313 and 448 tok/s against vLLM 245 and 423, `breport.log`) and reported B4 and
B8 as NOT COMPLETE; the machine then went read-only twice and was destroyed.

Not in the record [U]: HBM/L2 traffic, hardware occupancy counters, dispatch
counts, and a task-level ordered dependency chain.

## 3. GLM-5.3-Flash code

**Verdict:** the code is fixture-only verified and has never run real weights; the full model is blocked on a second GPU and 305 GiB of staging.

- Fixture correctness: **480/480** exact greedy tokens on a synthetic 4-layer,
  small-dimension checkpoint [M]. This is the correctness floor only.
- Real weights: never ran [U].
- 4-layer slice, ~29 GB, `--layers 4`, ~30 min of GPU: prepared, not run [U].
- Full 45 layers: 305.71 GiB payload, 163.96 GiB/rank, needs 2 GPUs; the single
  box could not stage it (610 GB peak against 501 GB free) [U].

## Unmeasured, common to all three goals

No task trace, hardware counter or profiler capture exists for the Qwen batch
decode path; the DeepSeek B8 trace and every GLM real-weight run are [U]. The
three Qwen experiments in section 1 are estimates, not measurements.

Sources: `docs/QWEN-BATCH-BASELINE-2026-10-18.md`,
`docs/DS-BATCH-ATTRIBUTION-2026-10-18.md`, `docs/EXPERIMENTS-2026-10-18.md`,
`examples/vllm_llm/results/EXPERIMENT-LOG.md`; raw records `bench_b*.json`,
`bench_plain_b*.json`, `/root/vllm-bench.log`, `trace_report.json`.
