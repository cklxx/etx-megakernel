# Batch 2/4/8 against vLLM, 2026-10-14 — box session 20261014T142757Z-03ff3e

Condition: one MI300X VF (192 GB, ROCm 7.14), bf16, context 1024, 32 greedy tokens per request, five interleaved
rounds per configuration, vLLM `vllm/vllm-openai-rocm@bb44b39a` with AITER at the same batch size. Arms per
docs/BATCH-2026-10-14.md section 10: B = 1 is the kept arm `alld4r0l1kvsmw`; B = 2 `x2` and B = 4 `x4` are the kept
recipe plus M1 and K1; B = 8 (`x8`) was built but its gate failed.

Result, throughput and per-request latency, five rounds:

| B | ETX tok/s | vLLM tok/s | throughput | spread | per-request latency | spread |
|---|---|---|---|---|---|---|
| 1 | 274 | 206 | **WIN +32.98%** | 1.50% | WIN +28.80% | 1.47% |
| 2 | 403 | 354 | **WIN +13.95%** | 0.60% | WIN +10.11% | 0.59% |
| 4 | 410 | 567 | **LOSS −27.70%** | 6.18% | LOSS −29.62% | 6.02% |
| 8 | — | — | NOT COMPLETE | — | — | — |

Raw number lines (`runner.log`):

    BREPORT b=1 vs vLLM aiter: throughput WIN median +32.98% spread 1.50% (274 vs 206 tok/s); per-request latency WIN median +28.80% spread 1.47%
    BREPORT b=2 vs vLLM aiter: throughput WIN median +13.95% spread 0.60% (403 vs 354 tok/s); per-request latency WIN median +10.11% spread 0.59%
    BREPORT b=4 vs vLLM aiter: throughput LOSS median -27.70% spread 6.18% (410 vs 567 tok/s); per-request latency LOSS median -29.62% spread 6.02%
    BREPORT b=8 NOT COMPLETE next: rerun B=8 after a local fix (NOT COMPLETE)

Gates (one gated invocation per arm, per-request golden 32/32 tokens and 27/27 layers):

    BGATE arm=x2 b=2 PASS
    BGATE arm=x4 b=4 PASS
    BGATE arm=x8 b=8 NOT COMPLETE: request 5: 3 layers within the gate (need 27)

`brounds.json`, round 1: `order [["etx",1],["etx",2],["etx",4],["vllm","aiter"]]`, `skipped ["x8"]`.

| B | ETX step_ms | worst request interval_ms | vLLM step_ms (VSEL) |
|---|---|---|---|
| 1 | 3.649801 | 3.76888 | 4.849658 |
| 2 | 4.963636 | 5.13440 | 5.620490 |
| 4 | 9.755924 | 10.02248 | 7.613653 |
| 8 | — | — | 9.355833 |

Skipped by dependency: `bgate_x8`, and Qwen's five jobs — `qwen_preflight` failed on the staged tree's own `.venv`
(`ModuleNotFoundError: No module named 'numpy'`), so `qwen_build_kept`, `qwen_build_kept_l1`, both certificates and
`qwen_pairs_kept_l1` never ran. The fix (create `<tree>/.venv` right after every stage, plus `libhipcub-dev` and the
torch headers for the vLLM build) is recorded in the agent skill `etx-agent-loop`, `references/box-prep-pitfalls.md`.

Seal: `VERIFIED: session 20261014T142757Z-03ff3e, 107 files match, 0 bad, 60 left on the box by design`.

Interpretation and the next decision are in docs/BATCH-2026-10-14.md section 11: replication leads at B = 1 and B = 2
and loses at B = 4, so the crossing sits between B = 2 and B = 4 and the column-batched (stage B2) work now has its
trigger measurement.
