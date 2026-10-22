# Qwen3-30B-A3B — MI300X / gfx942

## Implementations

| Implementation | Source of truth | Integration |
| --- | --- | --- |
| DSL GEMV, MoE, top-k and attention prologue | [`etx/ops/`](../../../../etx/ops/), selected by [`examples/vllm_llm/model.py`](../../../../examples/vllm_llm/model.py) | ETX-native tiles |
| vLLM HIP specializations | [`etx/importer/vllm_kernels.py`](../../../../etx/importer/vllm_kernels.py), [`etx/importer/spec.py`](../../../../etx/importer/spec.py) | Compiled IR import and generated adapters |
| vLLM Triton MoE gate/up | [`moe-up/vllm-triton-bf16/`](moe-up/vllm-triton-bf16/README.md) | Library-owned captured IR, metadata and reference code object; original path is a compatibility symlink |
| Remaining vLLM Triton captures | [`examples/vllm_llm/triton/`](../../../../examples/vllm_llm/triton/), [`etx/importer/specs/`](../../../../etx/importer/specs/) | Shape-specific IR and metadata import; not arbitrary-shape binaries |

Build and host integration remain in [`examples/vllm_llm/`](../../../../examples/vllm_llm/). The operator contract is documented in [`OPERATOR-SPEC.md`](../../../../docs/core/OPERATOR-SPEC.md).

## Qualified historical evidence

The [VLBEST2 session](../../../../docs/qwen/VLBEST2-2026-10-10.md) records a hybrid implementation: DSL GEMV/MoE/top-k/attention prologue plus imported vLLM attention, attention-reduce and RMSNorm.

Conditions: Qwen3-30B-A3B BF16, batch 1, context 1024, 32 greedy tokens, one MI300X VF, ROCm 7.14.0, five paired rounds. Recorded host-wall medians: fused 4.471 ms, same-kernel HIP graph 4.654 ms, strongest tested legal vLLM configuration 4.749 ms. See the report for the distinct per-round aggregation bases and paired verdicts.

All final rounds matched 32/32 tokens. The 97/97 imported-launch checks at positions 0 and 1040 were transferred between identical build identities, not rerun in that session. This qualifies the recorded build and conditions, not every revision of the linked sources, every prompt, or an individual fastest kernel.

The [historical all-imported comparison](../../../../docs/qwen/WHY-DSL-TILES-2026-10-01.md) recorded fusion losses and lacked its imported-launch-check binary. Keep it separate from the qualified hybrid result.
