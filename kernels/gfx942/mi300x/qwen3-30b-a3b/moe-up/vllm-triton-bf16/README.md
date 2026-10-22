# MoE gate/up — vLLM Triton BF16 capture

An actual captured kernel, not a placeholder or a new numerical implementation. This directory owns the original artifacts; `examples/vllm_llm/triton/fused_moe_up` is a relative compatibility symlink so existing builds and reference launches consume this same copy.

## Operation and supported contract

Computes the selected experts' concatenated gate/up projections. It does **not** include SiLU/multiply, down projection, expert summation or routing selection. The caller supplies correctly padded token/expert dispatch metadata.

| Property | Captured/supported value |
| --- | --- |
| Model / target | Qwen3-30B-A3B / MI300X, gfx942, wave64 |
| Forward batch | 1; not a general batched MoE capture |
| Model dimensions | H=2048, expert intermediate=768, experts=128, top-k=8 |
| GEMM output / reduction dimensions | N=1536 (gate + up), K=2048 |
| Precision | BF16 inputs/weights, FP32 dot accumulation, BF16 output |
| Triton tiles | BM=16, BN=64, BK=128 |
| Routing weight multiplication | Disabled (`MUL_ROUTED_WEIGHT=False`) |
| Original launch | 192 blocks, 256 threads/block (4 waves) |
| Shared memory | 20,480 bytes/block |
| Global/profile scratch | Both zero in capture metadata |

The authoritative, fingerprint-bound contract is the `fused_moe_up` record in [`moe-decode-v1.json`](../../../../../../etx/importer/specs/moe-decode-v1.json). [`launch_spec.py`](../../../../../../etx/importer/launch_spec.py) validates the fingerprints, target, shared memory and TTIR dot shapes, and derives the launch. Do not generalize the supported shape from the operator name.

## Files and provenance

- `fused_moe_kernel.ttir`: captured Triton IR, with source locations in `vllm/model_executor/layers/fused_moe/fused_moe.py`.
- `fused_moe_kernel.llir`: captured AMDGPU LLVM IR used by ETX's importer.
- `fused_moe_kernel.json`: original compiler metadata, reporting Triton 3.8.0 and gfx942.
- `fused_moe_kernel.hsaco`: original executable code object used for standalone reference launches.
- `CACHE_DIR`: original Triton cache-entry identifier.

These five files were migrated byte-for-byte from `examples/vllm_llm/triton/fused_moe_up`. Their repository introduction is commit `46870977d4493e5c12214d11012ecee1278b359b`. The existing registry pins SHA256 for the four kernel/metadata artifacts; the migration test also pins `CACHE_DIR`.

The captured files do not establish the exact upstream vLLM source revision or include its license/notice bundle. This migration adds no new upstream download and claims no relicensing. Before externally redistributing this capture, recover the matching upstream provenance and required notices. Regeneration from upstream Python source is not claimed reproducible from this bundle alone.

## Use and local validation

Existing builds continue to select export `fused_moe_up` via the compatibility path. ETX imports the LLVM IR into a device function; it does not embed the original `.hsaco` as a callable function. The `.hsaco` remains the independently launched reference. The adapter must supply the checked arguments, logical block/thread IDs, LDS and barrier semantics.

From the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_kernel_library.py tests/test_launch_spec.py tests/test_operator_spec.py
```

This checks canonical ownership, artifact integrity, the launch contract and actual IR transformation without a GPU. It does **not** establish GPU numerical correctness or performance on a new build. Device compilation and same-build imported/original and full-model checks remain required.

## Evidence and limits

The [historical all-imported experiment](../../../../../../docs/qwen/WHY-DSL-TILES-2026-10-01.md) used this kernel family and matched 32/32 generated tokens, but did not build its imported-launch checker and recorded a fusion loss. It is not a per-kernel latency certificate.

The later [VLBEST2 hybrid win](../../../../../../docs/qwen/VLBEST2-2026-10-10.md) used **DSL MoE**, not this imported MoE kernel. Do not attach that result to this entry. No isolated timing or fastest-kernel verdict is claimed here.
