# GLM-5.3-Flash — MI325X / gfx942

## Implementations

| Implementation | Source of truth | Integration |
| --- | --- | --- |
| Native FP8-weight readers, GEMV, normalization, routing, expert MLP, mHC, KDA and sparse MLA | [`examples/glm53/native.h`](../../../../examples/glm53/native.h), [`tiles.hip`](../../../../examples/glm53/tiles.hip) | ETX-native; not imported vLLM kernels |
| Graph, implementation options and host execution | [`model.py`](../../../../examples/glm53/model.py), [`options.py`](../../../../examples/glm53/options.py), [`model_host.hip`](../../../../examples/glm53/model_host.hip) | PP2 with whole-layer ownership; inspect the selected build options for task decomposition |

Model scope: text-only GLM-5.3-Flash, 45 layers (34 KDA and 11 sparse MLA), four residual streams, hidden width 4096. The native numerical path decodes block-scaled FP8 weights to BF16 and uses FP32 accumulation. See the [model description](../../../../docs/qwen/QWEN-BATCH-GLM53.md) for precision and state contracts.

## Evidence and status

**No correctness-qualified fastest kernel is established for this entry.**

The [October 22 session notes](../../../../docs/glm/EXPERIMENTS-GLM-MULTIGPU-2026-10-22.md) identify an MI325X node and record a two-card, batch-1, full-45-layer diagnostic trace of the original one-workgroup-per-request/layer implementation. That trace is not a qualified performance baseline and does not validate later changes in the linked sources.

The [execution design](../../../../docs/glm/GLM-EXECUTION-DESIGN-2026-10-19.md) distinguishes compiler-surviving variants from measured winners. The [baseline protocol](../../../../docs/glm/BASELINE-PROTOCOL.md) describes required acceptance; a protocol document is not evidence that its gates passed. The [capacity design](../../../../docs/glm/PARALLEL-CAPACITY-2026-10-22.md) contains estimates, not measured kernel throughput.

## External candidate, not a compatible implementation

AITER FlyDSL `aiter.ops.flydsl.glm5_mono` is discussed in the [mono-decode review](../../../../docs/core/VLLM-MONO-DECODE-PR-2026-10-19.md). Its documented envelope is GLM-5 `glm_moe_dsa`, gfx950, TP4/TP8 and pure decode widths 1/2/4/8. It is not a drop-in GLM-5.3-Flash / gfx942 / PP2 kernel and is not a measured winner for this entry. No upstream source is vendored here.
