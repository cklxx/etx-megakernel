# Plan: one generic decoder codebase for AMD and NVIDIA

Date: 2026-10-08. Status: plan. No GPU measurement supports a new number in this file.
Each number below cites its source document.

## 1. Request and scope

The client asks for one codebase that runs most decoder models.
A new model must need a config and optional better tiles. It must not need a new kernel.
The target models are Qwen3-30B-A3B, Qwen3-8B, Mixtral-8x7B and DeepSeek-V2-Lite.
The client asks for AMD and, if possible, NVIDIA by Friday.

The team has no AMD GPU and no NVIDIA GPU at this time.
Thus, the Friday scope contains only work that local tests can verify.

## 2. Current state

| Model | Path | ETX ms/token | Baseline ms/token | Source |
|---|---|---|---|---|
| Qwen3-30B-A3B | `examples/vllm_llm`, DSL tiles | 4.823 | vLLM default 4.863 | `docs/ROADMAP.md:66` |
| Mixtral-8x7B | `examples/vllm_llm`, DSL tiles | 10.365 | vLLM 8.686 | `docs/GENERALITY-2026-10-02.md:258` |
| DeepSeek-V2-Lite | `examples/dsv2lite`, MLA | 3.76-3.78 | fleet hand-written 3.557 | `README.md:92` |
| Qwen3-8B | `examples/llm`, generic tiles | 13.4 | not measured | `README.md:117` |

The Mixtral trace attributes the gap to MoE task widths that use Qwen values.
The `moe_up` phase has 896 tasks on 304 workers. It needs three serial waves.
The `moe_down` phase has 128 tasks. It leaves 58% of the workers idle.
Source: `docs/GENERALITY-2026-10-02.md:262-265`.

The NVIDIA path has four blockers:
1. The CUDA emission has no test.
2. The importer reads AMDGPU IR only.
3. The DSL tiles use wave64 builtins. Examples are `__builtin_amdgcn_update_dpp`, `ds_swizzle` and `ds_bpermute` in `etx/ops/gemv.py`.
4. `etx/frontends/tileop.py` has `cuda_link`, but no model uses it.

## 3. Client reply

Send this text. It contains the corrections from the ETX group review.

> Thanks, the goal is clear and we agree with it: one codebase that handles most decoder models, so a new model needs a config and optionally better tiles, not a new kernel.
>
> **What we will deliver by Friday (AMD):**
> - A generic decoder path driven by the model's config.json, with one entry point for all models
> - Qwen3-8B, Qwen3-30B-A3B and Mixtral-8x7B running through that common path
> - A task-code registry with pluggable tiles. Each task-code variant declares its own applicability (shape limits, dtype, LDS/VGPR budget, backend); the registry selects by these constraints, never by model name, and falls back to the generic or imported kernel when none match. Selection is unit-tested on CPU for all three models.
> - The NVIDIA backend structure started inside the same codebase, but not yet validated
>
> **DeepSeek-V2-Lite:** its attention (MLA) is structurally different from the GQA models, so for Friday it stays on its existing specialized path. We will fold it into the common path after the GQA models are stable.
>
> **Current constraint:** we have no AMD or NVIDIA GPU access right now, so Friday's work is verified locally (unit tests and build checks), not on hardware.
>
> **After Friday, once GPUs are available:**
> - End-to-end correctness on hardware for all three models
> - Performance reported against two baselines: the same kernels launched one by one, and vLLM's best configuration. For reference, in our current measurements we are at parity with vLLM on Qwen3-30B-A3B and about 1.19x behind on Mixtral-8x7B. Our trace attributes the Mixtral gap to MoE task widths that are still sized for Qwen's shapes, which is exactly the per-shape selection the registry is meant to handle. We will not commit to a specific gain until it is measured on hardware.
> - NVIDIA backend validated on hardware before we call it supported
>
> To get there, could you let us know whether you can provide MI300X access, which NVIDIA GPU you want targeted (H100 or B200), and roughly how many GPU-hours we can use?

The reply follows these rules:
1. The Friday list contains only work that local tests can verify.
2. The reply calls NVIDIA started, not supported.
3. The reply gives the Qwen result and the Mixtral result together.
4. The reply promises no gain before a hardware measurement.

## 4. Architecture

### 4.1 Variant declaration

Each tile variant declares its applicability and its resource needs as data.
The declaration lives beside the tile generator in Python.

| Field | Content | Example |
|---|---|---|
| `export` | The tile symbol. | `dsl_gemv_bigk` |
| `backend` | The target architectures. | `gfx942` |
| `dtype` | The weight and activation types. | `bf16` |
| `shape` | A predicate on the operator shape. | `K % 256 == 0` |
| `resources` | VGPR count, LDS bytes and workgroup size. | `vgpr=128, lds=29736, threads=1024` |
| `params` | The free task parameters and their legal values. | rows per task, task width, K chunk |
| `cost` | An estimate of body time for a given occupancy. | a function of bytes and waves |

At this time, `examples/vllm_llm/host.hip` holds the applicability rules as `die()` calls.
For example, `host.hip:354` stops the run for `dsl_gemv` when `K > 4096`.
The plan moves each rule into the declaration of its variant.
After the move, a shape outside every DSL variant selects the imported kernel. The run does not stop.

The `params` field addresses the Mixtral gap.
The task width is a parameter of the variant. It is not a constant from the Qwen shapes.

### 4.2 Joint selection

The megakernel uses one VGPR count and one LDS size for all tiles.
Each value is the maximum over all selected tiles.
Thus, a local choice for one phase changes the occupancy of every phase.
`docs/OPTIMISATIONS.md` records one case: the LDS arena of the imported kernels used the maximum of all kernels.

The selection has two layers:
1. Each variant declares its rules and resources. Section 4.1 gives the fields.
2. The planner selects one variant and one parameter set for each phase of the whole model.

The constraint uses a maximum, not a sum. Thus, the planner can solve the problem exactly by enumeration:
1. Make a list of budget levels. Each level gives a workgroup size, a VGPR limit and an LDS limit.
2. For each level, select the fastest legal variant for each phase. Use the `cost` field.
3. Add the phase times for that level.
4. Select the level with the smallest total.

The work is the product of levels, phases and variants. This product is small.

The workgroup size is a budget variable. At 1024 threads, each SIMD holds 4 waves.
The 512 registers per lane then give a limit of 128 VGPRs.
At 512 threads, the limit is 256 VGPRs. The spills stop, but each CU holds half as many waves.
Only a GPU measurement can decide this tradeoff. Before Friday, the planner accepts the workgroup size as an option.

`etx/tools/stage_sim.py` in PR #9 shows one coupling of this kind.
The hidden time of the o_proj stage depends on the LDS that other tiles use.

### 4.3 Publication through the feature manifest

The planner writes its selection into the feature manifest, `etx/codegen/manifest.py`.
The host and the device code read the selection from the manifest only.
The host keeps its checks. Each check reads the manifest. It does not derive the decision again.
This rule prevents the failure that `manifest.py` records: one side of the build derives a different decision.

### 4.4 Tile primitive layer

The DSL tile generators emit HIP C directly. This blocks NVIDIA.
The plan adds a small layer of tile primitives:

| Primitive | gfx942 lowering | sm_90 lowering |
|---|---|---|
| `wave_sum` | DPP, `ds_swizzle`, `ds_bpermute` on 64 lanes | `__shfl_xor_sync` on 32 lanes |
| `stage_copy` | `buffer_load ... lds` | `cp.async` |
| `dot_loop` | the stream loop of `examples/llm/gemv.h` | the same loop with warp32 indices |
| `epilogue` | bf16 store with the coherence mode of `etx/ops/coh.py` | bf16 store |

Each primitive has one lowering for each machine model in `etx/machine/arch/`.
The tile generators call the primitives only.
The first target is the reduction in `etx/ops/gemv.py`.
A CPU test compares the emitted source for gfx942 before and after the change. The two sources must be equal.

### 4.5 TileLang boundary

The design lists TileLang as a frontend. `etx/frontends` contains only `triton_host.py` at this time.

ETX takes two ideas from TileLang:
1. Tile code uses primitives. A per-target lowering produces the instructions. Section 4.4 applies this idea.
2. A tile has a configuration space. A search selects the configuration for each shape. Section 4.1 applies this idea through `params`.

ETX does not use the kernel shell that TileLang generates.
`T.Kernel` sets its own thread count, shared memory allocation and grid.
In the megakernel, these values are fixed and shared by all tiles.
A TileLang adapter must accept the fixed thread count.
It must use the LDS offset that ETX allocates.
It must not emit its own grid loop.

The best first TileLang tile is attention, not GEMV.
`etx/ops/gemv.py` records that the GEMV tile is limited by its skeleton, not by its stream.
Without the weight stream, the critical path keeps 63% of the body.
A matrix instruction addresses 0.70% of one measured regression.
Attention is the largest critical body in round 11: 310.84 us, from `docs/LATENCY-2026-10-07.md:241`.

TileLang publishes an MLA decode tile for MI300X.
It reports 0.73x to 1.21x of aiter-asm at batch 64 and batch 128 as a standalone kernel.
Source: [TileLang MLA on AMD](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/amd/README.md).
These numbers do not apply to batch 1 inside the megakernel. A GPU session must measure them again.

A TileLang attention tile replaces a production kernel.
`CLAUDE.md` limits the claim to production kernels. Thus, this step needs a decision from the user.

## 5. Friday deliverables

All items run on CPU.

1. Variant declarations for the current DSL tiles and imported kernels of `examples/vllm_llm`.
2. A planner that does the joint selection of section 4.2.
3. Selection tests for Qwen3-8B, Qwen3-30B-A3B and Mixtral-8x7B. Each test reads the model config only.
4. A fallback test. A shape outside every DSL variant must select the imported kernel.
5. A budget test. A variant above the budget must not change the selection of other phases.
6. The planner writes its selection into the feature manifest. A test checks that the host reads it from there.
7. The `wave_sum` primitive with gfx942 and sm_90 lowerings. The gfx942 source must stay equal.
8. One entry point for the three GQA models.

DeepSeek-V2-Lite stays on `examples/dsv2lite`.

## 6. After GPU access

1. Run end-to-end correctness for the three models. The gate is 32/32 tokens against the HF greedy reference.
2. Measure each model against two baselines. The first baseline is the same tiles without fusion. The second baseline is the best vLLM configuration.
3. Sweep the workgroup size and the Mixtral task widths.
4. Validate the sm_90 lowerings on hardware before the team calls NVIDIA supported.
5. If the user approves, build the TileLang adapter and measure an attention tile.

## 7. Decisions for the user

1. Approve or reject a TileLang attention tile as a replacement for the imported attention.
2. Select the NVIDIA target: H100 or B200.
