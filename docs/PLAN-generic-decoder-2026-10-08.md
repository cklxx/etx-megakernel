# Plan: one generic decoder codebase for AMD and NVIDIA

Date: 2026-10-08. Status: plan. No GPU measurement supports a new number in this file.
Some numbers are estimates. Each estimate has a label.
Each number below cites its source document.

## 1. Request and scope

The client asks for one codebase that runs most decoder models.
A new model must need a config and optional better tiles. It must not need a new kernel.
The target models are Qwen3-30B-A3B, Qwen3-8B, Mixtral-8x7B and DeepSeek-V2-Lite.
The client asks for AMD and, if possible, NVIDIA by Friday.

The team has no AMD GPU and no NVIDIA GPU at this time.
Thus, the Friday scope contains only work that local tests can verify.

## 2. Metric and floors

### 2.1 Metric

Batch-1 decode reads every active weight once per token.
Thus, the time per token has a lower limit: bytes per token divided by the attainable bandwidth.
This file reports each result as a fraction of the attainable bandwidth of its own GPU.
A ratio against vLLM is not comparable across vendors. vLLM has a different maturity on each GPU.

The denominator uses two definitions. The file gives both:
1. The measured copy ceiling. On MI300X, BabelStream gives 4.3 TB/s. Source: `docs/SURVEY-2026-09-29.md:68`.
2. The rate that kernel authors call attainable. On MI300X, this rate is 3.5 TB/s. Source: `docs/SURVEY-2026-09-29.md:71`.

An NVIDIA result uses the BabelStream copy ceiling of its own GPU. It does not use the vendor peak.

### 2.2 Two floors

The time per token has two floors:
1. The bandwidth floor is the bytes per token divided by the attainable bandwidth.
2. The latency floor is the time of the dependency chain at its minimum. It is the number of edges times the minimum cost of one edge, plus the minimum body time of each phase on the chain.

The file gives two combined floors:
1. The overlap floor is the larger of the two floors. It assumes that the weight stream fully overlaps the low-occupancy phases. This needs complete prefetch across each barrier.
2. The serial floor assumes no prefetch. For each phase, take the larger of its bytes divided by the bandwidth and its minimum body time. Add the results of all phases. Then add the synchronization cost.

The serial floor minus the overlap floor is the maximum gain from prefetch across barriers.
Section 3 uses this difference to rank the levers.
The overlap floor also needs a place to hold the prefetched bytes. The LDS, the L2 and the Infinity Cache have limited capacity.

A dense model is near its bandwidth floor. A MoE model with small experts can be near its latency floor.

For Qwen3-30B-A3B, the measured handoff cost is 9 edges per layer at 1.00 us each, or 0.433 ms per token.
Source: `docs/BUDGET-2026-10-01.md:267`. The minimum body times are not known.
Thus, the latency floor is at least 0.433 ms. Nobody computed the full latency floor yet.
The next CPU task is an estimate of both combined floors from the round11 trace. The estimate needs the bytes and the minimum body time of each phase.

### 2.3 Bandwidth utilization at this time

All rows use batch 1, bf16 and one MI300X. Times are wall time per token.

| Model | Bytes per token | Bandwidth floor at 4.3 / 3.5 TB/s | System | ms/token | Utilization at 4.3 / 3.5 TB/s |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 6.1875 GB, ledger, lower bound | 1.439 / 1.768 ms | ETX megakernel, DSL tiles | 4.823 | 29.8% / 36.7% |
| Qwen3-30B-A3B | same | same | vLLM default | 4.863 | 29.6% / 36.4% |
| Qwen3-8B | 15.29 GB, estimate | 3.556 / 4.369 ms | vLLM best | 4.91 | 72.4% / 89.0% |
| Qwen3-8B | same | same | ETX, generic tiles, without fusion | 9.45 | 37.6% / 46.2% |
| Qwen3-8B | same | same | ETX megakernel, generic tiles | 10.07 | 35.3% / 43.4% |
| Mixtral-8x7B | not recorded | 7.3 to 8.5 ms, README method | vLLM | 8.686 | 84% to 98% of the floor |
| Mixtral-8x7B | not recorded | same | ETX megakernel, DSL tiles | 10.365 | 70% to 82% of the floor |
| DeepSeek-Coder-V2-Lite | not recorded | not computed | ETX with fleet tiles | 3.78 | not computed |

Sources and methods:
1. Qwen3-30B-A3B bytes: `docs/BANDWIDTH-2026-10-07.md:21`. The ledger counts known traffic only. Thus, the utilization is a lower bound.
2. Qwen3-8B bytes: an estimate from `examples/llm/configs/qwen3-8b.json`. The estimate counts all weights except the embedding table, which contributes one row per token. It adds 0.15 GB of KV cache at a 1024-token context. No GPU measurement confirms it.
3. Qwen3-8B times: `README.md:130`, dated 2026-09-25.
4. Mixtral floor and times: `docs/GENERALITY-2026-10-02.md:258-260`. The README gives the floor without a byte count. The utilization column divides the floor by the time.

### 2.4 NVIDIA references

| Reference | Model class | Result | Use |
|---|---|---|---|
| Hazy No Bubbles, B200 and H100 | Llama-1B, dense | 78% of HBM bandwidth, batch 1 | Reference for dense models only |
| Best NVIDIA megakernel systems | MoE and dense | 1.2x to 1.7x of vLLM or SGLang | Temporary reference for MoE |

Sources: `docs/SURVEY-2026-09-29.md:34` and `docs/SURVEY-2026-09-29.md:60-61`.

The 78% result does not apply to Qwen3-30B-A3B. Qwen3-30B-A3B reads 8 small experts per layer. Each expert GEMV is small and latency-bound.
The MoE target on NVIDIA is "to be measured". The measurement uses the same model, bf16 and batch 1 on H100 with vLLM or SGLang.

### 2.5 What the table shows

1. Qwen3-30B-A3B has the most room. ETX and vLLM both use about 30% of the copy ceiling.
2. On gfx942, every published batch-1 MoE result is between 20% and 26%. The highest is 36%. Source: `docs/SURVEY-2026-09-29.md:69-70`.
3. On Qwen3-8B, vLLM uses 72% to 89%. The room above vLLM is small. The ETX target is the bandwidth floor, not a large margin over vLLM.
4. On Mixtral-8x7B, vLLM runs near the floor. The same rule applies.

## 3. Priority

The analysis uses the round11 trace of Qwen3-30B-A3B. Source: `docs/LATENCY-2026-10-07.md`.
The device span is 5,081 us. The bandwidth floor at 4.3 TB/s is 1,439 us. The excess is about 3.6 ms.

Two items own most of the excess:
1. Five low-occupancy phases take 1,931 us, or 38% of the span. They are router, prep, topk, reduce and attn. A small part of the CUs works during these phases. The HBM is mostly idle.
2. The gaps on the critical path take 877 us.

The two items can overlap. Their sum is an upper bound.

The rank of the levers comes from the floors of section 2.2:
1. The serial floor minus the overlap floor gives the maximum gain from prefetch across barriers.
2. The measured time minus the serial floor gives the maximum gain from phase merges, shorter bodies and fewer gaps without prefetch.
The larger value selects the first lever. Both values come from the same CPU estimate.

The first lever is to stream the weights of the next phases during the low-occupancy phases.
The second lever is to merge phases on the critical path.
fleet measured one form of the first lever: idle-worker prefetch into the Infinity Cache. It gave no gain. Source: fleet-mi300x `docs/STATUS.md:72`.
Before new work on this lever, find the cause of that result.

The o_proj stage into LDS predicts 0.117 to 0.171 ms. This gain is small against the 3.6 ms excess. Appendix A keeps it at low priority.

## 4. Current state

All rows use batch 1, bf16 and one MI300X.

Table A gives the current tiles of each model.

The rows of Table A differ in date, tiles and context length. Do not compare one row with another row. Section 2.3 gives the comparable metric.

| Model | Tiles | Date | ETX ms/token | Baseline ms/token | Source |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | DSL tiles | 2026-10-02 | 4.823 | vLLM default 4.863 | `README.md:46-50` |
| Mixtral-8x7B | DSL tiles | 2026-10-02 | 10.365 | vLLM 8.686 | `docs/GENERALITY-2026-10-02.md:258` |
| DeepSeek-Coder-V2-Lite | fleet hand-written tiles | 2026-09-25 | 3.78 | vLLM AITER 4.12, fleet 3.57 | `README.md:132` |
| Qwen3-8B | generic, untuned tiles | 2026-09-25 | 10.07 | vLLM best 4.91 | `README.md:130` |

Table B gives the generic, untuned tiles at a 1024-token context. Source: `README.md:123-135`.

| Model | vLLM, best of default and AITER | ETX tiles without fusion | ETX megakernel |
|---|---|---|---|
| Qwen3-8B | 4.91 | 9.45 | 10.07 |
| Qwen3-30B-A3B | 4.84 | 9.07 | 10.48 |

Table B gives three facts:
1. Qwen3-8B is about 2x slower than vLLM. Nobody measured Qwen3-8B again with the DSL tiles.
2. With the generic tiles, the megakernel is 6.6% slower than the same tiles without fusion. The fusion benefit is negative for Qwen3-8B at this time.
3. The README trace attributes the gap to the tiles, not to synchronization.

Qwen3-30B-A3B changed from 10.48 ms to 4.823 ms between the two tables.
This is the only measured evidence of a large gain from tuned tiles in this repository.
The same change also added the coherence design. Thus, the tiles do not own all of the gain.
DeepSeek-Coder-V2-Lite with tuned fleet tiles is 8% faster than vLLM. Source: `README.md:134`.
The fleet runtime runs the same tiles at 3.57 ms. ETX is 6% slower than fleet on this model. Source: `README.md:132`.

The Mixtral trace attributes the gap to MoE task widths that use Qwen values.
The `moe_up` phase has 896 tasks on 304 workers. It needs three serial waves.
The `moe_down` phase has 128 tasks. It leaves 58% of the workers idle.
Source: `docs/GENERALITY-2026-10-02.md:262-265`.

The NVIDIA path has four blockers:
1. The CUDA emission has no test.
2. The importer reads AMDGPU IR only.
3. The DSL tiles use wave64 builtins. Examples are `__builtin_amdgcn_update_dpp`, `ds_swizzle` and `ds_bpermute` in `etx/ops/gemv.py`.
4. `etx/frontends/tileop.py` has `cuda_link`, but no model uses it.

## 5. Client reply

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
> - Performance reported against two baselines: the same kernels launched one by one, and vLLM's best configuration.
> - Every result also reported as a fraction of the measured copy bandwidth of its GPU. This makes AMD and NVIDIA results comparable.
>
>   Our current measurements use batch 1 on one MI300X:
>   - Qwen3-30B-A3B: ETX and vLLM both use about 30% of the measured copy bandwidth. Published batch-1 MoE results on this GPU are between 20% and 36%. This model class has the most room, and this is where we expect the considerable gains. Our tuned tiles took this model from about 2.2x slower than vLLM to parity. That change also included a new memory-coherence design, so the tiles do not own all of the gain.
>   - Qwen3-8B: vLLM already uses about 72% of the copy bandwidth. Our generic, untuned tiles use about 35%, and the fused kernel is about 6% slower than the same tiles launched one by one. For dense models, our target is the bandwidth limit, not a large margin over vLLM.
>   - Mixtral-8x7B: vLLM runs near the bandwidth floor. We are about 1.19x slower. Our trace attributes the gap to MoE task widths that still use the Qwen shapes. The registry exists to select these widths per shape.
>   - DeepSeek-Coder-V2-Lite: with hand-tuned tiles, ETX is about 8% faster than vLLM. The same tiles in fleet's hand-written runtime run at 3.57 ms, so ETX's fused build is about 6% behind them.
>
>   We will not commit to a specific gain before a hardware measurement. For MoE models, we will measure the NVIDIA reference ourselves under the same conditions.
> - NVIDIA backend validated on hardware before we call it supported
>
> To get there, could you let us know whether you can provide MI300X access, which NVIDIA GPU you want targeted (H100 or B200), and roughly how many GPU-hours we can use?

The reply follows these rules:
1. The Friday list contains only work that local tests can verify.
2. The reply calls NVIDIA started, not supported.
3. The reply gives the good results and the bad results together. It includes Qwen3-8B, the core model of the scope.
4. The reply defines "considerable gains" with bandwidth utilization. It does not use a ratio against vLLM alone.
5. The reply promises no gain before a hardware measurement.

## 6. Architecture

### 6.1 Variant declaration

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

### 6.2 Joint selection

The megakernel uses one VGPR count and one LDS size for all tiles.
Each value is the maximum over all selected tiles.
Thus, a local choice for one phase changes the occupancy of every phase.
`docs/OPTIMISATIONS.md` records one case: the LDS arena of the imported kernels used the maximum of all kernels.

The selection has two layers:
1. Each variant declares its rules and resources. Section 6.1 gives the fields.
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

### 6.3 Publication through the feature manifest

The planner writes its selection into the feature manifest, `etx/codegen/manifest.py`.
The host and the device code read the selection from the manifest only.
The host keeps its checks. Each check reads the manifest. It does not derive the decision again.
This rule prevents the failure that `manifest.py` records: one side of the build derives a different decision.

### 6.4 Tile primitive layer

The DSL tile generators emit HIP C directly. This blocks NVIDIA.
The plan adds a small layer of tile primitives:

| Primitive | gfx942 lowering | sm_90 lowering |
|---|---|---|
| `wave_sum` | DPP, `ds_swizzle`, `ds_bpermute` on 64 lanes | `__shfl_xor_sync` on 32 lanes |
| `stage_copy` | `buffer_load ... lds` | `cp.async` |
| `dot_loop` | the stream loop of `examples/llm/gemv.h` | the same loop with warp32 indices |
| `epilogue` | bf16 store with the coherence mode of `etx/ops/coh.py` | bf16 store |

Each primitive has one lowering for each machine model in `etx/machine/arch/`.
The acceptance test for hardware independence uses the metric of section 2.1.
The same code must reach the same utilization on MI300X and on H100. Each GPU uses its own copy ceiling.
The tile generators call the primitives only.
The first target is the reduction in `etx/ops/gemv.py`.
A CPU test compares the emitted source for gfx942 before and after the change. The two sources must be equal.

### 6.5 TileLang boundary

The design lists TileLang as a frontend. `etx/frontends` contains only `triton_host.py` at this time.

ETX takes two ideas from TileLang:
1. Tile code uses primitives. A per-target lowering produces the instructions. Section 6.4 applies this idea.
2. A tile has a configuration space. A search selects the configuration for each shape. Section 6.1 applies this idea through `params`.

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

## 7. Friday deliverables

All items run on CPU.

1. Variant declarations for the current DSL tiles and imported kernels of `examples/vllm_llm`.
2. A planner that does the joint selection of section 6.2.
3. Selection tests for Qwen3-8B, Qwen3-30B-A3B and Mixtral-8x7B. Each test reads the model config only.
4. A fallback test. A shape outside every DSL variant must select the imported kernel.
5. A budget test. A variant above the budget must not change the selection of other phases.
6. The planner writes its selection into the feature manifest. A test checks that the host reads it from there.
7. The `wave_sum` primitive with gfx942 and sm_90 lowerings. The gfx942 source must stay equal.
8. One entry point for the three GQA models.

DeepSeek-V2-Lite stays on `examples/dsv2lite`.

## 8. After GPU access

1. Run end-to-end correctness for the three models. The gate is 32/32 tokens against the HF greedy reference.
2. Measure each model against two baselines. The first baseline is the same tiles without fusion. The second baseline is the best vLLM configuration.
3. Measure Qwen3-8B with the DSL tiles. The current number uses the generic tiles.
4. Sweep the workgroup size and the Mixtral task widths.
5. Validate the sm_90 lowerings on hardware before the team calls NVIDIA supported.
6. On H100, measure vLLM or SGLang on Qwen3-30B-A3B in bf16 at batch 1. This gives the MoE reference of section 2.4.
7. Run BabelStream on each GPU. Use the result as the denominator of section 2.1.
8. If the user approves, build the TileLang adapter and measure an attention tile.

## 9. Decisions for the user

1. Approve or reject a TileLang attention tile as a replacement for the imported attention.
2. Select the NVIDIA target: H100 or B200.

## Appendix A. o_proj stage into LDS

Priority: low. Section 3 gives the reason.

`etx/tools/stage_sim.py` estimates on CPU the gain from a stage of o_proj weights into free LDS during the attention chain.
The estimate is 0.117 to 0.171 ms per token on Qwen3-30B-A3B. Details: `docs/STAGE-oproj-2026-10-07.md`.
Do not start GPU work on this lever before the work of section 3.
