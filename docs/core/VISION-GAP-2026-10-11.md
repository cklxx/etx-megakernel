# Vision gap, 2026-10-11

Question: how much of the ETX vision does this repository hold today, and how much remains?
Baseline: commit `13c7af0`, branch `wf/o55vision`. The vision is the owner's text, kept at `docs/core/VISION-2026-10-11.md`.
This file makes no new performance claim. Each entry carries `[measured]`, `[estimated]` or `[unknown]`.
`[measured]` here also covers a fact read from code or from a results file.
The `results_*` directories cited below are in the main checkout `~/code/etx/`, not in this worktree.

Grades: **achieved** means the mechanism exists and runs end to end on every in-scope model.
**partly** means the mechanism exists, but only for a subset or with a placeholder.
**not started** means no mechanism has the component's defining property.

## Part 1. Status table

| # | Component | Grade | Evidence | What remains |
|---|---|---|---|---|
| 1 | Model front end | partly | `examples/llm/model.py:82-125` (config.json to Dims); `examples/vllm_llm/decode.py:1-13` (one entry, MLA refused); `examples/vllm_llm/model.py:265-267` | Move the entry out of the benchmark directory. Refuse unread config keys. Fold or retire `examples/llm` and `examples/dsv2lite` as runtime paths. |
| 2 | Operator library | partly | `docs/core/OPERATOR-SPEC.md:3` (21 imported operators); `etx/select/variants.py:283-290` (MoE imports for 2 shapes only); `examples/llm/tiles.hip:138,286,341` (generic-tile limits) | One definition per operator. A default variant that is correct for any in-scope shape. |
| 3 | Task code registry | partly | `etx/select/variants.py:42-62` (contract), `:186-195` (placeholder cost); `etx/select/planner.py:79-112`; `examples/vllm_llm/model.py:265-267` (GEMV phases only) | Add tile coordinates and coherence mode to the contract. Add a per-task reference test at registration. Select every phase. Register Fleet tiles. Use real costs. |
| 4 | Shape-generic host and runtime | partly | `examples/vllm_llm/host.hip:1` ("Qwen decode"), `:180-186` (`ETX_VL_BATCH`), `:468` (batched folded MoE refused); 205 distinct `ETX_VL_*` names (rg, `13c7af0`) | Derive every switch from the model description. Remove the `ETX_VL_*` switches. Prove correctness at batch > 1. |
| 5 | Automatic tuning | not started | `examples/vllm_llm/autoconf.py:1-31` (simulated search of 5 switches); `examples/vllm_llm/model.py:72` (`ETX_VL_CU` default 256); `examples/llm/model.py:39` (`WORKERS = 304`) | Time candidates on the GPU per model. Cache the result. Replace the Qwen-tuned constants. |
| 6 | One-command validation | partly | `examples/vllm_llm/decode.py:71-78,121-135` (prep, build, run, `--repeat 2`, golden required); `examples/llm/prep.py:226` (separate golden step); `examples/vllm_llm/pairs.py:1-18`; `etx/tools/cpu_check.py:1` | One command that also makes the HF golden, runs vLLM, pairs the runs and writes the report. |
| 7 | Chip backend boundary | partly | `etx/machine/model.py:1-8` (machine as data, no arch branches); `etx/machine/arch/sm_90.yaml`; `tests/test_prims.py:30,39-44`; `etx/ops/gemv.py:177-178` (amdgcn builtins remain) | Lower every tile through primitives. Import non-AMDGPU IR. Run one NVIDIA build on hardware. |

Summary: 0 of 7 achieved, 6 partly, 1 not started [measured, by the grades above].

## Part 2. Mechanism per component

**1. Model front end.** `load_dims` reads `config.json` and refuses MLA, mixed layouts and shared experts (`examples/llm/model.py:98-102`).
It derives q/k norm from `model_type` (`examples/llm/model.py:106`) and PhiMoE features from one name (`:124`) [measured].
`prep.py` ignores any `rope_scaling` type except `longrope` (`examples/vllm_llm/prep.py:99-102`) [measured].
A CPU dry run of a Llama-3.1-8B-shaped config with `rope_type: llama3` printed a plan and exited 0, with no warning [measured, CPU, this session].
The generic entry lives in `examples/vllm_llm/`, which the vision keeps as a benchmark only.
Next step: refuse an unknown `rope_scaling` type in `examples/llm/model.py` `load_dims`, then move `decode.py` under `etx/`.

**2. Operator library.** Operators exist in two forms: imported records (`etx/importer/spec.py`) and DSL generators (`etx/ops/`).
The imported fallback is shape-keyed. Fused MoE serves two shapes (`etx/select/variants.py:283-290`), attention serves G = 4, 6, 8 (`:295-296`).
The generic tiles of `examples/llm/tiles.hip` have shape limits, for example HD ≤ 192 (`:138`) and E ≤ 256 (`:286`) [measured].
No operator has one definition with a default correct for any in-scope shape.
Next step: register the generic `examples/llm/tiles.hip` tiles as a third variant kind in `etx/select/variants.py` (today `:55` allows only `dsl` and `imported`).

**3. Task code registry.** `Variant` declares export, phase, backend, dtype, shape rule, resources, cost and params (`etx/select/variants.py:42-52`).
The planner picks the cheapest legal variant per phase over budget levels (`etx/select/planner.py:57-76`).
Every cost is the same placeholder, so ties decide (`etx/select/variants.py:186-195`) [measured].
The contract lacks tile coordinates and coherence mode. Registration runs no reference test. `host --check` is a GPU-time check, not a registration gate.
`examples/vllm_llm/model.py:265-267` applies the planner to GEMV phases only. Fleet tiles (`examples/dsv2lite/`) are not registered.
Next step: in `examples/vllm_llm/model.py`, take every phase's variant from `etx_select.plan`, not only `GEMV_PHASES`.

**4. Shape-generic host and runtime.** The host reads shapes from the packed manifest and checks imported argument blocks (`docs/core/OPERATOR-SPEC.md:14-19`).
Build behaviour still depends on 205 distinct `ETX_VL_*` names [measured, rg on `13c7af0`]. An example is `ETX_VL_CU` (`examples/vllm_llm/model.py:72`).
Batch exists as `ETX_VL_BATCH` (`examples/vllm_llm/host.hip:180-186`). A batched MoE with the top-k fold stops (`:468`).
No result in this repository shows 32/32 at batch > 1 [unknown].
Next step: publish `CU` and the fold choices through the planner manifest (`etx/codegen/manifest.py`) and delete their `ETX_VL_*` reads in `examples/vllm_llm/model.py`.

**5. Automatic tuning.** `autoconf.py` ranks five build switches on the XCD simulator (`examples/vllm_llm/autoconf.py:25-31`). It times nothing on the GPU [measured].
`tile_ab.py` runs manual GPU A/B pairs. `examples/sessions/vlbest3.py:1` tunes vLLM's Triton MoE, the baseline, not ETX.
The Mixtral gap is attributed to Qwen-tuned MoE task widths (`docs/GENERALITY-2026-10-02.md:262-265`) [measured].
No cache of tuned sizes exists.
Next step: add a GPU timing loop over the `params` already declared in `etx/select/variants.py:253,272`, with a cache keyed by config and machine (new file `etx/select/tune.py`).

**6. One-command validation.** `decode.py` runs prep, build and run with `--repeat 2` (`examples/vllm_llm/decode.py:71-78`).
It stops when no HF golden exists (`:124-126,133-135`). The golden comes from a separate command (`examples/llm/prep.py:226`).
vLLM arms, pairing and the report live in session scripts and `pairs.py`, not in `decode.py`.
`python -m etx.tools.cpu_check` is a one-command CPU gate only (`etx/tools/cpu_check.py:1`).
Next step: add `golden`, `vllm` and `report` steps to `steps()` in `examples/vllm_llm/decode.py`.

**7. Chip backend boundary.** The machine is data: visibility, capabilities, costs and primitives per YAML (`etx/machine/model.py:1-8,42-57`).
YAML files exist for gfx90a, gfx942, gfx950, sm_90 and sm_100. `wave_sum` has an sm_90 lowering (`tests/test_prims.py:30,39-44`) [measured].
The GEMV generator still emits `__builtin_amdgcn_readfirstlane` (`etx/ops/gemv.py:177-178`). The importer reads AMDGPU IR only (`docs/core/PLAN-generic-decoder-2026-10-08.md:271-275`).
No NVIDIA build has run on hardware [unknown].
Next step: route `readfirstlane` through `etx/ops/prims.py` and compile `gemv.emit_tiles(machine="sm_90")` with `nvcc` in a CPU test.

## Part 3. Distance to the done-criterion

The criterion: "two unseen models run with minimal intervention and they match or beat TileRT" (`docs/core/VISION-2026-10-11.md:36-37`).

**Measured today.**
- Correctness on one GPU, 32 tokens equal to HF greedy: Qwen3-30B-A3B (`results_vlbest2_2026-10-10/report.log:6-9`), Mixtral-8x7B (`docs/GENERALITY-2026-10-02.md:185`), Phi-3.5-MoE (`docs/GENERALITY-2026-10-02.md:347`) [measured].
- Those two extra models were not run with minimal intervention. Mixtral needed rounds 38-42 of host fixes (`docs/GENERALITY-2026-10-02.md:199-201`) [measured].
- Phi-3.5-MoE needed new model-local code (`docs/GENERALITY-2026-10-02.md:292`) and a race fix (`:334-344`) [measured].
- Qwen3.6-35B-A3B was refused at the config (`docs/GENERALITY-2026-10-02.md:220-225`) [measured].
- Against vLLM, Qwen3-30B-A3B leads S3 by −0.278 ms, spread 0.015 ms (`results_vlbest2_2026-10-10/report.log:11`) [measured].
- Fusion is ESTABLISHED at 4.02% on the step-mean basis (`results_vlbest2_2026-10-10/report.log:10`) [measured].
- Earlier sessions agree: vlbest 4.27%, −0.300 ms (`results_vlbest_2026-10-09/session-iv3fdc3m/report.log:10-11`); nvref2 3.51%, −0.449 ms against default (`results_nvref2_2026-10-09/report.log:13-14`); nvref 2.79% (`results_nvref_2026-10-09/fusion_pairs.txt:7`) [measured].
- With vLLM's kernels only, fusion loses by 0.171 and 0.284 ms (`results_2026-10-01/round24/FINDING.md:27-28`) [measured].
- Other models against vLLM, not under the pairs rule: Phi-3.5-MoE 6.46 vs 11.04 ms (`docs/GENERALITY-2026-10-02.md:352-355`); Mixtral 10.365 vs 8.686 ms (`:258-259`); Qwen3-8B 10.07 vs 4.91 ms, generic tiles (`README.md:140`) [measured].

**Unknown today.**
- U1. Does an unseen in-scope model reach 32/32 with no code change? No unseen model has run since Phi-3.5-MoE [unknown].
- U2. Does such a model match or beat a baseline on time? [unknown].
- U3. Where does TileRT stand? The repository has no TileRT measurement, reference, or mention outside `docs/core/VISION-2026-10-11.md` [unknown].
- Public TileRT pages report results on 8x NVIDIA B200 for DeepSeek-V3.2 and GLM-5 (web search, 2026-10-11; not verified here) [unknown].
- Those models use MLA, which `decode.py:13` refuses. TileRT has no MI300X result to compare against [unknown].

**Cheapest experiment per unknown.** Rate: $1.99 per GPU-hour, owner-supplied (`docs/qwen/NVREF2-2026-10-09.md:160`).
- U1, step 1: a CPU dry run of two unseen configs with `decode.py --dry-run`. Cost: 0 GPU minutes [measured; one run this session].
- U1, step 2: one session that downloads, packs, builds and runs both models for 32 tokens against HF greedy.
  Cost: at most one vlbest2-sized session, 55 minutes, about $1.82 [estimated; 55 minutes is owner-supplied, `docs/qwen/VLBEST2-2026-10-10.md:146`].
  The vlbest2 runner itself took 44m03s (`results_vlbest2_2026-10-10/runner.log:302`) [measured].
- U2: one vlbest2-shaped session per model, five interleaved pairs against vLLM. Cost: about 2 × 55 minutes, about $3.65 [estimated].
  The full vlbest search took 104 minutes, about $3.45 (owner-supplied in the brief; runner 42m27s, `results_vlbest_2026-10-09/session-iv3fdc3m/runner.log:306`) [estimated].
- U3, step 1: a desk check of the GPUs and models TileRT supports. Cost: 0 GPU minutes.
  If no in-scope model runs on both TileRT and MI300X, the owner must restate the comparison. One option is the fraction of each GPU's copy bandwidth (`docs/core/PLAN-generic-decoder-2026-10-08.md:17-30`).
- U3, step 2: a TileRT run on NVIDIA hardware. No NVIDIA session exists in this repository, so its cost is [unknown].

Distance in one line: 0 of 2 unseen models attempted, and 0 TileRT comparisons, against 3 seen models correct and 1 model ahead of vLLM [measured].

Not verifiable in this shape: GPU behaviour of any unseen model, batch > 1 correctness, NVIDIA builds, and any TileRT figure.
