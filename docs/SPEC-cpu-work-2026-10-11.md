# Spec: the no-GPU workstream, 2026-10-11

Executor: an implementer working without follow-up questions. Repository: `a50d6d4`, run from the repository root. `PY=~/code/etx/.venv/bin/python`.
`results_*` paths are in `~/code/etx/`. vLLM paths are in `~/code/vllm/` (`v0.27.0-5-gf46a9dfe2`). This spec makes no performance claim.
Risk names refer to `docs/REPORT-2026-10-11.md`: R1 thin margin (`:41`), R2 fusion lever never engaged (`:42`), R3 S1 lost (`:43`), R4 vision 0 of 7 (`:44`), R5 unreplicated (`:45`), R6 session cost (`:46`), R7 no NVIDIA run (`:47`), R8 no TileRT baseline (`:48`).
CPU gate for every item: `$PY -m etx.tools.cpu_check --out build/cpu_check` must write `"cpu_core_passed": true` to `build/cpu_check/acceptance.json`. At `a50d6d4` it does: 1073 passed, 16 skipped, 3 min 33 s [measured].
Instance fingerprint `FP` (the plan that `examples/vllm_llm/model.py` builds), used by items B2 to B4:
```
for c in examples/llm/configs/{qwen3-30b-a3b,qwen3-8b,mixtral-8x7b}.json build/unseen/*/hf/config.json; do env -i PATH=$PATH HOME=$HOME ETX_LLM_CONFIG=$c $PY -c "import json,hashlib,sys;from examples.llm.model import load_dims;from examples.vllm_llm import model as M;i=M.instances(load_dims(sys.argv[1]));print('FP',sys.argv[1],len(i),hashlib.sha256(json.dumps(i,sort_keys=True,default=str).encode()).hexdigest()[:16])" $c 2>&1 | tail -1; done
```
At `a50d6d4` [measured]: qwen3-30b-a3b `434 e68e856650430532` (also with `ETX_VL_CHECK=1`, the timed build's switch, `results_vlbest2_2026-10-10/build_chkx.log:2`); qwen3-8b `362 c586df49f38d5c3d`; mixtral-8x7b `322 14bbfbe336a339c0`.

## Part A. Why S4f, S5f and S6f replaced 0 patterns

**A.1 Conditions in the arm's log.** Each line below is in `results_vlbest2_2026-10-10/a_r1_S4f.log`, unless another file is named. S5f and S6f print the same banner (`a_r1_S5f.log:68`, `a_r1_S6f.log:71`) [measured].
- The pass is on: `Enabled custom fusions: ... qk_norm_rope_kvcache` (`:12`) and `fuse_qk_norm_rope_kvcache=True` (`:244`). The pass runs before RoPE+KV fusion (`vllm/compilation/passes/pass_manager.py:197-208`). That order rules out a competing fusion.
- The dtype is bf16 (`:14`). The only dtype exit (`qk_norm_rope_kvcache_fusion.py:436-440`) prints a warning, and no such warning appears [measured].
- The backend is ROCM_AITER_FA (`:19`), or UNIFIED for S6f (`a_r1_S6f.log:19`). Both support the fusion when `VLLM_ROCM_USE_AITER=1` (`rocm_aiter_fa.py:1451-1452`, `rocm_aiter_unified_attn.py:314-315`, `_aiter_ops.py:1584,1700`). The env sets it (`:244`) [measured].
- The head-size exits (`qk_norm_rope_kvcache_fusion.py:447-466`) print warnings. None appears. Qwen's eps of 1e-6 is in the registered list (`:467`) [measured].
- The compile range (1, 256) applies (`:488-489`). The banner prints once, right before that range (`:70-72`) [measured].
- The pattern has three more requirements. It expects `vllm.ir.ops.rms_norm` nodes (`:182,186`). It expects the custom rotary op (`custom_ops: ['+rotary_embedding']`, `:7`). It expects the literal layer-name string (`:194`). The log cannot show whether the graph meets them. A pass that registered no pattern prints the same 0 [measured].

**A.2 The difference, from source.** `qk_norm_rope_kvcache_fusion.py` stores `layer.layer_name` as a `str` (`:138`). It bakes that string into the pattern (`:194,272`). `get_inputs` has no layer-name input (`:158-170`). The file imports no LayerName helper (`:23`).
The sibling adds `_encode_layer_name(...)` as a pattern input (`rope_kvcache_fusion.py:139-140,144,331-337`). It picks the input form when `_USE_LAYERNAME` is set (`:260-263,391-394`) and registers that pattern once (`:437-452`).
The model passes `_encode_layer_name(self.layer_name)` (`vllm/model_executor/layers/attention/attention.py:564-573`). `LayerName` is registered with `hoist=True`, so the name becomes a graph input (`vllm/utils/torch_utils.py:845-870`).
`_USE_LAYERNAME` is set when torch is 2.11 or newer and `VLLM_USE_LAYERNAME` is set (`torch_utils.py:832-837`). That variable defaults to "1" (`vllm/envs.py:1348`). The ROCm build pins `torch==2.11.0` (`requirements/build/rocm.txt:6`) [measured, source]. The image's torch version is [unknown].

**A.3 Conclusion.** A CPU probe copies `LayerName` and builds both pattern forms with torch's own matcher [measured, CPU, torch 2.14.1]:
`PROBE torch=2.14.1 names=str literal_matched=1 input_matched=None` and `PROBE torch=2.14.1 names=layername literal_matched=0 input_matched=1`.
With LayerName on, the literal-string pattern matches nothing, and the sibling's input form matches. This is enough to explain 0 on all 48 layers [estimated: the probe is torch 2.14.1 on CPU, not the image].
The other A.1 conditions are not excluded until a rerun with `VLLM_USE_LAYERNAME=0` counts a nonzero number.

**Step A-1, $0, CPU: the probe.** The probe is `/Users/ckl/code/.wt/etx-cpuspec/build/layername_probe.py` (28 lines, sha256 `b9ff3686…a99`). It stays uncommitted under `build/` (`.gitignore:2`).
Command: `$PY build/layername_probe.py str; $PY build/layername_probe.py layername`. Go: the two lines above. Then do A-3. No-go (`literal_matched=1` with `layername`): the hypothesis is false, and the next suspect is the op-form conditions of A.1.
**Step A-2, $0, CPU, optional: the image's torch.** `docker run --rm --platform linux/amd64 --entrypoint python3 vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7 -c "import torch; print('IMAGE-TORCH', torch.__version__)"` (digest: `examples/vllm_llm/vlbest_report.py:21`).
The deciding line is `IMAGE-TORCH 2.11` or newer. A lower version refutes the hypothesis. The image size and whether it runs under emulation are [unknown]. I did not run this step.
**Step A-3, CPU code, rerun on a GPU later.** In `examples/vllm_llm/vlbest_report.py` `config()`, set `env['VLLM_USE_LAYERNAME'] = '0'` for S4, S5 and S6 (`:42-46`). The env echo check (`:83-84`) then proves the variable reached the engine.
Extend `tests/test_arm_configs.py` for the new key. Acceptance: `$PY -m pytest -q tests/test_arm_configs.py tests/test_vlbest_report.py tests/test_vlbest_session.py`, then the CPU gate.
GPU go/no-go, in the next session, by the existing proof rule (`vlbest_report.py:95`): `replaced 48 pattern(s)` in each first-round fusion log. If it still prints 0, set `compilation_config.debug_dump_path`, dump the graph and patterns, and stop.
Do not touch: `~/code/vllm`, the image, or any vLLM patch. A legal arm is a vLLM configuration, and this variable is one (`envs.py:1348`).
Risks: A-3 makes R2 testable. Only a GPU rerun retires R2. A-3 retires none of R1, R3 or R5 to R8.

## Part B. The specification, in execution order

**B1. Refuse an unknown RoPE type.** File: `examples/llm/model.py` `load_dims` (`:82-125`).
Today `theta` is the only RoPE key read (`:111`). `prep.py` acts on `type == "longrope"` only (`examples/vllm_llm/prep.py:99`) [measured].
Change: add one helper `rope_kind(c)` beside `load_dims`. It reads `rope_scaling` and, for newer transformers configs, `rope_parameters`, from either the `type` or the `rope_type` key. It returns `"default"` for null, absent or `default`, and `"longrope"` for longrope. It raises `ValueError` naming the file and the type otherwise.
`load_dims` calls it. When `rope_parameters` carries `rope_theta`, `load_dims` reads that value. Today a missing `rope_theta` silently becomes 10000 (`:111`). `prep.py:99` calls the same helper.
Acceptance: add `tests/test_llm.py::test_unknown_rope_type_is_refused`. It covers a Llama-3.1-8B-shaped config with `rope_type: llama3` (refused), `phi-3.5-moe.json` (longrope, accepted) and `qwen3-30b-a3b.json` (accepted).
Run `$PY -m pytest -q tests/test_llm.py tests/test_decode_entry.py`. Then run `$PY -m examples.vllm_llm.decode build/unseen/llama3.1-8b-shape/hf/config.json --dry-run`.
Expected: the dry run exits non-zero, and its last line is `ValueError: ... rope ... llama3`. Today it exits 0 and prints a plan [measured]. The FP for all three known models is unchanged. Then the CPU gate.
Go: all of the above. No-go: any known model is refused. Do not touch: the RoPE math in `prep.py` or `tiles.hip`, and llama3 scaling support.
Risks: retires one silent-wrong-plan path inside R4 (front end). It does not retire U1 (`docs/VISION-GAP-2026-10-11.md:89`).

**B2. Take every phase from the planner.** File: `examples/vllm_llm/model.py` `instances` (`:249-367`). Today only `GEMV_PHASES` go through `etx_select.plan` (`:262-267`). The attention export is a dict on G (`:289`). The norm, RoPE, cache, top-k and expert exports are hard-coded (`:297,324-330,349-357`) [measured].
Change: call `etx_select.plan(d, "gfx942", structure=Structure(fold_norms=FOLD, fold_topk=FOLD_TOPK, attn_head=DSL_ATTNHEAD))` once over the full `REGISTRY`.
For every phase key the plan holds, `add()` uses `plan.phases[key].export`. An explicit `ETX_VL_*` A/B override (`DSL_ATTN`, `QKNR`, `DSL`, `DSL_TOPK`, `IMPORTED`) still wins, unchanged. `NoLegalVariant` raises `SystemExit` naming the phase. Today the error is a bare `KeyError: 5` for G = 5 [measured].
Acceptance: FP unchanged for all three known models, under the default env and under `ETX_VL_CHECK=1`. Then `$PY -m pytest -q tests/test_decode_entry.py tests/test_select_models.py tests/test_select_planner.py tests/test_select_fallback_budget.py`. At `a50d6d4` that set plus `tests/test_prims.py tests/test_llm.py tests/test_arm_configs.py` gives 98 passed [measured]. Then the CPU gate.
Go: every FP digest is identical. No-go: any digest changes. Revert and report the first differing instance.
Do not touch: `host.hip`, `build.sh`, the `ETX_VL_*` defaults, `plan_offsets`.
Risks: retires the "GEMV phases only" gap of R4, component 3 (`docs/VISION-GAP-2026-10-11.md:19`). No timing risk.

**B3. Generic tiles as a third variant kind.** File: `etx/select/variants.py`. `Variant.__post_init__` allows `dsl` and `imported` (`:55-56`). The planner ranks `dsl` 0 and everything else 1 (`etx/select/planner.py:62`).
Change: allow `kind="generic"`. Rank it `dsl` 0, `imported` 1, `generic` 2 in `_select_level` (`planner.py:62`), so it wins only when nothing else is legal. Add a `_gen()` declarer.
Register `llm_attn` for phase `attn` with predicate `HD % 8 == 0 and HD <= 192 and G <= 16` (`examples/llm/tiles.hip:138,199,203`). Add other phases only with a predicate that cites a `tiles.hip` line (`:286`, `:341`).
In `model.py` (after B2), a `generic` selection raises `SystemExit("phase attn: only the generic tile llm_attn is legal; the vLLM host cannot launch it")`. The generic tiles use another argument ABI (`LlmParams`, `tiles.hip:16-20`).
Acceptance: `$PY -m pytest -q tests/test_select_models.py tests/test_select_planner.py tests/test_decode_entry.py`, plus a new test that a G = 5 config plans `attn llm_attn generic`. FP for the three known models is unchanged. Then the CPU gate.
Go: all pass. No-go: any known model selects a `generic` variant.
Do not touch: `tiles.hip`, `host.hip`, the imported shape keys (`variants.py:283-296`).
Risks: R4, component 2 (`docs/VISION-GAP-2026-10-11.md:18`), at selection level only. It gives a correct fallback in the plan, not a build that runs. It does not retire U1.

**B4. Unseen-model dry runs, $0.** Configs: Mistral-7B-Instruct-v0.3 (dense, G = 4, no q/k norm) and Qwen3-14B (dense, G = 5, q/k norm). Fetch each one with `curl -fsSL https://huggingface.co/<repo>/raw/main/config.json -o build/unseen/<name>/hf/config.json`. The repos are `mistralai/Mistral-7B-Instruct-v0.3` and `Qwen/Qwen3-14B`. The names are `mistral-7b-v0.3` and `qwen3-14b`.
Command: `$PY -m examples.vllm_llm.decode build/unseen/<name>/hf/config.json --dry-run; echo exit=$?`, then FP. The dry run plans only. It does not run `model.py`, so FP is part of the pass (`decode.py:113-127`).
Today [measured]: Mistral exits 0, and FP gives 354 instances. Qwen3-14B exits 1 with `NoLegalVariant: ... phase 'attn'` (`planner.py:109`), and FP gives `KeyError: 5` (`model.py:289`).
After B1 to B3 [estimated]: Mistral is unchanged. Qwen3-14B's dry run exits 0 with `attn llm_attn generic`, and FP raises B3's `SystemExit`.
A pass is exit 0 plus an FP with no error. A refusal is a non-zero exit, or a `SystemExit` that names the missing piece. That is correct behaviour, not a bug. It means the model needs work before a GPU session.
Go to report step 7 only for configs that pass. Today that is Mistral only. Do not count a Llama-3.1 config as a pass: B1 refuses it.
Do not touch: any code to make a config pass. A code change during this step makes the model no longer "unseen".
Risks: first evidence on U1 (`REPORT-2026-10-11.md:51`). It retires nothing measured. GPU correctness (32/32) stays [unknown].

**B5. NVIDIA CPU half.** Files: `etx/ops/prims.py`, `etx/machine/arch/{gfx942,sm_90}.yaml`, `etx/ops/gemv.py`. Today `emit_tiles(machine="sm_90")` still contains 14 `__builtin_amdgcn_readfirstlane` calls (`gemv.py:177-178,752,811,853`) [measured].
Change: add a primitive `uniform` → `etx_uniform` (`int etx_uniform(int v)`). Its gfx942 lowering is `__builtin_amdgcn_readfirstlane(v)`, and its sm_90 lowering is `__shfl_sync(0xffffffffu, v, 0)`. Replace each call site with the mark.
Relax `tests/test_prims.py:34` for the `int` signature. Extend `test_sm_90_source_has_no_amd_code` (`:39-44`) to require zero `__builtin_amdgcn` in the whole sm_90 source.
Add a compile test with `/opt/homebrew/opt/llvm/bin/clang -x cuda --cuda-gpu-arch=sm_90 --cuda-device-only -nocudainc -nocudalib -Xclang -target-feature -Xclang +ptx80 -S`. It compiles the sm_90 lowerings behind a 4-line shim that maps `__shfl_*_sync` to `__nvvm_shfl_sync_*` builtins.
Expected: `.target sm_90` and `shfl.sync.bfly` lines. This works today for `wave_sum` and `wave_sum0` [measured, `build/sm90/`]. Skip when clang lacks `nvptx64`, and list the test in `EXTERNAL_TESTS` (`etx/tools/cpu_check.py:19`).
Acceptance: `$PY -m pytest -q tests/test_prims.py`. The gfx942 emitted source keeps the same device ISA. To check, compare `--save-temps` `.s` before and after with the local LLVM. The CPU gate must also pass.
Go: hardware becomes a separate decision. No-go: any gfx942 ISA difference.
Do not touch: `__hip_atomic_*`, `__shfl`/`__shfl_xor`, the runtime headers. The whole sm_90 source does not compile without CUDA headers [measured: `etx/abi.h`, `import.h`, `tile_util.h` includes].
Risks: the CPU part of R7 only. R7's hardware part and R8 stay.

**B6. Larger vision items, not part of this workstream.** Each needs its own spec. Each step must keep Qwen at 32/32 (`REPORT-2026-10-11.md:64`).
- Switches. First step: `CU` (`model.py:72`, `variants.py:96`) becomes one `Graph.codegen` key published by `etx/codegen/manifest.py` (`guard_block`, `:53`). Acceptance: FP is unchanged, and a `-DETX_VL_CU` on the command line hits the guard's `#error`. Risk: R4, component 4.
- `decode.py` steps. First step: a `golden` step in `steps()` (`decode.py:71-78`) that runs `examples/llm/prep.py --golden-only --context 1024` (`prep.py:226`) and fills the gap noted at `decode.py:124-126`. `vllm` and `report` come later. Risk: R4, component 6.
- `etx/select/tune.py` (new). First step: a CPU-only enumerator over the declared `params` (`variants.py:253,272-282`) with a JSON cache keyed by config hash and machine. GPU timing comes later, at a cost of [unknown]. Risk: R4, component 5.

Not verifiable in this shape: the image's torch version, the S4f/S5f/S6f rerun count, any GPU build of an unseen model, and an NVIDIA compile against real CUDA headers.
