# Section 7 items 1 and 2: the `etx.select` interface

Date: 2026-10-08. Owner: Grok. Branch: `grok/sec7-variants-planner`, based on `fix/sec7-456`.
Contract: `docs/SEC7-interface-fix.md`. This file adds only what that contract does not say.

## 1. Files

| File | Content |
|---|---|
| `etx/select/variants.py` | `Budget`, `Resources`, `Variant`, `Structure`, `REGISTRY`, `register()`, `variants_for()`, `phase_shapes()`, `default_structure()`, `load_cfg()`, the declarations. |
| `etx/select/planner.py` | `plan()`, `Plan`, `NoLegalVariant`, `DEFAULT_BUDGETS`, `BUDGET_LEVELS`. |
| `etx/select/__main__.py` | `python -m etx.select [--exports or --manifest] <config.json>`. |
| `tests/test_select_planner.py` | Smoke tests on the real `REGISTRY`. |

`etx/select/__init__.py` exports every name above.

## 2. Signatures

```python
plan(model_cfg, machine="gfx942", budget=None, *, variants=None, structure=None) -> Plan
phase_shapes(model_cfg, structure=None) -> dict[str, dict[str, int]]
default_structure(dims) -> Structure
register(v: Variant) -> Variant            # appends to REGISTRY; a second (export, phase) pair raises ValueError
variants_for(phase, variants=None) -> list[Variant]
Variant.legal(machine: str, shape, budget) -> bool   # rule 1 of the contract, section 5

Plan.budget, Plan.phases, Plan.cost       # as in the contract
Plan.machine: str
Plan.reasons: dict[str, str]              # phase key -> why; a fallback reason starts with "fallback"
Plan.manifest() -> dict                   # the contract's section 6 format; phase keys sorted, as selection_of()
Plan.resources() -> {"vgpr", "lds", "threads"}   # the maximum over the selected variants
Plan.exports(kind=None) -> list[str]      # kind="imported" gives the import list for build.sh
```

`Variant` has the contract's fields. It also has `rule: str = ""`. The field holds the `host.hip` line that the predicate replaces.

## 3. Rules

1. `model_cfg` is a `Dims`, a path to `config.json`, or the config dict.
2. The planner plans the phases of the model that `variants` declares. `REGISTRY` declares every phase of the five configs in `examples/llm/configs` that the tests use. A test list can declare only some phases. The planner then ignores the other phases.
3. The planner does not read the environment.
4. All variants of one phase have the same cost at this time. The cost is the phase floor plus the weight bytes at 4.3 TB/s. Thus the tie rule selects: a legal DSL variant first, then the first declared variant. A GPU measurement must replace this cost.

## 4. Phase set (`Structure`)

A fold removes a phase. Section 6.2 selects one variant for each phase of a fixed set. Thus the planner does not select folds. `Structure` gives them:

| Field | Effect | Default (`default_structure`) |
|---|---|---|
| `fold_norms` | No `in_norm` after layer 0, no `post_norm`, `final_norm` or `moe_sum`. `qkv`, `router`, `lm_head` get `fold=1`. Key `qkv@0` appears. | MoE models and PhiMoE. A dense model keeps its norms, because `dsl_gemv_fold` needs E > 0 (`host.hip:394`). |
| `fold_topk` | No `topk` phase. | `examples/vllm_llm/folds.decide()`. |
| `attn_head` | `prep` (`dsl_attnprep`) replaces `q_norm`, `k_norm`, `rope` and `cache`. | q/k norm and G + 2 <= 16, as in `model.py` `_attn_head`. |

A shape has the `Dims` fields as integers, and `fold`, `fold_topk` and `first_only`. A GEMV phase also has `M` and `K`. For `moe_up`, `M` is `MI`, as in `model.py`. The cost doubles it for the gate rows and the up rows.

## 5. Results on the real REGISTRY

1. Qwen3-30B-A3B and Mixtral-8x7B: the plan equals the exports of `model.py` `instances()` in the default build. A test checks this.
2. Qwen3-8B: `down` has K = 12288. It selects `wvsplitk_y2` with a fallback reason. `host.hip:358` stops this run today.
3. Qwen2.5-1.5B: `qkv` has a bias. It selects `wvsplitk_y2` (`host.hip:361`).
4. At `Budget(512, 256, 65536)`, `plan` raises `NoLegalVariant` for `qkv`. The DSL tiles and wvSplitK need 1024 threads.

## 6. Limits

1. Imported VGPR is 128 (the launch bound). Imported LDS is 0. `imports.json` has the true LDS. The CPU registry does not read it.
2. `fused_moe_up` and `fused_moe_down` need the `moe_act` phase (`silu_and_mul`). The planner does not add it. A plan that selects them is not complete.
3. `params` lists the task parameters. The planner does not search them yet.
4. `cost` does not use the budget. Occupancy needs a GPU measurement.
5. Only `gfx942` is in `backend`. No tile has an sm_90 lowering yet.
6. `model.py` and `build.sh` do not call the planner yet.
