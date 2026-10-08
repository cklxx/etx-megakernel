# Section 7 interface: variants, planner and manifest

Date: 2026-10-08. Status: contract. Owner of this file: fix.
Source: `docs/PLAN-generic-decoder-2026-10-08.md` sections 6.1 to 6.3 and section 7.

Items 1 and 2 (Grok, branch `grok/sec7-variants-planner`) supply the code below. Items 3 to 6 (lucky, fix) test against it.
A change to a name or a signature in this file needs a change to this file first.

## 1. Terms

| Term | Meaning |
|---|---|
| phase | One plan role, for example `qkv`, `topk`, `attn`. The role column of `vl_plan.txt` gives it. |
| layer-0 phase | `<role>@0`. It exists only when layer 0 uses a different export than the other layers of the role. Example: `qkv@0` is `dsl_gemv`, `qkv` is `dsl_gemv_fold`. |
| variant | One export that can run one phase, with its rules and resources. |
| kind | `dsl` (an `etx.ops` tile, the export starts with `dsl_`) or `imported` (a vLLM kernel from `etx.importer`). |
| budget | One level of section 6.2: workgroup size, VGPR limit, LDS limit. |
| selection plan | One variant for each phase, and the budget that the planner selected. Type `etx.select.Plan`. It is not `etx.passes.plan.Plan`. |
| machine | A machine model of `etx/machine/arch/` (`gfx942`, `sm_90`), as a name or as `MachineModel`. The same names as `etx.ops.prims`. |

## 2. Package

All code is in the package `etx/select/`. `etx/select/__init__.py` exports every name below.
- Item 1 (Grok): the variant declaration, the registry.
- Item 2 (Grok): `plan()` and `Plan`.
- Item 6 manifest side (fix): `etx/codegen/manifest.py` and `etx/runtime/include/etx/selection.h`.

## 3. Data types and registry (item 1)

```python
@dataclass(frozen=True)
class Budget:
    threads: int      # workgroup size: 1024 or 512
    vgpr: int         # VGPR limit per lane: 128 at 1024 threads, 256 at 512 threads
    lds: int          # LDS bytes per workgroup: 65536 on gfx942

@dataclass(frozen=True)
class Resources:
    vgpr: int
    lds: int
    threads: int = 0  # the workgroup size that the variant needs; 0 = any size

@dataclass(frozen=True)
class Variant:
    export: str                                   # tile symbol, for example "dsl_gemv"
    phase: str                                    # phase key, for example "qkv" or "qkv@0"
    kind: str                                     # "dsl" | "imported"
    backend: tuple[str, ...]                      # machine names, for example ("gfx942",)
    dtype: str                                    # "bf16"
    shape: Callable[[Mapping[str, int]], bool]    # applicability predicate on the phase shape
    resources: Resources
    cost: Callable[[Mapping[str, int], Budget], float]   # body time estimate, us per layer
    params: Mapping[str, tuple] = field(default_factory=dict)   # free parameters and their legal values

REGISTRY: list[Variant]                                   # every variant of examples/vllm_llm
def phase_shapes(model_cfg) -> dict[str, dict[str, int]]  # phase key -> shape
```

- `model_cfg` is `examples.llm.model.Dims` from `load_dims(config_path)`. A path to `config.json` is also accepted. A selection test reads the model config only.
- A phase shape has the `Dims` fields that the phase uses, as integers (`H`, `NH`, `NKV`, `HD`, `G`, `E`, `TOPK`, `MI`, `INTER`, `V`, `qk_norm`), and `M` and `K` for a GEMV phase.
- The `shape` predicate holds the rule that `host.hip` has as a `die()` today. Example: `dsl_gemv` needs `K % 256 == 0 and K <= 4096` (`host.hip:358`).

## 4. Planner (item 2)

```python
def plan(model_cfg, machine, budget, *, variants=None) -> Plan

@dataclass(frozen=True)
class Plan:
    budget: Budget                         # the selected budget level
    phases: Mapping[str, Variant]          # phase key -> variant
    cost: float                            # sum of the selected costs, us per layer
    def manifest(self) -> dict             # the format of section 6

class NoLegalVariant(ValueError): ...      # the message names the phase and the budget
DEFAULT_BUDGETS: tuple[Budget, ...] = (Budget(1024, 128, 65536),)
```

- `budget` is one `Budget` or a sequence of `Budget` levels. One `Budget` runs step 2 of section 6.2 for that level only. A sequence runs steps 1 to 4. `None` means `DEFAULT_BUDGETS`.
- `variants=None` means `REGISTRY`. The tests give their own list here.
- `plan` raises `NoLegalVariant` when no level is legal. Ties between levels go to the earlier level.

## 5. Rules that the tests check

1. Legal variant: `machine name in v.backend`, `v.shape(shape)` is true, `v.resources.vgpr <= budget.vgpr`, `v.resources.lds <= budget.lds`, and `v.resources.threads in (0, budget.threads)`.
2. At one level, each phase gets its legal variant with the smallest `cost(shape, budget)`. Ties go to `dsl`, then to the earlier variant in the list.
3. The constraint is a maximum, not a sum. Two legal variants of different phases never make a level illegal together.
4. Fallback (item 4): when no `dsl` variant of a phase is legal, the phase gets its `imported` variant. `plan` does not raise.
5. Budget isolation (item 5): a variant that is not legal at a level does not change the choice of any phase at that level. A variant above every level does not change the selection plan.
6. The planner does not read the environment.

## 6. Manifest format (item 6)

The manifest hook puts `Plan.manifest()` into `Graph.codegen["selection"]` (`manifest.SELECTION`):

```python
{"budget": {"threads": 1024, "vgpr": 128, "lds": 65536},
 "phases": {"qkv": ("dsl_gemv_fold", "dsl"), "qkv@0": ("dsl_gemv", "dsl"), "attn": ("attn_mfma16_g8", "imported"), ...}}
```

- `manifest.selection_of(rows, budget)` makes this value from plan instances `(role, layer, export)`. `model.py` uses it until `plan()` exists.
- `manifest.selection_block()` writes it into `plan_data.h` only:

```c
#define ETX_SEL_N_PHASES 10
#define ETX_SEL_THREADS 1024
#define ETX_SEL_VGPR 128
#define ETX_SEL_LDS 65536
static const char* const etx_sel_phase[10]  = {"attn", "in_norm", ...};   // sorted phase keys
static const char* const etx_sel_export[10] = {"attn_mfma16_g8", ...};
static const int32_t     etx_sel_kind[10]   = {1, 1, ...};                // 0 dsl, 1 imported
```

- A `-DETX_SEL_N_PHASES` on the command line is a compile `#error`.
- The host reads the selection through `etx/runtime/include/etx/selection.h`: `etx_sel_find()`, `etx_sel_find_instance(role, layer)` and `etx_sel_check(role, layer, export)`.
- `examples/vllm_llm/host.hip` calls `etx_sel_check` for each plan instance. It stops when the plan file and the manifest name different exports. It also stops when a selected phase has no instance.
- The megakernel source does not contain the block. Its tile calls come from the same graph. Thus the gfx942 source stays byte-identical when the selection equals the current choice.

## 7. What waits for items 1 and 2

1. The item 4 and item 5 tests use a stub with the signatures above (`tests/select_stub.py`). They use `etx.select` when it exists.
2. When `etx.select` exists, `model.py` builds its instances from `plan(...)` and publishes `Plan.manifest()`.
3. `build.sh` derives the import list again from `ETX_VL_DSL_*`. This is a second derivation of the selection. The planner must give the import list, for example `python -m etx.select --exports <config>`.
