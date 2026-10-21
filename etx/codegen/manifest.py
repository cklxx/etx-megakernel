"""The feature manifest: what the emitter actually emitted, published to every translation unit that must agree
with it.

A feature that changes device code or host behaviour and that more than one side must agree about belongs here and
nowhere else. The failure mode it exists to prevent is a `-D` on one compiler invocation and an independent
re-derivation of the same decision somewhere else; every occurrence has been silent, producing a build that looks
healthy and is wrong:

  2026-10-01 round 14  ETX_VL_NB      the tile was emitted without it while the plan batched, so extra requests
                                      would have read column 0.
  2026-10-01 round 19  ETX_DEV_RESET  build.sh appended the define to an array that a later `if` overwrote whenever
                                      ETX_VL_COH was set, so the kernel's reset tail was preprocessed out while the
                                      host believed it ran: 0/32 tokens, no diagnostic.
  2026-10-01 round 20  ETX_DEV_RESET  the same define never reaching the separate device-code compile stage.
  2026-10-01 round 30  the top-k fold model.py derived "do not fold" while build.sh's define defaulted to "fold", so
                                      the plan gave the top-k its own phase and the expert tiles recomputed it
                                      anyway: 0.63 ms, tokens still matched, nothing failed.

The rule now has teeth. `guard_block()` is emitted into both generated artefacts -- `megakernel_d0.hip` for the
device translation unit and `plan_data.h` for the host one -- ahead of anything that reads a feature macro. It
`#error`s if the macro arrived from a command line, then defines it from the plan. A reviewer who adds a feature the
old way gets a compile failure naming this file, not a wrong answer.

Each entry maps the plan's `Graph.codegen` key to the macro the sources test and how it is spelled:
  "flag"  the macro is defined only when the feature is on, so sources may use #ifdef (the tile sources do);
  "value" the macro always has an integer value, so sources may use #if or read it as a number.
"""
from __future__ import annotations

# codegen key -> (macro the sources test, spelling, default when the plan says nothing)
FEATURES: dict[str, tuple[str, str, int]] = {
    "moe_block_m": ("ETX_EMIT_MOE_BLOCK_M", "value", 16),  # captured MoE argument/grid contract
    "argpf": ("ETX_EMIT_ARGPF", "value", 0),        # next-task argument prefetch (kernel.py)
    "bar2": ("ETX_EMIT_BAR2", "value", 0),          # two-barrier static loop (kernel.py)
    "dev_reset": ("ETX_EMIT_DEV_RESET", "value", 0),  # the kernel resets its own per-step state (host.h)
    "fold_topk": ("ETX_EMIT_FOLD_TOPK", "value", 0),  # the top-k folded into the expert tasks (host.hip)
    "last_router": ("ETX_EMIT_LAST_ROUTER", "value", 0), # G01 plan/host argument binding; default off
    "coh": ("ETX_DSL_COH", "flag", 0),              # the DSL tiles' coherent activation access (etx/ops/coh.py)
    "coh_l1": ("ETX_DSL_COH_L1", "flag", 0),        # ...with AGENT-scope loads (coherence mode "l1", p4 sets it)
    "nb": ("ETX_VL_NB", "value", 1),                # activation columns the tiles were emitted for (host.hip)
    "requests": ("ETX_VL_REQUESTS", "value", 1),    # concurrent decode requests the plan carries (host.hip)
    "uc": ("ETX_VL_UC", "flag", 0),                 # cross-XCD tensors in uncached memory (host.hip, lowering)
    "dsl_bigk": ("ETX_EMIT_DSL_BIGK", "value", 0),  # MoE K > 4096 chunked in the DSL tiles (gemv.py, host.hip guards)
    "dsl_ext": ("ETX_EMIT_DSL_EXT", "value", 0),    # PhiMoE-class: LayerNorm fold (mode 4) + output bias; the two
                                                    # extra arg pointers exist only when ETX_VL_DSL_EXT sized the
                                                    # layout at emit time -- model.py asserts the two agree
}

# Macros that used to be passed as -D for a feature now in FEATURES. Trapped by name so an old build script, an old
# shell array, or a copied command line fails loudly instead of silently disagreeing with the plan.
RETIRED: tuple[str, ...] = ("ETX_DSL_COH", "ETX_VL_NB", "ETX_VL_REQUESTS", "ETX_VL_UC", "ETX_COHERENT_DEVICE_DATA",
                            "ETX_DEV_RESET", "ETX_VL_NO_TOPK_FOLD")


def guard_block(codegen: dict) -> list[str]:
    """The trap and the definitions, in that order. Emitted near the top of every generated artefact."""
    out = ["", "// ---- feature manifest (etx/codegen/manifest.py): the emitter's decisions, not the command line's."]
    out.append("// Passing any of these as -D is the bug class this block exists to catch; the plan decides them.")
    for mac in RETIRED:
        out.append(f"#ifdef {mac}")
        out.append(f'#error "{mac} must not be defined on the command line: the plan publishes it '
                   '(etx/codegen/manifest.py). Set the feature in examples/vllm_llm/model.py so the emitter and '
                   'every translation unit agree by construction."')
        out.append("#endif")
    for key, (mac, kind, default) in FEATURES.items():
        v = codegen.get(key, default)
        v = int(v) if not isinstance(v, bool) else int(bool(v))
        if kind == "flag":
            if v:
                out.append(f"#define {mac} 1")
        else:
            out.append(f"#define {mac} {v}")
    out.append("")
    return out


# ---- variant selection (docs/core/PLAN-generic-decoder-2026-10-08.md sections 6.2, 6.3; docs/core/SEC7-interface-fix.md)
#
# The planner selects one variant (one export) for each phase. `Graph.codegen[SELECTION]` carries that choice, and
# `selection_block()` publishes it in plan_data.h. The host reads it there and nowhere else: it compares every plan
# instance with it (etx/runtime/include/etx/selection.h). The device code needs no copy. Its tile calls come from the
# same graph, so the megakernel source does not change when the selection is published.
#
# A phase key is the plan role ("qkv", "topk", ...). A layer-0 instance whose export differs from the other layers
# of its role has its own key "<role>@0" (layer 0 has no previous expert rows, so its qkv is not folded).

SELECTION = "selection"
SELECTION_KINDS = ("dsl", "imported")       # index = the value in etx_sel_kind
LAYER0 = "@0"


def kind_of(export: str) -> str:
    """dsl: a tile that etx.ops emits. imported: a kernel that etx.importer imports from vLLM."""
    return "dsl" if export.startswith("dsl_") else "imported"


def phase_key(role: str, layer: int, phases) -> str:
    """The key that a plan instance looks up. The C++ twin is etx_sel_find_instance() in etx/selection.h."""
    return role + LAYER0 if layer == 0 and role + LAYER0 in phases else role


def selection_of(rows, budget: dict) -> dict:
    """The selection that a list of plan instances (role, layer, export) carries, in the manifest format.

    Raises ValueError when one phase has two exports outside the layer-0 rule: such a plan has no single
    selection to publish."""
    by_role: dict[str, dict[bool, set[str]]] = {}
    for role, layer, export in rows:
        by_role.setdefault(role, {}).setdefault(layer == 0, set()).add(export)
    phases: dict[str, tuple[str, str]] = {}
    for role, split in by_role.items():
        first, rest = split.get(True, set()), split.get(False, set())
        for name, exports in ((role + LAYER0, first), (role, rest)):
            if len(exports) > 1:
                raise ValueError(f"phase {name}: more than one export {sorted(exports)}")
        if rest:
            (e,) = rest
            phases[role] = (e, kind_of(e))
            if first and first != rest:
                (e0,) = first
                phases[role + LAYER0] = (e0, kind_of(e0))
        else:
            (e0,) = first
            phases[role] = (e0, kind_of(e0))
    for key in ("threads", "vgpr", "lds"):
        if int(budget[key]) <= 0:
            raise ValueError(f"budget {key} must be positive")
    return {"budget": {k: int(budget[k]) for k in ("threads", "vgpr", "lds")}, "phases": dict(sorted(phases.items()))}


def selection_block(selection: dict | None) -> list[str]:
    """plan_data.h only (host translation unit). Empty when the plan publishes no selection."""
    if not selection:
        return []
    phases = selection["phases"]
    budget = selection["budget"]
    for phase, (export, kind) in phases.items():
        if kind not in SELECTION_KINDS:
            raise ValueError(f"phase {phase}: kind {kind!r} is not one of {SELECTION_KINDS}")
        if not phase or not export or '"' in phase + export:
            raise ValueError(f"phase {phase!r}: bad name or export {export!r}")
    n = len(phases)
    out = ["", "// ---- variant selection (etx/codegen/manifest.py): the planner's choice per phase. The host reads it",
           "// here only (etx/selection.h). It is not a command-line decision."]
    out.append("#ifdef ETX_SEL_N_PHASES")
    out.append('#error "ETX_SEL_N_PHASES must not be defined on the command line: the plan publishes the selection '
               '(etx/codegen/manifest.py)."')
    out.append("#endif")
    out.append(f"#define ETX_SEL_N_PHASES {n}")
    out.append(f"#define ETX_SEL_THREADS {budget['threads']}")
    out.append(f"#define ETX_SEL_VGPR {budget['vgpr']}")
    out.append(f"#define ETX_SEL_LDS {budget['lds']}")
    dims = max(1, n)
    names = ", ".join(f'"{p}"' for p in phases) or '""'
    exports = ", ".join(f'"{e}"' for e, _ in phases.values()) or '""'
    kinds = ", ".join(str(SELECTION_KINDS.index(k)) for _, k in phases.values()) or "0"
    out.append(f"static const char* const etx_sel_phase[{dims}] = {{{names}}};")
    out.append(f"static const char* const etx_sel_export[{dims}] = {{{exports}}};")
    out.append(f"static const int32_t etx_sel_kind[{dims}] = {{{kinds}}};   // 0 dsl, 1 imported")
    out.append("")
    return out
