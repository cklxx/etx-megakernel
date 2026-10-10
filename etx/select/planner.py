"""The joint selection of docs/PLAN-generic-decoder-2026-10-08.md section 6.2 (section 7 item 2).

The megakernel uses one VGPR count, one LDS size and one workgroup width for all tiles: each is the maximum over the
selected tiles. The constraint is a maximum, so the planner solves it exactly by enumeration:
  1. take a list of budget levels (workgroup size, VGPR limit, LDS limit);
  2. at each level, give each phase its legal variant with the smallest cost;
  3. add the phase costs of the level;
  4. keep the level with the smallest total (ties: the earlier level).
The planner does not read the environment. Contract: docs/SEC7-interface-fix.md, docs/SEC7-interface-grok.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .variants import Budget, Structure, Variant, REGISTRY, load_cfg, phase_shapes

DEFAULT_BUDGETS: tuple[Budget, ...] = (Budget(1024, 128, 65536),)
# Section 6.2: the workgroup size is a budget variable. 512 threads double the VGPR limit and halve the waves per CU;
# only a GPU measurement decides it, so this sweep is an option, not the default.
BUDGET_LEVELS: tuple[Budget, ...] = (Budget(1024, 128, 65536), Budget(512, 256, 65536))


class NoLegalVariant(ValueError):
    """No budget level has a legal variant for every phase. The message names the phase and the budget."""


@dataclass(frozen=True)
class Plan:
    budget: Budget                         # the selected budget level
    phases: Mapping[str, Variant]          # phase key -> variant, in launch order
    cost: float                            # sum of the selected costs, us per phase instance
    machine: str = "gfx942"
    reasons: Mapping[str, str] = field(default_factory=dict)   # phase key -> why this variant

    def manifest(self) -> dict:
        """The value that item 6 puts into Graph.codegen["selection"] (docs/SEC7-interface-fix.md section 6)."""
        b = self.budget
        return {"budget": {"threads": b.threads, "vgpr": b.vgpr, "lds": b.lds},
                "phases": {k: (v.export, v.kind) for k, v in sorted(self.phases.items())}}   # sorted, as selection_of

    def resources(self) -> dict:
        """What the megakernel must declare: the maximum over the selected variants (section 6.2)."""
        vs = list(self.phases.values())
        return {"vgpr": max(v.resources.vgpr for v in vs), "lds": max(v.resources.lds for v in vs),
                "threads": self.budget.threads}

    def exports(self, kind: str | None = None) -> list[str]:
        """The distinct exports of the plan, sorted; `kind="imported"` gives build.sh its import list."""
        return sorted({v.export for v in self.phases.values() if kind is None or v.kind == kind})


def _machine_name(machine) -> str:
    return machine if isinstance(machine, str) else getattr(machine, "name")


def _select_level(shapes, machine, budget, variants):
    """Step 2 at one level: phase -> (variant, cost, reason), or (None, failing phase)."""
    chosen = {}
    for phase, s in shapes.items():
        cands = [v for v in variants if v.phase == phase]
        legal = [(v.cost(s, budget), {"dsl": 0, "imported": 1, "generic": 2}[v.kind], i, v)
                 for i, v in enumerate(cands) if v.legal(machine, s, budget)]
        if not legal:
            return None, phase
        # Generic is a last resort even if a future cost model makes it cheaper.
        specialized = [t for t in legal if t[3].kind != "generic"]
        legal = specialized or legal
        c, _, _, v = min(legal, key=lambda t: t[:3])          # cost, then dsl first, then declaration order
        dsl_ok = any(t[3].kind == "dsl" for t in legal)
        dsl_any = any(x.kind == "dsl" for x in cands)
        if v.kind == "imported" and dsl_any and not dsl_ok:
            why = "fallback: no DSL variant is legal for this shape and budget"
        elif len(legal) == 1:
            why = f"only legal variant ({c:.2f} us)"
        else:
            why = f"cheapest of {len(legal)} legal variants ({c:.2f} us; ties: dsl, imported, generic, then declaration order)"
        chosen[phase] = (v, c, why)
    return chosen, None


def plan(model_cfg, machine="gfx942", budget: Budget | Sequence[Budget] | None = None, *,
         variants: Sequence[Variant] | None = None, structure: Structure | None = None) -> Plan:
    """Select one variant for each phase of the model (section 6.2).

    model_cfg: examples.llm.model.Dims, a path to config.json, or the config dict. The model config is the only
               model input; the selection reads nothing else.
    machine:   a name of etx/machine/arch/ ("gfx942", "sm_90") or a MachineModel.
    budget:    one Budget (that level only), a sequence (steps 1-4), or None (DEFAULT_BUDGETS).
    variants:  None = REGISTRY. Tests pass their own list; only the phases it declares are planned.
    structure: the fold / attention-prep choices that fix the phase set; None = default_structure(cfg).
    """
    m = _machine_name(machine)
    levels = DEFAULT_BUDGETS if budget is None else ((budget,) if isinstance(budget, Budget) else tuple(budget))
    if not levels:
        raise ValueError("plan: empty budget list")
    vs = list(REGISTRY if variants is None else variants)
    # The phases planned are the model's phases that `variants` declares. REGISTRY declares every phase of every
    # supported config (tests/test_select_planner.py checks it); a test list may declare only the phases it tests.
    declared = {v.phase for v in vs}
    shapes = {k: s for k, s in phase_shapes(load_cfg(model_cfg), structure).items() if k in declared}
    best, fails = None, []
    for b in levels:
        chosen, bad = _select_level(shapes, m, b, vs)
        if chosen is None:
            fails.append(f"phase {bad!r} at {b}")
            continue
        total = sum(c for _, c, _ in chosen.values())
        if best is None or total < best[0]:                   # strict: ties keep the earlier level
            best = (total, b, chosen)
    if best is None:
        raise NoLegalVariant(f"no legal variant on {m}: " + "; ".join(fails))
    total, b, chosen = best
    return Plan(budget=b, phases={k: v for k, (v, _, _) in chosen.items()}, cost=total, machine=m,
                reasons={k: r for k, (_, _, r) in chosen.items()})
