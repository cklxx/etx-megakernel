"""STUB of etx.select: the interface of docs/SEC7-interface-fix.md sections 3 to 5, and nothing more.

Items 1 and 2 (Grok, branch grok/sec7-variants-planner) own the real code. The item 4 and item 5 tests load the
real package when it exists (`load()`) and this stub only when it does not. Delete this file when etx.select lands.
The stub has no REGISTRY entries: the tests give their own variants through `variants=`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping


@dataclass(frozen=True)
class Budget:
    threads: int
    vgpr: int
    lds: int


@dataclass(frozen=True)
class Resources:
    vgpr: int
    lds: int
    threads: int = 0


@dataclass(frozen=True)
class Variant:
    export: str
    phase: str
    kind: str
    backend: tuple
    dtype: str
    shape: Callable
    resources: Resources
    cost: Callable
    params: Mapping = field(default_factory=dict)


class NoLegalVariant(ValueError):
    pass


DEFAULT_BUDGETS = (Budget(1024, 128, 65536),)
REGISTRY: list = []


@dataclass(frozen=True)
class Plan:
    budget: Budget
    phases: Mapping
    cost: float

    def manifest(self) -> dict:
        return {"budget": {"threads": self.budget.threads, "vgpr": self.budget.vgpr, "lds": self.budget.lds},
                "phases": {p: (v.export, v.kind) for p, v in sorted(self.phases.items())}}


def _dims(model_cfg):
    if isinstance(model_cfg, (str, Path)):
        from examples.llm.model import load_dims
        return load_dims(str(model_cfg))
    return model_cfg


def phase_shapes(model_cfg) -> dict:
    d = _dims(model_cfg)
    base = {k: int(getattr(d, k)) for k in ("H", "NH", "NKV", "HD", "E", "TOPK", "MI", "INTER", "V")}
    base.update(G=d.G, qk_norm=int(d.qk_norm))
    gemv = {"qkv": (d.NQKV, d.H), "o_proj": (d.H, d.NH * d.HD), "lm_head": (d.V, d.H)}
    if d.moe:
        gemv["router"] = (d.E, d.H)
        out = {p: dict(base, M=m, K=k) for p, (m, k) in gemv.items()}
        out["topk"] = dict(base)
        out["moe_up"] = dict(base, M=2 * d.MI, K=d.H)
        out["moe_down"] = dict(base, M=d.H, K=d.MI)
        return out
    gemv.update(gate_up=(2 * d.INTER, d.H), down=(d.H, d.INTER))
    return {p: dict(base, M=m, K=k) for p, (m, k) in gemv.items()}


def _legal(v, shape, budget, machine) -> bool:
    r = v.resources
    return (machine in v.backend and bool(v.shape(shape)) and r.vgpr <= budget.vgpr and r.lds <= budget.lds
            and r.threads in (0, budget.threads))


def _at(shapes, variants, budget, machine):
    phases, total = {}, 0.0
    for phase, shape in shapes.items():
        legal = [v for v in variants if v.phase == phase and _legal(v, shape, budget, machine)]
        if not legal:
            return None, phase
        order = {id(v): n for n, v in enumerate(legal)}
        best = min(legal, key=lambda v: (v.cost(shape, budget), v.kind != "dsl", order[id(v)]))
        phases[phase] = best
        total += best.cost(shape, budget)
    return Plan(budget, phases, total), None


def plan(model_cfg, machine, budget, *, variants=None) -> Plan:
    name = machine if isinstance(machine, str) else machine.name
    levels = (budget,) if isinstance(budget, Budget) else tuple(budget or DEFAULT_BUDGETS)
    variants = list(REGISTRY if variants is None else variants)
    shapes = phase_shapes(model_cfg)
    best, missing = None, []
    for level in levels:
        p, phase = _at(shapes, variants, level, name)
        if p is None:
            missing.append(f"{phase} at {level}")
        elif best is None or p.cost < best.cost:
            best = p
    if best is None:
        raise NoLegalVariant("no legal variant: " + "; ".join(missing))
    return best


def load():
    """The real etx.select when it provides `plan`, else this stub. Returns (module, is_stub).

    A bare `import etx.select` is not the test: a leftover etx/select/ directory (for example only __pycache__)
    imports as an empty namespace package. Only a module with `plan` and the contract types counts as real."""
    import importlib

    try:
        real = importlib.import_module("etx.select")
    except ImportError:
        real = None
    names = ("plan", "Plan", "Budget", "Resources", "Variant", "NoLegalVariant")
    if real is not None and getattr(real, "__file__", None) and all(hasattr(real, n) for n in names):
        return real, False
    import tests.select_stub as stub
    return stub, True
