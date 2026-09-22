"""L2 verification passes (design §07 6.4).

All six checks must be green before a graph enters L4.  The first four are
purely about dependency semantics and run here; scope monotonicity and
co-residency need placement results and run in etx.passes.plan.verify_plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .instantiate import instantiate, Instance
from .types import Graph, InitKind


@dataclass
class Issue:
    check: str
    message: str
    severity: str = "error"     # error | warning

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.message}"


class VerifyError(Exception):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("\n".join(str(i) for i in issues))


def _reachable(graph: Graph) -> dict[str, set[str]]:
    """grid -> set of grids reachable through events (transitively)."""
    succ: dict[str, set[str]] = {g.name: set() for g in graph.grids}
    for g in graph.grids:
        for ev in g.out_edges:
            for c in graph.consumers_of(ev):
                succ[g.name].add(c.name)
    reach: dict[str, set[str]] = {}
    for g in graph.grids:
        seen: set[str] = set()
        stack = list(succ[g.name])
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(succ[n])
        reach[g.name] = seen
    return reach


def verify(graph: Graph, bindings: Mapping[str, int],
           runtime: Mapping[str, Any] | None = None, strict: bool = True) -> list[Issue]:
    issues: list[Issue] = []
    order = {g.name: i for i, g in enumerate(graph.grids)}

    # 1. acyclic at grid level: an in-edge event must be produced by an earlier grid
    for g in graph.grids:
        for ev in g.in_edges:
            if ev not in graph.events:
                issues.append(Issue("exists", f"grid {g.name!r} waits on unknown event {ev!r}"))
                continue
            prods = graph.producers_of(ev)
            if not prods:
                issues.append(Issue("acyclic", f"event {ev!r} consumed by {g.name!r} has no producer"))
            for p in prods:
                if order[p.name] >= order[g.name]:
                    issues.append(Issue("acyclic", f"grid {g.name!r} waits on {ev!r} produced by later grid {p.name!r} (cycle or forward reference)"))
        for ev in g.out_edges:
            if ev not in graph.events:
                issues.append(Issue("exists", f"grid {g.name!r} notifies unknown event {ev!r}"))
    reach = _reachable(graph)
    for g in graph.grids:
        if g.name in reach[g.name]:
            issues.append(Issue("acyclic", f"grid {g.name!r} reaches itself"))
    if any(i.severity == "error" for i in issues):
        if strict:
            raise VerifyError(issues)
        return issues

    # instantiate for the given bindings (needed by checks 2 and 3)
    try:
        inst: Instance = instantiate(graph, bindings, runtime)
    except KeyError as e:
        issues.append(Issue("instantiate", str(e)))
        if strict:
            raise VerifyError(issues)
        return issues

    # 2. count conservation
    for ev, e in graph.events.items():
        if e.is_runtime_count:
            if e.runtime_init_by is None:
                issues.append(Issue("count", f"event {ev!r} has runtime wait_count but no runtime_init_by grid"))
            elif e.runtime_init_by not in order:
                issues.append(Issue("count", f"event {ev!r} runtime_init_by unknown grid {e.runtime_init_by!r}"))
            else:
                for c in graph.consumers_of(ev):
                    if order[e.runtime_init_by] >= order[c.name]:
                        issues.append(Issue("count", f"event {ev!r} counts written by {e.runtime_init_by!r} after consumer {c.name!r}"))
            if e.runtime_count is not None:
                # the runtime expression must agree with the producers enumerated from the same sample
                bad = [coord for coord in inst.producers if coord[0] == ev and len(inst.producers[coord]) != inst.wait_counts[coord]]
                if bad:
                    c0 = bad[0]
                    issues.append(Issue("count", f"event {ev}{list(c0[1])}: runtime_count gives {inst.wait_counts[c0]} but {len(inst.producers[c0])} producers notify it ({len(bad)} coords)"))
            continue
        bad = 0
        for coord in inst.producers:
            if coord[0] != ev:
                continue
            fan_in = len(inst.producers[coord])
            if fan_in != inst.wait_counts[coord]:
                bad += 1
                if bad <= 3:
                    issues.append(Issue("count", f"event {ev}{list(coord[1])}: wait_count={inst.wait_counts[coord]} but {fan_in} producers notify it"))
        if bad > 3:
            issues.append(Issue("count", f"event {ev!r}: {bad} coordinates with mismatched counts (first 3 shown)"))

    # 3. runtime values are written by an earlier grid on a dependency path
    for g in graph.grids:
        for m in list(g.in_edges.values()) + list(g.out_edges.values()):
            for rt in m.runtime:
                writers = [w for w in graph.grids if rt in w.writes]
                if not writers:
                    issues.append(Issue("runtime-first", f"runtime tensor {rt!r} used by {g.name!r} is written by no grid"))
                    continue
                for w in writers:
                    if order[w.name] >= order[g.name]:
                        issues.append(Issue("runtime-first", f"runtime tensor {rt!r} is written by {w.name!r} after it is used by {g.name!r}"))
                    elif g.name not in reach[w.name]:
                        issues.append(Issue("runtime-first", f"{g.name!r} reads runtime tensor {rt!r} but no event path orders it after writer {w.name!r}"))
    for ev, e in graph.events.items():
        if e.is_runtime_count and e.runtime_init_by in order:
            for c in graph.consumers_of(ev):
                if c.name not in reach[e.runtime_init_by] and c.name != e.runtime_init_by:
                    issues.append(Issue("runtime-first", f"event {ev!r} counts are initialised by {e.runtime_init_by!r} but consumer {c.name!r} is not ordered after it"))

    # 4. unique writer per tensor (tile-level coverage is a frontend obligation)
    writers: dict[str, list[str]] = {}
    for g in graph.grids:
        for t in g.writes:
            writers.setdefault(t, []).append(g.name)
    for t, ws in writers.items():
        if len(ws) > 1:
            issues.append(Issue("coverage", f"tensor {t!r} written by {ws}; each output must have exactly one writer grid"))

    if strict and any(i.severity == "error" for i in issues):
        raise VerifyError(issues)
    return issues
