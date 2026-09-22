"""Pass 6: memory planning and event allocation by scope.

Event memory type comes from the machine's visibility table for the event's
scope; an illegal combination (system scope on coarse-grained memory on AMD)
is a compile error here, never a silent downgrade at runtime.  Intermediate
activations get a placement hint: domain-local when every reader is in the
producer's domain, so the target's L2 can hold them.
"""
from __future__ import annotations

from ..ir.types import Scope
from .plan import Plan


def run(plan: Plan) -> None:
    m = plan.machine
    for ep in plan.events.values():
        ep.memory = m.memory_for(ep.scope)      # raises if the lowering table lacks the scope
        if ep.scope == Scope.SYSTEM and not m.supports("fp_atomic_over_fabric"):
            plan.say(f"P6: {ep.name}: system scope on {m.name}: integer counters only (fp atomics do not cross the fabric)")
    inst = plan.inst
    for name, t in plan.graph.tensors.items():
        if t.role not in ("activation", "scratch"):
            plan.tensor_placement[name] = "weights_hbm" if t.role == "weight" else t.role
            continue
        readers = plan.graph.readers_of(name)
        writers = [g for g in plan.graph.grids if name in g.writes]
        doms: set[tuple[int, int]] = set()
        for g in readers + writers:
            for c in inst.tasks[g.name]:
                ti = plan.task((g.name, c))
                doms.add((ti.device, ti.domain))
        if len(doms) == 1:
            plan.tensor_placement[name] = "domain_local"
        elif len({d for d, _ in doms}) == 1:
            plan.tensor_placement[name] = "device"
        else:
            plan.tensor_placement[name] = "fine_grained"
    plan.say("P6: events " + ", ".join(f"{e.name}:{e.memory}" for e in plan.events.values()) +
             "; tensors " + ", ".join(f"{k}:{v}" for k, v in plan.tensor_placement.items()))
