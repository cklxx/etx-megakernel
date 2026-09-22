"""Pipeline: verify -> P1 -> P2 -> P3 (-> P2 again) -> P4 -> P5 -> P6 -> P7 -> verify_plan."""
from __future__ import annotations

import copy
from typing import Any, Mapping

from ..ir.instantiate import instantiate
from ..ir.types import Graph
from ..ir.verify import verify
from ..machine.model import MachineModel, load_machine
from . import p1_tiling, p2_affinity, p3_event_elim, p4_schedule_mode, p5_queues, p6_memory, p7_prefetch
from .plan import PassOptions, Plan, verify_plan


def compile_graph(graph: Graph, machine: MachineModel | str, bindings: Mapping[str, int],
                  runtime: Mapping[str, Any] | None = None, options: PassOptions | None = None) -> Plan:
    m = load_machine(machine) if isinstance(machine, str) else machine
    options = options or PassOptions()
    g = copy.deepcopy(graph)                 # passes rewrite the graph; keep the caller's intact
    verify(g, bindings, runtime)
    plan = Plan(arch=m.name, graph=g, inst=instantiate(g, bindings, runtime), machine=m,
                bindings=dict(bindings), options=options, runtime=runtime)
    p1_tiling.run(plan)
    p2_affinity.run(plan)
    if options.event_elimination:
        for _ in range(8):
            if not p3_event_elim.run(plan):
                break
            verify(g, bindings, runtime)
            plan.inst = instantiate(g, bindings, runtime)
            plan.events.clear()
            p2_affinity.run(plan)
    p4_schedule_mode.run(plan)
    p5_queues.run(plan)
    p6_memory.run(plan)
    p7_prefetch.run(plan)
    errors = verify_plan(plan)
    if errors:
        raise RuntimeError("plan verification failed:\n  " + "\n  ".join(errors))
    for n in m.notes:
        plan.say(f"note: {n}")
    return plan
