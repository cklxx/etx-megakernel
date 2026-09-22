"""Pass 7: cross-barrier weight prefetch (lever 3).

Weights do not depend on any event.  For every worker whose next static task
is known, the current task's epilogue can start streaming the next task's
weight slice while the worker will otherwise spin on an event.  Where the
bytes land is a capability of the machine (TMA / wide direct-to-LDS / L2
warm-up only); whether it is worth it is a cost decision against the
occupancy the prefetch buffer costs (fleet-mi300x measured -4% when added
without re-planning resources).
"""
from __future__ import annotations

from .plan import Plan


def _method(plan: Plan) -> str | None:
    caps = plan.machine.capabilities
    if caps.get("tma"):
        return "tma"
    ac = caps.get("async_copy_to_lds", "none")
    if ac == "wide":
        return "lds"
    if ac == "dword_only":
        return "l2_warm"
    return None


def run(plan: Plan) -> None:
    plan.prefetch.clear()
    if not plan.options.prefetch:
        plan.say("P7: prefetch disabled by options")
        return
    method = _method(plan)
    if method is None:
        plan.say(f"P7: no async copy capability on {plan.machine.name}; prefetch skipped")
        return
    lds_budget = plan.machine.resources["lds_kb"] * 1024
    skipped = 0
    for (dev, w), q in plan.static_queues.items():
        for a, b in zip(q, q[1:]):
            gb = plan.graph.grid(plan.tasks[b].grid)
            if not gb.weight_args:
                continue
            need = gb.resource.lds_bytes + gb.resource.prefetch_bytes
            if method == "lds" and need > lds_budget:
                skipped += 1
                continue
            plan.prefetch.append({"device": dev, "worker": w, "after": a, "task": b,
                                  "weights": list(gb.weight_args), "method": method})
    plan.say(f"P7: {len(plan.prefetch)} prefetch entries via {method}; {skipped} skipped for LDS budget")
