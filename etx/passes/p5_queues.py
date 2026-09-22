"""Pass 5: queue layout and static-queue materialisation.

Static tasks: one pre-ordered queue per worker, round-robin inside the task's
domain, in program order (so producers always precede consumers).
Dynamic / hybrid tasks: sized local queues per domain; the global queue only
exists for pure-dynamic tasks whose consumers cross domains.  Consumer lists
per event coordinate are materialised for pushes (MPK-style ranges would be
the in-kernel alternative; kept host-side in this version).
"""
from __future__ import annotations

from .plan import Plan


def run(plan: Plan) -> None:
    wpd = plan.workers_per_domain
    rr: dict[tuple[int, int], int] = {}
    plan.static_queues.clear()
    for d in range(plan.n_devices):
        for w in range(plan.workers_per_device()):
            plan.static_queues[(d, w)] = []
    local_count: dict[tuple[int, int], int] = {}
    global_count: dict[int, int] = {}
    for t in plan.tasks:                    # program order
        key = (t.device, t.domain)
        if t.mode == "static":
            k = rr.get(key, 0)
            w = t.domain * wpd + (k % wpd)
            rr[key] = k + 1
            t.worker = w
            plan.static_queues[(t.device, w)].append(t.id)
        else:
            local_count[key] = local_count.get(key, 0) + 1
            if t.mode == "dynamic":
                global_count[t.device] = global_count.get(t.device, 0) + 1
    for key, n in local_count.items():
        plan.local_queue_capacity[key] = n + 16
    for d in range(plan.n_devices):
        plan.global_queue_capacity[d] = global_count.get(d, 0) + 16
    # consumer lists for pushes
    plan.ev_consumers.clear()
    for ev_coord, cons in plan.inst.consumers.items():
        dyn = [plan.task_index[c] for c in cons if plan.tasks[plan.task_index[c]].mode != "static"]
        if dyn:
            plan.ev_consumers[(ev_coord[0], plan.linear(ev_coord[0], ev_coord[1]))] = dyn
    n_static = sum(len(q) for q in plan.static_queues.values())
    plan.say(f"P5: static tasks={n_static} in {len(plan.static_queues)} worker queues; local queue capacities={dict(plan.local_queue_capacity)}; global={dict(plan.global_queue_capacity)}; push lists={len(plan.ev_consumers)}")
