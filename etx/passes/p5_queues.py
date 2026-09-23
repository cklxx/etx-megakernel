"""Pass 5: queue layout and static-queue materialisation.

Static tasks: one pre-ordered queue per worker, round-robin inside the task's
domain, in program order (so producers always precede consumers).
Dynamic / hybrid tasks: sized local queues per domain; the global queue only
exists for pure-dynamic tasks whose consumers cross domains.  Consumer lists
per event coordinate are materialised for pushes (MPK-style ranges would be
the in-kernel alternative; kept host-side in this version).
"""
from __future__ import annotations

from ..ir.types import Scope
from .plan import Plan


def _decide_relay(plan: Plan) -> None:
    """Per-domain relay for DEVICE-scope events (fleet's scheduler mirror, generalised).

    Without it every consumer polls the one global counter word: on MI300X 296
    workers polling one line cost ~2 us per phase transition on the full model.
    With it, one workgroup per domain polls the global words and copies changes
    into a domain-local mirror that the domain's consumers poll, and it performs
    the domain-level half of the acquire (the L2 invalidate) once per domain, so
    consumers only drop their own L1.  Costs one workgroup per domain.
    """
    plan.relay = False
    plan.relay_words = {}
    opt = plan.options.relay
    if opt == "off" or plan.n_domains <= 1:
        return
    threshold = int(plan.machine.costs.get("relay_min_pollers") or 64)
    worst = 0
    words: dict[int, list[int]] = {}
    for ev, ep in plan.events.items():
        if ep.scope != Scope.DEVICE:
            continue
        for (e, c), cons in plan.inst.consumers.items():
            if e != ev or not cons:
                continue
            worst = max(worst, len(cons))
            for d in sorted({plan.tasks[plan.task_index[t]].device for t in cons}):
                words.setdefault(d, []).append(ep.offset + plan.linear(ev, c))
    if not words:
        return
    if opt == "auto" and worst < threshold:
        plan.say(f"P5: no relay: at most {worst} consumers poll one DEVICE-scope word (< {threshold})")
        return
    plan.relay = True
    plan.relay_words = {d: sorted(set(w)) for d, w in words.items()}
    plan.workers_per_domain -= 1
    plan.say(f"P5: relay on: up to {worst} consumers poll one DEVICE-scope word (>= {threshold}); one workgroup per domain "
             f"mirrors {max(len(w) for w in plan.relay_words.values())} words, workers/domain -> {plan.workers_per_domain}")


def run(plan: Plan) -> None:
    _decide_relay(plan)
    wpd = plan.workers_per_domain
    rr: dict[tuple[int, int], int] = {}
    plan.static_queues.clear()
    for d in range(plan.n_devices):
        for w in range(plan.workers_per_device()):
            plan.static_queues[(d, w)] = []
    local_count: dict[tuple[int, int], int] = {}
    global_count: dict[int, int] = {}
    from ..ir.edgemap import EdgeMap
    pins = {g.name: EdgeMap.parse(g.worker_map) for g in plan.graph.grids if g.worker_map}
    for t in plan.tasks:                    # program order
        key = (t.device, t.domain)
        if t.mode == "static":
            if t.grid in pins:              # explicit slot: (xcd, w) -> worker w of its domain
                slot = pins[t.grid].targets(t.coord, (wpd,), None, plan.bindings)[0][0] % wpd
                w = t.domain * wpd + slot
            else:
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
