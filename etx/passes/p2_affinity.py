"""Pass 2: event-affinity partitioning (the chiplet-aware core).

Assign every task instance to an exec domain so that as few edges as possible
cross domains, then give every event the lowest scope that is still correct.
First version: greedy.  Grids without dependencies are chunked contiguously
across domains (consecutive tiles share an L2); dependent tasks follow the
majority domain of their producers unless that domain is already overloaded.
"""
from __future__ import annotations

from collections import Counter

from ..ir.types import Scope
from .plan import EventPlan, Plan, TaskInst


def run(plan: Plan) -> None:
    g_all = plan.graph
    inst = plan.inst
    nd = plan.n_domains
    m = plan.machine
    plan.tasks.clear()
    plan.task_index.clear()
    plan.type_ids = {g.name: i for i, g in enumerate(g_all.grids)}
    for name in plan.inlined:                       # inlined producers keep a type id for their argument table
        plan.type_ids[name] = len(plan.type_ids)
    load: dict[tuple[int, int], int] = Counter()
    dom_of: dict[tuple[str, tuple[int, ...]], tuple[int, int]] = {}

    from ..ir.edgemap import EdgeMap
    wpd = max(1, plan.workers_per_domain)
    for g in g_all.grids:
        coords = inst.tasks[g.name]
        n = max(1, len(coords))
        per_dom = n / nd
        # Per-grid cap. A grid that fits in one wave of workers must not put more tasks on a domain than
        # ceil(n / nd): each extra task there is a second task on some worker and doubles the phase
        # (measured on MI300X, 2026-09-25: domain 0 got 39-49 of ~300 tasks, the fused Qwen3-8B ran
        # 45% slower than the same tiles unfused). Larger grids keep the imbalance tolerance.
        cap = -(-n // nd) if n <= nd * wpd else plan.options.affinity_imbalance * per_dom
        gload: Counter = Counter()
        pin = EdgeMap.parse(g.domain_map) if g.domain_map else None
        for lin, c in enumerate(coords):
            tid = (g.name, c)
            chunk = min(nd - 1, int(lin * nd // n))
            dom = chunk
            if pin is not None:
                dom = pin.targets(c, (nd,), None, inst.bindings)[0][0] % nd      # explicit placement wins
            elif g.in_edges:
                votes: Counter = Counter()
                for ev in inst.task_in[tid]:
                    prods = inst.producers[ev]
                    if len(prods) > wpd:          # a barrier (producers on every domain): no locality to follow
                        continue
                    for p in prods:
                        pd = dom_of.get(p)
                        if pd is not None and pd[0] == g.device:
                            votes[pd[1]] += 1
                for best, _ in votes.most_common():
                    if gload[best] + 1 <= cap:
                        dom = best
                        break
                else:
                    if gload[dom] + 1 > cap:      # the contiguous chunk is full: least-loaded domain of this grid
                        dom = min(range(nd), key=lambda d: (gload[d], d))
            elif gload[dom] + 1 > cap:
                dom = min(range(nd), key=lambda d: (gload[d], d))
            dom_of[tid] = (g.device, dom)
            gload[dom] += 1
            load[(g.device, dom)] += 1
            t = TaskInst(id=len(plan.tasks), grid=g.name, coord=c, type_id=plan.type_ids[g.name],
                         device=g.device, domain=dom, duration_us=g.duration_us)
            plan.task_index[tid] = t.id
            plan.tasks.append(t)

    # event scopes ------------------------------------------------------------
    offset = 0
    for name, e in g_all.events.items():
        shape = inst.event_shapes[name]
        numel = 1
        for d in shape:
            numel *= d
        cross_dom = cross_dev = 0
        scope = Scope.DOMAIN
        counts: list[int] = []
        for coord_key, prods in inst.producers.items():
            if coord_key[0] != name:
                continue
            cons = inst.consumers[coord_key]
            for p in prods:
                for c in cons:
                    pd, cd = dom_of[p], dom_of[c]
                    if pd[0] != cd[0]:
                        cross_dev += 1
                    elif pd[1] != cd[1]:
                        cross_dom += 1
            parties = {dom_of[x] for x in prods + cons}
            devices = {x[0] for x in parties}
            if len(devices) > 1:
                scope = Scope.SYSTEM
            elif len(parties) > 1 and scope < Scope.DEVICE:
                scope = Scope.DEVICE
        # flattened counts in linear order, and per-domain producer shares (last-arriver flush)
        import itertools
        share: dict[tuple[int, int], list[int]] = {}
        for lin, c in enumerate(itertools.product(*[range(d) for d in shape])):
            counts.append(inst.wait_counts[(name, c)])
            for p in inst.producers[(name, c)]:
                key = dom_of[p]
                share.setdefault(key, [0] * numel)[lin] += 1
        eff = m.effective_scope(scope)
        plan.events[name] = EventPlan(name=name, shape=shape, scope=eff, offset=offset, numel=numel,
                                      cross_domain_edges=cross_dom, cross_device_edges=cross_dev,
                                      counts=counts, runtime_init=e.is_runtime_count, share=share)
        e.scope = eff
        offset += numel
    per_domain = Counter(t.domain for t in plan.tasks)
    plan.say(f"P2: tasks per domain {dict(sorted(per_domain.items()))}; events: " +
             ", ".join(f"{n}={ep.scope.name}(x{ep.cross_domain_edges}/{ep.cross_device_edges})" for n, ep in plan.events.items()))
