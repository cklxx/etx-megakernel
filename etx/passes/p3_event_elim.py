"""Pass 3: event elimination by recomputation (lever 1, design §05 / §09).

An event exists because a producer's output is consumed by many tasks.  When
the producer is a pure function of inputs that are already visible to the
consumers, and it is cheap, every consumer recomputes it in its prologue and
the event plus the producer task disappear.  Redundant parallel microseconds
buy back serial synchronisation microseconds.

Rewrite (conservative):
  * consumer.prologue += producer (and the producer's own prologue chain)
  * consumer drops the eliminated in-edge and inherits the producer's in-edges
    with the "*" (all coordinates) map, which waits for everything the
    producer would have waited for -- correct for any coordinate mapping
  * the producer grid and the event are removed from the graph
The caller re-instantiates and re-runs Pass 2 afterwards.
"""
from __future__ import annotations

from ..ir.edgemap import EdgeMap
from ..ir.types import Graph, Scope, TaskGrid
from .plan import Plan


def _consumers_only_use(graph: Graph, p: TaskGrid, consumers: set[str]) -> bool:
    for t in p.writes:
        for r in graph.grids:
            if r.name == p.name:
                continue
            if (t in r.reads or t in r.args) and r.name not in consumers:
                return False
        for r in graph.grids:
            for m in list(r.in_edges.values()) + list(r.out_edges.values()):
                if t in m.runtime:
                    return False        # runtime tensors must be produced exactly once
    return True


def run(plan: Plan) -> bool:
    """Returns True if the graph changed."""
    graph, m, inst = plan.graph, plan.machine, plan.inst
    changed = False
    total_workers = plan.workers_per_device()
    for name in list(graph.events):
        if name not in graph.events:
            continue
        prods = graph.producers_of(name)
        cons = graph.consumers_of(name)
        if len(prods) != 1 or not cons:
            continue
        p = prods[0]
        if not p.pure or p.out_to_in is None:
            continue
        if any(c.device != p.device for c in cons):
            plan.say(f"P3: keep {name}: consumers span devices")
            continue
        if not _consumers_only_use(graph, p, {c.name for c in cons}):
            plan.say(f"P3: keep {name}: {p.name} output used outside its consumers or referenced by a runtime map")
            continue
        n_p = len(inst.tasks[p.name])
        ep = plan.events[name]
        gain = m.t_sync_us(ep.scope) + p.duration_us * (1.0 - min(1.0, n_p / max(1, total_workers)))
        cost = p.duration_us
        weight_reads = [t for t in p.reads if graph.tensors.get(t) and graph.tensors[t].role == "weight"]
        if weight_reads:
            n_cons = sum(len(inst.tasks[c.name]) for c in cons)
            per_domain = n_cons / max(1, plan.n_domains)
            l2_bytes = m.leaf_domain.l2_mb * 1e6
            bytes_tile = float(p.bytes_per_tile if isinstance(p.bytes_per_tile, int) else 0)
            if bytes_tile * per_domain > l2_bytes:
                cost += bytes_tile * n_cons / m.hbm_bw_bytes_per_us() / max(1, total_workers)
                plan.say(f"P3: {name}: recompute would re-read {bytes_tile * n_cons / 1e6:.1f} MB of weights from HBM")
        if gain <= cost:
            plan.say(f"P3: keep {name}: gain {gain:.2f} us <= recompute {cost:.2f} us")
            continue
        # rewrite ---------------------------------------------------------------
        plan.inlined[p.name] = p
        for c in cons:
            c.prologue = c.prologue + p.prologue + [(p.name, p.out_to_in)]
            del c.in_edges[name]
            for ev in p.in_edges:
                if ev not in c.in_edges:
                    c.in_edges[ev] = EdgeMap.parse("*")
            for t in p.reads:
                if t not in c.reads:
                    c.reads.append(t)
            for t in p.args:
                if t not in c.args:
                    c.args.append(t)
            c.duration_us += cost
        graph.grids = [g for g in graph.grids if g.name != p.name]
        del graph.events[name]
        for t in p.writes:
            if t in graph.tensors:
                graph.tensors[t].role = "recomputed"
        plan.eliminated_events.append(name)
        plan.say(f"P3: eliminate {name}: {p.name} inlined into {[c.name for c in cons]} (gain {gain:.2f} us > recompute {cost:.2f} us; consumers inherit {list(p.in_edges)} with '*')")
        changed = True
    return changed
