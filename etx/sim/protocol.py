"""Host-side protocol simulator.

Executes a Plan as the generated kernel would: every worker runs the same
loop (static head if ready, else pop from its local / the global queue, else
spin on the static head), events are counters with the machine's signalling
latency, pushes cost the pusher.  Used for three things:

  * deadlock detection before anything touches a GPU (which events never
    reached zero, which tasks were blocked and why)
  * makespan and per-worker timelines for the cost model (static vs dynamic
    vs hybrid on this topology)
  * the critical-path budget: time in tasks vs time waiting vs scheduling

It is a model, not a cycle simulator: durations come from the TileOp
annotations (duration_us, duration_cv), signalling from the cost table.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..ir.types import Scope
from ..passes.plan import Plan

INF = float("inf")


@dataclass
class Segment:
    device: int
    worker: int
    start: float
    end: float
    kind: str           # task | wait | sched
    task: int = -1


@dataclass
class SimResult:
    makespan_us: float
    segments: list[Segment]
    deadlock: bool
    blocked: list[str]
    busy_us: float
    wait_us: float
    sched_us: float
    n_workers: int
    per_grid_us: dict[str, float] = field(default_factory=dict)

    @property
    def utilisation(self) -> float:
        return self.busy_us / max(1e-9, self.makespan_us * self.n_workers)

    def summary(self) -> str:
        s = (f"makespan {self.makespan_us:.1f} us | busy {self.busy_us:.1f} wait {self.wait_us:.1f} sched {self.sched_us:.1f} "
             f"| utilisation {self.utilisation * 100:.1f}% over {self.n_workers} workers")
        if self.deadlock:
            s += "\nDEADLOCK: " + "; ".join(self.blocked[:8])
        return s


def simulate(plan: Plan, seed: int = 0, cv_scale: float = 1.0) -> SimResult:
    rng = random.Random(seed)
    inst = plan.inst
    m = plan.machine
    wpdev = plan.workers_per_device()
    workers = [(d, w) for d in range(plan.n_devices) for w in range(wpdev)]
    free = {k: 0.0 for k in workers}
    sptr = {k: 0 for k in workers}
    squeue = {k: list(plan.static_queues.get(k, [])) for k in workers}
    local_q: dict[tuple[int, int], list[tuple[int, float]]] = {}      # (device, domain) -> [(task, ready_at)]
    global_q: dict[int, list[tuple[int, float]]] = {d: [] for d in range(plan.n_devices)}
    done: dict[int, float] = {}
    remaining_local: dict[int, int] = {}
    remaining_all: dict[int, int] = {}
    ev_ready: dict[tuple[str, tuple[int, ...]], float] = {}
    ev_pending: dict[tuple[str, tuple[int, ...]], int] = {}
    segs: list[Segment] = []

    # per-task durations
    dur: dict[int, float] = {}
    for t in plan.tasks:
        g = plan.graph.grid(t.grid)
        cv = g.duration_cv * cv_scale
        f = max(0.05, 1.0 + rng.gauss(0.0, cv)) if cv > 0 else 1.0
        dur[t.id] = t.duration_us * f

    for ev, prods in inst.producers.items():
        ev_pending[ev] = len(prods)
        if not prods:
            ev_ready[ev] = 0.0
    for t in plan.tasks:
        tid = (t.grid, t.coord)
        prods_all = {plan.task_index[p] for e in inst.task_in[tid] for p in inst.producers[e]}
        prods_local = {p for p in prods_all if plan.tasks[p].domain == t.domain and plan.tasks[p].device == t.device}
        remaining_all[t.id] = len(prods_all)
        remaining_local[t.id] = len(prods_local) if t.mode == "hybrid" else len(prods_all)
        if t.mode != "static" and remaining_local[t.id] == 0:
            _push(plan, local_q, global_q, t, 0.0)

    def deps_ready(task_id: int) -> float:
        t = plan.tasks[task_id]
        r = 0.0
        for ev in inst.task_in[(t.grid, t.coord)]:
            if ev not in ev_ready:
                return INF
            r = max(r, ev_ready[ev])
        return r

    total = len(plan.tasks)
    n_done = 0
    pop_cost = m.t_pop_us()
    while n_done < total:
        best = None      # (start, key, task, source, popcost)
        for key in workers:
            d, w = key
            dom = plan.worker_domain(w)
            # static head
            if sptr[key] < len(squeue[key]):
                tid = squeue[key][sptr[key]]
                r = deps_ready(tid)
                if r < INF:
                    s = max(free[key], r)
                    if best is None or s < best[0]:
                        best = (s, key, tid, "static", 0.0)
            # local queue (hybrid + dynamic-local)
            q = local_q.get((d, dom), [])
            for i, (tid, at) in enumerate(q):
                r = deps_ready(tid) if plan.tasks[tid].mode == "hybrid" else 0.0
                if r == INF:
                    continue
                s = max(free[key], at, r)
                c = pop_cost * (1.0 + 0.01 * plan.workers_per_domain)
                if best is None or s + c < best[0]:
                    best = (s + c, key, tid, ("local", i), c)
            for i, (tid, at) in enumerate(global_q[d]):
                s = max(free[key], at)
                c = pop_cost * (1.0 + 0.01 * wpdev)
                if best is None or s + c < best[0]:
                    best = (s + c, key, tid, ("global", i), c)
        if best is None:
            blocked = []
            for key in workers:
                if sptr[key] < len(squeue[key]):
                    tid = squeue[key][sptr[key]]
                    t = plan.tasks[tid]
                    missing = [f"{e[0]}{list(e[1])}(pending {ev_pending[e]})" for e in inst.task_in[(t.grid, t.coord)] if e not in ev_ready]
                    blocked.append(f"worker {key} head {t.grid}{list(t.coord)} waits on {missing}")
            for (d, dom), q in local_q.items():
                for tid, _ in q:
                    t = plan.tasks[tid]
                    missing = [f"{e[0]}{list(e[1])}" for e in inst.task_in[(t.grid, t.coord)] if e not in ev_ready]
                    blocked.append(f"local queue {(d, dom)} task {t.grid}{list(t.coord)} waits on {missing}")
            not_done = [t for t in plan.tasks if t.id not in done and t.mode != "static"]
            never = [f"{t.grid}{list(t.coord)} never pushed (remaining {remaining_local[t.id]})" for t in not_done
                     if all(t.id != x for q in local_q.values() for x, _ in q) and all(t.id != x for q in global_q.values() for x, _ in q)]
            blocked.extend(never[:8])
            return _result(plan, segs, INF, True, blocked, workers)
        start, key, tid, source, c = best
        d, w = key
        t = plan.tasks[tid]
        if source == "static":
            sptr[key] += 1
        elif source[0] == "local":
            local_q[(d, plan.worker_domain(w))].pop(source[1])
        else:
            global_q[d].pop(source[1])
        if c > 0:
            segs.append(Segment(d, w, start - c, start, "sched", tid))
        if start - c > free[key]:
            segs.append(Segment(d, w, free[key], start - c, "wait", tid))
        end = start + dur[tid]
        segs.append(Segment(d, w, start, end, "task", tid))
        free[key] = end
        done[tid] = end
        n_done += 1
        # arrive on out events
        for ev in inst.task_out[(t.grid, t.coord)]:
            ev_pending[ev] -= 1
            if ev_pending[ev] == 0:
                scope = plan.events[ev[0]].scope
                ev_ready[ev] = end + m.t_sync_us(scope)
                if ev in ev_ready and plan.events[ev[0]].scope == Scope.SYSTEM:
                    pass
        # pushes for dynamic consumers
        for ev in inst.task_out[(t.grid, t.coord)]:
            for c_tid in inst.consumers[ev]:
                cid = plan.task_index[c_tid]
                ct = plan.tasks[cid]
                if ct.mode == "static":
                    continue
                same_dom = (ct.device == t.device and ct.domain == t.domain)
                if ct.mode == "hybrid" and not same_dom:
                    continue            # hybrid: cross-domain producers are waited on, not pushed
                remaining_local[cid] -= 1
                if remaining_local[cid] == 0:
                    cross = not same_dom
                    pc = m.t_push_us(cross_domain=cross)
                    segs.append(Segment(d, w, free[key], free[key] + pc, "sched", cid))
                    free[key] += pc
                    _push(plan, local_q, global_q, ct, free[key])
    makespan = max(free.values())
    return _result(plan, segs, makespan, False, [], workers)


def _push(plan: Plan, local_q, global_q, t, at: float) -> None:
    if t.mode == "dynamic":
        global_q[t.device].append((t.id, at))
    else:
        local_q.setdefault((t.device, t.domain), []).append((t.id, at))


def _result(plan: Plan, segs: list[Segment], makespan: float, deadlock: bool, blocked: list[str], workers) -> SimResult:
    busy = sum(s.end - s.start for s in segs if s.kind == "task")
    wait = sum(s.end - s.start for s in segs if s.kind == "wait")
    sched = sum(s.end - s.start for s in segs if s.kind == "sched")
    per_grid: dict[str, float] = {}
    for s in segs:
        if s.kind == "task":
            g = plan.tasks[s.task].grid
            per_grid[g] = per_grid.get(g, 0.0) + (s.end - s.start)
    return SimResult(makespan_us=makespan, segments=segs, deadlock=deadlock, blocked=blocked,
                     busy_us=busy, wait_us=wait, sched_us=sched, n_workers=len(workers), per_grid_us=per_grid)
