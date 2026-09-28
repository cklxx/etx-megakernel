"""Bandwidth-aware simulation of a compiled plan on the XCDs of one GPU.

Where etx.sim.protocol gives every task a fixed duration, this one gives it a cost *shape* and lets the tasks that run at
the same time share the memory system, so what one XCD does changes what another gets:

  task = wait (its input events, each fired at its last producer's end + the release, seen after the hand-off of the
         event's scope) -> front (fixed: descriptor and argument round trips) -> chain (fixed: the kernel's dependent
         global round trips, barriers, reductions, arithmetic) -> stream (bytes, fluid: rate-limited per CU and sharing
         the GPU's bandwidth) -> end (fixed: stores, release)

Streams follow Little's law: all of them see one loaded latency L = max(idle round trip, bytes in flight on the GPU /
bandwidth), and each pulls its bytes in flight / (L + the work of a trip that is not overlapped with memory). So a GPU
full of streams is bandwidth-bound, a few streams are latency-bound, and an XCD that streams alone runs faster. Workers run their static queue in order, one task at a time (the generated kernel's loop).

The output has the layout of ETX_TRACE_RAW (type, coord x4, taken, ready, done, worker; 10 ns ticks), so the same tools
read a simulated and a measured schedule: etx.tools.timeline (per-XCD occupancy, critical path, HTML) and
examples/vllm_llm/compare.py ledger.

  from etx.sim.xcd import Cost, Hw, simulate
  rows = simulate(plan_json, cost_of_type, Hw(...))
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Callable

TICK_US = 0.01


@dataclass
class Cost:
    front_us: float = 0.0        # before the kernel's own work: descriptor / argument round trips
    chain_us: float = 0.0        # the kernel's fixed latency: dependent round trips, barriers, reductions, arithmetic
    stream_bytes: float = 0.0    # bytes this task streams from HBM (weights, KV)
    cu_rate: float = 30e3        # bytes/us one CU of this task can pull, when inflight is 0 (a fixed cap)
    end_us: float = 0.0          # stores and the release after the stream
    inflight: float = 0.0        # bytes this task keeps in flight; with it the rate follows Little's law:
    exposed_us: float = 0.0      # rate = inflight / (loaded latency + exposed work per trip)


@dataclass
class Hw:
    # GPU bandwidth when n CUs stream: bw_max * n / (n + n_half) bytes/us (fitted to the measured stream rates)
    bw_max: float = 4.3e6
    n_half: float = 20.0
    handoff_us: dict = field(default_factory=lambda: {"DOMAIN": 1.0, "DEVICE": 1.3})
    release_us: float = 0.0
    xcd_bw: float | None = None  # optional cap per XCD (bytes/us)
    domains: int = 8
    wake_us: float = 0.0         # a waiter notices its event up to this late (uniform): polling / wake-up spread
    lat_us: float = 0.2          # idle round trip; loaded latency = max(lat_us, bytes in flight on the GPU / bw_max)
    rate_cv: float = 0.0         # per-task spread of the per-CU stream cap (normal, coefficient of variation)
    seed: int = 0

    def total_bw(self, n: int) -> float:
        return self.bw_max * n / (n + self.n_half) if n > 0 else 0.0


def _rates(active: dict[int, list], hw: Hw) -> dict[int, float]:
    """Streams with `inflight` follow Little's law: every one of them sees the same loaded latency
    L = max(idle round trip, total bytes in flight / bw_max) and pulls inflight / (L + its exposed work per trip);
    streams given only a fixed cap are water-filled into what is left of the bandwidth."""
    if not active:
        return {}
    rate = {}
    little = {k: v for k, v in active.items() if v[3] > 0}
    if little:
        L = max(hw.lat_us, sum(v[3] for v in little.values()) / hw.bw_max)
        for k, v in little.items():
            rate[k] = v[3] / (L + v[4])
    fixed = {k: v for k, v in active.items() if v[3] <= 0}
    if fixed:
        left = max(0.0, hw.total_bw(len(active)) - sum(rate.values()))
        todo = {k: v[1] for k, v in fixed.items()}
        while todo:
            share = left / len(todo)
            low = {k: c for k, c in todo.items() if c <= share}
            if not low:
                for k in todo:
                    rate[k] = share
                break
            for k, c in low.items():
                rate[k] = c
                left -= c
                del todo[k]
    if hw.xcd_bw:
        by = {}
        for k, v in active.items():
            by.setdefault(v[2], []).append(k)
        for d, ks in by.items():
            s = sum(rate[k] for k in ks)
            if s > hw.xcd_bw:
                for k in ks:
                    rate[k] *= hw.xcd_bw / s
    return rate


def simulate(plan: dict, cost_of: Callable[[str, dict], Cost], hw: Hw, max_tasks: int | None = None):
    """Returns (rows, stats): rows in ETX_TRACE_RAW layout for every executed task."""
    names = {t["id"]: n for n, t in plan["types"].items()}
    descs = plan["descs"]
    n = len(descs) if max_tasks is None else min(max_tasks, len(descs))
    ev_scope = {e["name"]: e["scope"] for e in plan["events"]}
    ev_inst = plan["ev_inst"]                               # [event, coord, producers]
    task_in = plan["task_in"]
    remaining = [len(p[2]) for p in ev_inst]
    fire = [None] * len(ev_inst)                             # time the instance's last producer ended
    consumers_of = [[] for _ in ev_inst]
    for t in range(n):
        for e in task_in[t]:
            consumers_of[e].append(t)
    produces = [[] for _ in range(len(descs))]
    for i, (_, _, prods) in enumerate(ev_inst):
        for p in prods:
            produces[p].append(i)
    queues = {}
    for key, ids in plan["static_queues"].items():
        d, s = (int(x) for x in key.split(":"))
        queues[d * plan["workers_per_domain"] + s] = [i for i in ids if i < n]
    pos = {w: 0 for w in queues}
    owner = {t: w for w, q in queues.items() for t in q}
    cost = {}
    taken, ready, done = [0.0] * n, [None] * n, [None] * n
    worker_free = {w: 0.0 for w in queues}
    rng = random.Random(hw.seed)

    def ready_time(t):
        tt = 0.0
        for e in task_in[t]:
            if fire[e] is None:
                return None
            dom_same = all(descs[p]["domain"] == descs[t]["domain"] for p in ev_inst[e][2])
            sc = ev_scope[ev_inst[e][0]]
            h = hw.handoff_us["DOMAIN"] if (sc == "DOMAIN" or dom_same and sc != "DEVICE") else hw.handoff_us[sc]
            tt = max(tt, fire[e] + hw.release_us + h)
        return tt + (rng.uniform(0.0, hw.wake_us) if hw.wake_us and task_in[t] else 0.0)

    # event queue of (time, seq, kind, payload)
    evq, seq = [], 0

    def push(time, kind, payload):
        nonlocal seq
        heapq.heappush(evq, (time, seq, kind, payload))
        seq += 1

    def try_start(w, now):
        if pos[w] >= len(queues[w]):
            return
        t = queues[w][pos[w]]
        if ready[t] is not None:
            return                                            # already scheduled
        r = ready_time(t)
        if r is None:
            return                                            # woken when its last event fires
        taken[t] = worker_free[w]
        ready[t] = max(r, worker_free[w])
        c = cost_of(names[descs[t]["type"]], descs[t])
        cost[t] = c
        push(ready[t] + c.front_us + c.chain_us, "stream", t)

    active = {}                                               # task -> [bytes left, cap, domain, inflight, exposed]
    last = 0.0
    rates = {}
    for w in queues:
        try_start(w, 0.0)
    while evq or active:
        # next stream completion under the current rates
        t_next_stream, who = None, None
        for k, v in active.items():
            r = rates.get(k, 0.0)
            if r > 0:
                tc = last + v[0] / r
                if t_next_stream is None or tc < t_next_stream:
                    t_next_stream, who = tc, k
        t_next_ev = evq[0][0] if evq else None
        if t_next_ev is not None and (t_next_stream is None or t_next_ev <= t_next_stream):
            now, _, kind, t = heapq.heappop(evq)
        else:
            now, kind, t = t_next_stream, "stream_done", who
        for k, v in active.items():                           # advance the fluid
            v[0] -= rates.get(k, 0.0) * (now - last)
        last = now
        if kind == "stream":
            c = cost[t]
            if c.stream_bytes > 0:
                f = max(0.3, rng.gauss(1.0, hw.rate_cv)) if hw.rate_cv else 1.0
                active[t] = [c.stream_bytes, c.cu_rate * f, descs[t]["domain"], c.inflight * f, c.exposed_us]
            else:
                push(now + c.end_us, "done", t)
        elif kind == "stream_done":
            del active[t]
            push(now + cost[t].end_us, "done", t)
        elif kind == "done":
            done[t] = now
            w = owner[t]
            pos[w] += 1
            worker_free[w] = now
            woken = set()
            for e in produces[t]:
                remaining[e] -= 1
                if remaining[e] == 0:
                    fire[e] = now
                    for c2 in consumers_of[e]:
                        woken.add(c2)
            try_start(w, now)
            for c2 in woken:
                w2 = owner.get(c2)
                if w2 is not None and queues[w2][pos[w2]:pos[w2] + 1] == [c2]:
                    try_start(w2, now)
        rates = _rates(active, hw)
    rows = []                                                  # one row per plan task, as ETX_TRACE_RAW writes them
    for t in range(len(descs)):
        d = descs[t]
        coord = list(d["coord"]) + [0] * (4 - len(d["coord"]))
        if t >= n or done[t] is None:
            rows.append((d["type"], *coord[:4], 0, 0, 0, 0))
            continue
        rows.append((d["type"], *coord[:4], int(taken[t] / TICK_US) + 1, int(ready[t] / TICK_US) + 1, int(done[t] / TICK_US) + 1,
                     owner[t]))
    unfinished = sum(1 for t in range(n) if done[t] is None)
    busy = sum(done[t] - ready[t] for t in range(n) if done[t] is not None)
    return rows, {"makespan_us": max((x for x in done if x is not None), default=0.0), "unfinished": unfinished, "busy_us": busy}
