"""Bandwidth-aware simulation of a compiled plan on the XCDs of one GPU.

Where etx.sim.protocol gives every task a fixed duration, this one gives it a cost *shape* and lets the tasks that run at
the same time share the memory system, so what one XCD does changes what another gets:

  task = wait (its input events, each fired at its last producer's end + the release, seen after the hand-off of the
         event's scope) -> front + floor + etx (fixed: the hardware minimum a workgroup pays before its bytes, plus
         ETX's own acquire / argument / release cost) -> chain (fixed: the kernel's dependent global round trips,
         barriers, reductions) -> stream (bytes, fluid: rate-limited per CU and sharing the GPU's bandwidth) -> end

Since 2026-09-30 the fixed part is dominated by `floor_us` and `etx_us`, measured per phase rather than derived from
an instruction count: hw_bench stream shows 296 workgroups cost about 3 us even when they read 896 B each, and the
measured builds put ETX's own share on top of that (docs/MODEL-vllm-kernels.md section 19). The earlier model
charged a per-task tax of about 2 us and counted events, and predicted gains from folding that the hardware did not
pay: build 2 5.934 ms -> build 3 5.895 ms, against a predicted 0.9 ms.

Streams follow Little's law: all of them see one loaded latency L = max(idle round trip, bytes in flight on the GPU /
bandwidth), and each pulls its bytes in flight / (L + the work of a trip that is not overlapped with memory). So a GPU
full of streams is bandwidth-bound, a few streams are latency-bound, and an XCD that streams alone runs faster. Workers run their static queue in order, one task at a time (the generated kernel's loop).

Streams are split by where their bytes come from. Weights (6.1 GB per token) never fit in any cache and stream from
HBM; the KV cache (104 MB) fits in the 256 MB Infinity Cache (MALL) and streams from there at MALL rate
(docs/BUDGET-2026-10-01.md: attn 2.16 MB/layer in 6.28 us = 344 GB/s per task, MALL-resident). Cost.mall_bytes marks
the MALL part of a task's stream_bytes; HBM and MALL are separate bandwidth pools that share nothing, so an XCD
streaming KV does not take HBM bandwidth from an XCD streaming weights. Before this split the model charged every
byte against HBM, which over-predicted contention whenever attention overlapped GEMMs.

The output has the layout of ETX_TRACE_RAW (type, coord x4, taken, ready, done, worker; 10 ns ticks), so the same tools
read a simulated and a measured schedule: etx.tools.timeline (per-XCD occupancy, critical path, HTML) and
examples/vllm_llm/compare.py ledger.

  from etx.sim.xcd import Cost, Hw, simulate
  rows = simulate(plan_json, cost_of_type, Hw.mi300x())
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
    floor_us: float = 0.0        # the hardware minimum any workgroup of a full-device phase pays before its bytes
                                 # (etx.sim.phase.FLOOR_US, measured with no events and no ETX); it does not overlap
                                 # with the stream, and it is what makes a phase, not a task, the unit of cost
    etx_us: float = 0.0          # ETX's own per-task cost on top of the floor: acquire, argument path, release, wave
                                 # start. Fitted to the measured builds, not to an instruction count
    stream_bytes: float = 0.0    # bytes this task streams (weights from HBM, KV from MALL; see mall_bytes)
    mall_bytes: float = 0.0      # of stream_bytes, this many are served from the Infinity Cache (MALL) and never
                                 # touch HBM: the KV cache (104 MB) fits in MALL (256 MB) and streams at MALL rate,
                                 # so it contends with other MALL streams only (docs/BUDGET-2026-10-01.md: attn
                                 # 2.16 MB/layer in 6.28 us = 344 GB/s per task, MALL-resident). Clamped to
                                 # stream_bytes by simulate; 0.0 keeps the old all-HBM behaviour.
    cu_rate: float = 30e3        # bytes/us one CU of this task can pull alone, when inflight is 0 (a fixed cap); the
                                 # new model takes it from etx.sim.phase.cu_rate(row mapping)
    agg_bw: float = 0.0          # bytes/us the whole device sustains for streams of this shape (0: hw.bw_max). It is
                                 # not one number: a workgroup reading 28 KB keeps a shallower queue than one reading
                                 # 1 MB and the device only reaches 3.5 TB/s at the deep end (etx.sim.phase.agg_bw).
                                 # Applies to the HBM pool; the MALL pool uses hw.mall_bw_max.
    end_us: float = 0.0          # stores and the release after the stream
    inflight: float = 0.0        # bytes this task keeps in flight; with it the rate follows Little's law:
    exposed_us: float = 0.0      # rate = inflight / (loaded latency + exposed work per trip)
    touch_bytes: float = 0.0     # the touch hook: this many of stream_bytes are pulled (same inflight, same bandwidth
                                 # sharing) from the moment the worker takes the task, before its events fire; the body's
                                 # stream of the rest starts when both the touch and front + chain are done.
                                 # Touch prefetches weights (HBM pool); MALL bytes are never touched.

    @property
    def fixed_us(self) -> float:
        return self.front_us + self.chain_us + self.floor_us + self.etx_us


@dataclass
class Hw:
    # GPU bandwidth when n CUs stream: bw_max * n / (n + n_half) bytes/us (fitted to the measured stream rates)
    bw_max: float = 4.3e6
    n_half: float = 20.0
    handoff_us: dict = field(default_factory=lambda: {"DOMAIN": 1.0, "DEVICE": 1.3})
    release_us: float = 0.0
    xcd_bw: float | None = None  # optional cap per XCD (bytes/us), applied to HBM + MALL together
    domains: int = 8
    wake_us: float = 0.0         # a waiter notices its event up to this late (uniform): polling / wake-up spread
    lat_us: float = 0.2          # idle round trip; loaded latency = max(lat_us, bytes in flight on the GPU / bw_max)
    rate_cv: float = 0.0         # per-task spread of the per-CU stream cap (normal, coefficient of variation)
    seed: int = 0
    # Infinity Cache (MALL): the on-package cache the KV cache streams from. A separate bandwidth pool from HBM.
    mall_bw_max: float = 2.0e6   # bytes/us aggregate MALL bandwidth. FIT, not measured: 344 GB/s per attn task
                                 # (docs/BUDGET-2026-10-01.md) x ~6 concurrent KV-streaming tasks, conservative.
                                 # Refit with an hw_bench MALL stream sweep (cf. etx.sim.phase.SPAN_US) before
                                 # trusting contention predictions between many concurrent attention tasks.
    mall_lat_us: float = 0.15    # MALL idle round trip, below HBM's (on-package, no HBM PHY)
    mall_n_half: float = 8.0     # MALL saturates with fewer streams than HBM (fewer clients share it)

    def total_bw(self, n: int) -> float:
        return self.bw_max * n / (n + self.n_half) if n > 0 else 0.0

    def mall_total_bw(self, n: int) -> float:
        return self.mall_bw_max * n / (n + self.mall_n_half) if n > 0 else 0.0

    @classmethod
    def mi300x(cls, **overrides) -> "Hw":
        """MI300X preset. Every number names its source; refit when the workload changes (sim_fit.py)."""
        return cls(
            bw_max=4.3e6,          # fitted to the measured stream rates
            n_half=20.0,
            handoff_us={"DOMAIN": 1.0, "DEVICE": 1.3},   # hw_bench handoff, padded counters
            lat_us=0.2,            # idle HBM round trip
            mall_bw_max=2.0e6,     # FIT, see field doc
            mall_lat_us=0.15,
            mall_n_half=8.0,
            **overrides,
        )


def _rates(active: dict, hw: Hw, pool_of: dict) -> dict:
    """Per-pool fluid model. 'hbm' and 'mall' streams share nothing with each other; inside one pool the rule is
    unchanged: streams with `inflight` follow Little's law (every one of them sees the pool's loaded latency
    L = max(idle round trip, total bytes in flight / pool bw_max) and pulls inflight / (L + its exposed work per
    trip)); streams given only a fixed cap are water-filled into what is left of the pool's bandwidth."""
    if not active:
        return {}
    rate = {}
    pools: dict[str, dict] = {}
    for k, v in active.items():
        pools.setdefault(pool_of.get(k, "hbm"), {})[k] = v
    for pool, sub in pools.items():
        if pool == "mall":
            bw_max, lat, n_half = hw.mall_bw_max, hw.mall_lat_us, hw.mall_n_half
        else:
            bw_max, lat, n_half = hw.bw_max, hw.lat_us, hw.n_half
        little = {k: v for k, v in sub.items() if v[3] > 0}
        if little:
            L = max(lat, sum(v[3] for v in little.values()) / bw_max)
            for k, v in little.items():
                rate[k] = v[3] / (L + v[4])
        fixed = {k: v for k, v in sub.items() if v[3] <= 0}
        if fixed:
            # the device's own limit for the shapes streaming right now: the smallest transfer sets it, since it is
            # the one whose queue is shallowest (etx.sim.phase.agg_bw). HBM pool only; MALL uses mall_bw_max.
            caps = [v[5] for v in sub.values() if len(v) > 5 and v[5] > 0]
            total = min(caps) if caps else bw_max * len(sub) / (len(sub) + n_half)
            left = max(0.0, total - sum(rate[k] for k in little))
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


@dataclass
class ReplayTiming:
    """Trace-calibrated timing residuals, in us; not a hardware timing model.

    Keep untimed arrive/fetch work with its producer, and ready overhead with its
    consumer. Fixed-body what-ifs may move these costs, but must not delete them.
    Event publication still starts at body-done, before the worker's next fetch.
    """
    queue_after_us: list[float]
    ready_after_us: list[float]
    initial_taken_us: dict[int, float] = field(default_factory=dict)
    event_after_us: list[float] | None = None  # optional notification branch before max(event, worker)


def simulate(plan: dict, cost_of: Callable[[str, dict], Cost], hw: Hw, max_tasks: int | None = None,
             *, replay: ReplayTiming | None = None):
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
    worker_free = {w: replay.initial_taken_us.get(w, 0.0) if replay else 0.0 for w in queues}
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

    touch_done = {}                                           # task -> time its touch stream finished (None while in flight)
    stream_pending = set()                                    # tasks whose body stream waits for their touch
    parts_left: dict[int, int] = {}                           # task -> stream parts (HBM and/or MALL) not yet done

    def try_start(w, now):
        if pos[w] >= len(queues[w]):
            return
        t = queues[w][pos[w]]
        if ready[t] is not None:
            return                                            # already scheduled
        if t not in cost:
            cost[t] = cost_of(names[descs[t]["type"]], descs[t])
        c = cost[t]
        if c.touch_bytes > 0 and t not in touch_done:         # the touch hook streams from the moment the worker is free
            touch_done[t] = None
            active[("touch", t)] = [min(c.touch_bytes, c.stream_bytes), c.cu_rate, descs[t]["domain"], c.inflight, c.exposed_us, c.agg_bw]
            pool_of[("touch", t)] = "hbm"                     # touch prefetches weights, never KV
        r = ready_time(t)
        if r is None:
            return                                            # woken when its last event fires
        taken[t] = worker_free[w]
        if replay and replay.event_after_us is not None:
            r += replay.event_after_us[t]
        ready[t] = max(r, worker_free[w])
        if replay:
            ready[t] += replay.ready_after_us[t]
        push(ready[t] + c.fixed_us, "stream", t)

    def part_done(t) -> bool:
        """One stream part of task t finished; True when the task's whole stream is done."""
        parts_left[t] -= 1
        return parts_left[t] <= 0

    def start_stream(t, now):
        c = cost[t]
        mall = max(0.0, min(c.mall_bytes, c.stream_bytes))
        touched = min(c.touch_bytes, c.stream_bytes) if t in touch_done else 0.0
        hbm_left = max(0.0, c.stream_bytes - mall - touched)
        dom = descs[t]["domain"]
        parts = 0
        if hbm_left > 0:
            f = max(0.3, rng.gauss(1.0, hw.rate_cv)) if hw.rate_cv else 1.0
            active[t] = [hbm_left, c.cu_rate * f, dom, c.inflight * f, c.exposed_us, c.agg_bw]
            pool_of[t] = "hbm"
            parts += 1
        if mall > 0:
            f = max(0.3, rng.gauss(1.0, hw.rate_cv)) if hw.rate_cv else 1.0
            active[("mall", t)] = [mall, c.cu_rate * f, dom, c.inflight * f, c.exposed_us, c.agg_bw]
            pool_of[("mall", t)] = "mall"
            parts += 1
        parts_left[t] = parts
        if parts == 0:
            push(now + c.end_us, "done", t)

    active = {}                                               # stream key -> [bytes left, cap, domain, inflight, exposed, agg_bw]
    pool_of: dict = {}                                        # stream key -> "hbm" | "mall"
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
            if t in touch_done and touch_done[t] is None:
                stream_pending.add(t)                         # the touch loads complete at the barrier before the body
            else:
                start_stream(t, now)
        elif kind == "stream_done":
            del active[t]
            del pool_of[t]
            if isinstance(t, tuple):                          # ("touch", task) or ("mall", task)
                tag, t = t
                if tag == "touch":
                    touch_done[t] = now
                    if t in stream_pending:
                        stream_pending.discard(t)
                        start_stream(t, now)
                    continue
                # tag == "mall": fall through to the join below with t = the task
            if part_done(t):
                push(now + cost[t].end_us, "done", t)
        elif kind == "done":
            done[t] = now
            w = owner[t]
            pos[w] += 1
            worker_free[w] = now + (replay.queue_after_us[t] if replay else 0.0)
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
        rates = _rates(active, hw, pool_of)
    rows = []                                                  # one row per plan task, as ETX_TRACE_RAW writes them
    for t in range(len(descs)):
        d = descs[t]
        coord = list(d["coord"]) + [0] * (4 - len(d["coord"]))
        if t >= n or done[t] is None:
            rows.append((d["type"], *coord[:4], 0, 0, 0, 0))
            continue
        tick = round if replay else int
        rows.append((d["type"], *coord[:4], tick(taken[t] / TICK_US) + 1, tick(ready[t] / TICK_US) + 1, tick(done[t] / TICK_US) + 1,
                     owner[t]))
    unfinished = sum(1 for t in range(n) if done[t] is None)
    busy = sum(done[t] - ready[t] for t in range(n) if done[t] is not None)
    return rows, {"makespan_us": max((x for x in done if x is not None), default=0.0), "unfinished": unfinished, "busy_us": busy}
