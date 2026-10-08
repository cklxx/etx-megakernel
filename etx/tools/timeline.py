"""Schedule inspection for ETX megakernel traces.

Run a build with `ETX_TRACE=1 ETX_TRACE_RAW=<file>`: the host dumps, for the last step, one row per task of
nine int64 {type, coord[0..3], taken, deps ready, body done, worker} in ETX_TIMER ticks (10 ns on AMD).
Together with the build's plan.json (types, descs, and the dependency lists `ev_inst` / `task_in`) this answers:

  * per task type: how many tasks, mean wait (taken -> deps ready) and body (ready -> done), first/last times
  * per XCD: the share of worker time in bodies, waiting on events, and idle (no task taken), and how far the
    XCDs finish each type apart (skew)
  * the critical path: from the last task back, each step's critical predecessor is either the producer that
    satisfied its dependencies last or the previous task on the same worker (queue order), whichever finished
    later; the path telescopes to the span, split into bodies, dependency gaps and queue gaps by task type
  * memory demand: each task's bytes (a spec: regex -> bytes per task) spread over its body, binned, against the
    achievable bandwidth: how long the memory system is nearly idle and which task types are running then

  python -m etx.tools.timeline build/dsv2lite/plan.json trace.bin --bytes bytes.json --grids '_5$' --html l5.html

The same analysis runs on a simulated schedule (`trace_from_sim`), so a planned layout can be inspected before it
runs, and a measured trace can be put next to the simulation of the same plan.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path

TICK_NS = 10.0          # ETX_TIMER on AMD: s_memrealtime, 100 MHz
ROW = 9                 # int64 per task in the raw trace


@dataclass
class Task:
    id: int
    type: int
    name: str           # type (grid) name
    domain: int
    worker: int
    taken: float        # us, relative to the window origin
    ready: float
    done: float
    nbytes: float = 0.0


@dataclass
class Trace:
    tasks: list[Task]                      # executed tasks inside the window
    by_id: dict[int, Task]
    type_names: list[str]
    n_domains: int
    workers_per_domain: int
    deps: list[list[int]] | None           # task id -> producer task ids (None: plan without dependency lists)
    t0: float
    t1: float
    notes: list[str] = field(default_factory=list)

    @property
    def span(self) -> float:
        return self.t1 - self.t0


# ------------------------------------------------------------------ loading
def load_plan(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def read_raw(path: str | Path) -> list[tuple[int, ...]]:
    data = Path(path).read_bytes()
    if len(data) % (8 * ROW):
        raise SystemExit(f"{path}: {len(data)} bytes is not a whole number of {ROW}-int64 rows")
    a = array("q"); a.frombytes(data)
    if sys.byteorder != "little":
        a.byteswap()
    return [tuple(a[i:i + ROW]) for i in range(0, len(a), ROW)]


def _deps_from_plan(plan: dict) -> list[list[int]] | None:
    if "ev_inst" not in plan or "task_in" not in plan:
        return None
    prods = [e[2] for e in plan["ev_inst"]]
    return [sorted({p for ev in evs for p in prods[ev]}) for evs in plan["task_in"]]


def bytes_spec(spec: dict[str, float] | None):
    rules = [(re.compile(k), float(v)) for k, v in (spec or {}).items()]

    def f(name: str) -> float:
        for rx, v in rules:
            if rx.search(name):
                return v
        return 0.0
    return f


def build_trace(plan: dict, rows: list[tuple[int, ...]], grids: str | None = None, spec: dict[str, float] | None = None,
                tick_ns: float = TICK_NS) -> Trace:
    names = [""] * len(plan["types"])
    for n, t in plan["types"].items():
        names[t["id"]] = n
    descs = plan["descs"]
    wpd = int(plan["workers_per_domain"])
    nb = bytes_spec(spec)
    notes = []
    if len(rows) != len(descs):
        notes.append(f"trace has {len(rows)} rows, plan {len(descs)} tasks: using the first {min(len(rows), len(descs))}")
    k = tick_ns / 1000.0
    all_tasks: list[Task] = []
    # Subtract integer ticks first. Absolute GPU timestamps can lose float precision.
    origin_ticks = min((r[5] for r in rows[:len(descs)] if r[5] and r[7]), default=0)
    for i, r in enumerate(rows[:len(descs)]):
        ty, taken, ready, done, worker = r[0], r[5], r[6], r[7], r[8]
        if taken == 0 or done == 0:
            continue                                            # not executed in the traced step
        d = descs[i]
        w = int(worker) & 0xFFFFFFFF
        dom = d["domain"] if d.get("mode", "static") == "static" else w // max(1, wpd)
        all_tasks.append(Task(i, int(ty), names[int(ty)] if 0 <= ty < len(names) else str(ty), int(dom), w,
                              (taken - origin_ticks) * k, (max(ready, taken) - origin_ticks) * k,
                              (max(done, ready, taken) - origin_ticks) * k))
    if not all_tasks:
        raise SystemExit("no executed task in the trace (was it written with ETX_TRACE=1?)")
    origin = min(t.taken for t in all_tasks)
    for t in all_tasks:
        t.taken -= origin; t.ready -= origin; t.done -= origin
        t.nbytes = nb(t.name)
    if grids:
        rx = re.compile(grids)
        sel = [t for t in all_tasks if rx.search(t.name)]
        if not sel:
            raise SystemExit(f"no task type matches {grids!r}")
        t0, t1 = min(t.ready for t in sel), max(t.done for t in sel)     # from the first task able to run
    else:
        t0, t1 = 0.0, max(t.done for t in all_tasks)
    tasks = [t for t in all_tasks if t.done > t0 and t.ready < t1]      # bodies overlapping the window
    return Trace(tasks, {t.id: t for t in all_tasks}, names, int(plan["domains"]), wpd, _deps_from_plan(plan), t0, t1, notes)


def trace_from_sim(plan_obj, sim_result) -> list[tuple[int, ...]]:
    """Raw-trace rows (ticks) from a simulated schedule of a compiled Plan: taken = the worker became free for the
    task (end of the preceding segment), ready = task start, done = task end."""
    rows = [(0,) * ROW for _ in plan_obj.tasks]
    per = 1000.0 / TICK_NS
    last_free: dict[tuple[int, int], float] = {}
    for s in sorted(sim_result.segments, key=lambda s: (s.device, s.worker, s.start)):
        key = (s.device, s.worker)
        if s.kind == "task":
            t = plan_obj.tasks[s.task]
            taken = min(last_free.get(key, s.start), s.start)
            c = list(t.coord) + [0] * (4 - len(t.coord))
            rows[s.task] = (plan_obj.type_ids[t.grid], *c[:4], int(round(taken * per)) + 1, int(round(s.start * per)) + 1,
                            int(round(s.end * per)) + 1, s.worker)
            last_free[key] = s.end
        elif s.kind == "sched":
            last_free[key] = s.end
    return rows


def plan_json_from_obj(plan_obj) -> dict:
    from ..codegen import emit_plan_json
    return json.loads(emit_plan_json(plan_obj))


# ------------------------------------------------------------------ analysis
def per_type(tr: Trace) -> list[dict]:
    acc: dict[str, dict] = {}
    for t in tr.tasks:
        a = acc.setdefault(t.name, {"type": t.name, "n": 0, "wait": 0.0, "body": 0.0, "first_taken": 1e30, "first_ready": 1e30,
                                    "last_ready": 0.0, "last_done": 0.0, "bytes": 0.0})
        a["n"] += 1; a["wait"] += t.ready - t.taken; a["body"] += t.done - t.ready; a["bytes"] += t.nbytes
        a["first_taken"] = min(a["first_taken"], t.taken); a["first_ready"] = min(a["first_ready"], t.ready)
        a["last_ready"] = max(a["last_ready"], t.ready); a["last_done"] = max(a["last_done"], t.done)
    out = sorted(acc.values(), key=lambda a: a["first_ready"])
    for a in out:
        a["wait"] /= a["n"]; a["body"] /= a["n"]
        for k in ("first_taken", "first_ready", "last_ready", "last_done"):
            a[k] -= tr.t0
    return out


def _clip(a: float, b: float, t0: float, t1: float) -> float:
    return max(0.0, min(b, t1) - max(a, t0))


def per_domain(tr: Trace) -> list[dict]:
    span = tr.span
    out = []
    for d in range(tr.n_domains):
        ts = [t for t in tr.tasks if t.domain == d]
        workers = {t.worker for t in ts}
        n_w = max(1, len(workers))
        body = sum(_clip(t.ready, t.done, tr.t0, tr.t1) for t in ts)
        wait = sum(_clip(t.taken, t.ready, tr.t0, tr.t1) for t in ts)
        out.append({"domain": d, "workers": len(workers), "body": body / (span * n_w), "wait": wait / (span * n_w),
                    "idle": max(0.0, 1.0 - (body + wait) / (span * n_w)),
                    "bytes": sum(t.nbytes * _clip(t.ready, t.done, tr.t0, tr.t1) / max(1e-9, t.done - t.ready) for t in ts)})
    return out


def domain_skew(tr: Trace) -> list[dict]:
    """Per type: when each XCD finishes its share (last done), relative to the earliest XCD."""
    acc: dict[str, dict[int, float]] = {}
    first: dict[str, float] = {}
    for t in tr.tasks:
        acc.setdefault(t.name, {})
        acc[t.name][t.domain] = max(acc[t.name].get(t.domain, 0.0), t.done)
        first[t.name] = min(first.get(t.name, 1e30), t.ready)
    out = []
    for name, m in sorted(acc.items(), key=lambda kv: first[kv[0]]):
        if len(m) < 2:
            continue
        lo, hi = min(m.values()), max(m.values())
        out.append({"type": name, "domains": len(m), "skew": hi - lo, "last_domain": max(m, key=m.get),
                    "first_domain": min(m, key=m.get)})
    return out


def critical_path(tr: Trace) -> list[dict]:
    """Steps from the window's last task back to the window start. Each step: the task, its body, and the gap to its
    critical predecessor (kind 'dep': the producer that finished last; 'queue': the previous task on its worker)."""
    prev_on_worker: dict[int, Task] = {}
    by_worker: dict[int, list[Task]] = {}
    for t in tr.by_id.values():
        by_worker.setdefault(t.worker, []).append(t)
    for ts in by_worker.values():
        ts.sort(key=lambda t: t.taken)
        for a, b in zip(ts, ts[1:]):
            prev_on_worker[b.id] = a
    last = max((t for t in tr.tasks if t.done <= tr.t1 + 1e-9), key=lambda t: t.done)
    steps = []
    t = last
    seen = set()
    while t is not None and t.id not in seen:
        seen.add(t.id)
        dep = None
        if tr.deps is not None and t.id < len(tr.deps):
            cands = [tr.by_id[p] for p in tr.deps[t.id] if p in tr.by_id]
            if cands:
                dep = max(cands, key=lambda p: p.done)
        q = prev_on_worker.get(t.id)
        pred, kind = None, "start"
        if dep is not None and (q is None or dep.done >= q.done):
            pred, kind = dep, "dep"
        elif q is not None:
            pred, kind = q, "queue"
        gap = (t.ready - pred.done) if pred is not None else (t.ready - tr.t0)
        steps.append({"task": t, "body": t.done - t.ready, "gap": max(0.0, gap), "kind": kind})
        if pred is None or pred.done <= tr.t0:
            if pred is not None:
                steps[-1]["gap"] = max(0.0, t.ready - tr.t0)
            break
        t = pred
    steps.reverse()
    return steps


def critical_by_type(steps: list[dict]) -> list[dict]:
    acc: dict[str, dict] = {}
    order: list[str] = []
    for s in steps:
        n = s["task"].name
        if n not in acc:
            acc[n] = {"type": n, "tasks": 0, "body": 0.0, "dep_gap": 0.0, "queue_gap": 0.0}; order.append(n)
        a = acc[n]; a["tasks"] += 1; a["body"] += s["body"]
        a["queue_gap" if s["kind"] == "queue" else "dep_gap"] += s["gap"]
    return [acc[n] for n in order]


def memory_demand(tr: Trace, bw_tbs: float = 4.3, bin_us: float = 0.5, idle_frac: float = 0.25) -> dict:
    n = max(1, int(tr.span / bin_us + 0.999))
    demand = [0.0] * n                      # bytes per bin
    active: list[dict[str, int]] = [dict() for _ in range(n)]
    for t in tr.tasks:
        a, b = max(t.ready, tr.t0), min(t.done, tr.t1)
        if b <= a:
            continue
        rate = t.nbytes / max(1e-9, t.done - t.ready)
        i0, i1 = int((a - tr.t0) / bin_us), min(n - 1, int((b - tr.t0) / bin_us))
        for i in range(i0, i1 + 1):
            lo, hi = tr.t0 + i * bin_us, tr.t0 + (i + 1) * bin_us
            ov = _clip(a, b, lo, hi)
            demand[i] += rate * ov
            if ov > 0:
                active[i][t.name] = active[i].get(t.name, 0) + 1
    cap = bw_tbs * 1e6 * bin_us                 # bytes per bin at the achievable bandwidth (TB/s = 1e6 B/us)
    util = [min(1.0, d / cap) for d in demand]
    low = [i for i, u in enumerate(util) if u < idle_frac]
    by_type: dict[str, float] = {}
    by_kind: dict[str, float] = {}
    for i in low:
        dom = max(active[i], key=active[i].get) if active[i] else "(no task)"
        by_type[dom] = by_type.get(dom, 0.0) + bin_us
        kind = re.sub(r"_-?\d+$", "", dom)
        by_kind[kind] = by_kind.get(kind, 0.0) + bin_us
    total = sum(t.nbytes * _clip(t.ready, t.done, tr.t0, tr.t1) / max(1e-9, t.done - t.ready) for t in tr.tasks)
    return {"bins": util, "bin_us": bin_us, "bytes": total, "floor_us": total / (bw_tbs * 1e6),
            "low_us": len(low) * bin_us, "low_by_type": sorted(by_type.items(), key=lambda kv: -kv[1]),
            "low_by_kind": sorted(by_kind.items(), key=lambda kv: -kv[1]), "bw_tbs": bw_tbs,
            "idle_frac": idle_frac, "mean_util": sum(util) / n}


def bandwidth_phases(tr: Trace, spec: dict[str, float], bw_tbs: float = 4.3) -> dict:
    """Charge each phase from its first ready time to the next phase's first ready time.

    This interval includes phase tails and handoffs. It does not isolate tile body time.
    Only byte rules in the spec count as known. Unknown phases retain their device time.
    """
    if bw_tbs <= 0 or not tr.tasks or tr.t0 != 0 or tr.notes:
        raise ValueError("bandwidth report needs a full, complete trace and positive bandwidth")
    rules = [re.compile(k) for k in spec]
    types = per_type(tr)
    phases: dict[str, dict] = {}
    for i, a in enumerate(types):
        name = a["type"]
        stem = re.sub(r"_\d+(?:_x)?$", "", name)
        known = any(rx.search(name) for rx in rules)
        # Charge the pre-first-ready delay to the first phase. The intervals then telescope.
        start = tr.t0 if i == 0 else a["first_ready"] + tr.t0
        end = types[i + 1]["first_ready"] + tr.t0 if i + 1 < len(types) else tr.t1
        if end < start:
            raise ValueError(f"phase {name} has negative device time")
        row = phases.setdefault(stem, {"phase": stem, "bytes": 0.0, "us": 0.0, "known": True})
        row["us"] += end - start
        row["bytes"] += a["bytes"]
        row["known"] &= known
    for row in phases.values():
        if row["known"]:
            row["floor_us"] = row["bytes"] / (bw_tbs * 1e6)
            row["wasted_us"] = row["us"] - row["floor_us"]
            row["gb_s"] = row["bytes"] / row["us"] / 1e3 if row["us"] else 0.0
            row["percent"] = row["gb_s"] / (bw_tbs * 1e3) * 100
        else:
            row["bytes"] = None
            row["floor_us"] = row["wasted_us"] = row["gb_s"] = row["percent"] = None
    ranked = sorted((r for r in phases.values() if r["known"]), key=lambda r: (-r["wasted_us"], r["phase"]))
    unknown = sorted((r for r in phases.values() if not r["known"]), key=lambda r: (-r["us"], r["phase"]))
    known_bytes = sum(r["bytes"] for r in ranked)
    floor_us = known_bytes / (bw_tbs * 1e6)
    return {"ranked": ranked, "unknown": unknown, "span_us": tr.span, "known_bytes": known_bytes,
            "floor_us": floor_us, "percent": floor_us / tr.span * 100, "bw_tbs": bw_tbs}


def latency_phases(tr: Trace) -> list[dict]:
    """Aggregate phase latency from existing task, dependency, and critical-path data.

    Active XCDs use the exact union of body intervals on each XCD.
    The denominator sums grid body spans, so gaps between layers do not dilute it.
    Dependency gaps start at the last producer's body end, before event publication.
    """
    if tr.deps is None:
        raise ValueError("latency report requires plan dependency lists")
    if (not tr.tasks or tr.notes or tr.t0 != 0 or len(tr.tasks) != len(tr.deps)
            or len(tr.by_id) != len(tr.deps)):
        raise ValueError("latency report requires a full, complete trace")
    groups: dict[str, list[Task]] = {}
    for t in tr.tasks:
        groups.setdefault(t.name, []).append(t)
    crit = {a["type"]: a for a in critical_by_type(critical_path(tr))}
    skew = {a["type"]: a["skew"] for a in domain_skew(tr)}
    acc = {}
    types = per_type(tr)
    for i, a in enumerate(types):
        name = a["type"]
        stem = re.sub(r"_\d+(?:_x)?$", "", name)
        row = acc.setdefault(stem, dict(phase=stem, tasks=0, grids=0, body=0., wait=0.,
            grid_span_us=0., xcd_us=0., start_spread_us=0., finish_skew_us=0., outgoing_gap_us=0.,
            cp_body_us=0., cp_dep_gap_us=0., cp_queue_gap_us=0., dep_gaps=[]))
        row["tasks"] += a["n"]
        row["grids"] += 1
        if i + 1 < len(types):
            row["outgoing_gap_us"] += types[i + 1]["first_ready"] - a["last_done"]
        row["body"] += a["body"] * a["n"]
        row["wait"] += a["wait"] * a["n"]
        row["grid_span_us"] += a["last_done"] - a["first_ready"]
        row["start_spread_us"] += a["last_ready"] - a["first_ready"]
        row["finish_skew_us"] += skew.get(name, 0.)
        c = crit.get(name, {})
        for key in ("body", "dep_gap", "queue_gap"):
            row[f"cp_{key}_us"] += c.get(key, 0.)
        domains: dict[int, list[tuple[float, float]]] = {}
        for t in groups[name]:
            domains.setdefault(t.domain, []).append((t.ready, t.done))
            producers = [tr.by_id[p] for p in tr.deps[t.id] if p in tr.by_id]
            if producers:
                row["dep_gaps"].append(t.ready - max(p.done for p in producers))
        for intervals in domains.values():
            end = -float("inf")
            for lo, hi in sorted(intervals):
                row["xcd_us"] += max(0., hi - max(lo, end))
                end = max(end, hi)
    for row in acc.values():
        row["body_us"] = row["body"] / row["tasks"]
        row["wait_us"] = row.pop("wait") / row["tasks"]
        span = row["grid_span_us"]
        row["active_tasks"] = row.pop("body") / span if span else 0.
        row["active_xcds"] = row.pop("xcd_us") / span if span else 0.
        gaps = sorted(row.pop("dep_gaps"))
        row["dep_gap_mean_us"] = sum(gaps) / len(gaps) if gaps else None
        row["dep_gap_p95_us"] = gaps[max(0, (95 * len(gaps) + 99) // 100 - 1)] if gaps else None
        row["dep_gap_max_us"] = gaps[-1] if gaps else None
    return list(acc.values())


def latency_report(tr: Trace, out=None) -> dict:
    out = out or sys.stdout
    rows = latency_phases(tr)
    print("Body and wait are task means. Active XCDs use each grid's body span.", file=out)
    print(f"{'phase':16} {'tasks':>7} {'body us':>9} {'wait us':>9} {'tasks live':>11} "
          f"{'XCDs live':>10} {'dep gap us':>11} {'p95 us':>9} {'CP body':>10} {'CP dep':>10} {'CP queue':>10}", file=out)
    for r in rows:
        gap = r['dep_gap_mean_us']
        p95 = r['dep_gap_p95_us']
        print(f"{r['phase']:16} {r['tasks']:7d} {r['body_us']:9.3f} {r['wait_us']:9.3f} "
              f"{r['active_tasks']:11.2f} {r['active_xcds']:10.2f} "
              f"{gap if gap is not None else float('nan'):11.3f} "
              f"{p95 if p95 is not None else float('nan'):9.3f} "
              f"{r['cp_body_us']:10.2f} {r['cp_dep_gap_us']:10.2f} {r['cp_queue_gap_us']:10.2f}", file=out)
    return {"latency": rows, "span_us": tr.span}


def bandwidth_report(tr: Trace, spec: dict[str, float], bw_tbs: float = 4.3, out=None) -> dict:
    out = out or sys.stdout
    r = bandwidth_phases(tr, spec, bw_tbs)
    print("Phase bytes include only the specified traffic. Device time includes tails and handoffs.", file=out)
    print(f"{'phase':16} {'GB':>10} {'us':>10} {'GB/s':>10} {'% of cap':>10} {'wasted us':>12}", file=out)
    for row in r["ranked"]:
        print(f"{row['phase']:16} {row['bytes'] / 1e9:10.4f} {row['us']:10.1f} {row['gb_s']:10.1f} "
              f"{row['percent']:10.1f} {row['wasted_us']:12.1f}", file=out)
    for row in r["unknown"]:
        print(f"{row['phase']:16} {'unknown':>10} {row['us']:10.1f} {'unknown':>10} "
              f"{'unknown':>10} {'unknown':>12}", file=out)
    print(f"TOTAL known {r['known_bytes'] / 1e9:.4f} GB; device span {r['span_us']:.1f} us; "
          f"floor {r['floor_us']:.1f} us at {bw_tbs} TB/s (estimate); "
          f"achieved {r['percent']:.1f}% (known bytes / full span)", file=out)
    if r["unknown"]:
        print("Unknown phase bytes exclude some traffic. Total achieved rate is a lower bound.", file=out)
    return r


# ------------------------------------------------------------------ report
def report(tr: Trace, bw_tbs: float = 4.3, out=None) -> dict:
    out = out or sys.stdout
    p = lambda s="": print(s, file=out)   # noqa: E731
    for n in tr.notes:
        p(f"note: {n}")
    p(f"window {tr.t0:.1f} .. {tr.t1:.1f} us (span {tr.span:.1f} us), {len(tr.tasks)} tasks, "
      f"{tr.n_domains} XCDs x {tr.workers_per_domain} workers")
    p("\nper type (us, relative to the window start)")
    p(f"  {'type':24} {'tasks':>6} {'wait':>7} {'body':>7} {'1st ready':>10} {'last ready':>11} {'last done':>10} {'MB':>8}")
    for a in per_type(tr):
        p(f"  {a['type']:24} {a['n']:6d} {a['wait']:7.2f} {a['body']:7.2f} {a['first_ready']:10.1f} {a['last_ready']:11.1f} "
          f"{a['last_done']:10.1f} {a['bytes'] / 1e6:8.2f}")
    p("\nper XCD: share of worker time in bodies / waiting for events / idle (no task taken)")
    for d in per_domain(tr):
        p(f"  XCD {d['domain']}  {d['workers']:3d} workers  body {d['body'] * 100:5.1f}%  wait {d['wait'] * 100:5.1f}%  "
          f"idle {d['idle'] * 100:5.1f}%  {d['bytes'] / 1e6:8.1f} MB")
    sk = domain_skew(tr)
    if sk:
        p("\nXCD finish skew per type (last XCD done - first XCD done)")
        for s in sk:
            p(f"  {s['type']:24} {s['skew']:6.2f} us  (first XCD {s['first_domain']}, last XCD {s['last_domain']}, {s['domains']} XCDs)")
    steps = critical_path(tr)
    body = sum(s["body"] for s in steps); gaps = sum(s["gap"] for s in steps)
    if tr.deps is None:
        p("\ncritical path: plan.json has no dependency lists (rebuild with this ETX); queue order only")
    p(f"\ncritical path: {len(steps)} tasks, {body:.1f} us in bodies + {gaps:.1f} us in gaps = {body + gaps:.1f} us "
      f"(span {tr.span:.1f})")
    p(f"  {'type':24} {'tasks':>6} {'body':>8} {'dep gap':>8} {'queue gap':>10}")
    for a in critical_by_type(steps):
        p(f"  {a['type']:24} {a['tasks']:6d} {a['body']:8.2f} {a['dep_gap']:8.2f} {a['queue_gap']:10.2f}")
    md = memory_demand(tr, bw_tbs)
    if md["bytes"] > 0:
        p(f"\nmemory: {md['bytes'] / 1e6:.1f} MB in the window, floor {md['floor_us']:.1f} us at {bw_tbs} TB/s "
          f"({md['floor_us'] / tr.span * 100:.0f}% of the span); mean demand {md['mean_util'] * 100:.0f}% of the bandwidth")
        p(f"  below {md['idle_frac'] * 100:.0f}% of the bandwidth for {md['low_us']:.1f} us ({md['low_us'] / tr.span * 100:.0f}% of the span); "
          "running then (by task kind, all layers):")
        for n, us in md["low_by_kind"][:10]:
            p(f"    {n:24} {us:7.1f} us  ({us / md['low_us'] * 100:4.1f}%)")
    else:
        p("\nmemory: no bytes spec (--bytes), demand not computed")
    return {"per_type": per_type(tr), "per_domain": per_domain(tr), "skew": sk, "critical": critical_by_type(steps),
            "critical_span": body + gaps, "memory": {k: v for k, v in md.items() if k != "bins"}}


_PALETTE = ["#3b6fb6", "#d9822b", "#3f9e5a", "#c0413b", "#8a63b8", "#8c6d4f", "#d36fa9", "#6f7c85", "#b3a52e", "#2aa3b5"]


def write_html(tr: Trace, path: str | Path, title: str = "ETX schedule", bw_tbs: float = 4.3) -> None:
    """One self-contained page: a row per worker grouped by XCD (body solid, event wait pale, critical path outlined)
    and the memory demand curve underneath. Canvas-drawn, so a whole token's tasks stay light."""
    types = sorted({t.name for t in tr.tasks}, key=lambda n: min(x.ready for x in tr.tasks if x.name == n))
    base = {}
    for n in types:
        stem = re.sub(r"_?-?\d+$", "", n)
        base.setdefault(stem, len(base))
    color = {n: _PALETTE[base[re.sub(r'_?-?\d+$', '', n)] % len(_PALETTE)] for n in types}
    workers = sorted({(t.domain, t.worker) for t in tr.tasks})
    row = {w: i for i, w in enumerate(workers)}
    crit = {s["task"].id for s in critical_path(tr)}
    md = memory_demand(tr, bw_tbs)
    data = {"t0": tr.t0, "t1": tr.t1, "rows": len(workers), "domains": [d for d, _ in workers],
            "tasks": [[row[(t.domain, t.worker)], round(t.taken, 3), round(t.ready, 3), round(t.done, 3), types.index(t.name),
                       1 if t.id in crit else 0] for t in tr.tasks],
            "types": types, "colors": [color[n] for n in types], "mem": [round(u, 3) for u in md["bins"]], "bin": md["bin_us"],
            "bw": bw_tbs}
    page = _HTML.replace("__TITLE__", html.escape(title)).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path(path).write_text(page, encoding="utf-8")


_HTML = """<title>__TITLE__</title>
<style>
:root{--bg:#f7f8fa;--fg:#1d232b;--muted:#5d6874;--grid:#d9dee4;--panel:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14181d;--fg:#e4e8ec;--muted:#95a1ad;--grid:#2b323a;--panel:#1b2026;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14181d;--fg:#e4e8ec;--muted:#95a1ad;--grid:#2b323a;--panel:#1b2026;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:13px/1.45 ui-sans-serif,system-ui,sans-serif;padding:16px}
h1{font-size:16px;margin:0 0 4px}.m{color:var(--muted)}#wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--grid);border-radius:4px}
#legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:10px 0}#legend span{display:inline-flex;align-items:center;gap:5px}
#legend i{width:10px;height:10px;display:inline-block;border-radius:2px}#tip{position:fixed;pointer-events:none;background:var(--panel);
border:1px solid var(--grid);padding:4px 6px;border-radius:3px;font:12px ui-monospace,monospace;display:none}
</style>
<h1>__TITLE__</h1>
<div class="m" id="sum"></div>
<div id="legend"></div>
<div id="wrap"><canvas id="c"></canvas></div>
<p class="m">One row per worker, grouped by XCD. Solid: task body. Pale: waiting for its events. Outlined: critical path.
Bottom: memory demand (bytes of the running tasks spread over their bodies) as a share of the achievable bandwidth.</p>
<div id="tip"></div>
<script>
const D=__DATA__;const cv=document.getElementById('c'),cx=cv.getContext('2d');
const RH=2,LW=64,W=Math.max(900,Math.min(4000,(D.t1-D.t0)*8)),MH=90,H=D.rows*RH+MH+30;
const dpr=window.devicePixelRatio||1;cv.width=(W+LW)*dpr;cv.height=H*dpr;cv.style.width=(W+LW)+'px';cv.style.height=H+'px';cx.scale(dpr,dpr);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const X=t=>LW+(t-D.t0)/(D.t1-D.t0)*W;
document.getElementById('sum').textContent=`window ${(D.t1-D.t0).toFixed(1)} us, ${D.tasks.length} tasks, ${D.rows} workers`;
document.getElementById('legend').innerHTML=D.types.map((n,i)=>`<span><i style="background:${D.colors[i]}"></i>${n}</span>`).join('');
function draw(){cx.fillStyle=css('--panel');cx.fillRect(0,0,W+LW,H);cx.fillStyle=css('--muted');cx.font='11px ui-sans-serif,system-ui';
 let last=-1;D.domains.forEach((d,r)=>{if(d!==last){cx.fillText('XCD '+d,4,r*RH+10);cx.fillStyle=css('--grid');cx.fillRect(LW,r*RH,W,1);cx.fillStyle=css('--muted');last=d;}});
 cx.save();cx.beginPath();cx.rect(LW,0,W,D.rows*RH+2);cx.clip();
 for(const [r,tk,rd,dn,ty,cr] of D.tasks){const y=r*RH;cx.globalAlpha=0.25;cx.fillStyle=D.colors[ty];cx.fillRect(X(tk),y,Math.max(0.5,X(rd)-X(tk)),RH);
  cx.globalAlpha=1;cx.fillRect(X(rd),y,Math.max(0.5,X(dn)-X(rd)),RH);}
 cx.strokeStyle=css('--fg');cx.lineWidth=1;for(const [r,tk,rd,dn,ty,cr] of D.tasks){if(cr)cx.strokeRect(X(rd),r*RH-1,Math.max(1,X(dn)-X(rd)),RH+2);}
 cx.restore();cx.fillStyle=css('--muted');
 const y0=D.rows*RH+20;cx.fillStyle=css('--grid');cx.fillRect(LW,y0,W,1);cx.fillRect(LW,y0+MH,W,1);cx.fillStyle=css('--muted');
 cx.fillText(D.bw+' TB/s',4,y0+8);cx.fillText('0',4,y0+MH);cx.beginPath();cx.strokeStyle=D.colors[0];
 D.mem.forEach((u,i)=>{const x=X(D.t0+i*D.bin),y=y0+MH-u*MH;i?cx.lineTo(x,y):cx.moveTo(x,y);});cx.stroke();
 for(let t=Math.ceil(D.t0/10)*10;t<=D.t1;t+=10){cx.fillText(t.toFixed(0),X(t)-6,H-4);}}
draw();matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
const tip=document.getElementById('tip');cv.addEventListener('mousemove',e=>{const b=cv.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top,r=Math.floor(y/RH);
 const t=D.t0+(x-LW)/W*(D.t1-D.t0);const k=D.tasks.find(q=>q[0]===r&&q[1]<=t&&q[3]>=t);
 if(!k){tip.style.display='none';return;}tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
 tip.textContent=`${D.types[k[4]]}  wait ${(k[2]-k[1]).toFixed(2)} us  body ${(k[3]-k[2]).toFixed(2)} us${k[5]?'  critical':''}`;});
</script>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m etx.tools.timeline", description=__doc__.split("\n\n")[0])
    ap.add_argument("plan", help="plan.json of the build that produced the trace")
    ap.add_argument("raw", help="ETX_TRACE_RAW file")
    ap.add_argument("--bytes", help="JSON file {regex over type names: bytes per task}")
    ap.add_argument("--grids", help="regex over type names: the window is from the first matching task ready to the last one done")
    ap.add_argument("--bw", type=float, default=4.3, help="achievable memory bandwidth, TB/s (default 4.3, MI300X)")
    ap.add_argument("--tick-ns", type=float, default=TICK_NS)
    ap.add_argument("--html", help="write the timeline page here")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--bandwidth-report", action="store_true", help="rank phase device time above the byte floor")
    ap.add_argument("--latency-report", action="store_true", help="phase task, XCD, and dependency latency")
    a = ap.parse_args(argv)
    if a.latency_report and (a.bandwidth_report or a.grids):
        ap.error("--latency-report requires a full step and no --bandwidth-report")
    if a.bandwidth_report and (not a.bytes or a.grids):
        ap.error("--bandwidth-report requires --bytes and a full step (no --grids)")
    plan = load_plan(a.plan)
    spec = json.loads(Path(a.bytes).read_text()) if a.bytes else None
    tr = build_trace(plan, read_raw(a.raw), a.grids, spec, a.tick_ns)
    if a.latency_report:
        res = latency_report(tr)
    else:
        res = bandwidth_report(tr, spec, a.bw) if a.bandwidth_report else report(tr, a.bw)
    if a.html:
        write_html(tr, a.html, f"{plan.get('graph', 'ETX')} {a.grids or 'all'}", a.bw)
        print(f"\nwrote {a.html}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
