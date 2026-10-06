"""Read raw traces and compare the device budget for each layer. No GPU is required."""
from __future__ import annotations

import argparse
import heapq
import json
import re
from pathlib import Path

from . import timeline as tl


METRIC_LIMITS = (
    "Type-id order is not dependency order. "
    "Negative overlaps can cancel positive gaps in both sums. "
    "Neither sum measures exposed device time. "
    "Do not use either sum alone for the 8.3 us rental gate."
)


def dependency_order(plan):
    """Sort phases by dependencies. Use the lowest type id to resolve ties."""
    deps = tl._deps_from_plan(plan)
    if deps is None:
        return None
    types = [d["type"] for d in plan["descs"]]
    parents = {ty: set() for ty in types}
    children = {ty: set() for ty in types}
    for target, producers in enumerate(deps):
        for source in producers:
            a, b = types[source], types[target]
            if a != b:
                parents[b].add(a)
                children[a].add(b)
    ready = [ty for ty in parents if not parents[ty]]
    heapq.heapify(ready)
    order = []
    while ready:
        ty = heapq.heappop(ready)
        order.append(ty)
        for child in sorted(children[ty]):
            parents[child].remove(ty)
            if not parents[child]:
                heapq.heappush(ready, child)
    # A task graph can be acyclic while its phase graph has a cycle.
    return order if len(order) == len(parents) else None


def layer_of(name):
    match = re.search(r"_(\d+)(?:_x)?$", name)
    return int(match[1]) if match else None


def analyze(plan, rows, tick_ns=tl.TICK_NS):
    """Use integer tick differences to reproduce host.h report_phases."""
    if len(rows) != len(plan["descs"]):
        raise ValueError("trace is not complete: task count does not match the plan")
    names = {v["id"]: k for k, v in plan["types"].items()}
    for desc, row in zip(plan["descs"], rows):
        if len(row) != tl.ROW or any(t <= 0 for t in row[5:8]):
            raise ValueError(f"trace is not complete: task {desc['id']} needs three positive timestamps")
        coord = (desc["coord"] + [0] * 4)[:4]
        if row[0] != desc["type"] or list(row[1:5]) != coord:
            raise ValueError(f"trace task {desc['id']} does not match the plan")
    active = rows
    if not active:
        raise ValueError("trace is not complete: no executed tasks")
    if any(not r[5] <= r[6] <= r[7] for r in active):
        raise ValueError("trace timestamps are not in task order")
    origin = min(r[5] for r in active)
    k = tick_ns / 1000
    groups = {}
    waits = {}
    workers = sorted({r[8] & 0xFFFFFFFF for r in active})
    for r in active:
        groups.setdefault(r[0], []).append(r)
        layer = layer_of(names[r[0]])
        if layer is not None:
            w = r[8] & 0xFFFFFFFF
            acc = waits.setdefault(layer, {})
            acc[w] = acc.get(w, 0) + r[6] - r[5]
    phases = []
    for ty, rs in sorted(groups.items()):
        first = min(r[6] for r in rs)
        phases.append(dict(type=names[ty], tasks=len(rs),
                           wait=sum(r[6] - r[5] for r in rs) / len(rs) * k,
                           body=sum(r[7] - r[6] for r in rs) / len(rs) * k,
                           first_ready=(first - origin) * k,
                           last_ready=(max(r[6] for r in rs) - first) * k,
                           last_done=(max(r[7] for r in rs) - first) * k))
    layers = {layer: dict(layer=layer, gap=0.0, edges=[],
                          worker_wait={str(w): ws.get(w, 0) * k for w in workers},
                          wait_per_worker=sum(ws.values()) * k / len(workers))
              for layer, ws in sorted(waits.items())}
    # Use type order, as in the host table. Charge the outgoing edge to its source layer.
    for a, b in zip(phases, phases[1:]):
        layer = layer_of(a["type"])
        if layer is not None:
            gap = b["first_ready"] - (a["first_ready"] + a["last_done"])
            layers[layer]["gap"] += gap
            layers[layer]["edges"].append(dict(source=a["type"], target=b["type"], gap=gap))
    order = dependency_order(plan)
    by_type = {ty: phase for ty, phase in zip(sorted(groups), phases)}
    for p in layers.values():
        p["dependency_gap"] = 0.0 if order is not None else None
        p["dependency_edges"] = []
    if order is not None:
        for src, dst in zip(order, order[1:]):
            a, b = by_type[src], by_type[dst]
            layer = layer_of(a["type"])
            if layer is not None:
                gap = b["first_ready"] - (a["first_ready"] + a["last_done"])
                layers[layer]["dependency_gap"] += gap
                layers[layer]["dependency_edges"].append(
                    dict(source=a["type"], target=b["type"], gap=gap))
    tr = tl.build_trace(plan, rows, tick_ns=tick_ns)
    steps = tl.critical_path(tr)
    return dict(span=(max(r[7] for r in active) - origin) * k, phases=phases,
                layers=layers, critical=tl.critical_by_type(steps),
                critical_span=sum(s["body"] + s["gap"] for s in steps),
                dependency_lists=tr.deps is not None,
                dependency_order=[names[ty] for ty in order] if order is not None else None,
                metric_limits=METRIC_LIMITS)


def compare(a, b, shared_body):
    """Return A minus B. Wait totals are worker time, not exposed device time."""
    if a["layers"].keys() != b["layers"].keys():
        raise ValueError("traces have different layer sets")
    out = []
    for layer, x in a["layers"].items():
        y = b["layers"][layer]
        shared = []
        body = []
        for result in (a, b):
            selected = {p["type"]: p["body"] for p in result["phases"]
                        if layer_of(p["type"]) == layer and re.search(shared_body, p["type"])}
            shared.append(set(selected))
            body.append(sum(selected.values()))
        if not shared[0] or shared[0] != shared[1]:
            raise ValueError(f"layer {layer}: shared body phases do not match")
        if x["worker_wait"].keys() != y["worker_wait"].keys():
            raise ValueError("traces have different worker sets")
        gap = x["gap"] - y["gap"]
        out.append(dict(layer=layer, gap=gap, body=body[0] - body[1],
                        excess=gap + body[0] - body[1],
                        wait_per_worker=x["wait_per_worker"] - y["wait_per_worker"],
                        worker_wait={w: v - y["worker_wait"][w] for w, v in x["worker_wait"].items()}))
    return out


def report(result, label):
    print(f"\ntrace {label}")
    print("Times are in us. Last-ready and last-done are relative to each phase first-ready.")
    print(f"phase trace (us): total span {result['span']:.1f}")
    print(f"  {'type':22} {'tasks':>6} {'wait':>9} {'body':>9} {'first-ready':>11} {'last-ready':>10} {'last-done':>10}")
    for p in result["phases"]:
        print(f"  {p['type']:22} {p['tasks']:6d} {p['wait']:9.2f} {p['body']:9.2f} "
              f"{p['first_ready']:11.1f} {p['last_ready']:10.1f} {p['last_done']:10.1f}")
    print("\nEach layer owns its outgoing edge. Negative gaps indicate overlap.")
    print(result["metric_limits"])
    print("Dependency order uses phase dependencies, with the lowest type id first for ties.")
    print("Consecutive phases in this order do not necessarily share a dependency.")
    if result["dependency_order"] is None:
        print("Dependency-order gap is unavailable: dependency lists are absent or phase dependencies have a cycle.")
    print("Wait totals measure worker time, not exposed device time.")
    print("layer type_id_gap_us dependency_gap_us wait_per_worker_us")
    for layer, p in result["layers"].items():
        dep = "unavailable" if p["dependency_gap"] is None else f"{p['dependency_gap']:.2f}"
        print(f"{layer} {p['gap']:.2f} {dep} {p['wait_per_worker']:.2f}")
    print("\nlayer worker wait_total_us")
    for layer, p in result["layers"].items():
        for w, wait in p["worker_wait"].items():
            print(f"{layer} {w} {wait:.2f}")
    if not result["dependency_lists"]:
        print("NOTE: plan has no dependency lists. The path uses queue order only.")
    print(f"\nexposed critical path (us): {result['critical_span']:.2f}")
    print("type tasks body_us dep_gap_us queue_gap_us")
    for p in result["critical"]:
        print(f"{p['type']} {p['tasks']} {p['body']:.2f} {p['dep_gap']:.2f} {p['queue_gap']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan")
    ap.add_argument("raw")
    ap.add_argument("--compare", nargs=2, metavar=("PLAN_B", "RAW_B"))
    ap.add_argument("--shared-body", default=r"^(attn|reduce)_", help="regex for shared body phases")
    ap.add_argument("--tick-ns", type=float, default=tl.TICK_NS)
    ap.add_argument("--json", help="save full results and per-edge gaps")
    args = ap.parse_args(argv)
    if args.tick_ns <= 0:
        ap.error("--tick-ns must be positive")
    def read(plan, raw):
        return analyze(tl.load_plan(plan), tl.read_raw(raw), args.tick_ns)
    try:
        a = read(args.plan, args.raw)
        b = read(*args.compare) if args.compare else None
        delta = compare(a, b, args.shared_body) if b else None
    except ValueError as exc:
        ap.error(str(exc))
    report(a, "A")
    if b:
        report(b, "B")
        print("\nA minus B (us). Excess = gap difference + shared mean body difference.")
        print("layer gap body excess wait_per_worker")
        for p in delta:
            print(f"{p['layer']} {p['gap']:.2f} {p['body']:.2f} {p['excess']:.2f} {p['wait_per_worker']:.2f}")
        print("\nA minus B: layer worker wait_total_us")
        for p in delta:
            for w, wait in p["worker_wait"].items():
                print(f"{p['layer']} {w} {wait:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(A=a, B=b, difference=delta), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
