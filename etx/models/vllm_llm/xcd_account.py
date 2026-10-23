#!/usr/bin/env python3
"""Per-XCD wait account of one traced decode step (ETX_TRACE_RAW), on top of `etx.tools.timeline`.

  python -m examples.vllm_llm.xcd_account PLAN TRACE --bytes bytes.json --out DIR [--layer 24]

Every worker's step is cut into disjoint intervals, so the classes of one worker add up to the span:

  start      before the worker's first task is taken (slot claim, domain discovery, launch skew)      (d)
  between    previous task done -> this task taken: the untimed arrive and the next static-queue fetch (d)
  dep        taken -> the last producer's body done: the producer is still running                    (b)
  handoff    last producer done (or taken, if later) -> deps ready: producer arrive, event visibility,
             poll, acquire, touch, ctx setup and one barrier; a task without producers puts its
             whole wait here                                                                            (b)
  bytes      the part of a body charged to memory: the phase's bytes at the full bandwidth plus the
             exposed load round trips of its ISA chain (an estimate, see PHASE_HOPS)                    (a)
  machinery  the rest of the body: tile loops, LDS, reductions, MFMA/VALU, store drain, barriers, and
             any shortfall of the stream below the bandwidth                                            (c)
  none       after the worker's last task: its static queue is empty                                   (e)

The trace stamps three times per task (taken, deps ready, body done). Nothing inside a body is stamped, so
bytes/machinery is a split by estimate, not by measurement; the other classes are measured.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from etx.tools import timeline as tl

CLASSES = ("start", "between", "dep", "handoff", "bytes", "machinery", "none")
TAXON = {"start": "d", "between": "d", "dep": "b", "handoff": "b", "bytes": "a", "machinery": "c", "none": "e"}

# Exposed load round trips per task from the round11 ISA (docs/qwen/FLOORS-2026-10-08.md:127-140), priced as in
# docs/qwen/FLOORS-2026-10-08.md:84-88: coherent 0.50 us; plain 0.104 us when warm, 0.50 us after an acquire;
# scalar 0.104 us. Store drains and scratch reloads are machinery, not bytes. Streaming phases count only the
# hops before their first weight load ("pre"); the rest of their chain overlaps the stream.
# stem: (coherent, plain, scalar, acquired, streaming)
PHASE_HOPS = {
    "prologue": (0, 2, 2, False, False),
    "in_norm": (0, 7, 1, True, False),
    "qkv": (0, 3, 1, False, True),
    "prep": (1, 4, 1, False, False),
    "attn": (0, 5, 2, True, False),
    "reduce": (0, 5, 2, True, False),
    "o_proj": (0, 3, 1, False, True),
    "router": (1, 4, 1, False, False),
    "topk": (1, 1, 1, False, False),
    "moe_up": (1, 4, 1, False, True),
    "moe_down": (1, 4, 1, False, True),
    "lm_head": (0, 0, 0, False, True),
    "argmax": (0, 1, 1, True, False),
}
RTT_COH, RTT_CACHED = 0.50, 0.104


def stem(name: str) -> str:
    return re.sub(r"_\d+(?:_x)?$", "", name)


def hop_us(name: str, rtt: float = RTT_COH) -> float:
    h = PHASE_HOPS.get(stem(name))
    if h is None:
        return 0.0
    coh, plain, scalar, acquired, _ = h
    return coh * rtt + plain * (rtt if acquired else RTT_CACHED) + scalar * RTT_CACHED


def phase_floor_us(tr: tl.Trace, bw_tbs: float) -> dict[str, float]:
    """Each type's bytes at the full bandwidth: the time the whole phase needs if it streams at bw_tbs."""
    acc: dict[str, float] = {}
    for t in tr.by_id.values():
        acc[t.name] = acc.get(t.name, 0.0) + t.nbytes
    return {k: v / (bw_tbs * 1e6) for k, v in acc.items()}


def split_body(t: tl.Task, floor: dict[str, float], rtt: float) -> tuple[float, float]:
    body = t.done - t.ready
    a = min(body, floor.get(t.name, 0.0) + hop_us(t.name, rtt))
    return a, body - a


def worker_intervals(tr: tl.Trace, bw_tbs: float = 4.3, rtt: float = RTT_COH) -> dict[int, list[tuple[str, float, float, str]]]:
    """worker -> [(class, start us, end us, task type)], disjoint and covering [0, span]."""
    floor = phase_floor_us(tr, bw_tbs)
    by_w: dict[int, list[tl.Task]] = {}
    for t in tr.by_id.values():
        by_w.setdefault(t.worker, []).append(t)
    out = {}
    for w, ts in by_w.items():
        ts.sort(key=lambda t: t.taken)
        iv: list[tuple[str, float, float, str]] = []
        prev = None
        for t in ts:
            if prev is None:
                iv.append(("start", tr.t0, t.taken, t.name))
            else:
                iv.append(("between", prev.done, t.taken, t.name))
            prods = [tr.by_id[p] for p in tr.deps[t.id] if p in tr.by_id] if tr.deps is not None else []
            if prods:
                p_done = max(p.done for p in prods)
                mid = min(max(p_done, t.taken), t.ready)
                iv.append(("dep", t.taken, mid, t.name))
                iv.append(("handoff", mid, t.ready, t.name))
            else:
                iv.append(("handoff", t.taken, t.ready, t.name))
            a, _ = split_body(t, floor, rtt)
            iv.append(("bytes", t.ready, t.ready + a, t.name))
            iv.append(("machinery", t.ready + a, t.done, t.name))
            prev = t
        iv.append(("none", prev.done, tr.t1, ""))
        out[w] = [x for x in iv if x[2] > x[1]]
    return out


def class_totals(ivs) -> dict[str, float]:
    tot = dict.fromkeys(CLASSES, 0.0)
    for c, a, b, _ in ivs:
        tot[c] += b - a
    return tot


def per_xcd(tr: tl.Trace, wiv) -> list[dict]:
    """Per XCD: share of its worker-time in each class, by phase kind too."""
    dom = {t.worker: t.domain for t in tr.by_id.values()}
    rows = []
    for d in range(tr.n_domains):
        ws = [w for w in wiv if dom[w] == d]
        tot = dict.fromkeys(CLASSES, 0.0)
        kind: dict[str, dict[str, float]] = {}
        for w in ws:
            for c, a, b, name in wiv[w]:
                tot[c] += b - a
                if name:
                    k = kind.setdefault(stem(name), dict.fromkeys(CLASSES, 0.0))
                    k[c] += b - a
        denom = len(ws) * tr.span
        rows.append({"xcd": d, "workers": len(ws), "share": {c: v / denom for c, v in tot.items()},
                     "worker_us": tot, "by_kind_worker_us": kind})
    return rows


def busy_union(tr: tl.Trace, d: int) -> dict:
    """Time the XCD has at least one body running, and its mean number of workers in a body."""
    iv = sorted((t.ready, t.done) for t in tr.tasks if t.domain == d)
    end, union = -1e30, 0.0
    for a, b in iv:
        union += max(0.0, b - max(a, end))
        end = max(end, b)
    body = sum(b - a for a, b in iv)
    return {"xcd": d, "any_body_us": union, "any_body_share": union / tr.span, "mean_workers_in_body": body / tr.span}


def series(tr: tl.Trace, wiv, bin_us: float, t0: float, t1: float) -> dict:
    """Per XCD and class: mean number of workers in the class per bin (bins of bin_us over [t0, t1])."""
    n = max(1, int((t1 - t0) / bin_us + 0.999))
    dom = {t.worker: t.domain for t in tr.by_id.values()}
    cls = ("body", "dep", "handoff", "between", "start", "none")
    fold = {"bytes": "body", "machinery": "body"}
    out = {d: {c: [0.0] * n for c in cls} for d in range(tr.n_domains)}
    for w, ivs in wiv.items():
        rows = out[dom[w]]
        for c, a, b, _ in ivs:
            a, b = max(a, t0), min(b, t1)
            if b <= a:
                continue
            row = rows[fold.get(c, c)]
            i0, i1 = int((a - t0) / bin_us), min(n - 1, int((b - t0) / bin_us))
            for i in range(i0, i1 + 1):
                lo = t0 + i * bin_us
                row[i] += tl._clip(a, b, lo, lo + bin_us) / bin_us
    return {"bin_us": bin_us, "t0": t0, "t1": t1, "xcd": {str(d): {c: [round(v, 3) for v in r] for c, r in rows.items()}
                                                         for d, rows in out.items()}}


def critical_account(tr: tl.Trace, bw_tbs: float = 4.3, rtt: float = RTT_COH) -> dict:
    """The critical path (it telescopes to the span) cut into the same classes, by phase kind."""
    floor = phase_floor_us(tr, bw_tbs)
    steps = tl.critical_path(tr)
    kinds: dict[str, dict[str, float]] = {}
    tot = {"bytes_bw": 0.0, "bytes_hops": 0.0, "machinery": 0.0, "dep_gap": 0.0, "queue_gap": 0.0}
    for s in steps:
        t = s["task"]
        a, c = split_body(t, floor, rtt)
        bw = min(a, floor.get(t.name, 0.0))
        k = kinds.setdefault(stem(t.name), {"tasks": 0, **dict.fromkeys(tot, 0.0)})
        k["tasks"] += 1
        for key, v in (("bytes_bw", bw), ("bytes_hops", a - bw), ("machinery", c),
                       ("queue_gap" if s["kind"] == "queue" else "dep_gap", s["gap"])):
            k[key] += v
            tot[key] += v
    return {"steps": len(steps), "total": tot, "sum_us": sum(tot.values()), "span_us": tr.span, "by_kind": kinds}


def critical_edges(tr: tl.Trace) -> list[dict]:
    """Critical-path gaps by edge: producer kind -> consumer kind, same or another XCD, and the consumer's producer
    count. The gap runs from the critical producer's body done to the consumer's deps ready, so it holds the producer's
    untimed arrive and the consumer's whole phase 1."""
    steps = tl.critical_path(tr)
    acc: dict[tuple, list[tuple[float, float]]] = {}
    nprod: dict[tuple, int] = {}
    for a, b in zip(steps, steps[1:]):
        p, t = a["task"], b["task"]
        k = (stem(p.name), stem(t.name), "same" if p.domain == t.domain else "cross")
        # late: the consumer's worker took it only after the producer finished (still busy with its own previous task)
        acc.setdefault(k, []).append((b["gap"], max(0.0, min(t.taken, t.ready) - p.done), t.worker == p.worker))
        nprod[k] = len(tr.deps[t.id]) if tr.deps is not None else 0
    out = []
    for k, v in acc.items():
        g = sorted(x[0] for x in v)
        late = sorted(x[1] for x in v)
        out.append({"producer": k[0], "consumer": k[1], "xcd": k[2], "edges": len(g), "producers": nprod[k],
                    "sum_us": sum(g), "median_us": g[len(g) // 2], "max_us": g[-1],
                    "late_taken": sum(1 for x in v if x[1] > 0), "same_worker": sum(1 for x in v if x[2]),
                    "median_before_taken_us": late[len(late) // 2], "before_taken_sum_us": sum(late)})
    return sorted(out, key=lambda e: -e["sum_us"])


def layer_table(tr: tl.Trace, layer: int) -> list[dict]:
    """Which XCD runs which phase of one layer, and when (us from the step start)."""
    rx = re.compile(rf"_{layer}(?:_x)?$")
    acc: dict[tuple[str, int], dict] = {}
    for t in tr.tasks:
        if not rx.search(t.name):
            continue
        a = acc.setdefault((t.name, t.domain), {"type": t.name, "xcd": t.domain, "tasks": 0, "first_ready": 1e30,
                                                 "last_done": 0.0, "body_sum": 0.0})
        a["tasks"] += 1
        a["first_ready"] = min(a["first_ready"], t.ready)
        a["last_done"] = max(a["last_done"], t.done)
        a["body_sum"] += t.done - t.ready
    return sorted(acc.values(), key=lambda a: (a["first_ready"], a["xcd"]))


_SVG_COLORS = {"body": "#3b6fb6", "dep": "#d9822b", "handoff": "#c0413b", "between": "#8a63b8", "start": "#6f7c85",
               "none": "#d9dee4"}


def write_svg(ser: dict, workers: int, path: Path, title: str, marks: list[tuple[float, str]]) -> None:
    """Stacked areas: per XCD, how many of its workers are in each class over time (one row per XCD)."""
    xs = ser["xcd"]
    n = len(next(iter(xs.values()))["body"])
    W, RH, LW, TOP = 1100, 70, 60, 34
    H = TOP + RH * len(xs) + 40
    sx = W / n
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W + LW + 10}" height="{H}" font-family="sans-serif" font-size="10">',
           f'<text x="4" y="14" font-size="12">{title}</text>']
    order = ("body", "dep", "handoff", "between", "start", "none")
    for k, c in enumerate(order):
        out.append(f'<rect x="{LW + k * 120}" y="20" width="10" height="10" fill="{_SVG_COLORS[c]}"/>'
                   f'<text x="{LW + k * 120 + 14}" y="29">{c}</text>')
    for r, (d, rows) in enumerate(sorted(xs.items(), key=lambda kv: int(kv[0]))):
        y0 = TOP + r * RH + RH - 4
        out.append(f'<text x="4" y="{y0 - RH / 2 + 8}">XCD {d}</text>')
        base = [0.0] * n
        for c in order:
            vals = rows[c]
            pts_top = [(LW + i * sx, y0 - (base[i] + vals[i]) / workers * (RH - 8)) for i in range(n)]
            pts_bot = [(LW + i * sx, y0 - base[i] / workers * (RH - 8)) for i in range(n)]
            poly = pts_top + [(x + sx, y) for x, y in pts_top[-1:]] + [(x + sx, y) for x, y in pts_bot[-1:]] + pts_bot[::-1]
            out.append('<polygon fill="{}" stroke="none" points="{}"/>'.format(
                _SVG_COLORS[c], " ".join(f"{x:.1f},{y:.1f}" for x, y in poly)))
            base = [b + v for b, v in zip(base, vals)]
    t0, bin_us = ser["t0"], ser["bin_us"]
    for t, label in marks:
        x = LW + (t - t0) / bin_us * sx
        out.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{H - 30}" stroke="#1d232b" stroke-width="0.4"/>'
                   f'<text x="{x + 1:.1f}" y="{H - 18}" transform="rotate(45 {x + 1:.1f} {H - 18})">{label}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out))


def account(plan: dict, rows, spec: dict | None, layer: int, bw_tbs: float = 4.3, rtt: float = RTT_COH) -> dict:
    tr = tl.build_trace(plan, rows, spec=spec)
    if tr.notes or tr.deps is None:
        raise SystemExit(f"need a full trace and a plan with dependency lists: {tr.notes}")
    wiv = worker_intervals(tr, bw_tbs, rtt)
    machine = class_totals([x for ivs in wiv.values() for x in ivs])
    nw = len(wiv)
    floor = phase_floor_us(tr, bw_tbs)
    return {
        "span_us": tr.span, "workers": nw, "xcds": tr.n_domains, "bw_tbs": bw_tbs, "rtt_us": rtt,
        "bytes_total": sum(t.nbytes for t in tr.by_id.values()),
        "floor_us": sum(floor.values()),
        "machine_worker_us": machine, "machine_share": {c: v / (nw * tr.span) for c, v in machine.items()},
        "taxonomy_share": {k: sum(v for c, v in machine.items() if TAXON[c] == k) / (nw * tr.span) for k in "abcde"},
        "per_xcd": per_xcd(tr, wiv),
        "busy": [busy_union(tr, d) for d in range(tr.n_domains)],
        "critical": critical_account(tr, bw_tbs, rtt),
        "critical_edges": critical_edges(tr),
        "layer": layer, "layer_table": layer_table(tr, layer),
        "_tr": tr, "_wiv": wiv,
    }


def text_report(r: dict, out=sys.stdout) -> None:
    p = lambda s="": print(s, file=out)   # noqa: E731
    span, nw = r["span_us"], r["workers"]
    p(f"span {span:.1f} us, {nw} workers ({r['xcds']} XCDs), bytes {r['bytes_total'] / 1e9:.4f} GB, "
      f"floor {r['floor_us']:.1f} us at {r['bw_tbs']} TB/s = {r['floor_us'] / span * 100:.1f}% of the span; RTT {r['rtt_us']} us")
    p("\nmachine worker-time by class (us summed over workers; share of workers x span)")
    for c in CLASSES:
        p(f"  {c:10} ({TAXON[c]})  {r['machine_worker_us'][c]:12.1f} worker-us  {r['machine_share'][c] * 100:6.2f}%")
    p("  taxonomy: " + "  ".join(f"({k}) {v * 100:.2f}%" for k, v in r["taxonomy_share"].items()))
    p(f"\nper XCD: share of its workers x span ({nw // r['xcds']} workers per XCD); any body = at least one body running")
    p(f"  {'XCD':>3} " + " ".join(f"{c:>9}" for c in CLASSES) + f" {'any body':>9} {'mean busy':>9}")
    for x, b in zip(r["per_xcd"], r["busy"]):
        p(f"  {x['xcd']:>3} " + " ".join(f"{x['share'][c] * 100:8.2f}%" for c in CLASSES)
          + f" {b['any_body_share'] * 100:8.2f}% {b['mean_workers_in_body']:9.2f}")
    p("\nper XCD and phase kind: body worker-us (bytes + machinery), whole step")
    kinds = sorted({k for x in r["per_xcd"] for k in x["by_kind_worker_us"]},
                   key=lambda k: list(PHASE_HOPS).index(k) if k in PHASE_HOPS else 99)
    p(f"  {'kind':10} " + " ".join(f"{'XCD ' + str(x['xcd']):>9}" for x in r["per_xcd"]))
    for k in kinds:
        vals = [x["by_kind_worker_us"].get(k, {}) for x in r["per_xcd"]]
        p(f"  {k:10} " + " ".join(f"{v.get('bytes', 0) + v.get('machinery', 0):9.1f}" for v in vals))
    c = r["critical"]
    p(f"\ncritical path: {c['steps']} tasks, {c['sum_us']:.1f} us (span {c['span_us']:.1f})")
    p(f"  {'kind':10} {'tasks':>5} {'bytes bw':>9} {'bytes hop':>9} {'machinery':>9} {'dep gap':>9} {'queue gap':>9} {'total':>9}")
    for k, v in sorted(c["by_kind"].items(), key=lambda kv: -sum(x for kk, x in kv[1].items() if kk != "tasks")):
        tot = sum(x for kk, x in v.items() if kk != "tasks")
        p(f"  {k:10} {v['tasks']:5d} {v['bytes_bw']:9.1f} {v['bytes_hops']:9.1f} {v['machinery']:9.1f} "
          f"{v['dep_gap']:9.1f} {v['queue_gap']:9.1f} {tot:9.1f}")
    t = c["total"]
    p(f"  {'TOTAL':10} {'':5} {t['bytes_bw']:9.1f} {t['bytes_hops']:9.1f} {t['machinery']:9.1f} "
      f"{t['dep_gap']:9.1f} {t['queue_gap']:9.1f} {c['sum_us']:9.1f}")
    stream = sum(v["machinery"] for k, v in c["by_kind"].items() if PHASE_HOPS.get(k, (0,) * 5)[4])
    p(f"  machinery in streaming phases (body above floor + hops) {stream:.1f} us, in latency phases "
      f"{t['machinery'] - stream:.1f} us")
    p("  shares of the span: " + "  ".join(f"{k} {v / c['span_us'] * 100:.1f}%" for k, v in t.items()))
    p("\ncritical-path gaps by edge (producer -> consumer; the first step's gap from the window start is not an edge)")
    p("  late = consumer taken after the producer's body done; before taken = median of that part of the gap")
    p(f"  {'producer':10} {'consumer':10} {'XCD':5} {'edges':>5} {'producers':>9} {'sum us':>8} {'median':>7} {'max':>6}"
      f" {'late':>5} {'same wkr':>8} {'before taken':>12}")
    for e in r["critical_edges"]:
        p(f"  {e['producer']:10} {e['consumer']:10} {e['xcd']:5} {e['edges']:5d} {e['producers']:9d} {e['sum_us']:8.1f} "
          f"{e['median_us']:7.2f} {e['max_us']:6.2f} {e['late_taken']:5d} {e['same_worker']:8d} {e['median_before_taken_us']:12.2f}")
    e = r["critical_edges"]
    p(f"  TOTAL edges {sum(x['edges'] for x in e)}, gaps {sum(x['sum_us'] for x in e):.1f} us, of which before taken "
      f"{sum(x['before_taken_sum_us'] for x in e):.1f} us on {sum(x['late_taken'] for x in e)} late edges")
    p(f"\nlayer {r['layer']}: which XCD runs which phase, us from the step start")
    p(f"  {'type':14} {'XCD':>3} {'tasks':>5} {'first ready':>11} {'last done':>9} {'mean body':>9}")
    for a in r["layer_table"]:
        p(f"  {a['type']:14} {a['xcd']:3d} {a['tasks']:5d} {a['first_ready']:11.2f} {a['last_done']:9.2f} "
          f"{a['body_sum'] / a['tasks']:9.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m examples.vllm_llm.xcd_account", description=__doc__.split("\n\n")[0])
    ap.add_argument("plan")
    ap.add_argument("raw")
    ap.add_argument("--bytes", help="timeline byte spec (examples.vllm_llm.bw_spec)")
    ap.add_argument("--out", required=True, help="directory for account.txt, account.json, series and SVG")
    ap.add_argument("--layer", type=int, default=24, help="the layer drawn and tabulated per XCD")
    ap.add_argument("--bw", type=float, default=4.3)
    ap.add_argument("--rtt", type=float, default=RTT_COH, help="coherent round trip, us (0.50 to 0.69 measured)")
    ap.add_argument("--bin", type=float, default=0.25, help="series bin, us, for the layer window")
    a = ap.parse_args(argv)
    plan = tl.load_plan(a.plan)
    spec = json.loads(Path(a.bytes).read_text()) if a.bytes else None
    r = account(plan, tl.read_raw(a.raw), spec, a.layer, a.bw, a.rtt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "account.txt").open("w") as f:
        text_report(r, f)
    tr, wiv = r.pop("_tr"), r.pop("_wiv")
    (out / "account.json").write_text(json.dumps(r, indent=1))
    rx = re.compile(rf"_{a.layer}(?:_x)?$")
    lt = [t for t in tr.tasks if rx.search(t.name)]
    t0, t1 = min(t.ready for t in lt) - 2.0, max(t.done for t in lt) + 2.0
    ser = series(tr, wiv, a.bin, t0, t1)
    (out / f"xcd_series_layer{a.layer}.json").write_text(json.dumps(ser, separators=(",", ":")))
    marks = sorted({(min(t.ready for t in lt if t.name == n), stem(n)) for n in {t.name for t in lt}})
    write_svg(ser, tr.workers_per_domain, out / f"xcd_layer{a.layer}.svg",
              f"Workers per XCD by class, layer {a.layer} ({t0:.1f}-{t1:.1f} us of the step), {a.bin} us bins", marks)
    step = series(tr, wiv, 5.0, tr.t0, tr.t1)
    (out / "xcd_series_step.json").write_text(json.dumps(step, separators=(",", ":")))
    write_svg(step, tr.workers_per_domain, out / "xcd_step.svg", f"Workers per XCD by class, whole step, 5 us bins", [])
    print(f"wrote {out}/account.txt, account.json, xcd_layer{a.layer}.svg, xcd_step.svg and the series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
