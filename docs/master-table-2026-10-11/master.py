#!/usr/bin/env python3
"""Master decomposition of one traced decode step on the kept KEY32-on build (CPU only, no GPU).

  .venv/bin/python docs/master-table-2026-10-11/master.py > docs/master-table-2026-10-11/master.txt

Inputs (untracked archive under ~/code/etx/): results_qwen300_2026-10-11/tr_on.bin, the borrowed plan
results_vlbest2_2026-10-10/b_all/plan.json (row-checked by the review, 0/60362 mismatches), the byte spec
docs/xcd-timeline-2026-10-11/bytes.json. Every number is a difference of trace stamps [M], the byte/hop model of
examples/vllm_llm/xcd_account.py [E], or a remainder of the two [R].

Views printed, each of which must total the critical path (= the span):
  BODY    per role: critical body [M] + gap charged to the role as consumer [M]
  WAIT    per role: bytes at 4.3 TB/s [E] + exposed hops [E] + machinery residual [R] + gap [M]
  LEDGER  per role: 48 x round11 floor [E] + allowance B [E] + layer/critical-task correction [R] + gap [M]
  FATE    per role: gap split into before-taken [M], free-consumer notification [M or R], late after-take [R]
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from etx.tools import timeline as tl  # noqa: E402
from examples.vllm_llm import xcd_account as xa  # noqa: E402

ARCH = Path(os.path.expanduser("~/code/etx"))
PLAN = ARCH / "results_vlbest2_2026-10-10/b_all/plan.json"
TRACE = ARCH / "results_qwen300_2026-10-11/tr_on.bin"
BYTES = ROOT / "docs/xcd-timeline-2026-10-11/bytes.json"
ROLES = ["prologue", "in_norm", "qkv", "prep", "attn", "reduce", "o_proj", "router", "topk", "moe_up", "moe_down",
         "lm_head", "argmax"]
LAYER_ROLES = ["qkv", "prep", "attn", "reduce", "o_proj", "router", "topk", "moe_up", "moe_down"]
# docs/FLOORS-2026-10-08.md:239-247 via docs/review-pool-analysis-2026-10-11/rebase.py:36-37
FLOOR = {"qkv": 7.55, "prep": 1.52, "attn": 2.92, "reduce": 2.81, "o_proj": 5.30, "router": 1.52,
         "topk": 1.21, "moe_up": 14.67, "moe_down": 9.48}
# grids whose phase 1 runs the touch hook (waves 1.. load the first weight trip while wave 0 waits)
TOUCH = {"qkv", "o_proj", "router", "lm_head"}


def main() -> int:
    plan = json.loads(PLAN.read_text())
    spec = json.loads(BYTES.read_text())
    tr = tl.build_trace(plan, tl.read_raw(TRACE), None, spec)
    steps = tl.critical_path(tr)
    floor = xa.phase_floor_us(tr, 4.3)
    span = tr.span
    print(f"trace {TRACE.name}: span {span:.3f} us, {len(tr.tasks)} tasks, path {len(steps)} tasks")

    # ---- per role, on the critical path
    R = {r: {"path_tasks": 0, "body": 0.0, "bw": 0.0, "hop": 0.0, "mach": 0.0, "gap": 0.0, "bt": 0.0,
             "free": 0.0, "free_n": 0, "late_after": 0.0, "late_n": 0, "edges_same": 0, "edges_cross": 0,
             "first_gap": 0.0} for r in ROLES}
    prev = None
    for s in steps:
        t = s["task"]
        r = xa.stem(t.name)
        a, c = xa.split_body(t, floor, xa.RTT_COH)
        bw = min(a, floor.get(t.name, 0.0))
        x = R[r]
        x["path_tasks"] += 1
        x["body"] += s["body"]
        x["bw"] += bw
        x["hop"] += a - bw
        x["mach"] += c
        x["gap"] += s["gap"]
        if prev is None:
            x["first_gap"] += s["gap"]                      # window start -> first task's deps ready
        else:
            p = prev
            bt = max(0.0, min(t.taken, t.ready) - p.done)   # consumer still busy when its producer finished
            if bt > 0:
                x["bt"] += bt
                x["late_after"] += s["gap"] - bt
                x["late_n"] += 1
            else:
                x["free"] += s["gap"]
                x["free_n"] += 1
            x["edges_same" if p.domain == t.domain else "edges_cross"] += 1
        prev = t

    # ---- per role, all tasks: XCDs, tasks/step, bytes/step, mean and median body, critical-task excess
    allt: dict[str, list] = {r: [] for r in ROLES}
    for t in tr.tasks:
        allt[xa.stem(t.name)].append(t)
    print("\nL1 ROLES (all tasks of the step; path = critical path)")
    print("role       tasks/step  XCDs        MB/step   mean body  median body | path: tasks  body[M]  bytes_bw[E]"
          "  hops[E]  mach[R]  gap[M]  total")
    tot = dict.fromkeys(["body", "bw", "hop", "mach", "gap"], 0.0)
    for r in ROLES:
        ts = allt[r]
        doms = sorted({t.domain for t in ts})
        mb = sum(t.nbytes for t in ts) / 1e6
        bodies = [t.done - t.ready for t in ts]
        x = R[r]
        for k in tot:
            tot[k] += x[k]
        print(f"{r:9s} {len(ts):11d}  {str(doms[0]) + '-' + str(doms[-1]) if len(doms) > 1 else str(doms[0]):10s}"
              f" {mb:9.3f} {statistics.mean(bodies):10.3f} {statistics.median(bodies):11.3f} |"
              f" {x['path_tasks']:10d} {x['body']:8.1f} {x['bw']:11.1f} {x['hop']:8.1f} {x['mach']:8.1f}"
              f" {x['gap']:7.1f} {x['body'] + x['gap']:7.1f}")
    print(f"{'TOTAL':9s} {len(tr.tasks):11d}  {'':10s} {sum(t.nbytes for t in tr.tasks) / 1e6:9.3f} {'':10s} {'':11s} |"
          f" {len(steps):10d} {tot['body']:8.1f} {tot['bw']:11.1f} {tot['hop']:8.1f} {tot['mach']:8.1f}"
          f" {tot['gap']:7.1f} {tot['body'] + tot['gap']:7.1f}")
    print(f"check: body + gap = {tot['body'] + tot['gap']:.3f} us vs span {span:.3f} us")

    # ---- gap fate split per consumer role
    print("\nGAP FATE (charged to the consumer role)")
    print("role       gap[M]  window-start  before-taken[M] (late n)  late after-take  free (n)  free-into-touch?")
    g = dict.fromkeys(["gap", "first_gap", "bt", "late_after", "free"], 0.0)
    free_touch = free_plain = 0.0
    for r in ROLES:
        x = R[r]
        for k in g:
            g[k] += x[k]
        if r in TOUCH:
            free_touch += x["free"]
        else:
            free_plain += x["free"]
        print(f"{r:9s} {x['gap']:7.1f} {x['first_gap']:13.1f} {x['bt']:16.1f} ({x['late_n']:2d}) {x['late_after']:16.1f}"
              f" {x['free']:8.1f} ({x['free_n']:2d})  {'yes' if r in TOUCH else 'no'}")
    print(f"{'TOTAL':9s} {g['gap']:7.1f} {g['first_gap']:13.1f} {g['bt']:16.1f}      {g['late_after']:16.1f} {g['free']:8.1f}")
    print(f"free edges into touch grids {free_touch:.1f} us, into non-touch grids {free_plain:.1f} us")
    print(f"check: window-start + before-taken + late after-take + free = "
          f"{g['first_gap'] + g['bt'] + g['late_after'] + g['free']:.3f} vs gap {g['gap']:.3f}")

    # ---- ledger view: 48 x floor + allowance B (rebase.py rule) + correction to the path body
    r11 = __import__("etx.tools.floors", fromlist=["round11_from_trace"]).round11_from_trace(PLAN, TRACE)
    print("\nLEDGER VIEW (layer roles; B = 48 x (critical body - floor), docs/PLAN-generic-decoder-2026-10-08.md:154)")
    print("role       floor  crit/layer  48xfloor[E]  B[E]     48xcrit  path body[M]  correction[R]")
    lt = dict.fromkeys(["f", "b", "corr"], 0.0)
    for r in LAYER_ROLES:
        c = r11[r]["crit_body"]
        f48, b = 48 * FLOOR[r], 48 * (c - FLOOR[r])
        corr = R[r]["body"] - 48 * c
        lt["f"] += f48; lt["b"] += b; lt["corr"] += corr
        print(f"{r:9s} {FLOOR[r]:6.2f} {c:11.3f} {f48:11.1f} {b:7.1f} {48 * c:10.1f} {R[r]['body']:12.1f} {corr:13.1f}")
    nonlayer = sum(R[r]["body"] for r in ROLES if r not in LAYER_ROLES)
    print(f"{'layers':9s} {'':6s} {'':11s} {lt['f']:11.1f} {lt['b']:7.1f} {'':10s} {'':12s} {lt['corr']:13.1f}")
    print(f"non-layer bodies [M] (prologue, in_norm, lm_head, argmax) {nonlayer:.1f}; gaps [M] {g['gap']:.1f}")
    print(f"check: {lt['f']:.1f} + {lt['b']:.1f} + {lt['corr']:.1f} + {nonlayer:.1f} + {g['gap']:.1f} = "
          f"{lt['f'] + lt['b'] + lt['corr'] + nonlayer + g['gap']:.3f} vs span {span:.3f}")

    # ---- per-layer-role statistics used by the L2 tables: slowest-task excess and start skew (all 48 layers)
    print("\nPER-LAYER ROLE STATS, layers 1-45 [M] (round11_from_trace): crit body, mean body, start skew of the last task")
    for r in LAYER_ROLES:
        x = r11[r]
        print(f"{r:9s} tasks/layer {x['tasks']:4d}  crit {x['crit_body']:.3f}  mean {x['mean_body']:.3f}  "
              f"crit-mean {x['crit_body'] - x['mean_body']:.3f}  start {x['start']:.3f}  period-share {x['delta']:.3f}")

    # ---- the [M]/[E]/[R] split of the step
    m = g["bt"] + free_plain
    e = tot["bw"] + tot["hop"]
    rr = tot["mach"] + g["late_after"] + free_touch + g["first_gap"]
    print("\nKNOWN/ESTIMATED/RESIDUAL split of the span (L3 resolution)")
    print(f"[M] before-taken {g['bt']:.1f} + free notification into non-touch grids {free_plain:.1f} = {m:.1f} us "
          f"({100 * m / span:.1f}%)")
    print(f"[E] bytes at 4.3 TB/s {tot['bw']:.1f} + exposed hops {tot['hop']:.1f} = {e:.1f} us ({100 * e / span:.1f}%)")
    print(f"[R] machinery {tot['mach']:.1f} + late after-take {g['late_after']:.1f} + free into touch grids "
          f"{free_touch:.1f} + window start {g['first_gap']:.1f} = {rr:.1f} us ({100 * rr / span:.1f}%)")
    print(f"check: {m:.1f} + {e:.1f} + {rr:.1f} = {m + e + rr:.3f} vs span {span:.3f}")
    print("per role [M] / [E] / [R] us:")
    for r in ROLES:
        x = R[r]
        mm = x["bt"] + (0.0 if r in TOUCH else x["free"])
        ee = x["bw"] + x["hop"]
        rrr = x["mach"] + x["late_after"] + (x["free"] if r in TOUCH else 0.0) + x["first_gap"]
        print(f"  {r:9s} {mm:7.1f} {ee:7.1f} {rrr:7.1f}  = {mm + ee + rrr:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
