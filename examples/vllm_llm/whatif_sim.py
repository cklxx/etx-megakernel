#!/usr/bin/env python3
"""What each lever is worth on the MoE step, from the XCD simulation (docs/MODEL-vllm-kernels.md section 8):

  levers    one request: the argument path per imported kernel, hardware barriers, cheaper hand-offs
  interleave  two requests in one megakernel, one workgroup per CU: the second request's tasks fill the CUs the first
            leaves idle (1-8-task launches, the attention block). Every worker's queue is the merge of both requests'
            queues, ordered by each task's start in a one-request simulation (request 1 shifted by an offset); every
            dependency has an earlier key than its dependant, so the merge cannot deadlock.

  python3 -m examples.vllm_llm.whatif_sim [--plan results_2026-09-28/plan_moe_xcd.json]
"""
from __future__ import annotations

import argparse
import copy
import json

from etx.sim.xcd import Hw, simulate
from examples.vllm_llm import kmodel as km
from examples.vllm_llm.xcd_run import cost_fn

TICK = 0.01


def hw_of(p, h_dom=1.0, h_dev=1.3):
    return Hw(bw_max=p.bw, n_half=0.0, handoff_us={"DOMAIN": h_dom, "DEVICE": h_dev}, wake_us=0.6, rate_cv=0.03, lat_us=p.t_v)


def two_requests(plan: dict, rows: list, offset_us: float) -> dict:
    """plan with a second copy of every task and event; queues merged by the one-request start times."""
    n, ne = len(plan["descs"]), len(plan["ev_inst"])
    two = copy.copy(plan)
    d1 = [dict(d, id=d["id"] + n) for d in plan["descs"]]
    two["descs"] = plan["descs"] + d1
    two["ev_inst"] = plan["ev_inst"] + [[e, c, [p + n for p in prods]] for e, c, prods in plan["ev_inst"]]
    two["task_in"] = plan["task_in"] + [[e + ne for e in evs] for evs in plan["task_in"]]
    start = [r[6] * TICK if r[6] else 1e12 for r in rows]         # ready time of every task alone
    q = {}
    for key, ids in plan["static_queues"].items():
        merged = [(start[i], 0, i) for i in ids] + [(start[i] + offset_us, 1, i + n) for i in ids]
        q[key] = [i for _, _, i in sorted(merged)]
    two["static_queues"] = q
    return two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="results_2026-09-28/plan_moe_xcd.json")
    ap.add_argument("--measured", default="results_2026-09-28/measured.json")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    base = km.fit(json.load(open(a.measured)))

    def run(front=None, hwbar=False, h_dom=1.0, h_dev=1.3, pl=None):
        p = km.Params(**{**base.__dict__, "hw": dict(base.hw)})
        if hwbar:
            p.hw["t_swbar"] = p.hw["t_bar"]
        if front is not None:
            p.A_task = front
        rows, st = simulate(pl or plan, cost_fn("qwen3-30b-a3b", p), hw_of(p, h_dom, h_dev))
        return rows, st["makespan_us"] / 1e3

    print("one request, ms per token (simulated; the simulation is ~12% optimistic against the measured 6.63)")
    cases = [("today", {}), ("hardware barriers", dict(hwbar=True)), ("argument path 2.1 -> 0.5 us", dict(front=0.5)),
             ("both", dict(hwbar=True, front=0.5)), ("both + hand-offs halved", dict(hwbar=True, front=0.5, h_dom=0.5, h_dev=0.65)),
             ("both + front 0 (arguments free)", dict(hwbar=True, front=0.0)),
             ("everything above + hand-offs 0 (bound)", dict(hwbar=True, front=0.0, h_dom=0.0, h_dev=0.0))]
    one = {}
    for name, kw in cases:
        rows, ms = run(**kw)
        one[name] = (rows, ms, kw)
        print(f"  {name:42s} {ms:6.3f} ms  {1e3 / ms:6.0f} tokens/s")
    print("vLLM measured: batch 1 4.898 ms (204 tokens/s), batch 2 5.724 ms per step (349 tokens/s), batch 4 563 tokens/s")
    print("\ntwo requests interleaved (one workgroup per CU), ms per step (two tokens), tokens/s")
    for name in ("today", "both"):
        rows, ms1, kw = one[name]
        for off in (0.0, 30.0, 60.0):
            ms2 = run(pl=two_requests(plan, rows, off), **kw)[1]
            print(f"  {name:10s} offset {off:4.0f} us: {ms2:6.3f} ms per step  {2e3 / ms2:6.0f} tokens/s  "
                  f"(x{ms1 * 2 / ms2:.2f} of one request; per-request latency {ms2:.2f} ms)")


if __name__ == "__main__":
    main()
