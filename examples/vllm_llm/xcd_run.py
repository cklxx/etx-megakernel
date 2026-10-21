#!/usr/bin/env python3
"""Simulate the fused ETX step of a vLLM-kernel plan on MI300X's XCDs with the kernel model's costs, and put it next to
the measured trace (docs/qwen/MODEL-vllm-kernels.md):

  task cost   the kernel model's ETX-task terms (examples/vllm_llm/kmodel.py): front = the argument path after the
              acquire, chain = scalar + vector round trips + barriers + reductions + issue, stream = bytes per task at
              a per-CU cap (bytes in flight / (idle round trip + exposed work per trip)), shared on the GPU
  hand-off    the microbenchmark: XCD-local 1.0 us, device-wide 1.3 us (hw_bench handoff, padded counters)

  python3 -m examples.vllm_llm.xcd_run PLAN.json --model qwen3-30b-a3b [--trace MEASURED.bin] [--out sim.bin]
      [--handoff-domain 1.0 --handoff-device 1.3]
Prints the simulated step time, and the per-role ledger of the simulated and (with --trace) the measured schedule.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

from etx.sim.xcd import Cost, Hw, simulate
from examples.vllm_llm import kmodel as km


def cost_fn(model: str, p: km.Params, variant: dict | None = None, plan: dict | None = None):
    """model "auto" with `plan`: every task's entry from the tile its type links (km.kern_for_symbol)"""
    base = km.MODELS["qwen3-30b-a3b" if model == "auto" else model]
    kern = {k.role: k for k in base}
    symbols = {n: t["symbol"] for n, t in plan["types"].items()} if (plan and model == "auto") else {}
    touched = {n for n, t in plan["types"].items() if t.get("touch")} if plan else set()
    variant = variant or {}
    t_l2 = p.hw.get("t_l2", 0.143)                    # the L2 round trip (hw_bench lat vector L2 idle)

    def role_of(name: str) -> str | None:
        m = re.match(r"(?:r\d+_)?(?:chain_)?(.+?)_(\d+)(_x)?$", name)   # r<k>_: request k of ETX_VL_REQUESTS > 1
        return m.group(1) if m else None

    cache = {}

    def cost(name: str, desc: dict) -> Cost:
        if name in cache:
            return cache[name]
        r = role_of(name)
        kk = kern
        if symbols and r is not None and not name.startswith("chain_"):
            k = km.kern_for_symbol(symbols[name], r, base)
            kk = dict(kern)
            if k is None: kk.pop(r, None)
            else: kk[r] = k
        if name.startswith("chain_"):                     # q/k norm + rope + cache run back to back in one task
            roles = re.match(r"chain_(.+)_\d+$", name).group(1).split("_")
            roles = [x if x in kk else f"{x}_{y}" for x, y in zip(roles, roles[1:] + [""]) if x in kk or f"{x}_{y}" in kk]
            parts = [kk[x] for x in roles]
            # every part reads its own arguments: each pays the argument path (measured: 18.6 us for the four);
            # a part whose blocks exceed the workgroup's slots runs in rounds
            ch = sum(km.terms(k, p, "etx_task")["total"] * k.chain_rounds for k in parts) - p.A_task
            c = Cost(front_us=p.A_task, chain_us=ch)
        elif r in kk:
            k = kk[r]
            t = km.terms(k, p, "etx_task")
            chain = t["scalar"] + t["chain"] + t["sync"] + t["issue"]
            if k.bytes_block:
                exposed = k.gap + k.sw_per_trip * (p.hw["t_swbar"] - p.hw["t_bar"])
                c = Cost(front_us=t["etx_args"], chain_us=chain, stream_bytes=k.bytes_block, mall_bytes=k.mall_block, inflight=k.inflight, exposed_us=exposed)
            else:
                c = Cost(front_us=t["etx_args"], chain_us=chain)
        else:                                              # prologue, argmax, lm_head, final norm: fixed guesses
            c = Cost(front_us=p.A_task, chain_us=2.0)
        if name in touched and c.stream_bytes > 0:         # touch hook (section 18): the first trip (one inflight of bytes)
            c = Cost(**{**c.__dict__, "touch_bytes": min(c.inflight, c.stream_bytes),   # streams before the events fire,
                        "chain_us": c.chain_us + t_l2})   # sharing the bandwidth; re-reading it after the acquire is an L2 trip
        for key, f in variant.items():                     # what-if: scale a term, e.g. {"front_us": 0.0}
            if isinstance(f, (int, float)):
                c = Cost(**{**c.__dict__, key: f})
        cache[name] = c
        return c
    return cost


def write_raw(rows, path):
    with open(path, "wb") as f:
        for r in rows:
            f.write(struct.pack("<9q", *[int(x) for x in r]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--model", default="qwen3-30b-a3b")
    ap.add_argument("--measured", default="results_2026-09-28/measured.json")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--handoff-domain", type=float, default=1.0)
    ap.add_argument("--handoff-device", type=float, default=1.3)
    ap.add_argument("--wake", type=float, default=0.0, help="wake-up spread, us (uniform)")
    ap.add_argument("--rate-cv", type=float, default=0.0, help="per-task spread of the stream cap")
    ap.add_argument("--hw-barriers", action="store_true", help="what-if: imported barriers as hardware s_barrier")
    ap.add_argument("--args-us", type=float, default=None, help="what-if: the argument path of every imported task, us")
    ap.add_argument("--what-if", default=None, help='JSON of Cost fields to override for every task, e.g. {"front_us": 0.3}')
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    p = km.fit(json.load(open(a.measured)))
    if a.hw_barriers:
        p.hw["t_swbar"] = p.hw["t_bar"]
    if a.args_us is not None:
        p.A_task = a.args_us
    hw = Hw(bw_max=p.bw, n_half=0.0, handoff_us={"DOMAIN": a.handoff_domain, "DEVICE": a.handoff_device},
            wake_us=a.wake, rate_cv=a.rate_cv, lat_us=p.t_v)
    rows, st = simulate(plan, cost_fn(a.model, p, json.loads(a.what_if) if a.what_if else None), hw)
    print(f"simulated step: {st['makespan_us'] / 1e3:.3f} ms ({st['unfinished']} tasks not run); "
          f"hand-off XCD-local {a.handoff_domain} us, device {a.handoff_device} us; BW {p.bw / 1e6:.1f} TB/s")
    out = a.out or "/tmp/etx_sim_trace.bin"
    write_raw(rows, out)
    vl = Path(a.plan).parent / Path(a.plan).name.replace("plan_", "vlplan_b_").replace(".json", ".txt")
    from examples.vllm_llm import compare
    for label, tr in (("simulated", out), ("measured", a.trace)):
        if not tr:
            continue
        print(f"\n== {label}")
        sys.argv = ["compare", "ledger", a.plan, tr, str(vl)]
        ns = argparse.Namespace(plan=a.plan, trace=tr, vl_plan=str(vl), vllm=None, kind="moe", model=a.model)
        compare.ledger(ns)


if __name__ == "__main__":
    main()
