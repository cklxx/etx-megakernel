#!/usr/bin/env python3
"""The kernel model's parameters from the microbenchmarks alone (docs/MODEL-vllm-kernels.md section 10), next to the
fitted ones, and the XCD simulation run with each set.

  python3 -m examples.vllm_llm.bench_params LOGDIR [--plan results_2026-09-28/plan_b_moe_xcd.json]

LOGDIR holds the session's logs of hw_bench lat / lf / launch / evx and imp_bench (hw_*.log, impbar_*.log; any file
with RESULT lines is read). Which measurement stands for which parameter (the mapping is the model's claim, stated here):

  t_v      vector round trip of data another kernel wrote: `lat vector MALL idle` (written back past the writer's L2)
  t_s      scalar round trip of a fresh kernel argument: `lat scalar MALL idle`
  F        kernel boundary in a HIP graph: `launch grid 304 threads 1024 hops -1` (an empty kernel per step)
  bw       the highest rate of `lf` (bytes in flight -> rate); the simulator's Little's-law cap takes t_v as the latency
  t_swbar  per barrier, the software slot barrier, fused_moe's shape: `impbar sw fused_moe`
  t_round  per barrier, the round protocol (hardware s_barrier with idle waves): `impbar rounds fused_moe`
  A_task   the acquire and the argument path after it, under load (tasks start while other CUs stream): one invalidate
           (`lat inv-only inv sc1 loaded`) plus two dependent hops; vector build (ETX_IMP_VECTOR_ARGS): the argument table
           and block miss the invalidated caches, 2 x (`lat vector L2 inv sc1 loaded` - the invalidate); scalar build:
           2 K$ hits (the K$ survives inv sc1, `lat scalar L1/K$ inv sc1` against `-`; the pre-wait touch put them there)
  handoff  not from evx: the isolated protocol measured 2.2 us (XCD) / 3.9 us (device) against the trace's 0.3-0.5 /
           1.5-2.1 (2026-09-28, results_2026-09-28b); until that is explained the trace's values stand
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RX = {
    "lat": re.compile(r"^RESULT lat (\S+)\s+(\S+)\s+(-|inv sc[01])\s+(loaded|idle)\s*:\s+([\d.]+) ns per hop"),
    "lf": re.compile(r"^RESULT lf bytes\s+(\d+) active\s+(\d+) inflight\s+(\d+) KB/WG \(\s*([\d.]+) MB total\): WG median\s+([\d.]+) us"
                     r"\s+span\s+([\d.]+)\s+per trip\s+([\d.]+) us\s+-> ([\d.]+) TB/s \(WG median\), ([\d.]+) TB/s"),
    "launch": re.compile(r"^RESULT launch grid\s+(\d+) threads\s+(\d+) hops\s+(-?\d+):\s+([\d.]+) us per kernel"),
    "evx": re.compile(r"^RESULT evx (xcd|device)\s+P\s+(\d+) C\s+(\d+) dirty\s+(\d+) KB\s*(late)?\s*: last arrival -> first\s+(-?[\d.]+)"
                      r"\s+median\s+(-?[\d.]+)\s+last\s+(-?[\d.]+) us; wait start -> wake median\s+(-?[\d.]+) us"),
    "impbar": re.compile(r"^RESULT impbar (rounds|sw)\s+(\S+).*?R\s+(\d+): block\s+([\d.]+) us(?:, per barrier\s+(-?[\d.]+) ns)?"),
    "sync": re.compile(r"^RESULT impbar (rounds|sw)\s+__syncthreads \(1024 threads\):\s+([\d.]+) ns per barrier"),
}


def parse(text: str) -> dict:
    """RESULT lines -> {kind: {key: value}}; values in us."""
    b = {k: {} for k in RX}
    for line in text.splitlines():
        line = line.strip()
        if (m := RX["lat"].match(line)):
            kind, state, inv, load, ns = m.groups()
            b["lat"][(kind, state, inv, load)] = float(ns) / 1e3
        elif (m := RX["lf"].match(line)):
            by, act, kb, mb, wg, span, trip, tb_wg, tb_span = m.groups()
            b["lf"][(int(by), int(act), int(kb))] = dict(inflight_mb=float(mb), wg_us=float(wg), trip_us=float(trip), tbs=float(tb_wg))
        elif (m := RX["launch"].match(line)):
            g, t, h, us = m.groups()
            b["launch"][(int(g), int(t), int(h))] = float(us)
        elif (m := RX["evx"].match(line)):
            sc, P, C, d, late, first, med, last, ready = m.groups()
            b["evx"][(sc, int(P), int(C), int(d), bool(late))] = dict(first=float(first), median=float(med), last=float(last), ready=float(ready))
        elif (m := RX["sync"].match(line)):
            b["sync"][m.group(1)] = float(m.group(2)) / 1e3
        elif (m := RX["impbar"].match(line)):
            proto, what, R, blk, per = m.groups()
            b["impbar"][(proto, what, int(R))] = dict(block=float(blk), per=float(per) / 1e3 if per else None)
    return b


def read_dir(d: str) -> dict:
    text = "\n".join(p.read_text(errors="replace") for p in sorted(Path(d).glob("*.log")))
    return parse(text)


def derive(b: dict) -> dict:
    """{name: (value, source)} for every parameter the microbenchmarks give; missing measurements are left out."""
    out = {}
    lat, lf = b["lat"], b["lf"]

    def put(name, v, src):
        if v is not None:
            out[name] = (v, src)
    put("t_v", lat.get(("vector", "MALL", "-", "idle")), "lat vector MALL idle")
    put("t_s", lat.get(("scalar", "MALL", "-", "idle")), "lat scalar MALL idle")
    put("F", b["launch"].get((304, 1024, -1)), "launch 304 x 1024, empty")
    if lf:
        best = max(lf.values(), key=lambda r: r["tbs"])
        put("bw", best["tbs"] * 1e6, f"lf max ({best['inflight_mb']:.2f} MB in flight)")
    for proto, name in (("sw", "t_swbar"), ("rounds", "t_round")):
        per = [v["per"] for (p, w, R), v in b["impbar"].items() if p == proto and w == "fused_moe" and R == 32 and v["per"] is not None]
        put(name, per[0] if per else None, f"impbar {proto} fused_moe R 32")
    inv = lat.get(("inv-only", "L1/K$", "inv sc1", "loaded"))
    v = lat.get(("vector", "L2", "inv sc1", "loaded"))
    if inv is not None and v is not None:
        put("A_task_vector", inv + 2 * (v - inv), "inv sc1 loaded + 2 x (vector L2 after inv sc1, loaded - inv)")
    k = lat.get(("scalar", "L1/K$", "-", "idle"))
    if inv is not None and k is not None:
        put("A_task_scalar", inv + 2 * k, "inv sc1 loaded + 2 x scalar K$ hit")
    e = b["evx"].get(("xcd", 1, 1, 0, True))
    put("ready_wait", e["ready"] if e else None, "evx late: wait on a complete event + acquire + barrier")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--plan", help="optional archived plan; requires --measured")
    ap.add_argument("--measured", help="optional archived measurements; requires --plan")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if bool(a.plan) != bool(a.measured):
        ap.error("--plan and --measured must be provided together")
    d = derive(read_dir(a.logdir))
    if not a.plan:
        print("microbenchmark-only: derived parameters; no fitted model or simulated latency")
        for key, (value, source) in d.items():
            print(f"{key}: {value} ({source})")
        if a.json:
            with open(a.json, "w") as stream:
                json.dump({"derived": {k: v[0] for k, v in d.items()},
                           "sources": {k: v[1] for k, v in d.items()}, "sim": {}}, stream, indent=1)
        return
    from etx.sim.xcd import Hw, simulate
    from examples.vllm_llm import kmodel as km
    from examples.vllm_llm.xcd_run import cost_fn

    fit = km.fit(json.load(open(a.measured)))
    fitted = {"t_v": fit.t_v, "t_s": fit.t_s, "F": fit.F, "bw": fit.bw, "t_swbar": fit.hw["t_swbar"], "t_round": fit.hw["t_bar"],
              "A_task_vector": fit.A_task, "A_task_scalar": None, "ready_wait": None}
    print(f"{'parameter':16s} {'fitted / assumed':>18s} {'microbenchmark':>16s}  source")
    for k, f in fitted.items():
        mv, src = d.get(k, (None, "not measured"))
        fs = "-" if f is None else f"{f / 1e6:.2f} TB/s" if k == "bw" else f"{f:.3f} us"
        ms = "-" if mv is None else f"{mv / 1e6:.2f} TB/s" if k == "bw" else f"{mv:.3f} us"
        print(f"{k:16s} {fs:>18s} {ms:>16s}  {src}")

    plan = json.load(open(a.plan))

    def sim(p, h_dom, h_dev):
        hw = Hw(bw_max=p.bw, n_half=0.0, handoff_us={"DOMAIN": h_dom, "DEVICE": h_dev}, wake_us=0.6, rate_cv=0.03, lat_us=p.t_v)
        return simulate(plan, cost_fn("qwen3-30b-a3b", p), hw)[1]["makespan_us"] / 1e3

    def params(build: str, micro: bool):
        p = km.Params(**{**fit.__dict__, "hw": dict(fit.hw)})
        h_dom, h_dev = 1.0, 1.3
        if micro:
            for k in ("t_v", "t_s", "F", "bw"):
                if k in d:
                    setattr(p, k, d[k][0])
            if "t_swbar" in d:
                p.hw["t_swbar"] = d["t_swbar"][0]
            h_dom, h_dev = 0.4, 1.8                   # the trace's hand-offs (evx does not reproduce them yet)
        if build in ("new", "hw"):                       # the round protocol: imported barriers cost the measured round
            p.hw["t_swbar"] = d["t_round"][0] if micro and "t_round" in d else p.hw["t_bar"]
        if build in ("new", "args"):
            p.A_task = d["A_task_scalar"][0] if micro and "A_task_scalar" in d else 0.5
        elif micro and "A_task_vector" in d:
            p.A_task = d["A_task_vector"][0]
        return p, h_dom, h_dev

    res = {}
    print(f"\nsimulated step, ms per token ({a.plan}); the fitted set is ~12% optimistic against today's 6.63 ms")
    for build in ("old", "hw", "args", "new"):
        f_ms = sim(*params(build, False))
        m_ms = sim(*params(build, True))
        res[build] = dict(fitted=f_ms, microbench=m_ms)
        print(f"  {build:5s} fitted {f_ms:6.3f}   microbenchmark-only {m_ms:6.3f}")
    if a.json:
        json.dump({"derived": {k: v[0] for k, v in d.items()}, "sources": {k: v[1] for k, v in d.items()}, "sim": res},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
