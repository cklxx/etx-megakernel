"""Fit the XCD simulator's two spread parameters (wake-up, stream-rate cv) to the measured ramps and tails of the MoE
xcd trace (docs/qwen/MODEL-vllm-kernels.md section 2). python3 examples/vllm_llm/sim_fit.py"""
import json, re, sys
from collections import defaultdict
from statistics import median
sys.path.insert(0, ".")
from etx.sim.xcd import Hw, simulate
from etx.tools import timeline as tl
from examples.vllm_llm import kmodel as km
from examples.vllm_llm.xcd_run import cost_fn

plan = json.load(open("results_2026-09-28/plan_moe_xcd.json"))
p = km.fit(json.load(open("results_2026-09-28/measured.json")))

def ledger(rows):
    tr = tl.build_trace(plan, rows)
    by = defaultdict(list)
    for t in tr.by_id.values(): by[t.name].append(t)
    order = sorted(by, key=lambda n: plan["types"][n]["id"])
    per = defaultdict(lambda: defaultdict(list)); prev = None
    for n in order:
        ts = by[n]; first = min(t.ready for t in ts); med = median(t.ready for t in ts); last = max(t.done for t in ts)
        body = sum(t.done - t.ready for t in ts) / len(ts)
        m = re.match(r"(.+?)_(\d+)(_x)?$", n)
        if m and prev is not None and 2 <= int(m.group(2)) <= 45:
            d = per[m.group(1)]
            d["h"].append(first - prev); d["r"].append(med - first); d["b"].append(body); d["t"].append(last - med - body); d["s"].append(last - prev)
        prev = last
    return {r: {k: sum(v) / len(v) for k, v in d.items()} for r, d in per.items()}

meas = ledger(tl.read_raw("results_2026-09-28/trace_moe_xcd.bin"))
best = []
for wake in (0.0, 0.2, 0.4, 0.6):
    for cv in (0.0, 0.03, 0.06, 0.1):
        hw = Hw(bw_max=p.bw, n_half=0.0, handoff_us={"DOMAIN": 1.0, "DEVICE": 1.3}, wake_us=wake, rate_cv=cv, lat_us=p.t_v)
        rows, st = simulate(plan, cost_fn("qwen3-30b-a3b", p), hw)
        sim = ledger(rows)
        err = sum((sim[r][k] - meas[r][k]) ** 2 for r in meas for k in ("r", "t"))
        span = sum(sim[r]["s"] for r in sim)
        best.append((err, wake, cv, st["makespan_us"] / 1e3, span))
        print(f"wake {wake} cv {cv}: step {st['makespan_us']/1e3:.3f} ms, layer {span:.1f} us, ramp+tail err {err:.1f}", flush=True)
print("measured layer", sum(meas[r]["s"] for r in meas))
