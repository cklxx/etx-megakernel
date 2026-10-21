#!/usr/bin/env python3
"""Every measured per-role number the kernel model is checked against, in one JSON (docs/qwen/MODEL-vllm-kernels.md):

  vllm       vLLM's launch of the kernel (rocprofv3, HIP graph: the duration is all the launch costs), untraced-scaled
  etx_unf    ETX's unfused launch of the imported kernel (rocprofv3, 8 steps of the prompt, layers 2..L-2)
  etx_task   ETX's fused task: body (ready -> done), and the hand-off / ramp / tail of its launch (ETX_TRACE, one step)

  python3 -m examples.vllm_llm.measured > results_2026-09-28/measured.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from statistics import median

from examples.vllm_llm.compare import kernels, read_vl_plan, vllm_roles

R = "results_2026-09-28"
SETS = {
    "qwen3-30b-a3b": dict(vllm="results_2026-09-27/vlnext/moe/prof", kind="moe", unf="rocprof_moe_base", unf_plan="vlplan_b_moe_base.txt",
                          task_plan="plan_moe_xcd.json", task_trace="trace_moe_xcd.bin"),
    "qwen2.5-1.5b": dict(vllm="results_2026-09-27/prof_qwen2.5-1.5b", kind="dense", unf="rocprof_q15_base", unf_plan="vlplan_b_q15_base.txt",
                         task_plan="plan_q15_xs1.json", task_trace="trace_q15_xs1.bin"),
}


def unfused(d, vl_plan):
    vl = read_vl_plan(vl_plan)
    ks = [k for k in kernels(d) if "etx_unfused" in k[0]]
    per = len(vl) + 2
    L = max(r["layer"] for r in vl)
    out = defaultdict(list)
    for s in range(1, len(ks) // per):                 # step 0 is cold
        for i in vl:
            if 2 <= i["layer"] <= L - 2:
                out[i["role"]].append(ks[s * per + 1 + i["n"]][1])
    return {r: median(v) for r, v in out.items()}


def tasks(plan_path, trace_path):
    from etx.tools import timeline as tl
    plan = tl.load_plan(plan_path)
    tr = tl.build_trace(plan, tl.read_raw(trace_path))
    by = defaultdict(list)
    for t in tr.by_id.values():
        by[t.name].append(t)
    order = sorted(by, key=lambda n: plan["types"][n]["id"])
    L = max(int(m.group(2)) for n in order for m in [re.match(r"(.+?)_(\d+)(_x)?$", n)] if m)
    rows = defaultdict(lambda: defaultdict(list))
    prev = None
    for n in order:
        ts = by[n]
        first = min(t.ready for t in ts); med = median(t.ready for t in ts); last = max(t.done for t in ts)
        body = [t.done - t.ready for t in ts]
        m = re.match(r"(.+?)_(\d+)(_x)?$", n)
        if m and prev is not None and 2 <= int(m.group(2)) <= L - 2:
            r = rows[m.group(1)]
            r["body_median"].append(median(body)); r["body_max"].append(max(body)); r["handoff"].append(first - prev)
            r["ramp"].append(med - first); r["span"].append(last - prev); r["tasks"].append(len(ts))
        prev = last
    return {role: {k: median(v) for k, v in d.items()} for role, d in rows.items()}


def main():
    out = {}
    for model, s in SETS.items():
        vl = read_vl_plan(f"{R}/{s['unf_plan']}")
        roles = [r["role"] for r in vl if r["layer"] == 2]
        out[model] = {"vllm": vllm_roles(s["vllm"], s["kind"], roles, model), "etx_unf": unfused(f"{R}/{s['unf']}", f"{R}/{s['unf_plan']}"),
                      "etx_task": tasks(f"{R}/{s['task_plan']}", f"{R}/{s['task_trace']}")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
