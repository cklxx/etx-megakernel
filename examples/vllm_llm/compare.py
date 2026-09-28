#!/usr/bin/env python3
"""Where the time of a fused (or unfused) ETX step with vLLM's kernels goes, role by role, next to vLLM's own launch of
the same kernel (docs/MODEL-moe.md: h = hand-off, p = what the imported form adds per task).

  ledger   ETX_TRACE of a fused run: for each launch B after launch A in plan order,
             handoff = B first ready - A last done      (the event: release, arrive, poll, acquire)
             ramp    = B median ready - B first ready    (how spread B's tasks start)
             body    = mean over B's tasks of done - ready
             tail    = B last done - (B median ready + body)
           so handoff + ramp + body + tail = B last done - A last done exactly, and a layer is the sum over its launches.
             python -m examples.vllm_llm.compare ledger PLAN.json TRACE.bin VL_PLAN.txt [--vllm DIR --kind dense|moe]
  unfused  rocprofv3 kernel trace of the unfused build (one launch per plan instance, HIP graph, no gaps: a duration is
           everything the launch costs, as in vLLM's trace): per role, ETX unfused - vLLM = p per launch.
             python -m examples.vllm_llm.compare unfused ROCPROF_DIR VL_PLAN.txt --vllm DIR --kind dense|moe

Roles are averaged over the middle layers (first and last two dropped). vLLM's per-role durations come from its own
trace (results_2026-09-27: prof_qwen2.5-1.5b / prof_qwen3-8b with custom_ops=all, the same kernels; vlnext/moe/prof,
vLLM's default configuration, whose norms and rope are Inductor kernels), less the tracer's per-launch overhead.
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

# vLLM traced -> untraced: the tracer adds about the same time to every launch, so subtract (busy time per token
# traced - untraced token time) / launches per token (vllm_profile.py analyze): 1.5B 0.083 ms / 331, 8B 0.041 / 455,
# 30B-A3B 0.415 / 843
OFFSET = {"qwen2.5-1.5b": (2.02 - 1.937) * 1e3 / 331, "qwen3-8b": (5.08 - 5.039) * 1e3 / 455, "qwen3-30b-a3b": (5.313 - 4.898) * 1e3 / 843}
# the 17 launches of one vLLM MoE layer (layer_trace.py), starting at the post-attention norm, and the ETX role of each
MOE_VLLM = ["post_norm", "router", "topk", "(fill)", "moe_up", "moe_act", "moe_down", "moe_sum", "(copy)", "in_norm", "qkv",
            "(qk_norm)", "(rope)", "cache", "attn", "reduce", "o_proj"]


def read_vl_plan(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#") or ln.startswith("total"):
            continue
        f = ln.split()
        rows.append({"n": int(f[0]), "export": f[1], "layer": int(f[2]), "role": f[3], "tasks": int(f[8])})
    return rows


def kernels(d):
    rows = []
    for f in glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True):
        rows += list(csv.DictReader(open(f)))
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    return [(r["Kernel_Name"], (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3) for r in rows]


def vllm_roles(d, kind, roles_in_layer, model):
    """{role: mean untraced us} for vLLM's launches of one decode layer."""
    ks = kernels(d)[-40000:]
    off = OFFSET.get(model, 0.0)
    out = defaultdict(list)
    if kind == "moe":
        anchor = [i for i, (n, _) in enumerate(ks) if "fused_add_rms_norm_moe_forward" in n]
        for a, b in zip(anchor, anchor[1:]):
            seq = ks[a:b]
            if len(seq) != len(MOE_VLLM):
                continue
            for role, (_, us) in zip(MOE_VLLM, seq):
                out[role].append(us - off)
        # Inductor runs q/k norm and rope as two kernels where ETX runs vLLM's three custom ops: compare the sums
        out["(qk_norm+rope)"] = [a + b for a, b in zip(out.pop("(qk_norm)", []), out.pop("(rope)", []))]
    else:
        # custom_ops=all: the same kernels in the same order; a layer from one reshape_and_cache to the next
        rot = roles_in_layer.index("cache")
        order = roles_in_layer[rot:] + roles_in_layer[:rot]
        anchor = [i for i, (n, _) in enumerate(ks) if "reshape_and_cache" in n]
        for a, b in zip(anchor, anchor[1:]):
            seq = ks[a:b]
            if len(seq) != len(order):
                continue
            for role, (_, us) in zip(order, seq):
                out[role].append(us - off)
    return {r: median(v) for r, v in out.items() if v}      # median: warm-up bursts carry outliers


def middle(rows):
    L = max(r["layer"] for r in rows)
    return [r for r in rows if 2 <= r["layer"] <= L - 2]


def ledger(a):
    from etx.tools import timeline as tl
    plan = tl.load_plan(a.plan)
    tr = tl.build_trace(plan, tl.read_raw(a.trace))
    by = defaultdict(list)
    for t in tr.by_id.values():
        by[t.name].append(t)
    order = sorted(by, key=lambda n: plan["types"][n]["id"])
    vl = read_vl_plan(a.vl_plan)
    # plan grid name -> (role, layer): the grid names model.py gives ("qkv_5", "post_norm_5_x", "chain_..._5")
    per_role = defaultdict(lambda: defaultdict(list))
    prev_done = None
    for n in order:
        ts = by[n]
        first = min(t.ready for t in ts); med = median(t.ready for t in ts); last = max(t.done for t in ts)
        body = mean(t.done - t.ready for t in ts)
        m = re.match(r"(.+?)_(\d+)(_x)?$", n)
        if m and prev_done is not None:
            role, layer = m.group(1), int(m.group(2))
            per_role[role]["layer"].append(layer)
            for k, v in (("handoff", first - prev_done), ("ramp", med - first), ("body", body), ("tail", last - med - body),
                         ("tasks", len(ts))):
                per_role[role][k].append(v)
        prev_done = last
    L = max(r["layer"] for r in vl)
    roles_in_layer = [r["role"] for r in vl if r["layer"] == 2]
    ref = vllm_roles(a.vllm, a.kind, roles_in_layer, a.model) if a.vllm else {}
    print(f"{'role':34s} {'tasks':>5s} {'handoff':>8s} {'ramp':>6s} {'body':>7s} {'tail':>6s} {'sum':>7s}   vLLM launch (untraced)")
    tot = defaultdict(float)
    for role in sorted(per_role, key=lambda r: order.index(next(n for n in order if n.startswith(r + "_")))):
        d = per_role[role]
        idx = [i for i, l in enumerate(d["layer"]) if 2 <= l <= L - 2]
        if not idx:
            continue
        v = {k: mean(d[k][i] for i in idx) for k in ("handoff", "ramp", "body", "tail", "tasks")}
        s = v["handoff"] + v["ramp"] + v["body"] + v["tail"]
        for k in ("handoff", "ramp", "body", "tail"):
            tot[k] += v[k]
        base = role.replace("chain_", "")
        rv = ref.get(role) or ref.get(base)
        print(f"{role:34s} {v['tasks']:5.0f} {v['handoff']:8.2f} {v['ramp']:6.2f} {v['body']:7.2f} {v['tail']:6.2f} {s:7.2f}   "
              + (f"{rv:6.2f}" if rv else "-"))
    print(f"{'layer':34s} {'':5s} {tot['handoff']:8.2f} {tot['ramp']:6.2f} {tot['body']:7.2f} {tot['tail']:6.2f} "
          f"{sum(tot.values()):7.2f}   " + (f"{sum(v for k, v in ref.items() if not k.startswith('(')):6.2f} (+ fill/copy "
                                               f"{sum(v for k, v in ref.items() if k in ('(fill)', '(copy)')):.2f})" if ref else ""))


def unfused(a):
    vl = read_vl_plan(a.vl_plan)
    ks = [k for k in kernels(a.rocprof) if "etx_unfused" in k[0]]
    per_step = len(vl) + 2                              # prologue, the instances, argmax
    steps = len(ks) // per_step
    per_role = defaultdict(list)
    for s in range(steps):
        seq = ks[s * per_step:(s + 1) * per_step]
        for inst, (_, us) in zip(vl, seq[1:-1]):
            if 2 <= inst["layer"] <= max(r["layer"] for r in vl) - 2:
                per_role[inst["role"]].append(us)
    roles_in_layer = [r["role"] for r in vl if r["layer"] == 2]
    ref = vllm_roles(a.vllm, a.kind, roles_in_layer, a.model)
    print(f"{steps} steps of {per_step} launches; per role: ETX unfused launch, vLLM launch, difference (p per launch)")
    tot_e = tot_v = 0.0
    for role in dict.fromkeys(roles_in_layer):
        e = mean(per_role[role]) if per_role[role] else float("nan")
        v = ref.get(role)
        if v is None and role in ("q_norm", "k_norm", "rope"):
            v = None
        n = roles_in_layer.count(role)
        tot_e += e * n
        if v is not None:
            tot_v += v * n
        print(f"  {role:12s} x{n}  ETX {e:7.2f} us   vLLM " + (f"{v:7.2f} us   p {e - v:+6.2f}" if v is not None else "      -"))
    print(f"  layer        ETX {tot_e:7.2f} us   vLLM {tot_v:7.2f} us (roles with a vLLM counterpart)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    l = sp.add_parser("ledger"); l.add_argument("plan"); l.add_argument("trace"); l.add_argument("vl_plan")
    u = sp.add_parser("unfused"); u.add_argument("rocprof"); u.add_argument("vl_plan")
    for p in (l, u):
        p.add_argument("--vllm", default=None); p.add_argument("--kind", choices=["dense", "moe"], default="dense")
        p.add_argument("--model", default="qwen2.5-1.5b")
    a = ap.parse_args()
    ledger(a) if a.cmd == "ledger" else unfused(a)


if __name__ == "__main__":
    main()
