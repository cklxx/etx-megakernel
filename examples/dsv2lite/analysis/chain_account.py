"""Brief 03 arithmetic only: archived CPU trace account, no builds or GPU calls.

Run from ETX: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python
  examples/dsv2lite/analysis/chain_account.py
All candidate values are conditional scenarios, not measured savings.
"""
import json
import os
import subprocess
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--native-chain":
    # Read-only analysis of the normal prefix, never the iso* copies. q_absorb
    # and attention retain their real overlap in these tasks.
    from etx.tools import timeline as tl
    trace = tl.build_trace(tl.load_plan(sys.argv[2]), tl.read_raw(sys.argv[3]))
    spans = []
    for layer in range(27):
        qkv = [t for t in trace.tasks if t.name == f"qkv_{layer}"]
        merge = [t for t in trace.tasks if t.name == f"merge_{layer}"]
        if not qkv or not merge:
            raise SystemExit(f"missing native chain at layer {layer}")
        spans.append(max(t.done for t in merge)-min(t.ready for t in qkv))
    value = sum(spans)
    print(f"native_qkv_ready_through_merge_done_sum_us={value:.6f} "
          f"pctP={100*value/2235:.6f} pct140={100*value/140:.6f}")
    raise SystemExit(0)

env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
a = json.loads(subprocess.check_output(
    [sys.executable, "-m", "examples.dsv2lite.phase_table300", "--json"],
    text=True, env=env))
P, checkpoint = 2235, 140


def triple(x):
    return f"{x:.3f} / {100*x/P:.2f}% / {100*x/checkpoint:.2f}%"


def row(name, value):
    print(f"{name}: {triple(value)}")


print("All triples: us/token / % of P=2235 / % of checkpoint=140")
for name, value in [("P", P), ("checkpoint", checkpoint),
                    ("floor", 5.20e9/4.2e6), ("device", 3473),
                    ("trace uplift to fixed HEAD", a["span"]-3473)]:
    row(name, value)
pa = a["phase_ledger"]
for p in ("qkv", "router", "attn", "merge", "oproj"):
    d = pa[p]
    n = 26 if p == "router" else 27
    hrt = sum(d[k] for k in ("handoff", "ramp", "tail"))
    total = sum(d.values())
    print(f"\n{p}: instances={n}, ratio={a['ratios'][p]}")
    for k in ("handoff", "ramp", "tail", "staging", "prologue", "floor", "stream"):
        row(k, d[k])
    row("H+R+T", hrt)
    row("phase total", total)
    row("nonfloor", total-d["floor"])
    if p != "attn":
        rep = {"qkv": 6.12, "router": 3.26, "merge": 3.42, "oproj": 4.96}[p]
        row("replica span x instances", rep*n)
        row("above replica", total-rep*n)
    if p in ("qkv", "router", "attn"):
        service = {"qkv": 51838976*27/4.2e6,
                   "router": 4849664*26/4.2e6,
                   "attn": 509607936/4.2e6}[p]
        residual = total-hrt-d["staging"]-d["prologue"]-service
        assert residual >= 0
        row("issued/requested service at 4.2 [E]", service)
        row("unassigned residual [U]", residual)
    if p == "merge":
        service = 4194304*27/4.2e6
        row("issued weights service [E]", service)
        for rt in (3, 5):
            hops = rt*.108*27
            barriers = .077*27
            residual = total-hrt-hops-barriers-service
            assert residual >= 0
            row(f"{rt} idle-L2-hop proxy [E]", hops)
            row("one barrier x layers [E]", barriers)
            row(f"residual at {rt} hops [U]", residual)

row("five-phase chain nonfloor", sum(sum(pa[p].values())-pa[p]["floor"]
    for p in ("qkv", "attn", "merge", "oproj", "router")))
print("\nConditional candidate arithmetic; independent charges need measurement")
for name, value in {
    "qkv staging barrier": 27*.077,
    "router staging barrier": 26*.077,
    "qkv norm barriers": 27*11*.077,
    "router norm barriers": 26*11*.077,
    "qkv norm residual after gamma and barriers": pa["qkv"]["prologue"]-27*(1+11*.077),
    "router norm residual after gamma and barriers": pa["router"]["prologue"]-26*(1+11*.077),
    "qkv staging residual after barrier": pa["qkv"]["staging"]-27*.077,
    "router staging residual after barrier": pa["router"]["staging"]-26*.077,
    "norm gamma gross": 53,
    "norm old tree barrier proxy": 53*10*.077,
    "norm old tree MODEL proxy": 53*1.5,
    "norm new tree proxy": 53*.4145,
    "norm package gross low": 53*(1+10*.077),
    "norm package gross high": 53*(1+1.5),
    "norm package net low": 53*(1+10*.077-.4145),
    "norm package net high": 53*(1+1.5-.4145),
    "qkv requested duplication equiv": 445906944/4.2e6,
    "qkv all issued excess equiv": (51838976-31457280)*27/4.2e6,
    "qkv one event per layer": 27,
    "router one event per layer": 26,
    "qkv publisher reader relief analogy": 27*(6.04-4.76),
    "router publisher reader relief analogy": 26*(6.04-4.76),
    "qkv minimal publisher extra": 27*(1+.108+.077),
    "router minimal publisher extra": 26*(1+.108+.077),
    "qkv publisher optimistic net": 27*((6.04-4.76)-(1+.108+.077)),
    "router publisher optimistic net": 26*((6.04-4.76)-(1+.108+.077)),
    "cache duplicate full equivalent": (509607936-31850496)/4.2e6,
    "cache two-head equivalent": (509607936/2)/4.2e6,
    "attention retain two hops": 27*2,
    "attention remove eight barriers": 27*8*.077,
    "attention in-place gross": 27*(2+8*.077),
    "attention scratch budget": 20,
    "attention in-place conditional net": 27*(2+8*.077)-20,
    "attention all barriers nominal": 27*20*.077,
    "router duplicate weights equiv": 47710208/4.2e6,
    "merge repeated partials equiv": 27*(8421376-526336)/4.2e6,
    "merge repeated partials less event": 27*(8421376-526336)/4.2e6-27,
    "KVA historical phase gross": 27*(23.2-18.6),
    "KVA historical attention penalty": 27*2,
    "KVA historical phase net": 27*(23.2-18.6-2),
    "KVA whole-token 0.6pct proxy": 3473*.006,
    "row split 0.8pct proxy": 3473*.008,
    "qkv no-final-load lower equivalent": 27*296*2*4*2*2048*2/4.2e6,
    "qkv no-final-load cost budget": 27*.5,
    "qkv no-final-load conditional net": 27*296*2*4*2*2048*2/4.2e6-27*.5,
    "qkv consumed padded rows equiv": 983040*27/4.2e6,
    "decider": 50, "hundreds marker": 100, "full validation gate": 70,
}.items():
    row(name, value)
print("decider per norm", 50/53)
print("decider per attention layer", 50/27)
print("attention effective-hop exposure threshold", (50/27-8*.077)/2)
print("hundred-us decider per qkv/attention layer", 100/27)
print("planning cap per decider: 5 warm GPU minutes; five pairs; not measured")
print("source-level norm LDS bytes", 3*8192 + 256*4 + 255*12 + 256*4)
print("f32 fold adds and bf16 roundings per WG", 2048*8, 2048*2)
