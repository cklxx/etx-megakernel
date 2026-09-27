"""Per-phase comparison of the one-request and two-request DeepSeek traces (design doc 15.13). Paths are the local
results directories of 2026-09-27; run from the repo root: PYTHONPATH=. .venv/bin/python examples/dsv2lite/phase_compare.py"""
import re, sys
from collections import defaultdict
from pathlib import Path
from etx.tools import timeline as tl
from examples.dsv2lite.bytes import spec
R = Path.home()/"code/etx"
TR = {"x1 (one request)": (R/"results_2026-09-27b/etx_plan.json", R/"results_2026-09-27b/etx_trace.bin"),
      "x2 none": (R/"results_2026-09-27c/x2n_plan.json", R/"results_2026-09-27c/x2n_trace.bin"),
      "x2 router": (R/"results_2026-09-27c/x2_plan.json", R/"results_2026-09-27c/x2_trace.bin")}
PH = ["qkv", "attn", "merge", "oproj", "router", "gate_up", "down"]
for label, (pj, tb) in TR.items():
    tr = tl.build_trace(tl.load_plan(pj), tl.read_raw(tb), spec=spec())
    # per request, per MoE layer 1..26: phase first-ready / last-done, body mean
    ph = defaultdict(lambda: [1e30, 0.0, 0.0, 0])      # (req, L, phase) -> first ready, last done, body sum, n
    for t in tr.tasks:
        m = re.match(r"(?:r(\d)_)?([a-z_]+?)_(\d+)$", t.name)
        if not m or m.group(2) not in PH: continue
        r, k, L = int(m.group(1) or 0), m.group(2), int(m.group(3))
        if not 1 <= L <= 26: continue
        a = ph[(r, L, k)]; a[0] = min(a[0], t.ready); a[1] = max(a[1], t.done); a[2] += t.done - t.ready; a[3] += 1
    reqs = sorted({k[0] for k in ph})
    print(f"\n== {label}: per request, mean over MoE layers 2-25 (us, relative to the request's layer start = its qkv first ready)")
    print(f"   {'phase':8} " + "  ".join(f"r{r} start  end   body" for r in reqs))
    lay = {}
    for k in PH:
        row = []
        for r in reqs:
            st = en = bo = 0.0; n = 0
            for L in range(2, 26):
                a = ph[(r, L, k)]; s0 = ph[(r, L, "qkv")][0]
                st += a[0] - s0; en += a[1] - s0; bo += a[2] / a[3]; n += 1
            row.append(f"{st/n:6.1f} {en/n:6.1f} {bo/n:6.1f}")
        print(f"   {k:8} " + "   ".join(row))
    for r in reqs:
        spans = [ph[(r, L + 1, "qkv")][0] - ph[(r, L, "qkv")][0] for L in range(2, 25)]
        print(f"   r{r}: layer period (qkv start to next qkv start) mean {sum(spans)/len(spans):6.1f} us")
    if len(reqs) == 2:
        offs = [ph[(1, L, "qkv")][0] - ph[(0, L, "qkv")][0] for L in range(2, 25)]
        print(f"   r1 qkv start minus r0 qkv start: mean {sum(offs)/len(offs):6.1f} us")
    md = tl.memory_demand(tr)
    print(f"   whole token: span {tr.span:.0f} us, bytes {md['bytes']/1e9:.2f} GB, mean bandwidth {md['bytes']/tr.span/1e6:.2f} TB/s, "
          f"below 25% of 4.3 TB/s for {md['low_us']/tr.span*100:.0f}% of the time")
