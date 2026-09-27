"""Where does the ETX port lose time against fleet itself? Same token, same tiles, two traces.

Inputs: fleet's `--trace` dump with its task graph, the ETX build's plan.json with its ETX_TRACE_RAW dump, and
optionally the ETX prologue stamps (a build with -DETX_DS_PRO=1 run with ETX_DS_PROLOGUE=<file>). Output: per layer
the span in both and the difference; per task kind the mean body and prologue (the part before the weight stream,
and its staging share) in both, for one layer or averaged over the MoE layers.

  python -m examples.dsv2lite.compare_fleet --fleet-trace fleet_trace.bin --fleet-graph taskgraph_d16.bin \
      --etx-plan ds_plan.json --etx-trace ds_trace.bin [--etx-prologue ds_pro.bin] [--fleet-src ~/fleet-mi300x/src]
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

from etx.tools import timeline as tl

TICK_US = 0.01
# ETX grid name stem -> fleet TaskKind name
KIND = {"embed": "EMBED", "qkv": "QKV_FUSED", "qabs": "Q_ABSORB", "attn": "ATTENTION", "merge": "MERGE_UV", "oproj": "O_PROJ",
        "router": "NORM_ROUTER", "norm": "NORM_ROUTER", "gate_up": "EXPERT_GATE_UP", "down": "EXPERT_DOWN",
        "dense_gu": "DENSE_GATE_UP", "dense_dn": "DENSE_DOWN", "lm_head": "LM_HEAD", "argmax": "ARGMAX"}


def load_fleet(trace: Path, graph: Path, src: Path):
    sys.path.insert(0, str(src / "host"))
    from taskgraph import PACK_FIELDS, PACK_FORMAT, TaskKind   # noqa: E402
    blob = graph.read_bytes()
    tasks = [dict(zip(PACK_FIELDS, struct.unpack_from(PACK_FORMAT, blob, 64 * i))) for i in range(len(blob) // 64)]
    tr = trace.read_bytes()
    st = [struct.unpack_from("<4Q", tr, 32 * i) for i in range(len(tr) // 32)]
    if len(st) != len(tasks):
        raise SystemExit(f"fleet trace has {len(st)} rows, graph {len(tasks)} descriptors")
    rows = [(t, s) for t, s in zip(tasks, st) if s[0]]
    t0 = min(s[1] for _, s in rows)
    out = []   # (layer, kind name, ready us, done us, prologue us, staging us)
    for t, s in rows:
        pro, stg = ((s[3] >> 32) & 0xFFFFFF) * TICK_US, ((s[3] >> 8) & 0xFFFFFF) * TICK_US
        out.append((t["layer"], TaskKind(t["kind"]).name, (s[1] - t0) * TICK_US, (s[2] - t0) * TICK_US, pro, stg))
    return out, TaskKind


def load_etx(plan: Path, trace: Path, prologue: Path | None, TaskKind):
    p = tl.load_plan(plan)
    tr = tl.build_trace(p, tl.read_raw(trace))
    pro = {}
    if prologue and prologue.exists():
        raw = prologue.read_bytes()
        words = struct.unpack(f"<{len(raw) // 8}Q", raw)
        descs = p["descs"]
        for t in tr.tasks:
            m = re.match(r"(.+?)(?:_(-?\d+))?$", t.name)
            stem, L = m.group(1), int(m.group(2)) if m.group(2) else -1
            if stem not in KIND:
                continue
            kind = TaskKind[KIND[stem]].value
            c = descs[t.id]["coord"] + [0, 0]
            slot = ((L + 1) * 32 + kind) * 1024 + c[0] * 64 + c[1]
            w0, stage, stamp = words[3 * slot: 3 * slot + 3]
            if w0:
                pro[t.id] = ((stamp - w0) * TICK_US if stamp else 0.0, (stage - w0) * TICK_US if stage else 0.0)
    out = []
    for t in tr.tasks:
        m = re.match(r"(.+?)(?:_(-?\d+))?$", t.name)
        stem, L = m.group(1), int(m.group(2)) if m.group(2) else -1
        if stem in ("lm_head", "argmax", "embed"):
            L = -1
        pr, sg = pro.get(t.id, (None, None))
        out.append((L, KIND.get(stem, stem), t.ready, t.done, pr, sg))
    return out, bool(pro)


def spans(rows):
    d = defaultdict(lambda: [1e30, 0.0])
    for L, k, r, e, *_ in rows:
        key = "lm_head" if k in ("LM_HEAD", "ARGMAX") else ("embed" if k == "EMBED" else L)
        d[key][0] = min(d[key][0], r); d[key][1] = max(d[key][1], e)
    return d


def kinds(rows, layers):
    d = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])
    for L, k, r, e, pr, sg in rows:
        if L not in layers:
            continue
        a = d[k]; a[0] += 1; a[1] += e - r
        if pr is not None:
            a[2] += pr; a[3] += sg; a[4] += 1
    return {k: (v[0], v[1] / v[0], v[2] / v[4] if v[4] else None, v[3] / v[4] if v[4] else None) for k, v in d.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fleet-trace", type=Path, required=True)
    ap.add_argument("--fleet-graph", type=Path, required=True)
    ap.add_argument("--fleet-src", type=Path, default=Path.home() / "fleet-mi300x" / "src")
    ap.add_argument("--etx-plan", type=Path, required=True)
    ap.add_argument("--etx-trace", type=Path, required=True)
    ap.add_argument("--etx-prologue", type=Path)
    ap.add_argument("--layer", type=int, default=None, help="per-kind table for one layer (default: mean over MoE layers 1-26)")
    a = ap.parse_args(argv)
    fl, TaskKind = load_fleet(a.fleet_trace, a.fleet_graph, a.fleet_src)
    ex, have_pro = load_etx(a.etx_plan, a.etx_trace, a.etx_prologue, TaskKind)
    fs, es = spans(fl), spans(ex)
    print(f"{'layer':>7} {'fleet span':>10} {'ETX span':>9} {'diff':>7}   (us)")
    tot = 0.0
    for key in ["embed"] + sorted(k for k in fs if isinstance(k, int)) + ["lm_head"]:
        if key not in es:
            continue
        f, e = fs[key][1] - fs[key][0], es[key][1] - es[key][0]
        tot += e - f
        print(f"{str(key):>7} {f:10.1f} {e:9.1f} {e - f:+7.1f}")
    fend = max(x[3] for x in fl); eend = max(x[3] for x in ex)
    print(f"token: fleet {fend:.1f} us, ETX {eend:.1f} us ({eend - fend:+.1f}); inside layers {tot:+.1f}, between layers {eend - fend - tot:+.1f}")
    layers = {a.layer} if a.layer is not None else set(range(1, 27))
    fk, ek = kinds(fl, layers), kinds(ex, layers)
    what = f"layer {a.layer}" if a.layer is not None else "MoE layers 1-26, per task"
    print(f"\nper task kind, {what} (us): body = ready -> done; prologue = before the weight stream, staging its first part")
    print(f"  {'kind':16} {'n':>5} {'fleet body':>10} {'ETX body':>9} {'diff':>6}   {'fleet pro/stage':>15} {'ETX pro/stage':>14}")
    for k in sorted(fk, key=lambda k: -fk[k][1]):
        if k not in ek:
            continue
        n, fb, fp, fsg = fk[k]; _, eb, ep, esg = ek[k]
        epro = f"{ep:6.2f}/{esg:5.2f}" if ep is not None else ("  (no stamps)" if not have_pro else "      -")
        print(f"  {k:16} {n:5d} {fb:10.2f} {eb:9.2f} {eb - fb:+6.2f}   {fp:7.2f}/{fsg:5.2f}   {epro:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
