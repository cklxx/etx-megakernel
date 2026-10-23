"""Time ledger of a DeepSeek-V2-Lite decode token: every microsecond of the token assigned to one named cause.

The token is a chain of phases per request: embed, then per layer q/kv_a -> attention -> merge -> o_proj -> norm/router ->
gate_up -> down, then lm_head -> argmax (q_absorb runs beside attention and is reported, not chained). For each phase P,
with `prev_end` the last completion of the phase before it:

  handoff   first task of P ready - prev_end          event latency (release, arrive, poll, acquire); negative when
                                                       per-XCD events let the fastest XCD start before the slowest ends
  ramp      median ready - first ready                 how spread the starts are (XCDs / workers noticing the event)
  staging   mean prologue staging of P's tasks         reading the upstream vector(s): partial folds, kv_a, routing logits
  prologue  mean rest of the prologue                  norm, top-k, q_absorb wait ... (before the weight stream)
  floor     P's weight bytes / 4.3 TB/s                the stream if every byte of the phase moved at full bandwidth
  stream    mean (body - prologue) - floor             stream time above the bandwidth floor (per-task size, depth, tails
                                                       of the GEMV loop, contention with other traffic)
  tail      last done - (median ready + mean body)     the slowest task / XCD beyond the typical one

so that prev_end + handoff + ramp + staging + prologue + floor + stream + tail = last done of P, exactly; summed over the
chain the categories add up to the token (embed ready -> argmax done). Without prologue stamps, staging and prologue are
0 and their time is inside `stream`.

  python -m examples.dsv2lite.ledger --etx PLAN TRACE [PROLOGUE] [--label x1] --fleet TRACE GRAPH [--fleet-src DIR]
  (any number of --etx / --fleet traces; a two-request ETX trace yields one ledger per request)
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from etx.tools import timeline as tl
from etx.models.declarations import declaration as _declaration
from . import paths
per_layer = _declaration(paths.ROOT, "dsv2lite.bytes").per_layer
fleet_export = _declaration(paths.ROOT, "dsv2lite.pinned").fleet_export

BW = 4.3e6          # bytes per us (4.3 TB/s)
TICK_US = 0.01
CATS = ["handoff", "ramp", "staging", "prologue", "floor", "stream", "tail"]
MOE_CHAIN = ["qkv", "attn", "merge", "oproj", "router", "gate_up", "down"]
DENSE_CHAIN = ["qkv", "attn", "merge", "oproj", "norm", "dense_gu", "dense_dn"]
FLEET_KIND = {"EMBED": "embed", "QKV_FUSED": "qkv", "Q_ABSORB": "qabs", "ATTENTION": "attn", "MERGE_UV": "merge",
              "O_PROJ": "oproj", "NORM_ROUTER": "router", "EXPERT_GATE_UP": "gate_up", "EXPERT_DOWN": "down",
              "DENSE_GATE_UP": "dense_gu", "DENSE_DOWN": "dense_dn", "LM_HEAD": "lm_head", "ARGMAX": "argmax"}
# fleet 3fb67a6 TaskKind, frozen rather than silently importing a working tree.
# D4 (examples/dsv2lite/graph.py build_moe_layer_d4): the shared XCDs run their expert stages as their own grids;
# in the chain they are the same phases, so a phase ends when its last task does, routed or shared
D4_PHASE = {"shgu": "gate_up", "sdown": "down"}
STAMP_KINDS = {"qkv": 0, "attn": 1, "merge": 2, "oproj": 3, "router": 4, "norm": 4,
               "gate_up": 5, "down": 6, "dense_gu": 8, "dense_dn": 9, "embed": 10,
               "lm_head": 11, "argmax": 12, "qabs": 14}
STAMPS2_MAGIC = 0x32504D4154535445
STAMPS2_INTERVALS = {
    "qkv": {"norm_tree": (4, 5), "gamma_apply": (6, 7), "norm": (3, 8), "q_gemv": (9, 10), "kv_a_gemv": (10, 11)},
    "router": {"norm_tree": (4, 5), "gamma_apply": (6, 7), "norm": (3, 8)},
    "norm": {"norm_tree": (4, 5), "gamma_apply": (6, 7), "norm": (3, 8)},
    "merge": {"M1_coefficients": (12, 13), "M2_partial_fold": (13, 14), "M3_w_uv_gemv": (14, 15)},
}


def read_prologue(path: Path):
    raw = path.read_bytes()
    if len(raw) % 8:
        raise ValueError("prologue file is not a whole number of u64 words")
    words = struct.unpack(f"<{len(raw) // 8}Q", raw)
    stride = 3
    if words and words[0] == STAMPS2_MAGIC:
        if len(words) < 4 or words[1:3] != (2, 16):
            raise ValueError("unsupported STAMPS2 header")
        _, _, stride, slots = words[:4]
        words = words[4:]
        if len(words) != slots * stride:
            raise ValueError("STAMPS2 payload length does not match header")
    elif len(words) % stride:
        raise ValueError("legacy prologue file has a partial slot")
    return words, stride


def subphase_intervals(row, phase: str) -> dict[str, float]:
    if len(row) != 16 or not row[0]:
        return {}
    values = {}
    for label, (a, b) in STAMPS2_INTERVALS.get(phase, {}).items():
        if row[a] and row[b]:
            if row[b] < row[a]:
                raise ValueError(f"nonmonotonic STAMPS2 interval {phase}/{label}")
            values[label] = (row[b] - row[a]) * TICK_US
    return values


@dataclass
class T:
    req: int
    layer: int          # -1: embed / lm_head / argmax
    phase: str
    ready: float
    done: float
    pro: float | None   # prologue us (None: not stamped)
    stage: float | None
    subphases: dict[str, float] = field(default_factory=dict)


def load_etx(plan: Path, trace: Path, prologue: Path | None, merge_d4: bool = True) -> list[T]:
    p = tl.load_plan(plan)
    tr = tl.build_trace(p, tl.read_raw(trace))
    stamps = {}
    if prologue and prologue.exists():
        words, stride = read_prologue(prologue)
        kinds = STAMP_KINDS
        if stride == 16 and any(t.name.startswith("iso") for t in tr.tasks):
            raise ValueError("STAMPS2 rejects isolate plans: repeated tasks collide with native slots")
        for t in tr.tasks:
            m = re.match(r"(?:r(\d)_)?([a-z_]+?)(?:_(-?\d+))?$", t.name)
            if not m or int(m.group(1) or 0) != 0 or m.group(2) not in kinds:
                continue
            L = int(m.group(3)) if m.group(3) else -1
            c = p["descs"][t.id]["coord"] + [0, 0]
            slot = ((L + 1) * 32 + kinds[m.group(2)]) * 1024 + c[0] * 64 + c[1]
            row = words[stride * slot: stride * slot + stride]
            if len(row) != stride:
                raise ValueError(f"prologue slot {slot} absent from file")
            w0, stg, pro = row[:3]
            if w0:
                stamps[t.id] = ((pro - w0) * TICK_US if pro else 0.0, (stg - w0) * TICK_US if stg else 0.0,
                                subphase_intervals(row, m.group(2)))
    out = []
    for t in tr.tasks:
        m = re.match(r"(?:r(\d)_)?([a-z_]+?)(?:_(-?\d+))?$", t.name)
        if not m:
            continue
        r, ph, L = int(m.group(1) or 0), m.group(2), int(m.group(3)) if m.group(3) else -1
        if merge_d4:
            ph = D4_PHASE.get(ph, ph)   # D4: the shared XCDs' expert stages join the routed ones
        pr, sg, sub = stamps.get(t.id, (None, None, {}))
        out.append(T(r, L, ph, t.ready, t.done, pr, sg, sub))
    return out


def load_fleet(trace: Path, graph: Path, src: Path) -> list[T]:
    sys.path.insert(0, str(src / "host"))
    from taskgraph import PACK_FIELDS, PACK_FORMAT, TaskKind
    blob = graph.read_bytes()
    tasks = [dict(zip(PACK_FIELDS, struct.unpack_from(PACK_FORMAT, blob, 64 * i))) for i in range(len(blob) // 64)]
    tr = trace.read_bytes()
    st = [struct.unpack_from("<4Q", tr, 32 * i) for i in range(len(tr) // 32)]
    rows = [(t, s) for t, s in zip(tasks, st) if s[0]]
    t0 = min(s[1] for _, s in rows)
    out = []
    for t, s in rows:
        ph = FLEET_KIND[TaskKind(t["kind"]).name]
        if t["layer"] < 1 and ph == "router":
            ph = "norm"
        L = t["layer"] if ph not in ("embed", "lm_head", "argmax") else -1
        pro, stg = ((s[3] >> 32) & 0xFFFFFF) * TICK_US, ((s[3] >> 8) & 0xFFFFFF) * TICK_US
        out.append(T(0, L, ph, (s[1] - t0) * TICK_US, (s[2] - t0) * TICK_US, pro, stg))
    return out


def phase_bytes(ph: str, L: int) -> float:
    b = per_layer(1024)
    key = {"attn": "attn", "norm": "router"}.get(ph, ph)
    return b.get(key, 0.0)


def ledger(tasks: list[T], req: int):
    """-> (rows: list of (layer, phase, dict of categories), token span, qabs spans)"""
    ts = [t for t in tasks if t.req == req]
    by = defaultdict(list)
    for t in ts:
        by[(t.layer, t.phase)].append(t)
    layers = sorted({t.layer for t in ts if t.layer >= 0})
    chain = [(-1, "embed")]
    for L in layers:
        chain += [(L, p) for p in (DENSE_CHAIN if (L, "dense_gu") in by else MOE_CHAIN)]
    chain += [(-1, "lm_head"), (-1, "argmax")]
    rows = []
    prev_end = min(t.ready for t in by[(-1, "embed")])
    t_start = prev_end
    for key in chain:
        g = by.get(key)
        if not g:
            continue
        ready = sorted(t.ready for t in g)
        first, med = ready[0], ready[len(ready) // 2]
        body = sum(t.done - t.ready for t in g) / len(g)
        stamped = [t for t in g if t.pro is not None]
        pro = sum(t.pro for t in stamped) / len(stamped) if stamped else 0.0
        stg = sum(t.stage for t in stamped) / len(stamped) if stamped else 0.0
        last = max(t.done for t in g)
        floor = phase_bytes(key[1], key[0]) / BW
        c = {"handoff": first - prev_end, "ramp": med - first, "staging": stg, "prologue": pro - stg,
             "floor": floor, "stream": body - pro - floor, "tail": last - (med + body)}
        rows.append((key[0], key[1], c))
        prev_end = last
    return rows, prev_end - t_start


def summarize(rows, span, label):
    tot = defaultdict(float)
    for _, _, c in rows:
        for k, v in c.items():
            tot[k] += v
    s = sum(tot.values())
    print(f"\n== {label}: token {span:.1f} us (ledger sum {s:.1f})")
    print("   " + "  ".join(f"{k} {tot[k]:7.1f}" for k in CATS))
    moe = defaultdict(lambda: defaultdict(float)); n = defaultdict(int)
    for L, ph, c in rows:
        if 2 <= L <= 25 and ph in MOE_CHAIN:
            n[ph] += 1
            for k, v in c.items():
                moe[ph][k] += v
    print(f"   per MoE layer (mean of layers 2-25), us:")
    print(f"   {'phase':8} " + " ".join(f"{k:>8}" for k in CATS) + f" {'total':>8}")
    layer = defaultdict(float)
    for ph in MOE_CHAIN:
        vals = {k: moe[ph][k] / n[ph] for k in CATS}
        for k in CATS:
            layer[k] += vals[k]
        print(f"   {ph:8} " + " ".join(f"{vals[k]:8.2f}" for k in CATS) + f" {sum(vals.values()):8.2f}")
    print(f"   {'layer':8} " + " ".join(f"{layer[k]:8.2f}" for k in CATS) + f" {sum(layer.values()):8.2f}")
    return tot, {ph: {k: moe[ph][k] / n[ph] for k in CATS} for ph in MOE_CHAIN}


def subphase_summary(tasks: list[T], req: int = 0):
    groups = defaultdict(list)
    for task in tasks:
        if task.req == req:
            for label, value in task.subphases.items():
                groups[(task.phase, label)].append(value)
    return [{"phase": phase, "part": label, "tasks": len(values), "mean_us": sum(values) / len(values),
             "min_us": min(values), "max_us": max(values)} for (phase, label), values in sorted(groups.items())]


def print_subphases(tasks: list[T], req: int = 0):
    rows = subphase_summary(tasks, req)
    if rows:
        print("\nSTAMPS2: thread-0 source intervals (us); instrumentation perturbs scheduling.")
        print("gamma_apply includes gamma loads + scale/bf16/store in default, apply only in early-gamma arms.")
        print("norm overlaps its tree/gamma_apply parts; do not add them. No workgroup completion fence is added.")
        print(f"{'phase':8} {'part':20} {'tasks':>7} {'mean_us':>10} {'min_us':>10} {'max_us':>10}")
        for row in rows:
            print(f"{row['phase']:8} {row['part']:20} {row['tasks']:7d} {row['mean_us']:10.3f} {row['min_us']:10.3f} {row['max_us']:10.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--etx", nargs="+", action="append", default=[], metavar="PLAN TRACE [PROLOGUE]")
    ap.add_argument("--fleet", nargs=2, action="append", default=[], metavar=("TRACE", "GRAPH"))
    ap.add_argument("--fleet-src", type=Path, default=None, help="default: the pinned 3fb67a6 export")
    ap.add_argument("--label", action="append", default=[])
    a = ap.parse_args(argv)
    labels = iter(a.label)
    for tr, gr in a.fleet:
        rows, span = ledger(load_fleet(Path(tr), Path(gr), a.fleet_src or fleet_export() / "src"), 0)
        summarize(rows, span, next(labels, f"fleet {tr}"))
    for e in a.etx:
        tasks = load_etx(Path(e[0]), Path(e[1]), Path(e[2]) if len(e) > 2 else None)
        lab = next(labels, f"etx {e[1]}")
        for r in sorted({t.req for t in tasks}):
            rows, span = ledger(tasks, r)
            summarize(rows, span, f"{lab} request {r}")
            print_subphases(tasks, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
