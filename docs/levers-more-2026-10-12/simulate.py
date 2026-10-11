"""CPU mechanism oracles and conditional trace replay; no GPU code is executed.

Run from the repository root with .venv/bin/python. Costs below are hypotheses,
not fits to intra-body measurements. Existing X7 inputs are read, never rewritten.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import random
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
X7 = ROOT / "docs/x7-schedule-whatif-2026-10-11"
sys.path.insert(0, str(X7))
import search as x7
from etx.sim.xcd import Cost, Hw, ReplayTiming, simulate

OUT = Path(__file__).resolve().parent


def dag(nodes):
    """Infinite issue capacity, finite dependencies: intentionally optimistic."""
    ends = {}
    for name, duration, parents in nodes:
        ends[name] = max((ends[p] for p in parents), default=0) + duration
    return max(ends.values())


def load_shapes(rtt, distance):
    # distance 0: source issue order; 1: independent scalar metadata early;
    # distance 2: dependent payload early too. Values/compute ordering unchanged.
    early = distance >= 1
    payload = distance >= 2
    prep = [("q", rtt, []), ("rms", .2, ["q"]),
            ("pos", rtt, [] if early else ["rms"]),
            ("cs", rtt, ["pos"]),
            ("rope", .2, ["rms", "cs"]),
            ("slot", rtt, [] if early else ["rope"]),
            ("store", .1, ["rope", "slot"])]
    reduce = [("seq", rtt, []), ("max", rtt, ["seq"]),
              ("tree", .1, ["max"]),
              ("sum", rtt, ["seq"] if early else ["tree"]),
              ("exp_tree", .1, ["sum", "tree"]),
              ("tmp", rtt, ["seq"] if payload else ["exp_tree"]),
              ("fma", .2, ["tmp", "exp_tree"])]
    attn = [("seq", rtt, []), ("kt", rtt, ["seq"]),
            ("q_k", rtt + .2, ["kt"]),
            ("vt", rtt, ["seq"] if early else ["q_k"]),
            ("v", rtt, ["vt"]), ("math", .4, ["v", "q_k"])]
    topk = [("logit0", rtt, []), ("logit1", rtt, [] if early else ["logit0"]),
            ("key_math", .1, ["logit0", "logit1"])]
    return {"prep": dag(prep), "reduce": dag(reduce), "attn": dag(attn), "topk": dag(topk)}


def bitonic(keys):
    a = list(keys)
    stages = 0
    k = 2
    while k <= len(a):
        j = k // 2
        while j:
            old = a[:]
            for i in range(len(a)):
                partner = i ^ j
                # Ascending final network; only immutable integer keys move.
                lo, hi = sorted((old[i], old[partner]))
                a[i] = lo if ((i & j) == 0) == ((i & k) == 0) else hi
            stages += 1
            j //= 2
        k *= 2
    return a[::-1], stages


def mechanism_oracles():
    rng = random.Random(20261012)
    # Reuse/vector-preserving swizzle preserves logical x and multiply order.
    for k, epl in [(2048, 8), (768, 4)]:
        x = [rng.getrandbits(32) for _ in range(k)]
        def perm(i):
            seg, off = divmod(i, 64 * epl)
            lane, element = divmod(off, epl)
            flip = (lane >> 4) & 1 if epl == 8 else 0
            return seg * 64 * epl + lane * epl + ((element // 4) ^ flip) * 4 + element % 4
        assert sorted(perm(i) for i in range(k)) == list(range(k))
        transposed = [0] * k
        for i in range(k):
            transposed[perm(i)] = x[i]
        for lane in range(64):
            stash = [x[seg * 64 * epl + lane * epl + i]
                     for seg in range(k // (64 * epl)) for i in range(epl)]
            for row in range(4):
                direct = [transposed[perm(seg * 64 * epl + lane * epl + i)]
                          for seg in range(k // (64 * epl)) for i in range(epl)]
                assert direct == stash
    # AMD's CDNA ds_read_b128 lane groups (CK documentation, see main report).
    read_groups = [list(range(a, a + 4)) + list(range(b, b + 4))
                   for a, b in ((0, 20), (4, 16), (8, 28), (12, 24),
                                (32, 52), (36, 48), (40, 60), (44, 56))]
    bank = {}
    for epl in (4, 8):
        def conflict(swizzle):
            counts = []
            for lanes in read_groups:
                for half in range(epl // 4):
                    offsets = [lane * epl + (half ^ (((lane >> 4) & 1) if swizzle and epl == 8 else 0)) * 4
                               for lane in lanes]
                    counts.append(max(Counter((off + word) % 32 for off in offsets for word in range(4)).values()))
            return max(counts)
        bank[str(epl)] = {"b128_original_conflict": conflict(False), "b128_swizzle_conflict": conflict(True)}
        assert conflict(True) == 1
    for n in range(512):
        # Unique tie-breaking low 7 bits, like KEY32; also zero/sentinel cases.
        keys = [(rng.randrange(65536) << 7) | (127 - e) for e in range(128)]
        if n == 0:
            keys = [127 - e for e in range(128)]
        if n == 1:
            keys = [0] * 128
        ordered, stages = bitonic(keys)
        assert ordered[:8] == sorted(keys, reverse=True)[:8]
    # The proposed two outstanding requests carry distinct destination registers,
    # and join before any conversion/max; completion order cannot mix payloads.
    for order in ((0, 1), (1, 0)):
        result = [None, None]
        values = [rng.getrandbits(16), rng.getrandbits(16)]
        for i in order:
            result[i] = values[i]
        assert result == values
    # Every address in the proposed prefetch is the original address after seq_len.
    for seq in range(1024, 1057):
        partitions = (seq + 255) // 256
        assert partitions in (4, 5)
        for tid in range(128):
            assert min(tid, partitions - 1) in range(partitions)
        # Separate K/V lane maps, including the clamped final partition.
        for part in range(partitions):
            for warp in range(4):
                for lane16 in range(16):
                    idx = part * 256 + 64 * warp + lane16
                    assert (idx // 16 if idx < seq else (seq + 15) // 16 - 1) < (seq + 15) // 16
            for depth in range(4):
                for row in range(4):
                    idx = part * 256 + depth * 64 + row * 16
                    assert (idx // 16 if idx < seq else (seq + 15) // 16 - 1) < (seq + 15) // 16
    print(f"ORACLE exact index reuse/swizzle: K=2048,768; bitonic 512/512, stages={stages}; clamped-address positions=33", flush=True)
    print("BANK b128 " + json.dumps(bank, sort_keys=True), flush=True)
    shapes = []
    for rtt in (.50, .69, .73):
        base = load_shapes(rtt, 0)
        for distance in (0, 1, 2, 4):
            after = load_shapes(rtt, distance)
            cuts = {role: round(base[role] - after[role], 6) for role in base}
            shapes.append(dict(rtt_us=rtt, distance=distance, body_cut_us=cuts))
            print(f"DAG rtt={rtt:.2f} distance={distance} cut_us={json.dumps(cuts, sort_keys=True)}", flush=True)
    return dict(bank_model=bank, selection_fixtures=512, selection_stages=stages, load_dags=shapes)


def main():
    mechanisms = mechanism_oracles()
    pbytes = gzip.decompress((X7 / "original-plan.json.gz").read_bytes())
    rawbytes = gzip.decompress((X7 / "measured.bin.gz").read_bytes())
    p = json.loads(pbytes)
    raw = x7.load_raw()
    inputs = json.loads((X7 / "inputs.json").read_text())
    assert hashlib.sha256(pbytes).hexdigest() == inputs["plan_sha256"]
    assert hashlib.sha256(rawbytes).hexdigest() == inputs["trace_sha256"]
    baseline, _ = x7.cp_account(p, raw)
    timing = ReplayTiming(inputs["queue_after_us"], inputs["ready_after_us"],
                          {int(k): v for k, v in inputs["initial_taken_us"].items()})
    names = {v["id"]: k for k, v in p["types"].items()}
    groups = defaultdict(list)
    for d in p["descs"]:
        groups[d["type"]].append(d["id"])
    records = []

    def run(label, plan, cuts=None, keep=False):
        cuts = cuts or {}
        floors = {"prep": 1.52, "reduce": 2.81, "attn": 2.92, "moe_up": 14.67, "topk": 1.21}
        body = []
        for d in plan["descs"]:
            role = x7.stem(names[d["type"]])
            old = inputs["body_us"][d["id"]]
            # Conservative clipping: do not cut a task already below the role's
            # phase floor; phase floors are not asserted to be per-task bounds.
            cut = min(cuts.get(role, 0), max(0, old - floors.get(role, 0)))
            body.append(old - cut)
        rows, stats = simulate(plan, lambda n, d: Cost(chain_us=body[d["id"]]),
                               Hw(handoff_us={"DOMAIN": 0., "DEVICE": 0.}), replay=timing)
        assert stats["unfinished"] == 0
        assert plan["ev_inst"] == p["ev_inst"] and plan["task_in"] == p["task_in"]
        assert all(abs((r[7] - r[6]) * .01 - body[t]) <= .010001 for t, r in enumerate(rows))
        latest = [max(rows[t][7] for t in ev[2]) for ev in plan["ev_inst"]]
        assert all(rows[t][6] >= max((latest[e] for e in ins), default=0) for t, ins in enumerate(plan["task_in"]))
        tr, cp = x7.cp_account(plan, rows, baseline.deps)
        if label == "baseline":
            origin = min(r[5] for r in raw)
            error = max(abs(rows[t][k] - (raw[t][k] - origin + 1)) for t in range(len(raw)) for k in (5, 6, 7))
            assert error == 0
            print(f"BASELINE tasks={len(rows)} all_stamp_error_ticks={error} span_us={tr.span:.3f}", flush=True)
        rec = dict(id=label, span_us=tr.span, gain_us=baseline.span - tr.span,
                   body_path_us=sum(v["body"] for v in cp.values()), gap_path_us=sum(v["gap"] for v in cp.values()),
                   cp=cp, body_cut_hypothesis_us=cuts, unfinished=stats["unfinished"])
        records.append(rec)
        print(f"SIM {label} span={tr.span:.3f} gain={rec['gain_us']:+.3f} us body={rec['body_path_us']:.3f} gap={rec['gap_path_us']:.3f}", flush=True)
        if keep:
            with gzip.GzipFile(filename=str(OUT / f"{label}.bin.gz"), mode="wb", mtime=0) as f:
                f.write(array('q', [v for r in rows for v in r]).tobytes())
            with gzip.GzipFile(filename=str(OUT / f"{label}-plan.json.gz"), mode="wb", mtime=0) as f:
                f.write(json.dumps(plan, separators=(',', ':')).encode())
        return rec

    run("baseline", p)
    # These are mechanism-derived CONDITIONAL body cuts, not prediction by fiat
    # that an implementation will achieve them. Also run zero and half exposure.
    shapes = load_shapes(.69, 0)
    early = load_shapes(.69, 2)
    lds_service = 524288 / (128 * 2100)  # MOE:48 nominal service assumption
    hypotheses = {
        "P": {"prep": shapes["prep"] - early["prep"]},
        "R": {"reduce": shapes["reduce"] - early["reduce"]},
        "A": {"attn": shapes["attn"] - early["attn"]},
        "T": {"topk": shapes["topk"] - early["topk"]},
        "S": {"moe_up": .5 * .5 * lds_service},
        "X": {"moe_up": .5 * .75 * lds_service},
    }
    for label, cuts in hypotheses.items():
        run(label + "-half", p, {k: .5 * v for k, v in cuts.items()})
        run(label, p, cuts)
    combined = {**hypotheses["P"], **hypotheses["R"], **hypotheses["A"], **hypotheses["T"], **hypotheses["X"]}
    # S and X overlap; explicitly use X rather than add both cuts.
    run("combined", p, combined, keep=True)
    for width in (6, 14, 38):
        for placement in ("original", "rotate_heads", "attention_wide"):
            c = copy.deepcopy(p)
            for d in c["descs"]:
                role = x7.stem(names[d["type"]])
                if placement == "rotate_heads" and role in ("prep", "attn", "reduce"):
                    d["domain"] = (d["domain"] + 4) % 8
                if placement == "attention_wide" and role == "attn":
                    d["domain"] = d["coord"][0] % 8
            queues = {f"0:{w}": [] for w in range(304)}
            for ids in groups.values():
                offsets = Counter()
                for t in ids:
                    domain = c["descs"][t]["domain"]
                    worker = domain * 38 + offsets[domain] % width
                    offsets[domain] += 1
                    c["descs"][t]["worker"] = worker
                    queues[f"0:{worker}"].append(t)
            c["static_queues"] = queues
            required = x7.shares(c)
            for e in c["events"]:
                if required[e["name"]] > 1:
                    e["scope"] = "DEVICE"
            x7.validate(c, p)
            # Promotion price zero is an optimistic bound. Losers stay losers
            # with any positive premium. No scope discount is granted.
            run(f"width{width}-{placement}-promotion0", c)
    for up_width in (18, 24):
        c = copy.deepcopy(p)
        queues = {f"0:{w}": [] for w in range(304)}
        for ids in groups.values():
            offsets = Counter()
            for t in ids:
                d = c["descs"][t]
                role = x7.stem(names[d["type"]])
                domain = d["domain"]
                start, width = ((0, up_width) if role == "moe_up" else
                                (up_width, 38 - up_width) if role == "moe_down" else (0, 38))
                worker = domain * 38 + start + offsets[domain] % width
                offsets[domain] += 1
                d["worker"] = worker
                queues[f"0:{worker}"].append(t)
        c["static_queues"] = queues
        x7.validate(c, p)
        run(f"split-up{up_width}-down{38-up_width}", c)
    report = dict(condition="CPU, trace-conditioned Qwen3-30B-A3B B1 C1024; not GPU measurements",
                  simulator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plan_sha256=inputs["plan_sha256"], trace_sha256=inputs["trace_sha256"],
                  mechanisms=mechanisms, results=records)
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
