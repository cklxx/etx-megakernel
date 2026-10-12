"""Reproduce the CPU-only Brief 02 trace account; writes no files.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m examples.dsv2lite.phase_table300
Use --ledger to run ledger.main with the pinned fleet parser, or --json for exact values.
Fleet's working tree is deliberately never imported: taskgraph is loaded from the pinned export.
"""
from __future__ import annotations

import json
import argparse
import sys
import types
from collections import defaultdict
from pathlib import Path


def pinned_parser():
    from examples.dsv2lite.pinned import fleet_export
    module = types.ModuleType("taskgraph")
    module.__file__ = str(fleet_export() / "src/host/taskgraph.py")
    sys.modules[module.__name__] = module
    source = Path(module.__file__).read_text()
    exec(compile(source, module.__file__, "exec"), module.__dict__)


def aggregate(rows):
    result = defaultdict(lambda: defaultdict(float))
    for _, phase, values in rows:
        for key, value in values.items():
            result[phase][key] += value
    return {phase: dict(values) for phase, values in result.items()}


def totals(rows, cats):
    return {c: sum(row[2][c] for row in rows) for c in cats}


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


def main():
    from examples.dsv2lite import ledger as ld
    from examples.dsv2lite.bytes import per_layer

    if "--stamps2" in sys.argv:
        ap = argparse.ArgumentParser(description="Print native STAMPS2 boundaries without historic data dependencies")
        ap.add_argument("--stamps2", nargs=3, metavar=("PLAN", "TRACE", "PROLOGUE"), required=True)
        ap.add_argument("--json", action="store_true")
        args = ap.parse_args()
        tasks = ld.load_etx(*(Path(p) for p in args.stamps2))
        if args.json:
            print(json.dumps(ld.subphase_summary(tasks), indent=2))
        else:
            ld.print_subphases(tasks)
        return 0
    pinned_parser()
    if len(sys.argv) > 1 and sys.argv[1] == "--ledger":
        return ld.main(sys.argv[2:])
    from examples.dsv2lite.pinned import data_sets
    data = data_sets()
    new = data / "results_2026-09-27d/sess2"
    old = data / "results_2026-09-27b"
    tasks = ld.load_etx(new / "x1s_plan.json", new / "x1s_trace.bin", None)
    stamped = ld.load_etx(old / "etx_plan.json", old / "etx_trace.bin", old / "etx_pro.bin")
    fleet = ld.load_fleet(old / "fleet_trace.bin", old / "taskgraph_d16.bin", Path("/pinned-in-memory"))
    rows, span = ld.ledger(tasks, 0)
    old_rows, _ = ld.ledger(stamped, 0)
    fleet_rows, fleet_span = ld.ledger(fleet, 0)
    old_agg = aggregate(old_rows)
    ratios = {}
    for phase, values in old_agg.items():
        rest = sum(values[c] for c in ("staging", "prologue", "stream"))
        assert rest > 0
        ratios[phase] = {c: values[c] / rest for c in ("staging", "prologue", "stream")}
    split = []
    for layer, phase, values in rows:
        c = dict(values)
        c["staging"] = values["stream"] * ratios[phase]["staging"]
        c["prologue"] = values["stream"] * ratios[phase]["prologue"]
        c["stream"] = values["stream"] - c["staging"] - c["prologue"]
        assert abs(sum(c.values()) - sum(values.values())) < 1e-9
        split.append((layer, phase, c))
    assert abs(sum(totals(split, ld.CATS).values()) - span) < 1e-8

    by = defaultdict(list)
    for task in tasks:
        by[(task.layer, task.phase)].append(task)
    t0 = min(t.ready for t in by[(-1, "embed")])
    stats = []
    for phase in {t.phase for t in tasks}:
        groups = [g for (_, p), g in by.items() if p == phase]
        group = [t for g in groups for t in g]
        worker = sum(t.done - t.ready for t in group)
        stats.append(dict(phase=phase, n=len(group), worker_us=worker,
                          per_task_us=worker / len(group),
                          in_layer_us=sum(max(t.done for t in g) - min(t.ready for t in g)
                                          for g in groups) / len(groups),
                          finishes_at_us=max(t.done for t in group) - t0))
    stats.sort(key=lambda s: s["finishes_at_us"])
    # Contiguous sections match fleet's boundary convention: last qkv belongs
    # to core; the tail begins immediately at qkv completion, including handoff.
    section_rows = defaultdict(list)
    for row in split:
        layer, phase, _ = row
        sec = ("prologue" if layer == 0 or phase == "embed" else
               "tail" if phase in ("lm_head", "argmax") or (layer == 26 and phase != "qkv")
               else "core")
        section_rows[sec].append(row)
    sections = {s: totals(r, ld.CATS) for s, r in section_rows.items()}
    assert abs(sum(sections["prologue"].values()) - (max(t.done for t in by[(0, "dense_dn")]) - t0)) < 1e-8
    assert abs(sum(sections["prologue"].values()) + sum(sections["core"].values()) -
               (max(t.done for t in by[(26, "qkv")]) - t0)) < 1e-8

    comparison = []
    for phase in ld.MOE_CHAIN:
        pair = []
        for rs in (split, fleet_rows):
            selected = [r for r in rs if 2 <= r[0] <= 25 and r[1] == phase]
            assert len(selected) == 24
            pair.append({k: v / len(selected) for k, v in totals(selected, ld.CATS).items()})
        comparison.append(dict(phase=phase, etx=pair[0], fleet=pair[1],
                               delta=sum(pair[0].values()) - sum(pair[1].values())))

    # bytes.py is unique-only. As-requested here means useful weight/cache
    # reads with OPS's explicit qkv x8-kva, cache x16-head, router x8 factors.
    # It excludes over-issue and activation traffic; no claim of measured HBM.
    b = per_layer(1024)
    byte_rows = []
    for phase, count in [("embed", 1), ("qkv", 27), ("qabs", 27), ("attn", 27),
                         ("merge", 27), ("oproj", 27), ("norm", 1), ("router", 26),
                         ("gate_up", 26), ("down", 26), ("dense_gu", 1),
                         ("dense_dn", 1), ("lm_head", 1), ("argmax", 1)]:
        unique = int(b.get(phase, 0)) * count
        requested = unique
        if phase == "qkv":
            requested += count * 7 * 2048 * (512 + 64) * 2
        if phase == "attn":
            requested *= 16
        if phase == "router":
            requested *= 8
        byte_rows.append(dict(phase=phase, count=count, unique=unique, requested=requested,
                              unique_43=unique / 4.3e6, unique_42=unique / 4.2e6,
                              requested_43=requested / 4.3e6, requested_42=requested / 4.2e6))
    ledger_bytes = sum(v["floor"] for _, _, v in rows) * ld.BW
    unique_total = sum(r["unique"] for r in byte_rows)
    # Ledger excludes concurrent qabs, but assigns router bytes to dense norm.
    assert abs(ledger_bytes - (unique_total - b["qabs"] * 27 + b["router"])) < 1e-6
    output = dict(span=span, fleet_span=fleet_span, tasks=len(tasks), phase_stats=stats,
                  ratios=ratios, phase_ledger=aggregate(split), sections=sections,
                  totals=totals(split, ld.CATS), unsplit_totals=totals(rows, ld.CATS),
                  comparison=comparison, fleet_totals=totals(fleet_rows, ld.CATS),
                  byte_rows=byte_rows, ledger_bytes=ledger_bytes,
                  bridge=dict(target_us=1e6 / 300,
                              traced_to_untraced=span - 3474, untraced_to_target=3474 - 1e6 / 300,
                              traced_to_target=span - 1e6 / 300, gate_2pct=3474 * .02))
    pa = output["phase_ledger"]
    metric_us = {
        "exact device gap": 3474 - 1e6 / 300,
        "rounded device gap": 141., "stale owner gap": 387.,
        "observed trace uplift": span - 3474,
        "traced to target": span - 1e6 / 300,
        "wall gap (not target basis)": 3507 - 1e6 / 300,
        "rounded wall gap": 174.,
        "2 percent at 3474": 3474 * .02, "rounded gate": 70.,
        "qkv stream excess": pa["qkv"]["stream"],
        "redundant kv_a @4.3": 27 * 7 * 2048 * 576 * 2 / 4.3e6,
        "redundant kv_a @4.2": 27 * 7 * 2048 * 576 * 2 / 4.2e6,
        "gate_up staging+prologue": pa["gate_up"]["staging"] + pa["gate_up"]["prologue"],
        "router staging": pa["router"]["staging"],
        "router+dense norm staging": pa["router"]["staging"] + pa["norm"]["staging"],
        "handoff+ramp+tail": sum(output["totals"][c] for c in ("handoff", "ramp", "tail")),
        "nonfloor": span - output["totals"]["floor"],
        "ledger denominator change": ledger_bytes / 4.2e6 - ledger_bytes / 4.3e6,
        "qabs minus phantom norm @4.2": (b["qabs"] * 27 - b["router"]) / 4.2e6,
        "fleet measured-model residual @4.2": (5.20e9 - unique_total) / 4.2e6,
        "fleet measured bytes floor @4.2": 5.20e9 / 4.2e6,
        "ledger-to-fleet floor difference": 5.20e9 / 4.2e6 - ledger_bytes / 4.3e6,
    }
    output["metrics"] = {k: {"us": v, "pct141": 100 * v / 141, "pct387": 100 * v / 387}
                         for k, v in metric_us.items()}
    output["byte_reconciliation"] = dict(
        parallel_qabs=b["qabs"] * 27, phantom_norm=b["router"],
        kva_repeats=27 * 7 * 2048 * 576 * 2,
        cache_head_repeats=27 * b["attn"] * 15, router_repeats=26 * b["router"] * 7,
        weights_and_embed=unique_total - 27 * b["attn"],
        fleet_residual_1024=5.20e9 - unique_total,
        fleet_model_1056=unique_total + 27 * 32 * 576 * 2,
        fleet_residual_1056=5.20e9 - unique_total - 27 * 32 * 576 * 2,
        cache_context_difference=27 * 32 * 576 * 2)
    if "--json" in sys.argv:
        print(json.dumps(output, indent=2))
        return 0
    print("PHASE STATS [M]")
    table(["phase", "n", "worker_us", "per_task_us", "in_layer_us", "finishes_at_us"],
          [[s["phase"], s["n"]] + [f"{s[k]:.2f}" for k in
           ("worker_us", "per_task_us", "in_layer_us", "finishes_at_us")] for s in stats])
    print("\nRATIOS [E] transferred staging / remainder-prologue / residual-stream")
    table(["phase", "staging %", "prologue %", "stream %"],
          [[p] + [f"{r[c] * 100:.3f}" for c in ("staging", "prologue", "stream")]
           for p, r in ratios.items()])
    for title, groups in (("PHASE LEDGER", output["phase_ledger"]), ("SECTION LEDGER", sections)):
        print("\n" + title)
        table(["phase/section"] + ld.CATS + ["total", "%141 / %387"],
              [[p] + [f"{v[c]:.3f}" for c in ld.CATS] + [f"{sum(v.values()):.3f}",
                f"{sum(v.values()) / 141 * 100:.2f}% / {sum(v.values()) / 387 * 100:.2f}%"]
               for p, v in groups.items()] +
              [["TOTAL"] + [f"{output['totals'][c]:.3f}" for c in ld.CATS] + [f"{span:.3f}",
                f"{span / 141 * 100:.2f}% / {span / 387 * 100:.2f}%"]])
    print("\nCATEGORY POOLS AND SECTION SHARES")
    table(["pool", "us", "%141", "%387", "%traced-token"],
          [[k, f"{v:.3f}", f"{v / 141 * 100:.2f}", f"{v / 387 * 100:.2f}", f"{v / span * 100:.2f}"]
           for k, v in list(output["totals"].items()) + [(s, sum(v.values())) for s, v in sections.items()]])
    print("\nFLEET COMPARISON: each category ETX / fleet, mean layers 2-25")
    layer_pair = {arm: {c: sum(r[arm][c] for r in comparison) for c in ld.CATS}
                  for arm in ("etx", "fleet")}
    table(["phase"] + ld.CATS + ["total E/F", "delta"],
          [[r["phase"]] + [f"{r['etx'][c]:.3f}/{r['fleet'][c]:.3f}" for c in ld.CATS] +
           [f"{sum(r['etx'].values()):.3f}/{sum(r['fleet'].values()):.3f}", f"{r['delta']:+.3f}"]
           for r in comparison] + [["layer"] +
           [f"{layer_pair['etx'][c]:.3f}/{layer_pair['fleet'][c]:.3f}" for c in ld.CATS] +
           [f"{sum(layer_pair['etx'].values()):.3f}/{sum(layer_pair['fleet'].values()):.3f}",
            f"{sum(r['delta'] for r in comparison):+.3f}"]])
    print("\nBYTES [E]: 1024 cached rows, useful requested excludes over-issue/activations")
    table(["phase", "instances", "unique B", "requested B", "unique us 4.3/4.2", "requested us 4.3/4.2"],
          [[r["phase"], r["count"], r["unique"], r["requested"],
            f"{r['unique_43']:.3f}/{r['unique_42']:.3f}",
            f"{r['requested_43']:.3f}/{r['requested_42']:.3f}"] for r in byte_rows])
    print("BYTE TOTALS", json.dumps({k: sum(r[k] for r in byte_rows) for k in
                                      ("unique", "requested", "unique_43", "unique_42", "requested_43", "requested_42")}))
    print("LEDGER_BYTES", ledger_bytes, "BRIDGE", json.dumps(output["bridge"]))
    print("FLEET_TOTALS", json.dumps(output["fleet_totals"]), "SPAN", fleet_span)
    print("\nDERIVED METRICS [E arithmetic; not demonstrated gains]")
    table(["quantity", "us", "%141", "%387"],
          [[name, f"{v['us']:.6f}", f"{v['pct141']:.4f}", f"{v['pct387']:.4f}"]
           for name, v in output["metrics"].items()])
    print("BYTE_RECONCILIATION", json.dumps(output["byte_reconciliation"]))
    print("MEAN_LAYER_ETX_FLEET_DELTA", *[sum(sum(r[arm].values()) for r in comparison)
                                             for arm in ("etx", "fleet")],
          sum(r["delta"] for r in comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
