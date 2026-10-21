#!/usr/bin/env python3
"""Validate a cost model against every ETX build of Qwen3-30B-A3B we have measured on one MI300X
(docs/qwen/MODEL-vllm-kernels.md section 19). One run each, the numbers as they were reported.

  python3 -m examples.vllm_llm.validate [--etx-us 6.6] [--mapping contig] [--fit]

--fit sweeps the one free constant (ETX's per-task cost) and prints the value that minimises the worst relative
error, so the fit is visible rather than hidden in a default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.vllm_llm import pmodel

ROOT = Path(__file__).resolve().parents[2]

# name, plan, measured ms (per step), tolerance, note, profile
BUILDS = [
    ("build3 fused",      "results_2026-09-30/plan_moe_dsl3.json",   5.895, 0.10, "DSL GEMV + head attention + fold + touch", "2026-09-30"),
    ("build2 fused",      "results_2026-09-30/plan_moe_dsl2.json",   5.934, 0.10, "DSL GEMV + head attention", "2026-09-30"),
    ("build3 unfused",    "results_2026-09-30/plan_moe_dsl3.json",   5.660, 0.10, "same tiles, one launch each", "2026-09-30"),
    ("build3 2 requests", "results_2026-09-30/plan_moe_dsl3x2.json", 8.988, 0.10, "two interleaved requests, per step", "2026-09-30"),
    ("qknr fused",        "results_2026-09-28b/plan_moe_xcd_qknrc_new.json", 5.875, 0.15, "imported vLLM kernels, fused q/k norm + rope", "2026-09-30"),
    ("qknr unfused",      "results_2026-09-28b/plan_moe_base_qknr.json",     5.623, 0.15, "the same, one launch each", "2026-09-30"),
    ("moe new",           "results_2026-09-28b/plan_moe_xcd_new.json",       6.313, 0.15, "imported vLLM kernels, xcd + chain", "2026-09-30"),
    ("moe old",           "results_2026-09-28b/plan_moe_xcd_old.json",       6.556, 0.15, "before the argument-path fixes", "2026-09-30"),
    ("tiles fused", "build/rocm22/qwen3-30b-a3b_default/plan.json", 4.393, 0.10,
     "report.log:9", "2026-10-08"),
    ("tiles unfused", "build/rocm22/qwen3-30b-a3b_default/plan.json", 4.617, 0.10,
     "fusion_pairs.txt:2-6", "2026-10-08"),
]
UNFUSED = {"build3 unfused", "qknr unfused", "tiles unfused"}
F_UNFUSED = 1.61        # hw_bench launch: the kernel boundary a separate launch pays instead of a hand-off


def run(etx_us: float, mapping: str, cfg: dict, profile: str | None = None):
    out = []
    for name, rel, meas, tol, note, row_profile in BUILDS:
        if profile is not None and row_profile != profile:
            continue
        p = ROOT / rel
        if not p.exists():
            out.append((name, None, meas, tol, note, "plan missing"))
            continue
        if row_profile == "2026-10-08":
            actual = p.with_name("vl_plan.txt")
            reference = ROOT / "results_nvref_2026-10-09/vlplan_all.txt"
            if (not actual.exists() or not reference.exists()
                    or actual.read_text().splitlines()[1:] != reference.read_text().splitlines()[1:]):
                out.append((name, None, meas, tol, note, "plan differs"))
                continue
        plan = json.load(open(p))
        fused = name not in UNFUSED
        # unfused: every phase is its own launch, so the kernel boundary replaces the hand-off
        hw = pmodel.hw() if fused else pmodel.hw(handoff_domain=F_UNFUSED, handoff_device=F_UNFUSED)
        from etx.sim.xcd import simulate
        _, st = simulate(plan, pmodel.cost_fn(plan, cfg, mapping, etx_us, fused=fused, profile=row_profile), hw)
        out.append((name, st["makespan_us"] / 1e3, meas, tol, note, ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etx-us", type=float, default=pmodel.ETX_US)
    ap.add_argument("--mapping", default="contig")
    ap.add_argument("--config", default=str(ROOT / "examples/llm/configs/qwen3-30b-a3b.json"))
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    etx = a.etx_us
    if a.fit:
        best = None
        for x in [round(3.0 + 0.2 * i, 1) for i in range(41)]:
            r = run(x, a.mapping, cfg, profile="2026-09-30")
            worst = max((abs(p - m) / m for _, p, m, _, _, e in r if p), default=9)
            if best is None or worst < best[0]:
                best = (worst, x)
        etx = best[1]
        print(f"fitted etx_us = {etx} us (worst relative error {best[0] * 100:.1f}%)")
    rows = run(etx, a.mapping, cfg)
    print(f"\n{'build':20s} {'predicted':>10s} {'measured':>9s} {'error':>7s} {'gate':>6s}  note")
    ok = True
    for name, pred, meas, tol, note, err in rows:
        if pred is None:
            print(f"{name:20s} {'-':>10s} {meas:9.3f} {'-':>7s} {'-':>6s}  {err}")
            ok = False
            continue
        e = (pred - meas) / meas
        good = abs(e) <= tol
        ok &= good
        print(f"{name:20s} {pred:10.3f} {meas:9.3f} {e * 100:+6.1f}% {'pass' if good else 'FAIL':>6s}  {note}")
    tiles = {name: pred for name, pred, *_ in rows if name.startswith("tiles ")}
    if all(tiles.get(name) is not None for name in ("tiles fused", "tiles unfused")):
        gain = tiles["tiles unfused"] - tiles["tiles fused"]
        good = abs(gain - 0.225) < 0.093
        ok &= good
        print(f"fusion (unfused - fused) {gain:.3f} ms against 0.225 ms: {'pass' if good else 'FAIL'}")
        for form, measured in (("fused", 4.526), ("unfused", 4.657)):
            wall = tiles[f"tiles {form}"] + pmodel.HOST_US[form] / 1e3
            print(f"wall {form} {wall:.3f} ms against {measured:.3f} ms (information only)")
    print(f"\nmodel: floor {pmodel.phase.FLOOR_US} us per workgroup, etx {etx} us per task, HBM "
          f"{pmodel.phase.BW_HBM / 1e6:.2f} TB/s, mapping {a.mapping}")
    print("gate: " + ("all builds within tolerance" if ok else "NOT met"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
