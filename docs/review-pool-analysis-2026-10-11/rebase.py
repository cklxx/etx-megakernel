#!/usr/bin/env python3
"""Rebase the plan's body allowances and the XCD idle share on three traces (CPU only, no GPU).

  .venv/bin/python docs/review-pool-analysis-2026-10-11/rebase.py > docs/review-pool-analysis-2026-10-11/rebase.txt

Allowance B = 48 x (critical body - round11 serial floor at 0.50 us) / 1000 ms, the lever-rule method of
docs/PLAN-generic-decoder-2026-10-08.md:154 with etx/tools/floors.py round11_from_trace (layers 1-45) and the
floors of docs/FLOORS-2026-10-08.md:239-247. On tr_all it must reproduce docs/PLAN-generic-decoder-2026-10-08.md:224-231.
Idle = 100% minus the mean of the eight per-XCD 'any body' shares of an xcd_account account.txt, times the span.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from etx.tools.floors import round11_from_trace  # noqa: E402
from etx.tools import timeline as tl  # noqa: E402

ARCH = Path(os.path.expanduser("~/code/etx"))
PLAN = ARCH / "results_vlbest2_2026-10-10/b_all/plan.json"
TRACES = {
    "tr_all": ARCH / "results_tiles/tr_all.bin",
    "tr_off": ARCH / "results_qwen300_2026-10-11/tr_off.bin",
    "tr_on": ARCH / "results_qwen300_2026-10-11/tr_on.bin",
}
ACCOUNTS = {
    "tr_all": ROOT / "docs/xcd-timeline-2026-10-11/account.txt",
    "tr_off": ROOT / "docs/dep-gap-ledger-2026-10-11/account.txt",
    "tr_on": ROOT / "docs/review-pool-analysis-2026-10-11/account_on.txt",
}
FLOOR = {"qkv": 7.55, "prep": 1.52, "attn": 2.92, "reduce": 2.81, "o_proj": 5.30, "router": 1.52,
         "topk": 1.21, "moe_up": 14.67, "moe_down": 9.48}


def row_mismatches(plan: dict, raw: Path) -> int:
    bad = 0
    for i, r in enumerate(tl.read_raw(raw)):
        d = plan["descs"][i]
        n = len(d["coord"])
        if r[0] != d["type"] or (r[8] & 0xFFFFFFFF) != d["worker"] or list(r[1:1 + n]) != d["coord"] or any(r[1 + n:5]):
            bad += 1
    return bad


def idle(account: Path) -> tuple[float, float, float, float]:
    lines = account.read_text().splitlines()
    span = float(re.search(r"span ([\d.]+)", lines[0]).group(1))
    rows = [l.split() for l in lines if re.match(r"^\s+[0-7]\s+\d", l)][:8]
    busy = [float(r[8].rstrip("%")) for r in rows]
    mean = sum(busy) / 8
    return span, sum(busy[:4]) / 4, sum(busy[4:]) / 4, (100 - mean) / 100 * span


def main() -> int:
    plan = json.loads(PLAN.read_text())
    crit = {}
    for name, raw in TRACES.items():
        if name != "tr_all":
            print(f"{name}: {row_mismatches(plan, raw)} plan-row mismatches of {len(tl.read_raw(raw))}")
        crit[name] = round11_from_trace(PLAN, raw)
    print("\nphase     floor  | critical body us/layer: all   off    on | B ms/step: all    off     on")
    tot = dict.fromkeys(TRACES, 0.0)
    for p, f in FLOOR.items():
        c = [crit[n][p]["crit_body"] for n in TRACES]
        b = [48 * (x - f) / 1000 for x in c]
        for n, x in zip(TRACES, b):
            tot[n] += x
        print(f"{p:9s} {f:5.2f}  | {c[0]:21.3f} {c[1]:6.3f} {c[2]:6.3f} | {b[0]:15.3f} {b[1]:7.3f} {b[2]:6.3f}")
    print("sum                                                      | "
          + " ".join(f"{tot[n]:7.3f}" for n in TRACES))
    print("\ntrace   span us  XCD0-3 busy%  XCD4-7 busy%  idle 8-XCD-equivalent us")
    for n, a in ACCOUNTS.items():
        s, lo, hi, i = idle(a)
        print(f"{n:7s} {s:7.1f}  {lo:12.3f}  {hi:12.3f}  {i:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
