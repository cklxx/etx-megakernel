#!/usr/bin/env python3
"""Per-batch critical-path attribution from a multibatch trace report.

Usage: python tools/ds_batch_attribution.py <trace_report.json>

Prints, per batch, the step time, tokens/s and the critical-path split into
body work, dependency gaps and queue gaps, then the same split for each
family and the B2 -> B4 deltas.
"""
from __future__ import annotations

import json
import sys


def totals(row):
    fam = row["trace"]["families"]
    return {
        "body": sum(f["cp_body_us"] for f in fam),
        "dep": sum(f["cp_dep_gap_us"] for f in fam),
        "queue": sum(f["cp_queue_gap_us"] for f in fam),
        "tasks": sum(f["tasks"] for f in fam),
    }


def main(path: str) -> int:
    data = json.load(open(path))
    first = {}
    for row in data["results"]:
        first.setdefault(row["batch"], row)
    print(f"{'batch':>5} {'step_us':>8} {'tok/s':>7} {'body':>8} {'dep':>7} {'queue':>7} {'tasks':>7}")
    prev = None
    for batch in sorted(first):
        row = first[batch]
        step = float(row["gate"]["step_ms"]) * 1000.0
        t = totals(row)
        print(f"{batch:>5} {step:>8.0f} {batch * 1e6 / step:>7.0f} "
              f"{t['body']:>8.0f} {t['dep']:>7.1f} {t['queue']:>7.1f} {t['tasks']:>7}")
        if prev is not None:
            print(f"      delta step {step - prev[0]:+.0f} us: "
                  f"body {t['body'] - prev[1]['body']:+.0f}, "
                  f"dep {t['dep'] - prev[1]['dep']:+.1f}, "
                  f"queue {t['queue'] - prev[1]['queue']:+.1f}")
            for f in row["trace"]["families"]:
                old = next((g for g in prev[2] if g["family"] == f["family"]), None)
                if old is None:
                    continue
                dq = f["cp_queue_gap_us"] - old["cp_queue_gap_us"]
                dd = f["cp_dep_gap_us"] - old["cp_dep_gap_us"]
                db = f["cp_body_us"] - old["cp_body_us"]
                if abs(dq) > 20 or abs(dd) > 20 or abs(db) > 100:
                    print(f"        {f['family']:<10} body {db:+.0f} dep {dd:+.1f} queue {dq:+.1f}")
        prev = (step, t, row["trace"]["families"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "examples/gpu-live/results/2026-10-16-multibatch/trace_report.json"))
