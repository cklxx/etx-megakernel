#!/usr/bin/env python3
"""Per-batch critical-path attribution from a multibatch trace report.

Usage: python3 tools/ds_batch_attribution.py <trace_report.json>

Prints, per batch, the step time, tokens/s and the critical-path split into
body work, dependency gaps and queue gaps, then the same split for each
family and the B2 -> B4 deltas.

Only the first round of each batch is used.  The three critical-path columns
sum to the traced span, which is shorter than the host step: the step also
contains launch and unwinding time outside the instrumented trace.
"""
from __future__ import annotations

import json
import sys


def load_report(path):
    with open(path) as handle:
        return json.load(handle)


def first_rounds(data):
    """One row per batch: the lowest round number present."""
    chosen = {}
    for row in data["results"]:
        batch = row["batch"]
        if batch not in chosen or row["round"] < chosen[batch]["round"]:
            chosen[batch] = row
    return [chosen[batch] for batch in sorted(chosen)]


def step_us(row):
    return float(row["gate"]["step_ms"]) * 1000.0


def tokens_per_s(batch, step):
    return batch * 1e6 / step


def totals(row):
    fam = row["trace"]["families"]
    return {
        "body": sum(f["cp_body_us"] for f in fam),
        "dep": sum(f["cp_dep_gap_us"] for f in fam),
        "queue": sum(f["cp_queue_gap_us"] for f in fam),
        "tasks": sum(f["tasks"] for f in fam),
    }


def family_deltas(prev_families, families):
    """Families whose body/dependency/queue contribution moved materially."""
    out = []
    for f in families:
        old = next((g for g in prev_families if g["family"] == f["family"]), None)
        if old is None:
            continue
        db = f["cp_body_us"] - old["cp_body_us"]
        dd = f["cp_dep_gap_us"] - old["cp_dep_gap_us"]
        dq = f["cp_queue_gap_us"] - old["cp_queue_gap_us"]
        if abs(dq) > 20 or abs(dd) > 20 or abs(db) > 100:
            out.append({"family": f["family"], "body": db, "dep": dd, "queue": dq})
    return out


def summarize(data):
    """Per-batch numbers plus the deltas from the previous batch."""
    rows = []
    prev = None
    for row in first_rounds(data):
        batch = row["batch"]
        step = step_us(row)
        t = totals(row)
        item = {
            "batch": batch,
            "step_us": step,
            "tokens_per_s": tokens_per_s(batch, step),
            "body": t["body"],
            "dep": t["dep"],
            "queue": t["queue"],
            "tasks": t["tasks"],
            "delta": None,
            "families": [],
        }
        if prev is not None:
            item["delta"] = {
                "step_us": step - prev["step_us"],
                "body": t["body"] - prev["body"],
                "dep": t["dep"] - prev["dep"],
                "queue": t["queue"] - prev["queue"],
            }
            item["families"] = family_deltas(prev["families"], row["trace"]["families"])
        prev = {
            "step_us": step,
            "body": t["body"],
            "dep": t["dep"],
            "queue": t["queue"],
            "families": row["trace"]["families"],
        }
        rows.append(item)
    return rows


def main(path: str) -> int:
    data = load_report(path)
    print(f"{'batch':>5} {'step_us':>8} {'tok/s':>7} {'body':>8} {'dep':>7} {'queue':>7} {'tasks':>7}")
    for item in summarize(data):
        print(f"{item['batch']:>5} {item['step_us']:>8.0f} {item['tokens_per_s']:>7.0f} "
              f"{item['body']:>8.0f} {item['dep']:>7.1f} {item['queue']:>7.1f} {item['tasks']:>7}")
        delta = item["delta"]
        if delta is not None:
            print(f"      delta step {delta['step_us']:+.0f} us: "
                  f"body {delta['body']:+.0f}, "
                  f"dep {delta['dep']:+.1f}, "
                  f"queue {delta['queue']:+.1f}")
            for f in item["families"]:
                print(f"        {f['family']:<10} body {f['body']:+.0f} "
                      f"dep {f['dep']:+.1f} queue {f['queue']:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "examples/gpu-live/results/2026-10-16-multibatch/trace_report.json"))
