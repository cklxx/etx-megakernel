"""Derive medians/ranges and scaling from the sealed continuation's nine gated traces.

This is trace attribution, never a candidate speedup or hardware-traffic claim.
Usage: python -m examples.dsv2lite.multibatch_trace_summary trace_report.json batch_report.json out.json
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def summarize(report, baseline):
    results = report["results"]
    if report["status"] != "COMPLETE" or {(r["round"], r["batch"]) for r in results} != {
            (k, b) for k in (1, 2, 3) for b in (1, 2, 4)} or len(results) != 9:
        raise ValueError("nine complete round/batch tuples required")
    out = dict(scope=report["scope"], candidate=report["candidate"], batches={})
    for b in (1, 2, 4):
        traces = [r["trace"] for r in results if r["batch"] == b]
        if len({t["tasks"] for t in traces}) != 1:
            raise ValueError("task count changes between rounds")
        families = sorted({r["family"] for t in traces for r in t["families"]})
        family_rows = {}
        keys = ("body_work_us", "wait_work_us", "cp_body_us", "cp_dep_gap_us", "cp_queue_gap_us")
        for name in families:
            rows = [next(r for r in t["families"] if r["family"] == name) for t in traces]
            family_rows[name] = dict(tasks=rows[0]["tasks"], **{k: stats([r[k] for r in rows]) for k in keys})
        totals = {k: stats([sum(r[k] for r in t["families"]) for t in traces]) for k in keys}
        totals["all_cp_gaps_us"] = stats([sum(r["cp_dep_gap_us"] + r["cp_queue_gap_us"] for r in t["families"]) for t in traces])
        totals["span_us"] = stats([t["span_us"] for t in traces])
        base = baseline["batches"][str(b)]
        if base["status"] != "COMPLETE":
            raise ValueError("incomplete retained baseline")
        out["batches"][str(b)] = dict(tasks=traces[0]["tasks"], totals=totals, families=family_rows,
            reused_baseline=dict(etx_step_ms=statistics.median(map(float, base["etx_step_ms"])),
                vllm_step_ms=statistics.median(map(float, base["vllm_step_ms"])),
                etx_request_interval_ms=statistics.median(map(float, base["etx_latency_ms"])),
                etx_tokens_per_s=base["etx_tokens_per_s"], vllm_tokens_per_s=base["vllm_tokens_per_s"],
                throughput=base["throughput"], latency=base["latency"], config=base["config"]))
    b2, b4 = (out["batches"][str(b)] for b in (2, 4))
    out["b4_over_b2"] = dict(tasks=b4["tasks"] / b2["tasks"],
        **{k: b4["totals"][k]["median"] / b2["totals"][k]["median"] for k in
           ("span_us", "body_work_us", "wait_work_us", "cp_body_us", "all_cp_gaps_us")})
    target = baseline["batches"]["4"]["next"]["bound_ms"]
    step = b4["reused_baseline"]["etx_step_ms"]
    out["registered_target"] = dict(etx_step_ms=step, target_ms=target,
        required_reduction_ms=step - target, required_reduction_pct=100 * (step - target) / step,
        note="Host target versus instrumented device trace is a scale comparison, not a performance prediction.")
    return out


def main() -> None:
    report, baseline, dest = map(Path, sys.argv[1:])
    dest.write_text(json.dumps(summarize(json.loads(report.read_text()), json.loads(baseline.read_text())),
                               indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
