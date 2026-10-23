"""Bounded continuation: complete task traces of unchanged, producer-bound B1/B2/B4 controls.

No candidate speedup or hardware traffic is inferred. Bodies include tile-internal waits.
Run by ds_multibatch_trace.py against a sealed baseline record; never writes that record.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
import time

from etx.session.batch import etx_result
from etx.session.record import sha256
from etx.tools.timeline import build_trace, latency_report, load_plan, read_raw
from . import paths
from .batch_session import BatchSession
from etx.models.dsv2lite.session import write_json, PINNED_INPUTS
from .reference_prepare import verify

ARMS = {1: "alld4r0l1kvsmw", 2: "x2", 4: "x4"}
STOP = 1792084930


def validate_rows(plan, rows):
    """Reject corruption before timeline's display-oriented timestamp clamping."""
    if len(rows) != len(plan["descs"]):
        raise ValueError("trace coverage differs from plan")
    if "task_in" not in plan or "ev_inst" not in plan:
        raise ValueError("missing dependency data")
    for i, (d, row) in enumerate(zip(plan["descs"], rows)):
        coord = (list(d["coord"]) + [0] * 4)[:4]
        if len(row) != 9 or row[0] != d["type"] or list(row[1:5]) != coord:
            raise ValueError(f"task {i}: type/coordinate mismatch")
        if not 0 < row[5] <= row[6] <= row[7]:
            raise ValueError(f"task {i}: invalid timestamp order")
        if d.get("mode", "static") != "static" or row[8] != d["worker"]:
            raise ValueError(f"task {i}: unexpected worker/mode")


def family(phase):
    return re.sub(r"^r\d+_", "", phase)


def aggregate(rows):
    result = {}
    for r in rows:
        name = family(r["phase"])
        a = result.setdefault(name, dict(family=name, tasks=0, body_work_us=0., wait_work_us=0.,
                                         cp_body_us=0., cp_dep_gap_us=0., cp_queue_gap_us=0.))
        a["tasks"] += r["tasks"]
        a["body_work_us"] += r["tasks"] * r["body_us"]
        a["wait_work_us"] += r["tasks"] * r["wait_us"]
        for key in ("cp_body_us", "cp_dep_gap_us", "cp_queue_gap_us"):
            a[key] += r[key]
    return sorted(result.values(), key=lambda r: -r["cp_body_us"])


def analyze(plan, raw):
    rows = read_raw(raw)
    validate_rows(plan, rows)
    tr = build_trace(plan, rows)
    report = latency_report(tr, out=io.StringIO())
    report["families"] = aggregate(report["latency"])
    report["tasks"] = len(rows)
    report["raw_sha256"] = sha256(raw)
    return report


def identity(base):
    result = {}
    for b, arm in ARMS.items():
        gate = json.loads((base / f"bgate_{arm}.json").read_text())
        if gate.get("passed") is not True:
            raise ValueError(f"baseline gate not passed: {arm}")
        expected = gate["identity"]["files"]
        actual = {name: sha256(base / f"b_{arm}" / name) for name in expected}
        if actual != expected:
            raise ValueError(f"baseline identity differs: {arm}")
        result[b] = gate["identity"]
    return result


def run(base):
    s = BatchSession()
    if time.time() >= STOP - 600:
        raise ValueError("insufficient execution/export reserve")
    before = identity(base)
    ready = json.loads((base / "references_ready.json").read_text())
    if ready.get("status") != "READY" or s.data != (base / "references").resolve():
        raise ValueError("incorrect reference root")
    for req in ready["requests"][:4]:
        r = req["request"]
        d = s.data if r == 0 else s.data / f"r{r}"
        if verify(d, req["receipt"]["recipe"]) != req["receipt"]:
            raise ValueError(f"reference receipt differs r{r}")
    for name, digest in PINNED_INPUTS.items():
        if sha256(s.data / name) != digest:
            raise ValueError(f"pinned input differs: {name}")
    write_json(s.logs / "bindings.json", dict(baseline=str(base), executable_identity=before,
               source=(s.root / "SOURCE_IDENTITY").read_text(), references=ready,
               driver_sha256=sha256(Path(__file__))))
    results = []
    for k, order in enumerate(((1, 2, 4), (4, 2, 1), (1, 2, 4)), 1):
        for b in order:
            if time.time() >= STOP - 420:
                raise ValueError("absolute stop/export reserve reached")
            arm = ARMS[b]
            name = f"trace_r{k}_b{b}"
            raw = s.logs / f"{name}.bin"
            text = s.run(name, [base / f"b_{arm}" / "run", *s.etx_args(b)],
                         env={"ETX_TRACE": "1", "ETX_TRACE_RAW": str(raw)}, timeout=90)
            gate = etx_result(text, b)
            report = analyze(load_plan(base / f"b_{arm}" / "plan.json"), raw)
            result = dict(round=k, batch=b, arm=arm, gate=gate, trace=report)
            write_json(s.logs / f"{name}.json", result)
            results.append(result)
            print(f"TRACE COMPLETE round={k} b={b} tasks={report['tasks']} span_us={report['span_us']:.3f}", flush=True)
    after = identity(base)
    if before != after:
        raise ValueError("baseline identity changed")
    write_json(s.logs / "trace_report.json", dict(status="COMPLETE", results=results,
        identity_before=before, identity_after=after, candidate="NOT IMPLEMENTED/NOT MEASURED",
        scope="instrumented final-token task trace; no accepted hardware traffic counters; no performance verdict"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path)
    run(ap.parse_args().baseline.resolve())


if __name__ == "__main__":
    main()
