"""Opt-in, read-only worker-assignment audit for the ETX GQA graph.

    ETX_DSL_WORKER_AUDIT=1 python -m etx.ops.worker_audit PLAN_JSON TRACE_RAW

The legal decision is P5's static round-robin worker slot, *within* the domain
selected by model.py's DSL_ATTNHEAD prep/attn/reduce branch. That branch has no
worker_map, and slot assignment does not change its bodies, counts or edges.
This module does not reassign anything: first establish critical queue blocking
on ETX's own trace. In particular, done -> next-taken includes an unstamped
arrive/fetch; it is not proof that another worker could remove that interval.

The output is historical trace attribution, not a performance prediction or a
certificate that a trace was produced by the current compiled tile bodies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from etx.runtime import trace as timeline


FLAG = "ETX_DSL_WORKER_AUDIT"
GQA = re.compile(r"(?P<prefix>r\d+_)?(?P<role>prep|attn|reduce)_(?P<layer>\d+)$")


def _validate(plan: dict, rows: list[tuple[int, ...]]) -> dict[int, int]:
    """Reject incomplete or mismatched evidence instead of truncating/clamping it."""
    if plan["devices"] != 1 or plan["arch"] != "gfx942":
        raise ValueError("worker audit requires one gfx942 device")
    descs = plan["descs"]
    if len(rows) != len(descs) or len(plan.get("task_in", [])) != len(descs):
        raise ValueError("complete trace and task dependency lists are required")
    if not plan.get("ev_inst"):
        raise ValueError("event producer lists are required")
    nworker = plan["domains"] * plan["workers_per_domain"]
    names = {v["id"]: n for n, v in plan["types"].items()}
    for i, (d, r) in enumerate(zip(descs, rows)):
        coord = (d["coord"] + [0] * 4)[:4]
        if (d["id"] != i or d["type"] not in names or d["mode"] != "static"
                or d["device"] != 0 or d["worker"] is None
                or not 0 <= d["worker"] < nworker
                or d["domain"] != d["worker"] // plan["workers_per_domain"]
                or r[0] != d["type"] or list(r[1:5]) != coord
                or r[8] & 0xFFFFFFFF != d["worker"]):
            raise ValueError(f"task {i}: trace/descriptor identity or static placement mismatch")
        if not 0 < r[5] <= r[6] <= r[7]:
            raise ValueError(f"task {i}: incomplete or nonmonotonic timestamps")
    previous: dict[int, int] = {}
    seen: set[int] = set()
    for key, queue in plan["static_queues"].items():
        device, worker = map(int, key.split(":"))
        if device != 0 or not 0 <= worker < nworker:
            raise ValueError(f"invalid queue {key}")
        for pos, tid in enumerate(queue):
            if tid in seen or not 0 <= tid < len(descs) or descs[tid]["worker"] != worker:
                raise ValueError(f"queue {key}: duplicate or misplaced task {tid}")
            seen.add(tid)
            if pos:
                pred = queue[pos - 1]
                if pred >= tid or rows[pred][7] > rows[tid][5]:
                    raise ValueError(f"queue {key}: program or trace order mismatch at task {tid}")
                previous[tid] = pred
    if len(seen) != len(descs):
        raise ValueError("static queues do not cover every task exactly once")
    deps = timeline._deps_from_plan(plan)
    for tid, producers in enumerate(deps):
        for pred in producers:
            if not 0 <= pred < tid or rows[pred][7] > rows[tid][6]:
                raise ValueError(f"task {tid}: invalid or incomplete dependency {pred}")
    return previous


def _gqa_tasks(plan: dict, deps: list[list[int]]) -> dict[int, str]:
    """Accept the source-confirmed head-local branch, not an arbitrary name match."""
    names = {v["id"]: n for n, v in plan["types"].items()}
    descs = plan["descs"]
    selected = {}
    for d in descs:
        name = names[d["type"]]
        match = GQA.fullmatch(name)
        if not match:
            continue
        role = match["role"]
        prefix, layer = match["prefix"] or "", match["layer"]
        stage_names = {r: f"{prefix}{r}_{layer}" for r in ("prep", "attn", "reduce")}
        if any(n not in plan["types"] for n in stage_names.values()):
            raise ValueError(f"{name}: incomplete GQA stages")
        shapes = {r: plan["types"][n]["grid"] for r, n in stage_names.items()}
        if any(len(s) != 1 for s in shapes.values()):
            raise ValueError(f"{name}: only the one-dimensional head-local GQA branch is audited")
        heads, attns = shapes["prep"][0], shapes["attn"][0]
        if not heads or attns % heads or shapes["reduce"][0] != heads:
            raise ValueError(f"{name}: inconsistent GQA geometry")
        parts = attns // heads
        head = d["coord"][0] // parts if role == "attn" else d["coord"][0]
        if head != d["domain"]:
            raise ValueError(f"{name}: cross-domain attention placement is outside this audit")
        if role != "prep":
            expected = "prep" if role == "attn" else "attn"
            producers = [descs[p] for p in deps[d["id"]]]
            count = 1 if role == "attn" else parts
            if len(producers) != count or any(
                    names[p["type"]] != stage_names[expected] or p["domain"] != head for p in producers):
                raise ValueError(f"{name}: dependencies do not match the head-local GQA graph")
        selected[d["id"]] = role
    return selected


def audit(plan: dict, rows: list[tuple[int, ...]]) -> dict:
    """Rebuild ETX's critical path and screen the legal same-domain slot decision."""
    previous = _validate(plan, rows)
    trace = timeline.build_trace(plan, rows)
    selected = _gqa_tasks(plan, trace.deps)
    steps = timeline.critical_path(trace)
    total = sum(s["body"] + s["gap"] for s in steps)
    if abs(total - trace.span) > 1e-6:
        raise ValueError("critical path does not telescope to the trace span")
    roles = {r: dict(tasks=0, critical_tasks=0, body_us=0., gap_us=0.,
                    queue_predecessors=0, preceding_body_delay_us=0., late_taken_us=0.)
             for r in ("prep", "attn", "reduce")}
    for role in selected.values():
        roles[role]["tasks"] += 1
    blocked = []
    for step in steps:
        task = step["task"]
        if task.id not in selected:
            continue
        result = roles[selected[task.id]]
        result["critical_tasks"] += 1
        result["body_us"] += step["body"]
        result["gap_us"] += step["gap"]
        result["queue_predecessors"] += step["kind"] == "queue"
        dep_done = max((trace.by_id[p].done for p in trace.deps[task.id]), default=trace.t0)
        pred = trace.by_id.get(previous.get(task.id))
        queue_done = pred.done if pred else trace.t0
        delay = max(0., queue_done - dep_done)
        result["preceding_body_delay_us"] += delay
        result["late_taken_us"] += max(0., task.taken - dep_done)
        if delay:
            blocked.append(dict(task=task.id, grid=task.name, worker=task.worker,
                                predecessor=pred.id, preceding_body_delay_us=delay))
    status = ("NO_ELIGIBLE_GQA_DECISION" if not selected else
              "REVIEW_CRITICAL_QUEUE_BLOCKING" if blocked else "STOP_NO_CRITICAL_QUEUE_BLOCKING")
    for values in roles.values():
        for key, value in values.items():
            if isinstance(value, float):
                values[key] = round(value, 6)
    return dict(enabled=True, status=status, tasks=len(rows), identity_mismatches=0,
                workers=plan["domains"] * plan["workers_per_domain"],
                critical_tasks=len(steps), span_us=round(trace.span, 6),
                critical_queue_predecessors=sum(s["kind"] == "queue" for s in steps),
                critical_body_us=round(sum(s["body"] for s in steps), 6),
                critical_gap_us=round(sum(s["gap"] for s in steps), 6),
                gqa=roles, candidates=blocked,
                qualified_saving_us=0.,
                notes=["Read-only audit: no worker, count, domain, edge or body changed.",
                       "done is before arrive; late_taken includes unstamped arrive/fetch, not certified savings.",
                       "A blocked predecessor is only a candidate; another worker and the full counterfactual queue require validation.",
                       "Identity checks cover trace/plan descriptors, not compiled body or current-baseline identity."])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("raw", type=Path)
    args = parser.parse_args(argv)
    enabled = os.environ.get(FLAG, "0")
    if enabled == "0":
        print(json.dumps(dict(enabled=False, status="OFF", flag=FLAG)))
        return 0
    if enabled != "1":
        parser.error(f"{FLAG} must be 0 or 1")
    try:
        result = audit(timeline.load_plan(args.plan), timeline.read_raw(args.raw))
    except (ValueError, KeyError, IndexError) as error:
        parser.error(str(error))
    result["inputs"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.plan, args.raw)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
