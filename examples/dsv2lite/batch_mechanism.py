"""Why DeepSeek B4 (and B8) do not scale past B2: the mechanism per step, not "more work".

Offline (no GPU): `offline` decomposes the retained 2026-10-15 task traces (examples/gpu-live/results/
2026-10-16-multibatch/trace_report.json) and the statically lowered plans into the quantities that set the traced span:

  span = W / (N * u)       W  summed task-body time (us), N  resident worker slots, u  slot busy fraction

This holds by definition of u, so the ratio of two batches' spans is exactly the product of the ratios of tasks,
per-task body, N and 1/u; `steps` turns that product into milliseconds in a fixed order (tasks, body, slots, busy).
Bytes are the shape model (bytes.py / batch_floor.py), labelled estimates; no hardware counter was accepted.

On the box (examples/sessions/ds_batch_mechanism.py), one fresh gated invocation per operation:
  gate <arm>        BatchSession's per-request gates (32/32 tokens, 27/27 layers, 32/32 warm-up)
  trace <arm>       ETX_TRACE on the gated binary, three fresh invocations; per worker slot: body, dependency wait,
                    queue blocking behind ANOTHER request's task on the same slot (the serialization counter)
  iso <arm>         the in-situ phase microbenchmark (graph._isolate) on the x4 recipe: a phase alone, beside the
                    other slot of its CU (r0 with r1), and queued behind the same slot (r0 with r2)
  counters <arm>    rocprofv3 --pmc, one counter group per fresh invocation, 64 megakernel dispatches each
  report            pre-registered predictions (docs/deepseek/DEEPSEEK-BATCH-MECHANISM-2026-10-18.md) against the measurements
Instrumented spans and counter runs never supply a performance verdict.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "examples/gpu-live/results/2026-10-16-multibatch/trace_report.json"
ARMS_BY_B = {1: "alld4r0l1kvsmw", 2: "x2", 4: "x4", 8: "x8"}
RATE_TBS = 4.3                     # CLAUDE.md: achievable MI300X stream rate, for floors only
# hw_bench stream, rows layout, 296 WGs (examples/dsv2lite/results/2026-09-27d/hw_stream.log:18): 1 MB per WG
STREAM_TBS_MEASURED = 3.51
# Counter groups, one rocprofv3 pass each; all listed by rocprofv3 1.3.2 / ROCm 7.14.0 on the 2026-10-15 box
# (examples/gpu-live/results/2026-10-15/b4/rocprof_metrics.raw.log).
COUNTER_GROUPS = {"fetch": ("FETCH_SIZE",), "write": ("WRITE_SIZE",), "l2": ("TCC_HIT_sum", "TCC_MISS_sum"),
                  "dram": ("TCC_EA0_RDREQ_DRAM_sum",), "issue": ("SQ_WAIT_INST_ANY", "SQ_BUSY_CYCLES", "SQ_WAVES"),
                  "active": ("GRBM_GUI_ACTIVE",)}
DISPATCHES = 64                    # 32 tokens x 2 repetitions, one megakernel dispatch per step
# The isolation spec of arm x4iso (arms.py): per phase, A alone, B beside the CU's other slot, C queued on the same slot
ISO_PHASES = ("gate_up_5", "qkv_5", "attn_5")
ISO_SPEC = ";".join(f"0:{p};0:{p},1:{p};0:{p},2:{p}" for p in ISO_PHASES)
ISO_REPS = 8


# ---------------------------------------------------------------- offline: retained traces + static plans
def plan_stats(arm: str) -> dict:
    """Lower an arm's graph exactly as its build does (arms.py env) and count what the plan fixes: grids, tasks,
    resident worker slots, tasks per slot, requests per slot. No GPU; ~5-65 s per arm."""
    import collections
    import os
    from etx.passes import PassOptions, compile_graph
    from examples.dsv2lite.arms import ARMS
    saved = dict(os.environ)
    try:
        os.environ.update(ARMS[arm].env)
        from examples.dsv2lite import graph
        g = graph.build(routing_cached=True)
        plan = compile_graph(g, "gfx942", {}, {}, PassOptions(relay="on"))
    finally:
        os.environ.clear()
        os.environ.update(saved)
    per = collections.Counter(t.worker for t in plan.tasks)
    reqs = collections.defaultdict(set)
    for t in plan.tasks:
        reqs[t.worker].add(_request(t.grid))
    return dict(arm=arm, grids=len(g.grids), tasks=len(plan.tasks), wg_per_cu=plan.wg_per_cu,
                slots=len(per), tasks_per_slot_mean=len(plan.tasks) / len(per), tasks_per_slot_max=max(per.values()),
                requests_per_slot=max(len(v) for v in reqs.values()), dispatches_per_step=1)


def _median_run(results: list[dict], b: int) -> dict:
    runs = sorted((r for r in results if r["batch"] == b), key=lambda r: r["trace"]["span_us"])
    if not runs:
        raise ValueError(f"no trace for B{b}")
    return runs[len(runs) // 2]


def batch_terms(run: dict, slots: int, step_bytes: float) -> dict:
    """One traced invocation -> W, N, u, critical-path split, host excess and the estimated delivered byte rate."""
    t = run["trace"]
    fam = {f["family"]: f for f in t["families"]}
    W = sum(f["body_work_us"] for f in fam.values())
    span = t["span_us"]
    cp = {k: sum(f[k] for f in fam.values()) for k in ("cp_body_us", "cp_dep_gap_us", "cp_queue_gap_us")}
    return dict(batch=run["batch"], arm=run["arm"], round=run["round"], tasks=t["tasks"], slots=slots,
                span_us=span, body_work_us=W, wait_work_us=sum(f["wait_work_us"] for f in fam.values()),
                body_per_task_us=W / t["tasks"], busy=W / (slots * span), **cp,
                host_step_ms=float(run["gate"]["step_ms"]), host_minus_span_us=1000 * float(run["gate"]["step_ms"]) - span,
                bytes_est=step_bytes, rate_est_tbs=step_bytes / (span * 1e6),
                families={k: dict(tasks=f["tasks"], body_per_task_us=f["body_work_us"] / f["tasks"],
                                  cp_body_us=f["cp_body_us"]) for k, f in fam.items()})


def steps(a: dict, b: dict) -> dict:
    """span_b - span_a in us, attributed in the fixed order tasks -> per-task body -> slots -> busy fraction.
    Each factor's term is the span change when that factor alone moves from a to b, the earlier ones already moved;
    the four terms add up to the measured change exactly (the last one absorbs nothing: the identity is exact)."""
    f = [("tasks", b["tasks"] / a["tasks"]), ("body_per_task", b["body_per_task_us"] / a["body_per_task_us"]),
         ("slots", a["slots"] / b["slots"]), ("busy", a["busy"] / b["busy"])]
    span, out = a["span_us"], {}
    for name, ratio in f:
        out[name] = dict(ratio=ratio, us=span * ratio - span)
        span *= ratio
    out["check_us"] = span - b["span_us"]          # rounding only
    out["total_us"] = b["span_us"] - a["span_us"]
    return out


def offline(traces: Path = TRACES, stats: dict | None = None) -> dict:
    from examples.dsv2lite.batch_floor import step_bytes
    report = json.loads(traces.read_text())
    if report.get("status") != "COMPLETE":
        raise ValueError("trace report is not COMPLETE")
    stats = stats or {b: plan_stats(ARMS_BY_B[b]) for b in (1, 2, 4, 8)}
    out = dict(source=str(traces.relative_to(ROOT)) if traces.is_relative_to(ROOT) else str(traces),
               scope="instrumented final-token traces (ETX_TRACE), median-span run per B; bytes are shape estimates",
               plans={str(b): s for b, s in stats.items()}, batches={})
    for b in (1, 2, 4):
        run = _median_run(report["results"], b)
        if run["trace"]["tasks"] != stats[b]["tasks"]:
            raise ValueError(f"B{b}: traced {run['trace']['tasks']} tasks, plan lowers {stats[b]['tasks']}")
        out["batches"][str(b)] = batch_terms(run, stats[b]["slots"], step_bytes(b)["replicated"])
    x = out["batches"]
    out["steps"] = {"1->2": steps(x["1"], x["2"]), "2->4": steps(x["2"], x["4"])}
    # B8 is not traced (its gate failed: docs/deepseek/BATCH-2026-10-14.md section 13a). Projection from the plan only [E]:
    # same slots, per-task body and busy fraction as B4, tasks from the lowered plan.
    b4, p8 = x["4"], stats[8]
    span8 = p8["tasks"] * b4["body_per_task_us"] / (p8["slots"] * b4["busy"])
    out["projection_b8"] = dict(kind="ESTIMATE: B4 body/task and busy fraction, B8 plan task count",
                                span_us=span8, per_token_us=span8 / 8, request_interval_us=span8,
                                note="excludes x8's 144 B/lane scratch (x4: 64) and the unresolved B8 hidden-state failure")
    return out


def render(o: dict) -> str:
    """The tables of docs/deepseek/DEEPSEEK-BATCH-MECHANISM-2026-10-18.md section 2, from offline()."""
    x, p = o["batches"], o["plans"]
    lines = ["| quantity | B1 | B2 | B4 | B8 (plan) |", "|---|---:|---:|---:|---:|"]
    def row(name, f, b8="—"):
        lines.append(f"| {name} | " + " | ".join(f(x[b]) for b in ("1", "2", "4")) + f" | {b8} |")
    row("megakernel dispatches per step", lambda v: "1", "1")
    lines.append("| grids per step | " + " | ".join(str(p[b]["grids"]) for b in ("1", "2", "4", "8")) + " |")
    row("tasks per step", lambda v: f"{v['tasks']}", str(p["8"]["tasks"]))
    row("resident worker slots N", lambda v: f"{v['slots']}", str(p["8"]["slots"]))
    lines.append("| requests per slot | " + " | ".join(str(p[b]["requests_per_slot"]) for b in ("1", "2", "4", "8")) + " |")
    lines.append("| tasks per slot (mean / max) | " + " | ".join(
        f"{p[b]['tasks_per_slot_mean']:.1f} / {p[b]['tasks_per_slot_max']}" for b in ("1", "2", "4", "8")) + " |")
    row("summed body W (ms)", lambda v: f"{v['body_work_us'] / 1000:.1f}")
    row("body per task (us)", lambda v: f"{v['body_per_task_us']:.2f}")
    row("slot busy fraction u = W/(N span)", lambda v: f"{v['busy']:.4f}")
    row("traced span (ms)", lambda v: f"{v['span_us'] / 1000:.4f}", f"{o['projection_b8']['span_us'] / 1000:.2f} [E]")
    row("span per generated token (ms)", lambda v: f"{v['span_us'] / 1000 / v['batch']:.4f}",
        f"{o['projection_b8']['per_token_us'] / 1000:.2f} [E]")
    row("critical path: bodies (ms)", lambda v: f"{v['cp_body_us'] / 1000:.4f}")
    row("critical path: dependency gaps (ms)", lambda v: f"{v['cp_dep_gap_us'] / 1000:.4f}")
    row("critical path: queue gaps (ms)", lambda v: f"{v['cp_queue_gap_us'] / 1000:.4f}")
    row("host step of the traced run minus span (ms)", lambda v: f"{v['host_minus_span_us'] / 1000:.4f}")
    row("replicated bytes per step [E] (GB)", lambda v: f"{v['bytes_est'] / 1e9:.3f}")
    row("bytes / span [E] (TB/s)", lambda v: f"{v['rate_est_tbs']:.3f}")
    lines += ["", "| step | total (ms) | tasks | body per task | slots | busy fraction |", "|---|---:|---:|---:|---:|---:|"]
    for k, s in o["steps"].items():
        lines.append(f"| B{k.replace('->', ' -> B')} | {s['total_us'] / 1000:+.4f} | " + " | ".join(
            f"x{s[n]['ratio']:.4f} = {s[n]['us'] / 1000:+.4f}" for n in ("tasks", "body_per_task", "slots", "busy")) + " |")
    fams = ("gate_up", "down", "qkv", "attn", "router", "oproj", "merge", "qabs", "lm_head")
    lines += ["", "| family | body/task B1 (us) | B2 | B4 | B2/B1 | B4/B2 |", "|---|---:|---:|---:|---:|---:|"]
    for f in fams:
        v = [x[b]["families"].get(f, {}).get("body_per_task_us") for b in ("1", "2", "4")]
        lines.append(f"| {f} | " + " | ".join("—" if a is None else f"{a:.2f}" for a in v) + " | "
                     + ("—" if v[0] is None else f"{v[1] / v[0]:.3f}") + f" | {v[2] / v[1]:.3f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- on the box: raw traces
def _request(name: str) -> int:
    m = re.match(r"(?:iso\d+x\d+_)?r(\d+)_", name)
    return int(m.group(1)) if m else 0


def worker_metrics(tr) -> dict:
    """Per worker slot, over the traced step: body, idle, and why each task started when it did. For a task with
    producers done at D and its slot's previous task done at P, the start S = body start:
      dependency wait   max(0, D - P)       the slot was free and the inputs were not
      queue blocking    max(0, min(S, P) - D)  the inputs were ready and the slot was still busy with its previous task;
                        `cross` when that previous task belongs to another request (the B4 slot sharing), else `same`.
    Sums are over all slots; per-slot medians and the busiest slot are kept."""
    by_worker: dict[int, list] = {}
    for t in tr.tasks:
        by_worker.setdefault(t.worker, []).append(t)
    tot = dict(body_us=0., dep_wait_us=0., queue_cross_us=0., queue_same_us=0.)
    per_slot_body = []
    for ts in by_worker.values():
        ts.sort(key=lambda t: t.taken)
        prev = None
        body = 0.
        for t in ts:
            body += t.done - t.ready
            prods = [tr.by_id[p] for p in (tr.deps[t.id] if tr.deps else []) if p in tr.by_id]
            D = max((p.done for p in prods), default=tr.t0)
            P = prev.done if prev is not None else tr.t0
            tot["dep_wait_us"] += max(0., min(t.ready, D) - P)
            blocked = max(0., min(t.ready, P) - D)
            if prev is not None and _request(prev.name) != _request(t.name):
                tot["queue_cross_us"] += blocked
            else:
                tot["queue_same_us"] += blocked
            prev = t
        per_slot_body.append(body)
        tot["body_us"] += body
    slots = len(by_worker)
    capacity = slots * tr.span
    return dict(slots=slots, span_us=tr.span, **tot,
                **{k.replace("_us", "_share"): tot[k] / capacity for k in tot},
                slot_body_median_us=statistics.median(per_slot_body), slot_body_max_us=max(per_slot_body))


def analyze_trace(plan_path: Path, raw: Path) -> dict:
    from examples.dsv2lite.multibatch_trace import analyze
    from etx.tools.timeline import build_trace, load_plan, read_raw
    plan = load_plan(plan_path)
    report = analyze(plan, raw)                         # validates coverage/types/timestamps before any number
    report["slots"] = worker_metrics(build_trace(plan, read_raw(raw)))
    report.pop("latency", None)
    return report


def iso_spans(tr) -> dict:
    """graph._isolate copies -> {block: {request: median span over repetitions}} (span = first body start -> last done)."""
    inst: dict = {}
    for t in tr.tasks:
        m = re.match(r"iso(\d+)x(\d+)_r(\d+)_([a-z_]+_-?\d+)$", t.name)
        if m:
            inst.setdefault((int(m.group(1)), m.group(4)), {}).setdefault(int(m.group(2)), []).append(t)
    if not inst:
        raise ValueError("no iso* tasks in the trace (build with ETX_DSV2_ISOLATE)")
    out = {}
    for (blk, phase), reps in sorted(inst.items()):
        spans = [max(t.done for t in ts) - min(t.ready for t in ts) for ts in reps.values()]
        reqs = sorted({_request(t.name.split("_", 1)[1]) for ts in reps.values() for t in ts})
        out[str(blk)] = dict(phase=phase, requests=reqs, reps=len(spans), span_us=statistics.median(spans))
    return out


def iso_ratios(blocks: dict) -> dict:
    """Blocks come in threes per phase (ISO_SPEC): A alone, B with the CU's other slot, C queued on the same slot."""
    out = {}
    for i, phase in enumerate(ISO_PHASES):
        a, b, c = (blocks[str(3 * i + k)] for k in range(3))
        if not (a["requests"] == [0] and b["requests"] == [0, 1] and c["requests"] == [0, 2]):
            raise ValueError(f"isolation blocks of {phase} are not A/B/C as registered: {a, b, c}")
        out[phase] = dict(alone_us=a["span_us"], other_slot_us=b["span_us"], same_slot_us=c["span_us"],
                          co_resident=b["span_us"] / a["span_us"], queued=c["span_us"] / a["span_us"])
    return out


# ---------------------------------------------------------------- on the box: counters
def counter_values(path: Path, names: tuple[str, ...]) -> dict:
    """Every named counter, every megakernel dispatch, exactly DISPATCHES each; no zero substitution."""
    from examples.dsv2lite.batch_attribution import counter_rows
    rows = [r for r in counter_rows(path) if "etx_megakernel" in r.get("Kernel_Name", "")]
    out = {}
    for name in names:
        sel = [r for r in rows if r.get("Counter_Name") == name]
        keys = {(r.get("Agent_Id"), r.get("Process_Id"), r["Dispatch_Id"]) for r in sel}
        if len(sel) != DISPATCHES or len(keys) != DISPATCHES:
            raise ValueError(f"{name}: {len(sel)} records / {len(keys)} dispatches, expected {DISPATCHES}")
        vals = [float(r["Counter_Value"]) for r in sel]
        if any(not math.isfinite(v) or v < 0 for v in vals):
            raise ValueError(f"{name}: nonfinite or negative value")
        spans = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000 for r in sel]
        out[name] = dict(median=statistics.median(vals), minimum=min(vals), maximum=max(vals),
                         dispatch_us_median=statistics.median(spans))
    return out


# ---------------------------------------------------------------- the session
def _session_class():
    from examples.dsv2lite.batch_session import BatchSession
    from examples.dsv2lite.inplace_session import write_json
    from examples.dsv2lite.reference_prepare import verify
    from etx.session import batch

    class MechanismSession(BatchSession):
        batches = (2, 4, 8)

        def __init__(self):
            super().__init__()
            ready = json.loads((self.logs / "references_ready.json").read_text())
            if ready.get("status") != "READY" or len(ready["requests"]) != 8:
                raise ValueError("eight producer-bound reference tuples required")
            self.data = self.logs / "references"
            for r in ready["requests"]:
                verify(self.data if r["request"] == 0 else self.data / f"r{r['request']}", r["receipt"]["recipe"])

        def passed(self, arm):
            p = self.logs / f"bgate_{arm}.json"
            return p.is_file() and json.loads(p.read_text()).get("passed") is True

        def trace(self, arm):
            b = next(b for b, a in ARMS_BY_B.items() if a == arm)
            out = self.fresh(self.logs / f"trace_{arm}.json")
            if not self.passed(arm):
                write_json(out, dict(status="NOT-RUN", reason=f"bgate_{arm} did not pass: no timing without correctness"))
                print(f"MTRACE arm={arm} NOT-RUN: gate not passed", flush=True)
                return
            runs = []
            for k in range(3):
                name = f"trace_{arm}_{k}"
                raw = self.logs / f"{name}.bin"
                gate = batch.etx_result(self.run(name, [self.out(arm) / "run", *self.etx_args(b)],
                                                 env=dict(ETX_TRACE="1", ETX_TRACE_RAW=str(raw)), timeout=180), b)
                rep = analyze_trace(self.out(arm) / "plan.json", raw)
                runs.append(dict(run=k, gate=gate, trace=rep))
                s = rep["slots"]
                print(f"MTRACE arm={arm} run={k} tasks={rep['tasks']} span_us={rep['span_us']:.2f} "
                      f"busy={s['body_share']:.4f} queue_cross={s['queue_cross_share']:.4f} "
                      f"dep_wait={s['dep_wait_share']:.4f}", flush=True)
            write_json(out, dict(status="COMPLETE", arm=arm, batch=b, runs=runs,
                                 scope="instrumented; never a performance verdict"))

        def iso(self, arm):
            from etx.tools.timeline import build_trace, load_plan, read_raw
            out = self.fresh(self.logs / f"iso_{arm}.json")
            raw = self.logs / f"iso_{arm}.bin"
            gate = batch.etx_result(self.run(f"iso_{arm}", [self.out(arm) / "run", *self.etx_args(4)],
                                             env=dict(ETX_TRACE="1", ETX_TRACE_RAW=str(raw)), timeout=240), 4)
            blocks = iso_spans(build_trace(load_plan(self.out(arm) / "plan.json"), read_raw(raw)))
            ratios = iso_ratios(blocks)
            write_json(out, dict(status="COMPLETE", gate=gate, blocks=blocks, ratios=ratios))
            for phase, r in ratios.items():
                print(f"MISO {phase} alone_us={r['alone_us']:.2f} other_slot=x{r['co_resident']:.3f} "
                      f"same_slot=x{r['queued']:.3f}", flush=True)

        def counters(self, arm):
            b = next(b for b, a in ARMS_BY_B.items() if a == arm)
            out = self.fresh(self.logs / f"counters_{arm}.json")
            result = dict(arm=arm, batch=b, groups={})
            if not self.passed(arm):
                result.update(status="NOT-RUN", reason=f"bgate_{arm} did not pass")
            for group, names in COUNTER_GROUPS.items():
                if result.get("status") == "NOT-RUN":
                    break
                name = f"counters_{arm}_{group}"
                cmd = ["rocprofv3", "--pmc", *names, "--output-format", "csv", "-d", str(self.logs / name), "--",
                       self.out(arm) / "run", *self.etx_args(b)]
                try:
                    batch.etx_result(self.run(name, cmd, timeout=600), b)
                    result["groups"][group] = dict(status="COMPLETE", values=counter_values(self.logs / name, names))
                except (OSError, ValueError, RuntimeError) as exc:      # a crashed profiler is a missing value
                    result["groups"][group] = dict(status="INCOMPLETE", error=str(exc))
                print(f"MCOUNT arm={arm} group={group} {result['groups'][group]['status']}", flush=True)
                write_json(out.with_suffix(".partial.json"), result)
            result.setdefault("status", "COMPLETE" if all(g["status"] == "COMPLETE" for g in result["groups"].values())
                              else "INCOMPLETE")
            write_json(out, result)

        def report(self):
            write_json(self.logs / "mechanism_report.json", evaluate(self.logs))
            print("MREPORT written", flush=True)

    return MechanismSession


# ---------------------------------------------------------------- pre-registered evaluation
def evaluate(logs: Path) -> dict:
    """Apply the predictions of docs/deepseek/DEEPSEEK-BATCH-MECHANISM-2026-10-18.md section 4; a missing input is INCOMPLETE."""
    def load(name):
        p = logs / name
        return json.loads(p.read_text()) if p.is_file() else None

    def med_run(t, key):
        return statistics.median(r["trace"]["slots"][key] for r in t["runs"])

    out = dict(predictions={})
    t2, t4 = load("trace_x2.json"), load("trace_x4.json")
    if t2 and t4 and t2.get("status") == t4.get("status") == "COMPLETE":
        body = med_run(t4, "body_us") / med_run(t2, "body_us")
        busy = (med_run(t2, "body_share"), med_run(t4, "body_share"))
        cross = (med_run(t2, "queue_cross_share"), med_run(t4, "queue_cross_share"))
        ok = 1.94 <= body <= 2.06 and abs(busy[1] - busy[0]) <= 0.02 and cross[1] - cross[0] >= 0.05
        out["predictions"]["P1_slot_serialization"] = dict(
            status="HOLDS" if ok else "FAILS", slot_body_ratio_b4_b2=body, busy=busy, queue_cross_share=cross,
            rule="B4/B2 summed slot body in [1.94, 2.06], busy fraction within 0.02, cross-request queue blocking "
                 "share rises by >= 0.05")
    else:
        out["predictions"]["P1_slot_serialization"] = dict(status="INCOMPLETE")
    c2, c4 = load("counters_x2.json"), load("counters_x4.json")
    try:
        f2, f4 = (c["groups"]["fetch"]["values"]["FETCH_SIZE"] for c in (c2, c4))
        d2, d4 = (c["groups"]["dram"]["values"]["TCC_EA0_RDREQ_DRAM_sum"] for c in (c2, c4))
        fetch = f4["median"] / f2["median"]
        rate = (d4["median"] / d4["dispatch_us_median"]) / (d2["median"] / d2["dispatch_us_median"])
        out["predictions"]["P2_traffic"] = dict(
            status="HOLDS" if 1.9 <= fetch <= 2.1 and 0.9 <= rate <= 1.1 else "FAILS", fetch_ratio_b4_b2=fetch,
            dram_request_rate_ratio_b4_b2=rate,
            rule="FETCH_SIZE per dispatch B4/B2 in [1.9, 2.1] (replicated reads, no cross-request L2/MALL reuse) and "
                 "DRAM read-request rate per dispatch-microsecond within 10% (HBM demand rate unchanged)")
    except (TypeError, KeyError):
        out["predictions"]["P2_traffic"] = dict(status="INCOMPLETE")
    iso = load("iso_x4iso.json")
    if iso and iso.get("status") == "COMPLETE":
        r = iso["ratios"]
        queued_ok = all(1.85 <= v["queued"] <= 2.15 for v in r.values())
        gu = r["gate_up_5"]["co_resident"]
        out["predictions"]["P3_queue_vs_co_residency"] = dict(
            queued="HOLDS" if queued_ok else "FAILS", gate_up_co_resident=gu,
            branch=("RESIDENCY-CAP: gate_up has headroom beyond two slots per CU" if gu < 1.6 else
                    "SATURATED: gate_up is at its rate with two slots; only fewer bytes help" if gu >= 1.85 else
                    "BETWEEN: neither branch registered"), ratios=r,
            rule="same-slot (r0, r2) span / alone in [1.85, 2.15] for every phase; gate_up other-slot (r0, r1) / alone "
                 "< 1.6 -> residency cap binding, >= 1.85 -> phase saturated at two slots")
    else:
        out["predictions"]["P3_queue_vs_co_residency"] = dict(status="INCOMPLETE")
    t8 = load("trace_x8.json")
    out["b8"] = (dict(status=t8.get("status"), reason=t8.get("reason")) if t8 else dict(status="INCOMPLETE"))
    if t8 and t8.get("status") == "COMPLETE" and t4 and t4.get("status") == "COMPLETE":
        out["b8"]["slot_body_ratio_b8_b4"] = med_run(t8, "body_us") / med_run(t4, "body_us")
    out["status"] = ("COMPLETE" if all(p.get("status", p.get("queued")) != "INCOMPLETE"
                                       for p in out["predictions"].values()) else "INCOMPLETE")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("operation", choices=("offline", "gate", "trace", "iso", "counters", "report"))
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--traces", type=Path, default=TRACES)
    ap.add_argument("--json", type=Path, help="offline: write the decomposition here")
    ap.add_argument("--markdown", type=Path, help="offline: write the tables here")
    a = ap.parse_args(argv)
    if a.operation == "offline":
        o = offline(a.traces)
        if a.json:
            a.json.write_text(json.dumps(o, indent=1) + "\n")
        text = render(o)
        if a.markdown:
            a.markdown.write_text(text)
        print(text, end="")
        return
    if (a.operation in ("gate", "trace", "iso", "counters")) != (a.arm is not None):
        ap.error(f"{a.operation}: arm argument mismatch")
    allowed = {"gate": ("x2", "x4", "x8"), "trace": ("x2", "x4", "x8"), "counters": ("x2", "x4", "x8"), "iso": ("x4iso",)}
    if a.arm is not None and a.arm not in allowed[a.operation]:
        ap.error(f"{a.operation} arm must be one of {allowed[a.operation]}")
    s = _session_class()()
    getattr(s, a.operation)(*([a.arm] if a.arm else []))


if __name__ == "__main__":
    main()
