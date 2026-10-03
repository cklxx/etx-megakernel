"""Reproducible CPU protocol benchmarks. No tensor execution or GPU timings."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess

from etx.passes import PassOptions, compile_graph
from etx.sim import simulate

MODES = ("static", "dynamic", "hybrid")
ROOT = Path(__file__).resolve().parents[2]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean(value):
    """Keep failed nonfinite observations valid JSON, without hiding the failure."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_config(config):
    expected = {"schema_version", "arch", "rounds", "seed", "cases"}
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("config must use schema_version 1 and exactly " + str(sorted(expected)))
    if config["arch"] != "gfx942":
        raise ValueError("initial frozen protocol supports gfx942 only")
    if type(config["rounds"]) is not int or not 2 <= config["rounds"] <= 100:
        raise ValueError("rounds must be an integer in [2, 100]")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(config["cases"], list) or not config["cases"]:
        raise ValueError("cases must be a nonempty list")
    names = set()
    for case in config["cases"]:
        if set(case) != {"name", "workload", "size", "duration_cv"}:
            raise ValueError("case requires name, workload, size, duration_cv")
        if not isinstance(case["name"], str) or not case["name"] or case["name"] in names:
            raise ValueError("case names must be nonempty and unique")
        names.add(case["name"])
        if case["workload"] not in ("splitk", "moe"):
            raise ValueError("workload must be splitk or moe")
        if type(case["size"]) is not int or not 1 <= case["size"] <= 256:
            raise ValueError("size must be an integer in [1, 256]")
        if not finite(case["duration_cv"]) or not 0 <= case["duration_cv"] <= 2:
            raise ValueError("duration_cv must be finite and in [0, 2]")


def make_plan(case, arch, mode):
    # Repository fixtures, not installed-package model frontends.
    from examples import moe_layer, splitk_sum
    mod = splitk_sum if case["workload"] == "splitk" else moe_layer
    if mod is moe_layer and os.environ.get("ETX_MOE_SMALL") == "1":
        raise ValueError("unset ETX_MOE_SMALL: frozen benchmark uses the default MoE dimensions")
    bindings = mod.bindings(**({"n": case["size"]} if mod is splitk_sum else {"B": case["size"], "seed": 12345}))
    graph = mod.build()
    for grid in graph.grids:
        grid.duration_cv = case["duration_cv"]
    # Keep exactly the same tasks/operators in every arm; only scheduling varies.
    return compile_graph(graph, arch, bindings, mod.runtime(bindings),
                         PassOptions(force_mode=mode, event_elimination=False, prefetch=False))


def contract(plan):
    dependencies = {}
    for task in plan.tasks:
        key = (task.grid, task.coord)
        dependencies[str(task.id)] = sorted({plan.task_index[p]
            for event in plan.inst.task_in[key] for p in plan.inst.producers[event]})
    return {"tasks": [asdict(t) for t in plan.tasks], "dependencies": dependencies,
            "workers_per_device": plan.workers_per_device(), "n_devices": plan.n_devices,
            "bindings": plan.bindings, "runtime": plan.runtime, "decisions": plan.log,
            "tensors": {name: asdict(t) for name, t in plan.graph.tensors.items()},
            "options": asdict(plan.options)}


def check_sample(spec, result):
    """Independent trace checks; does NOT certify tensor values or cache visibility."""
    errors = []
    if result.get("deadlock") is not False:
        errors.append("deadlock or missing completion status")
    span = result.get("makespan_us")
    if not finite(span) or span <= 0:
        errors.append("nonfinite or nonpositive makespan")
    for key in ("busy_us", "wait_us", "sched_us"):
        if not finite(result.get(key)) or result[key] < 0:
            errors.append(f"invalid {key}")
    segments = result.get("segments", [])
    counts = Counter(s["task"] for s in segments if s["kind"] == "task")
    expected = {t["id"] for t in spec["tasks"]}
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        errors.append("task coverage: missing, duplicate or unknown task")
    tasks = {}
    worker_ends = {}
    for segment in sorted(segments, key=lambda s: s["start"] if finite(s.get("start")) else -1):
        start, end = segment.get("start"), segment.get("end")
        if not finite(start) or not finite(end) or start < 0 or end < start:
            errors.append("invalid segment timestamp")
            continue
        if finite(span) and end > span + 1e-9:
            errors.append("segment extends beyond completion boundary")
        device, worker = segment["device"], segment["worker"]
        if not (0 <= device < spec["n_devices"] and 0 <= worker < spec["workers_per_device"]):
            errors.append("invalid worker placement")
        key = (device, worker)
        if start < worker_ends.get(key, 0) - 1e-9:
            errors.append("overlapping worker segments")
        worker_ends[key] = max(worker_ends.get(key, 0), end)
        if segment["kind"] == "task":
            tasks[segment["task"]] = segment
        elif segment["kind"] not in ("wait", "sched"):
            errors.append("unknown segment kind")
    if finite(span) and not math.isclose(span, max(worker_ends.values(), default=0),
                                       rel_tol=1e-9, abs_tol=1e-8):
        errors.append("makespan disagrees with final trace completion")
    for task, deps in spec["dependencies"].items():
        consumer = tasks.get(int(task))
        if consumer is None:
            continue
        for dep in deps:
            producer = tasks.get(dep)
            if producer and finite(producer["end"]) and finite(consumer["start"]):
                if consumer["start"] < producer["end"] - 1e-9:
                    errors.append("dependency order violation")
    for field, kind in (("busy_us", "task"), ("wait_us", "wait"), ("sched_us", "sched")):
        parts = [s for s in segments if s["kind"] == kind]
        if all(finite(s["start"]) and finite(s["end"]) for s in parts) and finite(result.get(field)):
            total = sum(s["end"] - s["start"] for s in parts)
            if not math.isclose(total, result[field], rel_tol=1e-9, abs_tol=1e-8):
                errors.append(f"{field} disagrees with trace")
    return sorted(set(errors))


def provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    # Bind all tracked and unignored new files, including scripts, HIP, headers,
    # formal inputs and symlink targets. A dirty flag alone cannot identify edits.
    paths = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()))
    hashes = {p: digest(os.readlink(ROOT / p).encode() if (ROOT / p).is_symlink()
                        else (ROOT / p).read_bytes()) for p in paths
              if (ROOT / p).is_file() or (ROOT / p).is_symlink()}
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": hashes, "python": platform.python_version(), "host": platform.platform(),
            "created_utc": datetime.now(timezone.utc).isoformat()}


def run(config, out):
    validate_config(config)
    # Avoid silently reusing/overwriting an earlier experiment.
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "kind": "protocol_simulation", "unit": "us",
                "timing_boundary": "simulated first dispatch to all workers complete",
                "warmup": 0, "warmup_reason": "fresh deterministic simulator; no device cache",
                "tensor_correctness": "not_executed", "gpu_measurement": False,
                "config": config, "provenance": provenance(), "artifacts": {}}
    specs = {}
    plan_errors = {}
    plans = {}
    for case in config["cases"]:
        for mode in MODES:
            key = f'{case["name"]}:{mode}'
            try:
                plans[key] = make_plan(case, config["arch"], mode)
                specs[key] = contract(plans[key])
            except Exception as exc:
                plan_errors[key] = f"{type(exc).__name__}: {exc}"
    write_json(out / "plans.json", specs)
    with (out / "samples.jsonl").open("w") as stream:
        for case in config["cases"]:
            for round_id in range(config["rounds"]):
                # Rotate order; all arms in a pair use the same task-duration seed.
                order = MODES[round_id % 3:] + MODES[:round_id % 3]
                for mode in order:
                    key = f'{case["name"]}:{mode}'
                    sample = {"case": case["name"], "mode": mode, "round": round_id,
                              "seed": config["seed"] + round_id, "errors": []}
                    try:
                        if key in plan_errors:
                            raise ValueError(plan_errors[key])
                        result = clean(asdict(simulate(plans[key], seed=sample["seed"])))
                        sample["result"] = result
                        sample["errors"] = check_sample(specs[key], result)
                    except Exception as exc:
                        sample["errors"] = [f"{type(exc).__name__}: {exc}"]
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
                    stream.flush()
                    print(f'{key} round={round_id} {"FAIL" if sample["errors"] else "ok"}', flush=True)
    for name in ("plans.json", "samples.jsonl"):
        manifest["artifacts"][name] = digest((out / name).read_bytes())
    write_json(out / "manifest.json", manifest)
    return report(out)


def report(out):
    """Recompute exclusively from saved contracts/traces; never trust cached verdicts."""
    # A failed rebuild must not leave an earlier successful report in place.
    write_json(out / "summary.json", {"kind": "protocol_simulation", "valid": False,
                                    "failures": ["report validation incomplete"], "comparisons": []})
    (out / "report.md").write_text("# INVALID protocol simulation\n\nReport validation incomplete; comparisons withheld.\n")
    manifest = json.loads((out / "manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "protocol_simulation"
            or manifest.get("gpu_measurement") is not False or manifest.get("unit") != "us"):
        raise ValueError("unsupported manifest or timing basis")
    config = manifest["config"]
    validate_config(config)
    for name in ("plans.json", "samples.jsonl"):
        if digest((out / name).read_bytes()) != manifest["artifacts"].get(name):
            raise ValueError(f"artifact checksum mismatch: {name}")
    specs = json.loads((out / "plans.json").read_text())
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    expected = {(c["name"], m, r) for c in config["cases"] for m in MODES for r in range(config["rounds"])}
    observed = Counter((s["case"], s["mode"], s["round"]) for s in samples)
    failures = []
    if set(observed) != expected or any(n != 1 for n in observed.values()):
        failures.append("incomplete, duplicate or unexpected paired samples")
    values = {}
    for sample in samples:
        key = f'{sample["case"]}:{sample["mode"]}'
        errors = list(sample["errors"])
        if sample["seed"] != config["seed"] + sample["round"]:
            errors.append("paired seed mismatch")
        if key not in specs or "result" not in sample:
            errors.append("missing plan or simulation result")
        else:
            errors.extend(check_sample(specs[key], sample["result"]))
        if errors:
            failures.append(f'{key} round={sample["round"]}: ' + "; ".join(sorted(set(errors))))
        else:
            values[(sample["case"], sample["mode"], sample["round"])] = sample["result"]["makespan_us"]
    lines = ["# CPU protocol simulation — not GPU performance", "",
             "No kernels or tensors executed; model numerical correctness is unverified.",
             "Samples vary modeled task durations, not hardware noise. No confidence or speedup claim for a GPU.", "",
             f'Source commit: `{manifest["provenance"]["commit"]}`; dirty: `{manifest["provenance"]["dirty"]}`.', ""]
    summary = {"kind": "protocol_simulation", "valid": not failures, "failures": failures, "comparisons": []}
    if failures:
        lines += ["**INVALID: all performance comparisons withheld.**", ""] + [f"- {f}" for f in failures]
    else:
        lines += ["| Case | Schedule | Median modeled µs | Min–max µs | Median paired reduction vs static |",
                  "|---|---|---:|---:|---:|"]
        for case in config["cases"]:
            for mode in MODES:
                vals = [values[(case["name"], mode, r)] for r in range(config["rounds"])]
                reductions = [100 * (1 - values[(case["name"], mode, r)] / values[(case["name"], "static", r)])
                              for r in range(config["rounds"])]
                row = {"case": case["name"], "mode": mode, "median_us": statistics.median(vals),
                       "min_us": min(vals), "max_us": max(vals), "pairs": len(vals),
                       "median_paired_reduction_pct": statistics.median(reductions)}
                summary["comparisons"].append(row)
                lines.append(f'| {row["case"]} | {mode} | {row["median_us"]:.3f} | '
                             f'{min(vals):.3f}–{max(vals):.3f} | {row["median_paired_reduction_pct"]:.2f}% |')
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return 0 if not failures else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    try:
        return run(json.loads(args.config.read_text()), args.out) if args.command == "run" else report(args.out)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"BENCH FAILED: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
