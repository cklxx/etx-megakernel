#!/usr/bin/env python3
"""Paired fused / HIP-graph runs of the same tiles: per-pair gain, median, spread, and the candidate 1 verdict.

  python -m examples.vllm_llm.pairs ~/sess/p1_fused.log ~/sess/p1_graph.log ... [--pairs 5] [--basis wall]

The files are read in the order given, as one log (a single file holding all runs works too). A run is the host's
`mode:` line followed by its `RESULT ... median_ms=<device> wall_ms=<wall>` line (examples/vllm_llm/host.hip):
`mode: ETX megakernel ...` is the fused arm, `mode: unfused ...` the HIP-graph arm. The session prints one
`ETX_ENV ETX_UNFUSED_SYNC=<v> ETX_UNFUSED_NOGRAPH=<v>` line before each run, because the host's mode line reads the
same with and without the graph. gpu_jobs headers (`# job <name> gpu=<id>`) separate the job logs.

Runs pair up in order: (1, 2), (3, 4), ... Each pair must hold one run of each arm. Relative gain of a pair:
(graph - fused) / graph, so a positive gain means the megakernel is faster. Spread: largest minus smallest gain.

Verdict (docs/NEXT-2026-10-03.md candidate 1): ESTABLISHED only if every pair's gain is positive, the median gain is
above 2%, and the median gain exceeds twice the spread. The exit code follows --basis (wall by default, as in
docs/qwen/DESIGN-acquire-2026-10-05.md section 4): 0 established, 1 not established, 2 refused. The device basis is
reported next to it. `VLLM ...` control lines are summarised, not paired.

Refused: an odd number of runs, a pair that is not one run of each arm, fewer or more pairs than --pairs, a RESULT
without its mode line, an unknown or stream-launch mode, a run without its ETX_ENV line or with debug
synchronization or graph-off set, a run that is not 32/32 correct, a zero time, or runs on different GPUs.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

MIN_GAIN = 0.02
SPREAD_FACTOR = 2.0
DEBUG_ENV = ("ETX_UNFUSED_SYNC", "ETX_UNFUSED_NOGRAPH")


class Refusal(Exception):
    pass


@dataclass
class Run:
    arm: str
    device_ms: float
    wall_ms: float
    job: str
    gpu: str | None
    line: int


@dataclass
class Log:
    runs: list[Run] = field(default_factory=list)
    controls: dict = field(default_factory=dict)


def kv(text: str) -> dict:
    return dict(re.findall(r"(\w+)=(\S*)", text))


def arm_of(mode: str, where: str) -> str:
    if mode.startswith("ETX megakernel"):
        return "fused"
    if mode.startswith("unfused"):
        if "stream launches" in mode:
            raise Refusal(f"{where}: unfused run used stream launches, not a HIP graph: {mode!r}")
        return "graph"
    raise Refusal(f"{where}: unknown mode {mode!r}")


def parse(lines: list[str]) -> Log:
    log = Log()
    job, gpu, mode, env = "-", None, None, None
    for n, line in enumerate(lines, 1):
        where = f"line {n} (job {job})"
        m = re.match(r"# job (\S+) gpu=(\S+)", line)
        if m:
            job, gpu, mode, env = m.group(1), m.group(2), None, None
            continue
        if line.startswith("ETX_ENV "):
            env = kv(line[len("ETX_ENV "):])
            continue
        if line.startswith("mode: "):
            mode = line[len("mode: "):].strip()
            continue
        if line.startswith("VLLM "):
            f = kv(line)
            if "median_ms" not in f:
                raise Refusal(f"{where}: VLLM line without median_ms")
            log.controls.setdefault(f"aiter={f.get('aiter', '?')} batch={f.get('batch', '?')} "
                                    f"context={f.get('context', '?')}", []).append(float(f["median_ms"]))
            continue
        if not line.startswith("RESULT "):
            continue
        f = kv(line)
        if mode is None:
            raise Refusal(f"{where}: RESULT without a preceding mode line, so its arm is unknown")
        arm = arm_of(mode, where)
        if env is None:
            raise Refusal(f"{where}: missing input: ETX_ENV line before the run (debug synchronization state unknown)")
        on = [k for k in DEBUG_ENV if env.get(k)]
        if on:
            raise Refusal(f"{where}: {', '.join(on)} set: debug synchronization or graph-off run is not a timing run")
        for k in ("matched", "gen", "median_ms", "wall_ms"):
            if k not in f:
                raise Refusal(f"{where}: missing input: RESULT field {k}")
        if int(f["gen"]) <= 0 or int(f["matched"]) != int(f["gen"]):
            raise Refusal(f"{where}: correctness gate failed: matched {f['matched']} of {f['gen']} tokens")
        dev, wall = float(f["median_ms"]), float(f["wall_ms"])
        if dev <= 0 or wall <= 0:
            raise Refusal(f"{where}: zero time (device {dev}, wall {wall})")
        log.runs.append(Run(arm, dev, wall, job, gpu, n))
        mode, env = None, None
    return log


def pairs(log: Log, expected: int) -> list[tuple[Run, Run]]:
    runs = log.runs
    if not runs:
        raise Refusal("no runs found")
    if len(runs) % 2:
        raise Refusal(f"{len(runs)} runs: an odd count cannot form pairs")
    gpus = {r.gpu for r in runs if r.gpu is not None}
    if len(gpus) > 1:
        raise Refusal(f"runs on different GPUs {sorted(gpus)}: pairs must share one device")
    out = []
    for i in range(0, len(runs), 2):
        a, b = runs[i], runs[i + 1]
        if {a.arm, b.arm} != {"fused", "graph"}:
            raise Refusal(f"pair {i // 2 + 1} (jobs {a.job}, {b.job}) is not interleaved: arms {a.arm}, {b.arm}")
        out.append((a, b) if a.arm == "fused" else (b, a))
    if len(out) != expected:
        raise Refusal(f"incomplete: {len(out)} of {expected} pairs")
    return out


def verdict(gains: list[float]) -> dict:
    med = median(gains)
    spread = max(gains) - min(gains)
    positive = all(g > 0 for g in gains)
    ok = positive and med > MIN_GAIN and med > SPREAD_FACTOR * spread
    return dict(median=med, spread=spread, positive=positive, established=ok)


def report(ps: list[tuple[Run, Run]], log: Log, basis: str, out=None) -> bool:
    out = out or sys.stdout
    gains = {"wall": [], "device": []}
    print(f"{'pair':>4} {'fused wall':>10} {'graph wall':>10} {'gain':>8}   {'fused dev':>9} {'graph dev':>9} {'gain':>8}",
          file=out)
    for i, (f, g) in enumerate(ps, 1):
        gw = (g.wall_ms - f.wall_ms) / g.wall_ms
        gd = (g.device_ms - f.device_ms) / g.device_ms
        gains["wall"].append(gw); gains["device"].append(gd)
        print(f"{i:4d} {f.wall_ms:10.3f} {g.wall_ms:10.3f} {gw:+8.2%}   {f.device_ms:9.3f} {g.device_ms:9.3f} {gd:+8.2%}", file=out)
    result = None
    for b in (basis, "device" if basis == "wall" else "wall"):
        v = verdict(gains[b])
        word = "ESTABLISHED" if v["established"] else "NOT ESTABLISHED"
        print(f"VERDICT {b} {word}: median gain {v['median']:+.2%} (needs > {MIN_GAIN:.0%}), "
              f"2 x spread {SPREAD_FACTOR * v['spread']:.2%} (needs < median), "
              f"every pair positive: {'yes' if v['positive'] else 'no'}"
              + ("" if b == basis else "  [secondary basis]"), file=out)
        if result is None:
            result = v["established"]
    for k, v in log.controls.items():
        print(f"control vLLM {k}: {len(v)} run(s), median {median(v):.3f} ms (vllm_bench wall basis; not paired)", file=out)
    return bool(result)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m examples.vllm_llm.pairs", description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", help="log files, in run order")
    ap.add_argument("--pairs", type=int, default=5, help="pairs the session planned (default 5)")
    ap.add_argument("--basis", choices=["wall", "device"], default="wall")
    a = ap.parse_args(argv)
    try:
        lines = []
        for p in a.logs:
            path = Path(p).expanduser()
            if not path.is_file():
                raise Refusal(f"missing input: log {path}")
            lines += path.read_text(errors="replace").splitlines()
        log = parse(lines)
        ps = pairs(log, a.pairs)
    except Refusal as e:
        print(f"REFUSED: {e}. No verdict.")
        return 2
    return 0 if report(ps, log, a.basis) else 1


if __name__ == "__main__":
    raise SystemExit(main())
