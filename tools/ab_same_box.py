"""Same-machine interleaved A/B measurement with an A/A control (docs/EXPERIMENTS.md, measurement rules).

The script runs three arms on one machine: A (the baseline), B (the candidate) and C (a second copy of A, the
control). Every round runs each arm once. The arm order rotates from round to round, so no arm always runs first.
Each run prints one timing value. A regular expression reads the value from the run output.

For each round the script computes two paired gains (lower is better):
  gain    = A / B - 1     the gain of the candidate
  control = A / C - 1     the gain of A over itself; this is the noise of the machine and of the order
The verdict uses only these paired gains, never a number from another session or another machine:
  WIN      every gain > 0 and median gain > max |control|
  LOSS     every gain < 0 and -median gain > max |control|
  NO-DIFF  otherwise
  INCOMPLETE  a run fails, a run prints no value, or the machine identity changes during the session

Output, in --out:
  runs/r<round>_<arm>.log   the full output of each run
  ab.log                    one ABRUN line per run, then one ABREPORT line; tools/extract_key.py reads it
  ab.json                   the same numbers
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import socket
import statistics
import subprocess
import sys
from pathlib import Path

ARMS = ("A", "B", "C")


def machine_identity() -> str:
    """Host name and GPU serial numbers. A failed probe gives an empty field, not an error."""
    parts = [socket.gethostname()]
    for cmd in (["rocm-smi", "--showserial", "--json"], ["nvidia-smi", "--query-gpu=serial,uuid", "--format=csv,noheader"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout.strip():
            parts.append(" ".join(r.stdout.split()))
    return " | ".join(parts)


def order(round_index: int) -> tuple[str, ...]:
    """Rotate the arm order: ABC, BCA, CAB, ABC, ..."""
    k = round_index % len(ARMS)
    return ARMS[k:] + ARMS[:k]


def spread(values: list[float]) -> dict:
    return dict(n=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values),
                spread=max(values) - min(values))


def verdict(gains: list[float], controls: list[float]) -> tuple[str, str]:
    noise = max(abs(c) for c in controls)
    med = statistics.median(gains)
    if all(g > 0 for g in gains) and med > noise:
        return "WIN", f"every gain > 0 and median {med:+.6%} > max |control| {noise:.6%}"
    if all(g < 0 for g in gains) and -med > noise:
        return "LOSS", f"every gain < 0 and |median| {-med:.6%} > max |control| {noise:.6%}"
    return "NO-DIFF", f"median {med:+.6%}, max |control| {noise:.6%}"


def run(args: argparse.Namespace, identity=machine_identity) -> dict:
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.metric)
    cmds = {"A": args.a, "B": args.b, "C": args.control or args.a}
    lines: list[str] = []
    before = identity()
    lines.append(f"ABMACHINE start {before}")
    values = {arm: [] for arm in ARMS}
    problems: list[str] = []
    for r in range(args.rounds):
        for arm in order(r):
            log = out / "runs" / f"r{r}_{arm}.log"
            p = subprocess.run(cmds[arm], shell=True, capture_output=True, text=True, timeout=args.timeout)
            text = p.stdout + p.stderr
            log.write_text(f"$ {cmds[arm]}\n{text}")
            found = pattern.findall(text)
            value = float(found[-1]) if found else None
            lines.append(f"ABRUN round={r} arm={arm} rc={p.returncode} value={value} src=runs/{log.name}")
            if p.returncode != 0:
                problems.append(f"round {r} arm {arm} rc={p.returncode}")
            elif value is None:
                problems.append(f"round {r} arm {arm} printed no value for {args.metric!r}")
            values[arm].append(value)
    after = identity()
    lines.append(f"ABMACHINE end {after}")
    if after != before:
        problems.append("the machine identity changed during the session")
    result = dict(label=args.label, rounds=args.rounds, commands=cmds, machine=before, metric=args.metric,
                  values=values, problems=problems)
    if problems:
        result.update(verdict="INCOMPLETE", rule="; ".join(problems))
        lines.append(f"ABREPORT label={args.label} verdict=INCOMPLETE reason={result['rule']}")
    else:
        gains = [a / b - 1 for a, b in zip(values["A"], values["B"])]
        controls = [a / c - 1 for a, c in zip(values["A"], values["C"])]
        v, rule = verdict(gains, controls)
        g, c = spread(gains), spread(controls)
        result.update(gains=gains, controls=controls, gain=g, control=c, verdict=v, rule=rule,
                      medians={arm: statistics.median(values[arm]) for arm in ARMS})
        lines.append(f"ABREPORT label={args.label} B vs A: gain median {g['median'] * 100:+.6f}% spread "
                     f"{g['spread'] * 100:.6f}% control median {c['median'] * 100:+.6f}% spread "
                     f"{c['spread'] * 100:.6f}% rounds={args.rounds} verdict={v}")
    (out / "ab.log").write_text("\n".join(lines) + "\n")
    (out / "ab.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="baseline command (shell)")
    ap.add_argument("--b", required=True, help="candidate command (shell)")
    ap.add_argument("--control", help="control command; default: the baseline command again (A/A)")
    ap.add_argument("--metric", required=True, help="regex with one group: the timing value, lower is better")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--label", default="ab")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.rounds < 3:
        ap.error("--rounds must be 3 or more: the rotation needs every arm in every position once")
    r = run(args)
    print((Path(args.out) / "ab.log").read_text(), end="")
    return 0 if r["verdict"] != "INCOMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
