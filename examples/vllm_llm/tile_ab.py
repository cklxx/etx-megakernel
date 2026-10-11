#!/usr/bin/env python3
"""Verdicts for the tile-flag session (examples/sessions/tiles.py, docs/TILES-2026-10-07.md section 6).

  python -m examples.vllm_llm.tile_ab ~/sess

Reads the session's job logs from the log directory:
  ab<round>_<flag>_ctl.log / ab<round>_<flag>_x.log   one timed run each: the control and the flag build, adjacent
  bitcmp.log                                           BITCMP <build> identical | DIFFERS <files>
  tr_<build>.log                                       ETX_TRACE phase table (optional)

A pair is the two runs of one round and one flag. Gain = flag - control on the device median (ms per token), so a
negative gain is faster. A flag is KEEP when the control is self-identical, the flag build is bit-identical to it,
every run is 32/32, there are at least MIN_PAIRS pairs, every pair is faster and the mean gain is at or below
-THRESH_MS. NEUTRAL when |mean| < THRESH_MS, SLOWER when the mean is at or above +THRESH_MS, MIXED otherwise.
A flag whose output is not bit-identical is REJECT whatever its time. Missing or incomplete data is INCOMPLETE.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import mean

THRESH_MS = 0.030   # above the 0.017-0.028 ms same-binary spread of five fused runs (results_2026-10-01/final, round 23)
MIN_PAIRS = 3
TYPES = ("qkv", "o_proj", "router", "moe_up", "moe_down", "lm_head")
RESULT = re.compile(r"^RESULT .*\bmatched=(\d+) gen=(\d+) median_ms=([0-9.]+) wall_ms=([0-9.]+)")
TRACE_ROW = re.compile(r"^\s+([a-z_]+?)_(\d+)(?:_x)?\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s")
TRACE_ROW1 = re.compile(r"^\s+([a-z_]+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s")


def result(path: Path) -> tuple[float, float, bool] | None:
    """(device ms, wall ms, 32/32) of the log's RESULT line, or None."""
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        m = RESULT.match(line)
        if m:
            return float(m.group(3)), float(m.group(4)), m.group(1) == m.group(2)
    return None


def bitcmp(path: Path) -> dict[str, bool]:
    out = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"^BITCMP (\S+) (identical|DIFFERS)", line)
            if m:
                out[m.group(1)] = m.group(2) == "identical"
    return out


def trace_bodies(path: Path) -> dict[str, float]:
    """Mean task body (us) per base type over its layers, from the ETX_TRACE phase table."""
    acc: dict[str, list[float]] = {}
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        m = TRACE_ROW.match(line)
        if m:
            acc.setdefault(m.group(1), []).append(float(m.group(5)))
            continue
        m = TRACE_ROW1.match(line)
        if m and m.group(1) in TYPES:
            acc.setdefault(m.group(1), []).append(float(m.group(4)))
    return {k: mean(v) for k, v in acc.items()}


def pairs(logdir: Path, flag: str) -> list[tuple[int, tuple, tuple]]:
    out = []
    for p in sorted(logdir.glob(f"ab*_{flag}_ctl.log")):
        rnd = int(re.match(r"ab(\d+)_", p.name).group(1))
        c, x = result(p), result(logdir / f"ab{rnd}_{flag}_x.log")
        if c and x:
            out.append((rnd, c, x))
    return out


def verdict(prs: list, ident: bool | None, ctl_ident: bool | None) -> str:
    if ident is False and ctl_ident:
        return "REJECT"
    if len(prs) < MIN_PAIRS or ident is None or ctl_ident is None:
        return "INCOMPLETE"
    if not all(c[2] and x[2] for _, c, x in prs):
        return "REJECT"
    g = [x[0] - c[0] for _, c, x in prs]
    m = mean(g)
    if m <= -THRESH_MS and all(v < 0 for v in g):
        return "KEEP" if ident and ctl_ident else "KEEP-UNVERIFIED"
    if abs(m) < THRESH_MS:
        return "NEUTRAL"
    if m >= THRESH_MS:
        return "SLOWER"
    return "MIXED"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdir")
    ap.add_argument("--flags", default="all,wsum,scalar,gu1,rbatch")
    a = ap.parse_args(argv)
    d = Path(a.logdir).expanduser()
    bc = bitcmp(d / "bitcmp.log")
    ctl_ident = bc.get("ctl2")
    print(f"control self-identical (ctl vs ctl2): {ctl_ident}")
    print(f"{'flag':8} {'pairs':>5} {'mean ms':>8} {'min':>7} {'max':>7} {'wall':>7}  bits       verdict")
    for f in a.flags.split(","):
        prs = pairs(d, f)
        g = [x[0] - c[0] for _, c, x in prs]
        gw = [x[1] - c[1] for _, c, x in prs]
        ident = bc.get(f)
        bits = {True: "identical", False: "DIFFERS", None: "missing"}[ident]
        if g:
            print(f"{f:8} {len(g):5} {mean(g):+8.3f} {min(g):+7.3f} {max(g):+7.3f} {mean(gw):+7.3f}  {bits:10} {verdict(prs, ident, ctl_ident)}")
        else:
            print(f"{f:8} {0:5} {'-':>8} {'-':>7} {'-':>7} {'-':>7}  {bits:10} INCOMPLETE")
    ctl_tr = trace_bodies(d / "tr_ctl.log")
    if ctl_tr:
        print("\ntraced mean task body per type (us), flag build minus control:")
        print(f"{'type':10} {'ctl':>7} " + " ".join(f"{f:>7}" for f in a.flags.split(",")))
        for t in TYPES:
            if t not in ctl_tr:
                continue
            row = []
            for f in a.flags.split(","):
                b = trace_bodies(d / f"tr_{f}.log")
                row.append(f"{b[t] - ctl_tr[t]:+7.2f}" if t in b else f"{'-':>7}")
            print(f"{t:10} {ctl_tr[t]:7.2f} " + " ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
