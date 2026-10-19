#!/usr/bin/env python3
"""Compare a local pytest run with tools/known_env_failures.txt.

LOCAL RUNS ONLY. Do not use this in CI to hide failures: CI must run the full suite and fail on every failure.

Input: a pytest junit XML file (pytest --junitxml=FILE) or a pytest -rA / -rfE text log.
Exit 1 when a failure or error is not in the list (new failure). Report list entries that now pass (stale);
with --strict, stale entries also exit 1.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LIST = Path(__file__).resolve().parent / "known_env_failures.txt"


def load_list(path: Path) -> dict[str, str]:
    known = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tid, _, reason = line.partition(" ")
        known[tid] = reason.strip()
    return known


def _junit_id(case: ET.Element) -> str:
    cls, name, file = case.get("classname", ""), case.get("name", ""), case.get("file")
    parts = cls.split(".")
    # classname "tests.test_x.TestC" -> tests/test_x.py::TestC
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith("test"):
            mod = "/".join(parts[:i]) + ".py"
            rest = parts[i:]
            return "::".join([file or mod, *rest, name]) if name else (file or mod)
    return f"{cls}::{name}"


def results(path: Path) -> tuple[set[str], set[str]]:
    """(failed or errored ids, passed ids)."""
    bad, ok = set(), set()
    text = path.read_text(errors="replace")
    if text.lstrip().startswith("<"):
        for case in ET.fromstring(text).iter("testcase"):
            tid = _junit_id(case)
            if case.find("failure") is not None or case.find("error") is not None:
                bad.add(tid)
            elif case.find("skipped") is None:
                ok.add(tid)
        return bad, ok
    for m in re.finditer(r"^(PASSED|FAILED|ERROR) (\S+)", text, re.M):
        (ok if m.group(1) == "PASSED" else bad).add(m.group(2))
    return bad, ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="pytest --junitxml file or -rA text log")
    ap.add_argument("--list", type=Path, default=LIST)
    ap.add_argument("--strict", action="store_true", help="stale entries also fail")
    a = ap.parse_args(argv)
    known = load_list(a.list)
    bad, ok = results(a.report)
    new = sorted(bad - known.keys())
    stale = sorted(t for t in known if t in ok)
    for t in sorted(bad & known.keys()):
        print(f"KNOWN  {t}  ({known[t]})")
    for t in stale:
        print(f"STALE  {t}  (passes now; remove it from {a.list.name})")
    for t in new:
        print(f"NEW    {t}")
    print(f"{len(bad)} failed/errored: {len(bad) - len(new)} known, {len(new)} new; {len(stale)} stale")
    return 1 if new or (a.strict and stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
