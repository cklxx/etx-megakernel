"""Rerun derived experiment files from committed inputs and compare the bytes.

The registry (examples/gpu-live/results/registry.json) lists each derived file, the
script that made it and its committed inputs. This check runs on a CPU. It fails when
a fresh output differs from the committed output, or when an input is missing.
It also runs tools/extract_key.py --check for the key.json and key.md files, and it prints the verdict of
each experiment. A verdict that is not COMPLETE (for example an INCOMPLETE A/B session) is flagged on a
NOT COMPLETE line. A key.json that has an INCOMPLETE A/B comparison but a COMPLETE verdict fails the check.

Usage: python tools/check_results.py [--registry PATH]
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def run_module_cli(base: Path, entry: dict, work: Path) -> Path:
    out = work / Path(entry["output"]).name

    def expand(arg):
        match = re.fullmatch(r"\{in:(.+)\}", arg)
        if match:
            path = base / match.group(1)
            if not path.is_file():
                raise FileNotFoundError(path)
            return str(path)
        return str(out) if arg == "{out}" else arg

    cmd = [sys.executable, "-m", entry["module"], *map(expand, entry["args"])]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exit {proc.returncode}: {proc.stderr.strip()[-500:]}")
    return out


def run_session(base: Path, entry: dict, work: Path) -> Path:
    """Run one CPU-only session step on a copy of its committed inputs."""
    module, name = entry["session"].split(":")
    cls = getattr(importlib.import_module(module), name)
    for dest, src in entry["inputs"].items():
        shutil.copyfile(base / src, work / dest)
    session = cls.__new__(cls)  # no GPU lease, no reference verification
    session.logs = work
    session.batches = tuple(entry["batches"])
    with contextlib.redirect_stdout(io.StringIO()):
        getattr(session, entry["method"])()
    return work / entry["produces"]


KINDS = {"module_cli": run_module_cli, "session": run_session}


def check_derived(registry: Path) -> int:
    base = registry.parent
    bad = 0
    for entry in json.loads(registry.read_text())["derived"]:
        expected = base / entry["output"]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fresh = KINDS[entry["kind"]](base, entry, Path(tmp))
                same = fresh.read_bytes() == expected.read_bytes()
            except Exception as exc:  # report every entry, then fail
                print(f"DERIVED ERROR {entry['output']}: {exc}", file=sys.stderr)
                bad += 1
                continue
        if same:
            print(f"DERIVED OK {entry['output']}")
        else:
            print(f"DERIVED MISMATCH {entry['output']}: fresh output differs from the committed file",
                  file=sys.stderr)
            bad += 1
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=ROOT / "examples/gpu-live/results/registry.json")
    a = ap.parse_args(argv)
    import extract_key
    bad = check_derived(a.registry)
    bad += extract_key.main(["--registry", str(a.registry), "--check"])
    flagged = []
    for entry in json.loads(a.registry.read_text())["key_extraction"]:
        path = a.registry.parent / entry["dir"] / "key.json"
        if not path.is_file():
            continue  # extract_key --check already reported it
        key = json.loads(path.read_text())
        value, rule = key["verdict"]["value"], key["verdict"]["rule"]
        print(f"VERDICT {value} {entry['dir']}: {rule}")
        if value != "COMPLETE":
            flagged.append(f"{entry['dir']}={value}")
        ab_bad = [c for c in key.get("comparisons", []) if c.get("outcome") == "INCOMPLETE"]
        if ab_bad and value == "COMPLETE":
            print(f"VERDICT ERROR {entry['dir']}: an A/B comparison is INCOMPLETE but the verdict is COMPLETE",
                  file=sys.stderr)
            bad += 1
    if flagged:
        print(f"NOT COMPLETE ({len(flagged)}): " + ", ".join(flagged))
    for item in json.loads(a.registry.read_text()).get("not_reproducible", []):
        print(f"NOT CHECKED {item['output']}: {item['reason']}")
    print("RESULTS CHECK", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
