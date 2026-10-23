"""Acquire: acquire contract."""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from etx.tools.contracts.source_control import compact

from .schema import CONSUMER, CONTRACT, PRODUCER, ROOT, Refusal
from .source import check_atomic_sub_device, check_bindings, check_cflags, check_lowering, check_plan, check_runtime, check_source
from .llvm import check_ir, load_ir

def run(build: Path | None, ir: Path | None, cflags: str | None, runtime: Path) -> tuple[list[str], dict]:
    if cflags is None:
        raise Refusal("missing input: --cflags. A -DETX_COHERENT_DEVICE_DATA or -DETX_NO_ALLWAVE_DRAIN on the device "
                      "compile removes a fence and leaves no trace in the build directory")
    files = {}
    for name in ("megakernel_d0.hip", "imp_tiles.hip", "etx_lowering.h", "plan.json"):
        path = (build / name) if build else None
        if path is None or not path.is_file():
            raise Refusal(f"missing input: {path or name}")
        files[name] = path.read_text()
    ir = ir or (build / "imports.bc")
    for name in ("primitives.h", "import.h"):
        if not (runtime / name).is_file():
            raise Refusal(f"missing input: {runtime / name}")
    errors: list[str] = []
    report = dict(cflags=check_cflags(cflags, errors))
    report["ir"] = check_ir(load_ir(ir), errors)
    report["bindings"] = check_bindings(errors)
    plan = json.loads(files["plan.json"])
    pairs = check_source(files["megakernel_d0.hip"], plan, errors)
    check_plan(plan, pairs, errors)
    report["pairs"] = pairs
    check_lowering(files["etx_lowering.h"], errors)
    vendor = runtime / "primitives_amdgcn.h"
    check_atomic_sub_device(vendor.read_text() if vendor.is_file() else "", errors)
    check_runtime((runtime / "primitives.h").read_text(), (runtime / "import.h").read_text(),
                  files["imp_tiles.hip"], errors)
    return errors, report


def summary(report: dict, out) -> None:
    for body, inv in report["ir"].items():
        if body == "attn_bodies":
            continue
        export = PRODUCER if PRODUCER in body else CONSUMER
        cells = [f"{CONTRACT[export][k].name}={'+'.join(f'{n} {c}' for c, n in sorted(v.items()))}"
                 for k, v in inv.items()]
        print(f"  {body}: {', '.join(cells)}", file=out)
    pairs = report["pairs"]
    if pairs:
        x = sum(p.get("crosses_xcd", False) for p in pairs)
        p0 = pairs[0]
        print(f"  {len(pairs)} pairs (layers {pairs[0]['layer']}-{pairs[-1]['layer']}); {x} cross XCDs; "
              f"e.g. {p0['producer']} -> {p0['consumer']} on {p0['event']} (word {p0['word']}, share {p0['share']}), "
              f"producer domains {p0.get('producer_domains')}, consumer domains {p0.get('consumer_domains')}", file=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m etx.tools.acquire_contract", description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, required=True, help="build directory (examples/vllm_llm/build.sh output)")
    ap.add_argument("--ir", type=Path, help="imports .ll or .bc (default: <build>/imports.bc)")
    ap.add_argument("--cflags", help="the device compile flags of the build, e.g. \"-DETX_DSL_NT -DETX_DSL_VECNORM\"")
    ap.add_argument("--runtime", type=Path, default=ROOT / "etx/runtime/include/etx")
    a = ap.parse_args(argv)
    try:
        errors, report = run(a.build.expanduser(), a.ir, a.cflags, a.runtime)
    except Refusal as e:
        print(f"REFUSED: {e}. No verdict.")
        return 2
    for e in errors:
        print(f"VIOLATION: {e}")
    if errors:
        print(f"FAIL: {len(errors)} contract violations. Keep the acquire; do not build the transformed pair.")
        return 1
    print(f"PASS: {PRODUCER} -> {CONSUMER} matches docs/core/CONTRACT-acquire-2026-10-05.md")
    summary(report, sys.stdout)
    return 0
