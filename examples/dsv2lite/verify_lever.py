"""Local verification record of one lever arm (no GPU): CPU gates, ROCm 7.2.4 resources, default-unchanged proof.

  python -m examples.dsv2lite.verify_lever <arm>      # arm = a lever, or lever+lever for a combination

Runs, in order, and fails closed on any step:
  1. the lever's CPU gates: tests/test_dsv2_lever_<name>.py for each lever in the arm, plus the stamps composition test;
  2. rocm_build_check.sh for `default` and the arm (the VM's compiler, offline container);
  3. default identity: the default build's ISA hash must equal the recorded results/levers/default.json;
  4. static ISA counts that the GPU cannot be asked for (non-temporal weight loads, full vmcnt(0) drains).
Writes examples/dsv2lite/results/levers/<arm>.json and prints the table that goes into the commit message.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from examples.dsv2lite.build_resources import parse
from examples.dsv2lite.pinned import FLEET_COMMIT, FLEET_SRC_SHA256, fleet_export

ROOT = Path(__file__).resolve().parents[2]
# The default arm's record (its isa.bin hash equals effb06e's default build): every lever run checks against it.
BASELINE = ROOT / "examples/dsv2lite/results/levers/default.json"


def run(cmd: list[str], **kw) -> str:
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    if p.returncode:
        sys.stdout.write(p.stdout)
        raise SystemExit(f"FAILED ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def static_counts(asm: Path) -> dict[str, int]:
    text = asm.read_text()
    return {"nt_weight_loads": len(re.findall(r"(?:flat|global)_load_dwordx4[^\n]*\bnt\b", text)),
            "vmcnt0_waits": len(re.findall(r"s_waitcnt[^\n]*vmcnt\(0\)", text)),
            "asm_lines": text.count("\n")}


def build(arm: str, out: Path) -> dict:
    run(["bash", "examples/dsv2lite/rocm_build_check.sh", str(fleet_export()), arm, str(out)])
    d = out / arm
    return dict(resources=parse((d / "resource.log").read_text()),
                isa_sha256=hashlib.sha256((d / "isa.bin").read_bytes()).hexdigest(),
                static=static_counts(d / "mk.s"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm")
    ap.add_argument("--out", type=Path, default=ROOT / "build/verify_lever")
    a = ap.parse_args(argv)
    levers = [] if a.arm == "default" else a.arm.split("+")
    record = dict(arm=a.arm, etx_commit=run(["git", "rev-parse", "HEAD"]).strip(),
                  etx_dirty=bool(run(["git", "status", "--porcelain", "--", ".", ":!examples/dsv2lite/results/levers"]).strip()),
                  fleet_commit=FLEET_COMMIT, fleet_src_sha256=FLEET_SRC_SHA256)
    tests = [f"tests/test_dsv2_lever_{name}.py" for name in levers if name != "stamps"] + ["tests/test_dsv2_stamps.py"]
    out = run([sys.executable, "-m", "pytest", "-q", "-rs", *tests])
    summary = out.strip().splitlines()[-1]
    if "skipped" in summary or "failed" in summary or "passed" not in summary:
        raise SystemExit(f"CPU gates not clean: {summary}")
    record["cpu_gates"] = dict(files=tests, summary=summary)
    default = build("default", a.out)
    if BASELINE.exists():
        want = json.loads(BASELINE.read_text())["default"]["isa_sha256"]
        if default["isa_sha256"] != want:
            raise SystemExit(f"default ISA changed: {default['isa_sha256']} != recorded {want}")
    record["default"] = default
    if levers:
        arm = build(a.arm, a.out)
        record["arm_build"] = arm
        record["delta_vs_default"] = {k: arm["resources"][k] - default["resources"][k] for k in default["resources"]}
        record["static_delta"] = {k: arm["static"][k] - default["static"][k] for k in default["static"]}
        if arm["resources"]["scratch"] > default["resources"]["scratch"]:
            record["verdict"] = f"REJECTED: adds scratch, {arm['resources']['scratch']} B/lane"
    record.setdefault("verdict", "PASS")
    path = ROOT / f"examples/dsv2lite/results/levers/{a.arm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if record["verdict"] != "PASS":
        print(f"wrote {path.relative_to(ROOT)}")
        raise SystemExit(f"arm {a.arm} {record['verdict']}")
    print(f"VERIFY arm={a.arm} gates='{summary}' default_isa={default['isa_sha256'][:16]}")
    print("arm | VGPR | AGPR | SGPR | scratch | LDS | waves/SIMD | NT loads | vmcnt(0)")
    rows = [("default", default)] + ([(a.arm, record["arm_build"])] if levers else [])
    for name, b in rows:
        r, s = b["resources"], b["static"]
        print(f"{name} | {r['vgpr']} | {r['agpr']} | {r['sgpr']} | {r['scratch']} | {r['lds']} | {r['occupancy']} | "
              f"{s['nt_weight_loads']} | {s['vmcnt0_waits']}")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
