"""Local verification record of one lever arm (no GPU): CPU gates, ROCm 7.2.4 resources, default-unchanged proof.

  python -m examples.dsv2lite.verify_lever <arm>      # lever, lever+lever, or alld5 (five levers + compiler D5)
  python -m examples.dsv2lite.verify_lever <arm> --compiler 7.2.4|7.14   # local ROCm, no docker (tools/rocm_build_local.sh)
  python -m examples.dsv2lite.verify_lever --registered d4 [--name d4]  # a registered arms.py arm (levers + flags + env)

Compiler modes: `docker` (default) is rocm_build_check.sh with the etx-rocm22 image and ~/code/rocm-headers (ROCm 7.2.4).
`7.2.4` and `7.14` run the same steps with a locally installed ROCm (tools/rocm_build_local.sh; roots ETX_ROCM_724,
ETX_ROCM_714). Every record has a `compiler` field: mode (docker | local), toolchain, compiler path, clang version line
and clang commit (docker: also the image id), so records from different compilers are never mistaken for each other.
In local mode a default ISA that differs from the docker baseline is recorded (`default_isa_matches_baseline`), not fatal.

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
import os
from pathlib import Path
import re
import subprocess
import sys

from examples.dsv2lite.build_resources import parse
from examples.dsv2lite.pinned import FLEET_COMMIT, FLEET_SRC_SHA256, fleet_export, source_sha

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
            "asm_lines": text.count("\n"),
            # D2: acquire invalidates, L2 write-backs and system-scope (sc0 sc1) loads in the megakernel
            "inv_sc1": len(re.findall(r"^\s*buffer_inv sc1\s*$", text, re.M)),
            "inv_sc0": len(re.findall(r"^\s*buffer_inv sc0\s*$", text, re.M)),
            "wbl2": len(re.findall(r"^\s*buffer_wbl2\b", text, re.M)),
            "sys_loads": len(re.findall(r"^\s*global_load\w*[^\n]*\bsc0 sc1\b", text, re.M))}


# lever -> the plan switch it needs (the lever and the plan are checked against each other by coh_check.py)
LEVER_ENV = {"coherent_loads": {"ETX_DSV2_COH": "1"}, "l1_loads": {"ETX_DSV2_COH": "l1"},
             "kchunk352": {"ETX_DSV2_KCHUNKS": "4", "ETX_CFLAGS": "-DETX_DSV2_KCHUNKS=1"}}
ARM_LEVERS = {"alld5": "gemv_no_terminal_all+norm_tree_all+argmax_tree+topk_rank+attn_prologue_live"}


COMPILERS = ("docker", "7.2.4", "7.14")
LOCAL_SCRIPT = "tools/rocm_build_local.sh"


def build_cmd(arm: str, out: Path, compiler: str, registered: bool = False, fleet: Path | None = None) -> list[str]:
    """The build command of one arm. registered: `arm` is an arms.py name (levers + flags + env from the registry)."""
    if compiler not in COMPILERS:
        raise ValueError(f"compiler {compiler!r}: one of {COMPILERS}")
    fleet = str(fleet or fleet_export())
    target = ["--arm", arm] if registered else [arm]
    if compiler == "docker":
        return ["bash", "examples/dsv2lite/rocm_build_check.sh", fleet, *target, str(out)]
    return ["bash", LOCAL_SCRIPT, compiler, fleet, *target, str(out)]


def compiler_record(d: Path, compiler: str) -> dict:
    """mode, toolchain, compiler path, clang version line and clang commit of the build in directory d."""
    if compiler != "docker":
        return json.loads((d / "compiler.json").read_text())
    first = (d / "compiler.txt").read_text().splitlines()[0] if (d / "compiler.txt").exists() else ""
    m = re.search(r"([0-9a-f]{40})", first)
    image = d.parent / "image.txt"
    return {"mode": "docker", "toolchain": "7.2.4", "image": image.read_text().strip() if image.exists() else None,
            "compiler_path": "/rocm/rocm-llvm/opt/rocm-7.2.4/lib/llvm/bin/clang++ (etx-rocm22)",
            "rocm_root": "~/code/rocm-headers", "clang_version": first, "clang_commit": m.group(1) if m else None}


def fleet_src_record(fleet: Path) -> dict:
    """Path and pinned.py source hash of the fleet export a build uses; refuses (SystemExit) when it is not the pin."""
    got = source_sha(Path(fleet) / "src")
    if got != FLEET_SRC_SHA256:
        raise SystemExit(f"fleet source {fleet}/src is not the pinned fleet {FLEET_COMMIT}: {got} != {FLEET_SRC_SHA256}")
    return {"path": str(fleet), "commit": FLEET_COMMIT, "src_sha256": got}


def build(arm: str, out: Path, compiler: str = "docker", registered: bool = False) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("ETX_DSV2_COH", "ETX_DSV2_KCHUNKS", "ETX_CFLAGS", "ETX_DSV2_D5")}
    build_arm = arm if registered else ARM_LEVERS.get(arm, arm)
    if arm == "alld5" and not registered:
        env["ETX_DSV2_D5"] = "1"
        out = out / arm
    if not registered:
        for lever in build_arm.split("+"):
            env.update(LEVER_ENV.get(lever, {}))
    if compiler != "docker":
        env["ETX_PYTHON"] = sys.executable
    fleet = fleet_export()
    fleet_src = fleet_src_record(fleet)
    run(build_cmd(build_arm, out, compiler, registered, fleet=fleet), env=env)
    d = out / build_arm
    return dict(compiler=compiler_record(d, compiler), fleet_src=fleet_src, resources=parse((d / "resource.log").read_text()),
                plan_sha256=hashlib.sha256((d / "plan.json").read_bytes()).hexdigest(),
                kernel_sha256=hashlib.sha256((d / "megakernel_d0.hip").read_bytes()).hexdigest(),
                isa_sha256=hashlib.sha256((d / "isa.bin").read_bytes()).hexdigest(),
                static=static_counts(d / "mk.s"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--registered", metavar="ARM", help="verify a registered arms.py arm (for example d4, alld4s)")
    ap.add_argument("--name", help="record name (results/levers/<name>.json); default: the arm")
    ap.add_argument("--compiler", choices=COMPILERS, default="docker")
    ap.add_argument("--record", type=Path, help="write the record here instead of results/levers/<name>.json")
    ap.add_argument("--keep-prior", action="store_true",
                    help="keep the existing record's content under `prior_record` (records of another format)")
    ap.add_argument("--out", type=Path, default=ROOT / "build/verify_lever")
    a = ap.parse_args(argv)
    if (a.arm is None) == (a.registered is None):
        ap.error("give exactly one of ARM or --registered")
    registered = a.registered is not None
    arm_name = a.registered if registered else a.arm
    name = a.name or arm_name
    if registered:
        from examples.dsv2lite.arms import ARMS
        spec = ARMS[arm_name]
        levers = list(spec.levers)
        record_extra = {"registered_arm": arm_name, "flags": list(spec.flags), "plan_env": dict(spec.env)}
    else:
        levers = [] if a.arm == "default" else ARM_LEVERS.get(a.arm, a.arm).split("+")
        record_extra = {}
    out_dir = a.out / a.compiler
    record = dict(arm=name, etx_commit=run(["git", "rev-parse", "HEAD"]).strip(),
                  etx_dirty=bool(run(["git", "status", "--porcelain", "--", ".", ":!examples/dsv2lite/results/levers"]).strip()),
                  fleet_commit=FLEET_COMMIT, fleet_src_sha256=FLEET_SRC_SHA256, **record_extra)
    tests = [f"tests/test_dsv2_lever_{lv}.py" for lv in levers if lv != "stamps"] + ["tests/test_dsv2_stamps.py"]
    if registered and any(f.startswith("-DETX_DSV2_D4") for f in spec.flags):
        tests += ["tests/test_dsv2_d4.py"]
    if a.arm == "alld5":
        tests += ["tests/test_dsv2_d5.py"]
        record["plan_env"] = {"ETX_DSV2_D5": "1"}
    out = run([sys.executable, "-m", "pytest", "-q", "-rs", *tests])
    summary = out.strip().splitlines()[-1]
    if "skipped" in summary or "failed" in summary or "passed" not in summary:
        raise SystemExit(f"CPU gates not clean: {summary}")
    record["cpu_gates"] = dict(files=tests, summary=summary)
    default = build("default", out_dir, a.compiler)
    record["compiler"] = default["compiler"]
    record["fleet_src"] = default["fleet_src"]
    if BASELINE.exists():
        want = json.loads(BASELINE.read_text())["default"]["isa_sha256"]
        if a.compiler == "docker" and default["isa_sha256"] != want:
            raise SystemExit(f"default ISA changed: {default['isa_sha256']} != recorded {want}")
        if a.compiler != "docker":
            record["default_isa_matches_baseline"] = default["isa_sha256"] == want
    record["default"] = default
    if levers or registered:
        arm = build(arm_name, out_dir, a.compiler, registered)
        record["arm_build"] = arm
        record["delta_vs_default"] = {k: arm["resources"][k] - default["resources"][k] for k in default["resources"]}
        record["static_delta"] = {k: arm["static"][k] - default["static"][k] for k in default["static"]}
        if arm["resources"]["scratch"] > default["resources"]["scratch"]:
            record["verdict"] = f"REJECTED: adds scratch, {arm['resources']['scratch']} B/lane"
    record.setdefault("verdict", "PASS")
    path = a.record or ROOT / f"examples/dsv2lite/results/levers/{name}.json"
    old = ROOT / f"examples/dsv2lite/results/levers/{name}.json"
    if a.keep_prior and old.exists():
        prior = json.loads(old.read_text())
        record["prior_record"] = prior.get("prior_record", prior)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if record["verdict"] != "PASS":
        print(f"wrote {path}")
        raise SystemExit(f"arm {name} {record['verdict']}")
    print(f"VERIFY arm={name} compiler={a.compiler} gates='{summary}' default_isa={default['isa_sha256'][:16]}")
    print("arm | VGPR | AGPR | SGPR | scratch | LDS | waves/SIMD | NT loads | vmcnt(0)")
    rows = [("default", default)] + ([(name, record["arm_build"])] if "arm_build" in record else [])
    for name, b in rows:
        r, s = b["resources"], b["static"]
        print(f"{name} | {r['vgpr']} | {r['agpr']} | {r['sgpr']} | {r['scratch']} | {r['lds']} | {r['occupancy']} | "
              f"{s['nt_weight_loads']} | {s['vmcnt0_waits']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
