"""Historical E1/E2 follow-up to ds_inplace: drop Q-all, then compare the same default sources under ROCm 7.2.3.

Builds, invocation orders and rules are registered in arms.py; measure/decide run the same five interleaved rounds
as the main session. E2 uses the box's existing rocm:latest image with its host runtime (libamdhip64.so.7).

  ETX_SESS_LOGDIR=~/sess_followup ETX_DSV2_FLEET=~/fleet-3fb67a6 ETX_DSV2_DATA=~/dsv2-data \
    .venv/bin/python -u -m examples.dsv2lite.followup_2026_10_13 /root/sess
"""
from __future__ import annotations

import json
import shutil
import statistics

from etx.session.experiment import measure, ns
from etx.models.declarations import declaration as _declaration
from . import paths
ARMS = _declaration(paths.ROOT, "dsv2lite.arms").ARMS
FOLLOWUPS = _declaration(paths.ROOT, "dsv2lite.arms").FOLLOWUPS
followup_decision = _declaration(paths.ROOT, "dsv2lite.arms").followup_decision
import sys
from pathlib import Path

from . import session as ins
from .build_resources import parse


def run(previous: str) -> int:
    s = ins.Session()
    prev = Path(previous)
    for arm in ("default", "all"):                     # the session's own binaries (same commit, same compiler)
        shutil.copytree(prev / f"b_{arm}", s.out(arm))
    s.build("allnoQ")                                   # examples/dsv2lite/arms.py: all without gemv_no_terminal_all
    # E2's external compiler build is explicitly registered (and rejected by the ETX-only local build check).
    src, dst = s.out("default"), s.out("default723")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("run", "*.o"))
    cmd = ("hipcc --version | head -2 && hipcc -O3 -std=c++17 --offload-arch=gfx942 -DFLEET_NT_WEIGHTS=1 "
           "-Rpass-analysis=kernel-resource-usage -I /root/fleet-3fb67a6/src -I /root/etx/etx/runtime/include "
           f"-I {dst} -I /root/etx {dst}/megakernel_d0.hip /root/etx/examples/dsv2lite/host.hip -o {dst}/run")
    text = s.run("build_default723", ["docker", "run", "--rm", "-v", "/root:/root", "-w", "/root/etx",
                                       "--entrypoint", "bash", ARMS["default723"].builder, "-c", cmd], timeout=900)
    res = parse(text)
    ins.write_json(dst / "resources.json", res)
    print(f"BUILD default723 (ROCm 7.2.3) resources={res}", flush=True)
    if res["scratch"] > 0:
        raise RuntimeError("default723 spills to scratch")
    result = {}
    for tag, (name, fields) in FOLLOWUPS.items():
        arm = ARMS[name]
        rs = measure(arm, s.full_invocation, tag=tag)
        result[tag] = dict(rounds=rs, medians_ms={a: statistics.median(ns(r[a]["device_ms"]) for r in rs) / 1e6
                                                for a in rs[0]})
        for key, ref in fields.items():
            result[tag][key] = followup_decision(arm, rs, ref)
            print(f"FOLLOWUP {tag} {ref} - {arm.cand}: {result[tag][key]}", flush=True)
    result["e2"]["resources_723"] = res
    ins.write_json(s.logs / "followup.json", result)
    print("FOLLOWUP medians " + json.dumps({k: result[k]["medians_ms"] for k in ("e1", "e2")}), flush=True)
    return 0


def main() -> int:
    return run(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
