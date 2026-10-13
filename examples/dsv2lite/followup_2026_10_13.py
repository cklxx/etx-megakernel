"""Follow-up on the 2026-10-13 ds_inplace box (same box, same inputs, rules fixed before the run).

E1, is gemv_no_terminal_all dead weight in the combination?  Arms: default, all, allnoQ (all without Q-all).
   Rule: drop Q-all if every paired (all - allnoQ) device difference is > 0 and its median > 2 x its spread.
E2, does the compiler explain ETX's body regression?  Arms: default (ROCm 7.14) and default723: the SAME generated
   sources compiled with the ROCm 7.2.3 hipcc of the box's rocm:latest image, run on the host's 7.14 runtime
   (same soname libamdhip64.so.7), so only the compiler differs.
   Rule: the 7.14 compiler costs time if every paired (default - default723) difference is > 0 and median > 2 x spread.
5 interleaved rounds each, order reversed on odd rounds, every invocation 32/32 + 27/27 (inplace_session.full_result).

  ETX_SESS_LOGDIR=~/sess_followup ETX_DSV2_FLEET=~/fleet-3fb67a6 ETX_DSV2_DATA=~/dsv2-data \
    .venv/bin/python -u -m examples.dsv2lite.followup_2026_10_13 /root/sess
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
from pathlib import Path

from examples.dsv2lite import inplace_session as ins
from examples.dsv2lite.build_resources import parse

ALL_NO_Q = tuple(x for x in ins.LEVERS_ALL if x != "gemv_no_terminal_all")


def verdict(name: str, diffs_us: list[float]) -> dict:
    median, spread = statistics.median(diffs_us), max(diffs_us) - min(diffs_us)
    v = dict(test=name, diffs_us=diffs_us, median_us=median, spread_us=spread,
             holds=min(diffs_us) > 0 and median > 2 * spread)
    print(f"FOLLOWUP {name}: median {median:.1f} us spread {spread:.1f} diffs {diffs_us} holds={v['holds']}", flush=True)
    return v


def rounds(s: ins.Session, tag: str, order: tuple[str, ...]) -> list[dict]:
    out = []
    for i in range(5):
        r = {}
        for arm in (order if i % 2 == 0 else tuple(reversed(order))):
            r[arm] = s.full_invocation(arm, f"{tag}_r{i}_{arm}")
        out.append(r)
    return out


def main(previous: str) -> int:
    s = ins.Session()
    prev = Path(previous)
    for arm in ("default", "all"):                     # the session's own binaries (same commit, same compiler)
        shutil.copytree(prev / f"b_{arm}", s.out(arm))
    ins.ARMS["allnoQ"] = ALL_NO_Q
    s.build("allnoQ")
    # E2 build: the default arm's generated sources, compiled by ROCm 7.2.3 in the image, flags as build.sh.
    src, dst = s.out("default"), s.out("default723")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("run", "*.o"))
    cmd = ("hipcc --version | head -2 && hipcc -O3 -std=c++17 --offload-arch=gfx942 -DFLEET_NT_WEIGHTS=1 "
           "-Rpass-analysis=kernel-resource-usage -I /root/fleet-3fb67a6/src -I /root/etx/etx/runtime/include "
           f"-I {dst} -I /root/etx {dst}/megakernel_d0.hip /root/etx/examples/dsv2lite/host.hip -o {dst}/run")
    text = s.run("build_default723", ["docker", "run", "--rm", "-v", "/root:/root", "-w", "/root/etx",
                                       "--entrypoint", "bash", "rocm:latest", "-c", cmd], timeout=900)
    res = parse(text)
    ins.write_json(dst / "resources.json", res)
    print(f"BUILD default723 (ROCm 7.2.3) resources={res}", flush=True)
    if res["scratch"] > 0:
        raise RuntimeError("default723 spills to scratch")
    e1 = rounds(s, "e1", ("default", "all", "allnoQ"))
    e2 = rounds(s, "e2", ("default", "default723"))
    dev = lambda rs, a, b: [round(1000 * (r[a]["device_ms"] - r[b]["device_ms"]), 1) for r in rs]
    med = lambda rs, a: statistics.median(r[a]["device_ms"] for r in rs)
    result = dict(
        e1=dict(rounds=e1, medians_ms={a: med(e1, a) for a in e1[0]},
                drop_q_all=verdict("E1 all - allnoQ", dev(e1, "all", "allnoQ")),
                allnoQ_vs_default=verdict("E1 default - allnoQ", dev(e1, "default", "allnoQ"))),
        e2=dict(rounds=e2, medians_ms={a: med(e2, a) for a in e2[0]}, resources_723=res,
                compiler_costs=verdict("E2 default - default723", dev(e2, "default", "default723"))))
    ins.write_json(s.logs / "followup.json", result)
    print("FOLLOWUP medians " + json.dumps({k: result[k]["medians_ms"] for k in ("e1", "e2")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
