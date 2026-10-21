"""Refuse a D2 build whose plan and tiles disagree (docs/deepseek/DSV2-D1-D5.md, D2).

  python -m examples.dsv2lite.coh_check <megakernel_d0.hip> <fleet src dir the build compiles>

A plan generated with ETX_DSV2_COH=1 drops the acquire invalidate of fleet's tiles; that is correct only if the tiles
load every payload coherently (lever coherent_loads). The reverse (patched tiles, plain plan) is correct but is not an
arm anyone asked for and would be measured under the wrong name. Both mismatches exit 2 before the compiler runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# (plan mark in megakernel_d0.hip, tile mark in the patched gemv.h, lever, switch value): D2 and D2b
MODES = [("coherent-load tile: no acquire invalidate", "#define FLEET_COH_LOADS 1", "coherent_loads", "1"),
         ("coherent-L1 tile: no acquire invalidate", "#define FLEET_L1_LOADS 1", "l1_loads", "l1")]
PLAN_COHERENCE = {"coherent_loads": "coherent_loads", "l1_loads": "coherent_l1"}   # lever -> plan.json types[*].coherence


def state(megakernel: Path, src: Path) -> list[tuple[bool, bool]]:
    gemv = (src / "kernels/gemv.h").read_text()
    # The plan's own record of each task type's coherence is authoritative. The kernel comment is the fallback for a
    # build without plan.json: large graphs (>= 512 types) compact their dispatch switches and drop those comments
    # (etx/codegen/kernel.py _compact_dispatch), so x4 / x8 kernels carry no mark although their tiles are marked.
    plan_json = megakernel.with_name("plan.json")
    if plan_json.is_file():
        marked = {t.get("coherence") for t in json.loads(plan_json.read_text())["types"].values()}
        return [(PLAN_COHERENCE[lever] in marked, tile in gemv) for _, tile, lever, _ in MODES]
    mk = megakernel.read_text()
    return [(plan in mk, tile in gemv) for plan, tile, _, _ in MODES]


def check(megakernel: Path, src: Path) -> str:
    on = []
    for (plan, tiles), (_, _, lever, value) in zip(state(megakernel, src), MODES):
        if plan and not tiles:
            raise SystemExit(f"COH-CHECK REFUSED: the plan drops the acquires of its tiles (ETX_DSV2_COH={value}) but "
                             f"the fleet tiles are not patched with {lever}: they would read stale cached payloads")
        if tiles and not plan:
            raise SystemExit(f"COH-CHECK REFUSED: the fleet tiles are patched with {lever} but the plan keeps every "
                             f"acquire (ETX_DSV2_COH={value} unset): not that arm")
        on.append(int(plan))
    if sum(on) > 1:
        raise SystemExit("COH-CHECK REFUSED: two coherence modes in one build")
    return "COH-CHECK OK " + " ".join(f"{m[2]}={o}" for m, o in zip(MODES, on))


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2:
        raise SystemExit(__doc__)
    try:
        print(check(Path(a[0]), Path(a[1])), flush=True)
    except SystemExit as e:
        print(e, file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
