"""Refuse a D2 build whose plan and tiles disagree (docs/DSV2-D1-D5.md, D2).

  python -m examples.dsv2lite.coh_check <megakernel_d0.hip> <fleet src dir the build compiles>

A plan generated with ETX_DSV2_COH=1 drops the acquire invalidate of fleet's tiles; that is correct only if the tiles
load every payload coherently (lever coherent_loads). The reverse (patched tiles, plain plan) is correct but is not an
arm anyone asked for and would be measured under the wrong name. Both mismatches exit 2 before the compiler runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

PLAN_MARK = "coherent-load tile: no acquire invalidate"
TILE_MARK = "#define FLEET_COH_LOADS 1"


def state(megakernel: Path, src: Path) -> tuple[bool, bool]:
    return PLAN_MARK in megakernel.read_text(), TILE_MARK in (src / "kernels/gemv.h").read_text()


def check(megakernel: Path, src: Path) -> str:
    plan, tiles = state(megakernel, src)
    if plan and not tiles:
        raise SystemExit("COH-CHECK REFUSED: the plan drops the acquires of coherent-load tiles (ETX_DSV2_COH=1) but the "
                         "fleet tiles are not patched with coherent_loads: they would read stale cached payloads")
    if tiles and not plan:
        raise SystemExit("COH-CHECK REFUSED: the fleet tiles are patched with coherent_loads but the plan keeps every "
                         "acquire (ETX_DSV2_COH unset): not the coh arm")
    return f"COH-CHECK OK coherent={int(plan)}"


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
