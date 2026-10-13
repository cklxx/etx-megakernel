"""Verify a fetched paid-hour session directory (examples/sessions/ds_hour.py) before the box is destroyed
(review 07, R10: rederive.py covers only the ds_inplace layout).

  .venv/bin/python -m examples.dsv2lite.verify_hour <fetched session dir>

1. the box's SHA256SUMS against the copy (etx/tools/verify_sums.py; binaries and patched sources stay on the box);
2. every pairs_<arm>.json re-derived from its raw invocation logs (<arm>_r<i>_<build>.raw.log, parsed by the same
   full_result() and reported by the same build_report() the session ran): rounds, medians, spreads and the verdict
   must be equal;
3. every stages_<arm>.json re-derived from its trace, stamps and the fetched plan of its stamped build.
Prints one line per arm; exit 0 only if everything matches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from etx.tools.verify_sums import verify
from examples.dsv2lite import inplace_session as ins

EXCLUDED = ["b_*/run", "b_*/fleet_src/*", "*.o"]


def rederive_pairs(d: Path, arm: str) -> tuple[bool, str]:
    stored = json.loads((d / f"pairs_{arm}.json").read_text())
    order, refs, cand = ins.arm_order(arm, fleet_available=(d / "b_fleet").exists())
    rounds = []
    for i in range(5):
        r = {}
        for cur in (order if i % 2 == 0 else tuple(reversed(order))):
            r[cur] = ins.full_result((d / f"{arm}_r{i}_{cur}.raw.log").read_text())
        rounds.append(r)
    again = json.loads(json.dumps(ins.build_report(arm, refs, cand, rounds)))
    keys = ("rounds", "device_ms", "wall_ms", "verdict")
    same = all(again.get(k) == stored.get(k) for k in keys)
    v = again["verdict"]
    first = again["device_ms"][refs[0]]
    return same, (f"{'OK  ' if same else 'DIFF'} pairs {arm:10s} {v['outcome']:22s} vs {refs[0]}: median "
                  f"{first['median_us']:.1f} us spread {first['spread_us']:.1f} every_positive={first['every_positive']}")


def rederive_stages(d: Path, arm: str) -> tuple[bool, str]:
    from examples.dsv2lite import stage_table as st
    stored = json.loads((d / f"stages_{arm}.json").read_text())
    us, n, span = st.measured(d / f"b_{arm}_stamps" / "plan.json", d / f"trace_{arm}.trace.bin",
                              d / f"trace_{arm}.stamps.bin")
    again = json.loads(json.dumps(dict(arm=arm, span_us=span, stages_us=us, instances=n)))
    same = all(again[k] == stored.get(k) for k in again)          # stored may add an attribution note
    return same, f"{'OK  ' if same else 'DIFF'} trace {arm:10s} span {span:.1f} us"


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 1:
        raise SystemExit(__doc__)
    d = Path(a[0])
    rc, lines = verify(d, EXCLUDED)
    print("\n".join(lines))
    ok = rc == 0
    for f in sorted(d.glob("pairs_*.json")):
        same, line = rederive_pairs(d, f.stem.removeprefix("pairs_"))
        ok &= same
        print(line)
    for f in sorted(d.glob("stages_*.json")):
        same, line = rederive_stages(d, f.stem.removeprefix("stages_"))
        ok &= same
        print(line)
    print("HOUR VERIFIED: every verdict and stage table re-derived from the raw files" if ok else "HOUR NOT VERIFIED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
