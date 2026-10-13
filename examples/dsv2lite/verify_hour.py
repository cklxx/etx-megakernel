"""Verify a fetched paid-hour session directory (examples/sessions/ds_hour.py) before the box is destroyed
(review 07, R10: rederive.py covers only the ds_inplace layout).

  .venv/bin/python -m examples.dsv2lite.verify_hour <fetched session dir>

1. the copy against the record's own SEAL and MANIFEST (etx/session/record.py; binaries stay on the box);
2. every pairs_<arm>.json re-derived from its raw invocation logs (<arm>_r<i>_<build>.raw.log, parsed by the same
   full_result() and reported by the same build_report() the session ran): rounds, medians, spreads and the verdict
   must be equal;
3. every stages_<arm>.json re-derived from its trace, stamps and the fetched plan of its stamped build.
Prints one line per arm. Exit 0 = every promised result present and re-derived; 3 = what exists verifies but a
promised result is missing (an incomplete hour); 2 = something does not match the box or does not re-derive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from etx.session import record
from examples.dsv2lite import inplace_session as ins


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


def expected_outputs() -> list[str]:
    """Every result file examples/sessions/ds_hour.py promises: one pairs report per pairs job, one stage table per trace
    job. A missing one is an incomplete hour, never a verified one (review 08, R13)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ds_hour", Path(__file__).resolve().parents[1] / "sessions/ds_hour.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = []
    for name, op in mod.GPU:
        kind, arm = op.split()
        out.append(f"pairs_{arm}.json" if kind == "pairs" else f"stages_{arm}.json")
    return out


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 1:
        raise SystemExit(__doc__)
    d = Path(a[0])
    rc, lines = record.verify(d)
    print("\n".join(lines))
    if rc == 2 or not (d / "SEAL").is_file():               # not what the box sealed, or no seal: not a result
        print("HOUR NOT VERIFIED")
        return 2
    ok = True
    for f in sorted(d.glob("pairs_*.json")):
        same, line = rederive_pairs(d, f.stem.removeprefix("pairs_"))
        ok &= same
        print(line)
    for f in sorted(d.glob("stages_*.json")):
        same, line = rederive_stages(d, f.stem.removeprefix("stages_"))
        ok &= same
        print(line)
    missing = [f for f in expected_outputs() if not (d / f).exists()]
    for f in missing:
        print(f"MISSING {f} (its job did not complete: see runner.log / status.json)")
    if not ok:
        print("HOUR NOT VERIFIED")
        return 2
    if missing or rc:
        print(f"HOUR INCOMPLETE: what exists is verified; {len(missing)} promised result(s) missing")
        return 3
    print("HOUR VERIFIED: every promised verdict and stage table present and re-derived from the raw files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
