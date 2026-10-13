"""Re-derive a ds_inplace session's verdicts from its raw per-invocation logs (not from its JSON).

  python -m examples.dsv2lite.rederive examples/dsv2lite/results/2026-10-13-ds_inplace [--stamps DIR]

1. SHA256SUMS: every listed file present here must match (files archived outside git, the STAMPS2 stamp files, are
   looked up in --stamps DIR, default $ETX_CACHE/results_<dirname> (pinned.CACHE), and otherwise reported as ARCHIVED).
2. Identity: source.json (ETX commit, fleet commit + source hash) and inputs.json (input hashes) printed.
3. Pairs: every <arm>_r<i>_<name>.raw.log is parsed with the session's own gate (inplace_session.full_result: 32/32,
   27/27, warm-up) and the paired device gains are recomputed with inplace_session.paired; they must equal the
   session's pairs_<arm>.json to 1e-6 us.
4. Stages (if the stamp files are available): stages_<arm>.json is recomputed from the trace + stamps with
   stage_table.measured and must equal it to 1e-6 us.
Prints one table and exits non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from examples.dsv2lite import inplace_session as ins
from examples.dsv2lite.pinned import CACHE

PAIR_ARMS = {"baseline": ("default", "fleet", ("default",)), "all": ("default", "all", ("default", "fleet")),
             **{a: ("default", a, ("default",)) for a in ins.SINGLE}}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--stamps", type=Path, default=None)
    a = ap.parse_args(argv)
    d = a.session
    stamps_dir = a.stamps or CACHE / f"results_{d.name}"
    bad = []

    archived = []
    for line in (d / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*").removeprefix("./")
        path = d / name if (d / name).exists() else stamps_dir / Path(name).name
        if not path.exists():
            archived.append(name)
        elif sha(path) != digest:
            bad.append(f"checksum {name}")
    print(f"SHA256SUMS: {len(archived)} archived-not-present, mismatches {sum(b.startswith('checksum') for b in bad)}")

    src, inp = json.loads((d / "source.json").read_text()), json.loads((d / "inputs.json").read_text())
    commit = re.search(r"^commit ([0-9a-f]{40})$", src["etx_identity"], re.M)[1]
    print(f"IDENTITY etx {commit[:12]} fleet {src['fleet_commit']} src {src['fleet_src_sha256'][:16]} "
          + " ".join(f"{k}={v[:12]}" for k, v in sorted(inp["sha256"].items())))

    print("PAIRS (device us, gain = reference - candidate; recomputed from the raw logs)")
    for arm, (ref0, cand, refs) in PAIR_ARMS.items():
        recorded = json.loads((d / f"pairs_{arm}.json").read_text())
        names = (ref0, cand) if arm != "all" else ("default", "all", "fleet")
        rounds = [{n: ins.full_result((d / f"{arm}_r{i}_{n}.raw.log").read_text()) for n in names} for i in range(5)]
        for ref in refs:
            got = ins.paired([1000 * (r[ref]["device_ms"] - r[cand]["device_ms"]) for r in rounds])
            want = recorded["device_ms"][ref]
            if any(abs(x - y) > 1e-6 for x, y in zip(got["gains_us"], want["gains_us"])):
                bad.append(f"pairs {arm} vs {ref}")
            go = got["go"] if arm == "all" else None
            med = {n: statistics.median(r[n]["device_ms"] for r in rounds) for n in names}
            print(f"  {arm:8s} vs {ref:7s} median {got['median_us']:7.1f} spread {got['spread_us']:5.1f} "
                  f"every_positive={int(got['every_positive'])} GO={go}  device medians "
                  + " ".join(f"{n}={v:.3f}" for n, v in med.items()))

    for arm in ins.STAMPED:
        st_file, trace = d / f"stages_{arm}.json", d / f"trace_{arm}.trace.bin"
        pro = stamps_dir / f"trace_{arm}.stamps.bin"
        plan = d / f"b_{arm}_stamps" / "plan.json"
        if not (st_file.exists() and trace.exists() and pro.exists() and plan.exists()):
            print(f"STAGES {arm}: skipped (needs {pro.name} from the archive)")
            continue
        from examples.dsv2lite import stage_table as st
        from examples.dsv2lite.phase_table300 import pinned_parser
        pinned_parser()
        us, _, span = st.measured(plan, trace, pro)
        want = json.loads(st_file.read_text())
        if abs(span - want["span_us"]) > 1e-6 or any(abs(us[k] - want["stages_us"][k]) > 1e-6 for k in us):
            bad.append(f"stages {arm}")
        print(f"STAGES {arm}: span {span:.1f} us, recomputed from trace + stamps, equal to stages_{arm}.json")

    fu = d / "followup"
    if (fu / "followup.json").exists():
        want = json.loads((fu / "followup.json").read_text())
        tests = {"e1": ("all", "allnoQ", "drop_q_all", ("default", "all", "allnoQ")),
                 "e2": ("default", "default723", "compiler_costs", ("default", "default723"))}
        for e, (ref, cand, key, names) in tests.items():
            rounds = [{n: ins.full_result((fu / f"{e}_r{i}_{n}.raw.log").read_text()) for n in names} for i in range(5)]
            diffs = [round(1000 * (r[ref]["device_ms"] - r[cand]["device_ms"]), 1) for r in rounds]
            if diffs != want[e][key]["diffs_us"]:
                bad.append(f"followup {e}")
            med = {n: statistics.median(r[n]["device_ms"] for r in rounds) for n in names}
            print(f"FOLLOWUP {e} {ref}-{cand}: median {statistics.median(diffs):.1f} spread {max(diffs) - min(diffs):.1f} "
                  f"diffs {diffs}  device medians " + " ".join(f"{n}={v:.3f}" for n, v in med.items()))

    print("RESULT " + ("OK: every verdict re-derived from raw logs" if not bad else f"MISMATCH {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
