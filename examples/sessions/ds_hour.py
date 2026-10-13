"""The next paid hour (plan: docs/DSV2-D1-D5.md, "The next paid hour"; every rule fixed in that file before its code).

GPU order (one pairs invocation 5.9 s, one traced token 7-8 s, from the 2026-10-13 runner records):
  pairs_l1         D2b rerun, 3-way default / relaxed / l1, D2's rule           15 inv, ~88 s
  pairs_alld4r0l1  best arm + D2b, 3-way default / alld4r0 / alld4r0l1, KEEP    15 inv, ~88 s
  pairs_alld3      D3, 2-way all / alld3, GO-or-DROP                            10 inv, ~59 s
  pairs_alld5      D5, 2-way all / alld5, KEEP-or-DROP (drop if neutral)        10 inv, ~59 s
  trace_default, trace_alld4r0, trace_alld4r0l1   stamped tokens: the per-stage record   3 inv, ~24 s
  pairs_alld4s     optional, last: Astra's batched D4 slices, 3-way default / alld4r0 / alld4s   15 inv, ~88 s
then a CPU job hashes the session directory ON THE BOX (SHA256SUMS) before anything is fetched.
The full prep chain of ds_inplace.py runs first (fresh droplet: 3 min 59 s to preflight on 2026-10-13); every build
runs in parallel during it. Every GPU job gates hard on preflight and on each binary it runs.
Run: examples/dsv2lite/stage_inplace.sh <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=1800 tools/session.sh start <ip> examples/sessions/ds_hour.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 1800
PREP = ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "prep_sums", "preflight",
        "build_default", "build_all", "build_relaxed", "build_default_stamps")
JOBS = [dict(d) for d in _full.JOBS if d["name"] in PREP]
_full.JOBS.clear()
add = _full.add
for arm in ("alld4r0", "l1", "alld4r0l1", "alld3", "alld5", "alld4s"):
    add(f"build_{arm}", f"build {arm}", after=("source",), timeout=1200)
for arm in ("alld4r0", "alld4r0l1"):
    add(f"build_{arm}_stamps", f"build {arm} --stamped", after=("source",), timeout=1200)

GPU = [  # (job, operation, binaries it runs)
    ("pairs_l1", "pairs l1", ("default", "relaxed", "l1")),
    ("pairs_alld4r0l1", "pairs alld4r0l1", ("default", "alld4r0", "alld4r0l1")),
    ("pairs_alld3", "pairs alld3", ("all", "alld3")),
    ("pairs_alld5", "pairs alld5", ("all", "alld5")),
    ("trace_default", "trace default", ("default_stamps",)),
    ("trace_alld4r0", "trace alld4r0", ("alld4r0_stamps",)),
    ("trace_alld4r0l1", "trace alld4r0l1", ("alld4r0l1_stamps",)),
    ("pairs_alld4s", "pairs alld4s", ("default", "alld4r0", "alld4s")),
]
earlier = []
for name, op, bins in GPU:
    add(name, op, lane="gpu", after=("preflight", *(f"build_{b}" for b in bins)), timeout=1200)
    if earlier:
        # strictly in this order, whatever each result: after ALL earlier GPU jobs, so a skip in the middle cannot let
        # a later arm run before an earlier one that is still waiting for its build (review 07, R07)
        _full.JOBS[-1]["after_any"] = list(earlier)
    earlier.append(name)
# box-side hashes of everything the session wrote, before any fetch (2026-10-13: no run directory had them)
# the runner's own live files (runner.log, status.json, startup.* -- review 07, R05) are written after this job; it
# waits for EVERY other job, not only the measurements, so no build is still writing (R06)
_full.job("hash_session", 'cd "$ETX_SESS_LOGDIR" && find . -type f ! -name SHA256SUMS ! -name runner.log '
          "! -name status.json ! -name 'startup.*' ! -name 'hash_session*' -print0 | sort -z | xargs -0 sha256sum "
          '> SHA256SUMS && wc -l SHA256SUMS', after=(), timeout=300)
_full.JOBS[-1]["after_any"] = [d["name"] for d in JOBS + _full.JOBS[:-1]]
JOBS += _full.JOBS
