"""Rerun of the arms the 2026-10-13 d1d5 run of ds_inplace.py did not measure (same job definitions, imported).

That run lost build_coh / build_d4 to a session bug (fixed in inplace_session.build) and, through a job-graph bug (fixed
in ds_inplace.py: every GPU measurement now gates hard on preflight), started the single-lever pairs before the data
existed. baseline, pairs_all and both traces completed and are kept in ~/sess_run1. This session reuses the prepared
inputs on the box (~/dsv2-data, re-verified by preflight against its SHA256SUMS) and runs, in order: pairs_coh,
pairs_d4, d4trace, then the eight single-arm pairs (relaxed last). No report job: report needs pairs_all, which is in
run 1; the two runs are combined after fetch.

Run: examples/dsv2lite/stage_inplace.sh <ip> && ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=3600 \\
     tools/session.sh start <ip> examples/sessions/ds_rerun_d1d5.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 3600
GPU_ORDER = ["pairs_coh", "pairs_d4", "d4trace", *(f"pairs_{a}" for a in _full.SINGLE)]
ARMS_BUILT = ["default", "relaxed", "coh", "d4", *(a for a in _full.SINGLE if a != "relaxed")]
KEEP = {"etx_env", "source", "preflight", *(f"build_{a}" for a in ARMS_BUILT), *GPU_ORDER}

JOBS = []
for d in _full.JOBS:
    if d["name"] not in KEEP:
        continue
    d = dict(d)
    d["after"] = [a for a in d["after"] if a in KEEP]
    if "after_any" in d:
        d["after_any"] = [a for a in d["after_any"] if a in KEEP]
    if d["name"] == "preflight":
        d["after"] = ["source"]          # the inputs exist; preflight re-verifies every file against SHA256SUMS
    JOBS.append(d)
assert [d["name"] for d in JOBS if d["lane"] == "gpu"] == GPU_ORDER
