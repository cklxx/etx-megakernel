"""D2b session (thresholds: docs/DSV2-D1-D5.md, "D2b"): l1 (three-way default / relaxed / l1, D2's rule) and
alld4r0l1 (three-way default / alld4r0 / alld4r0l1, KEEP vs alld4r0). The full prep chain of ds_inplace.py runs first
(idempotent on a box that already has the inputs), so this file works on a fresh droplet.
Run: examples/dsv2lite/stage_inplace.sh <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=3600 tools/session.sh start <ip> examples/sessions/ds_d2b.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 3600
PREP = ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "prep_sums", "preflight",
        "build_default", "build_all", "build_relaxed")
JOBS = [dict(d) for d in _full.JOBS if d["name"] in PREP]
_full.JOBS.clear()
for arm in ("alld4r0", "l1", "alld4r0l1"):
    _full.add(f"build_{arm}", f"build {arm}", after=("source",), timeout=1200)
_full.add("pairs_l1", "pairs l1", lane="gpu", after=("preflight", "build_default", "build_relaxed", "build_l1"),
          timeout=1200)
_full.add("pairs_alld4r0l1", "pairs alld4r0l1", lane="gpu",
          after=("preflight", "build_default", "build_alld4r0", "build_alld4r0l1"), timeout=1200)
_full.JOBS[-1]["after_any"] = ["pairs_l1"]
JOBS += _full.JOBS
