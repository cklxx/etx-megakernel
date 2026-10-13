"""Two measurement arms after the d1d5 results (thresholds: docs/DSV2-D1-D5.md, "Next two arms"): allrelaxed (all
levers + the unsafe relaxed acquire, three-way with default and all) and d4r0 (D4 placement, no slices). Same job
definitions as ds_inplace.py; inputs reused from the box.
Run: ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=1800 tools/session.sh start <ip> examples/sessions/ds_next_d1d5.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 1800
JOBS = []
for d in _full.JOBS:
    if d["name"] in ("etx_env", "source", "build_default", "build_all"):
        d = dict(d); d["after"] = [a for a in d["after"] if a != "apt"]; JOBS.append(d)
    if d["name"] == "preflight":
        d = dict(d); d["after"] = ["source"]; JOBS.append(d)
_full.JOBS.clear()
_full.add("build_allrelaxed", "build allrelaxed", after=("source",), timeout=1200)
_full.add("build_d4r0", "build d4r0", after=("source",), timeout=1200)
_full.add("pairs_allrelaxed", "pairs allrelaxed", lane="gpu",
          after=("preflight", "build_default", "build_all", "build_allrelaxed"), timeout=1200)
_full.add("pairs_d4r0", "pairs d4r0", lane="gpu", after=("preflight", "build_default", "build_d4r0"), timeout=900)
_full.JOBS[-1]["after_any"] = ["pairs_allrelaxed"]
# run 4: the candidate (docs/DSV2-D1-D5.md, alld4r0); ETX_DSV2_NEXT=alld4r0 selects it
import os
if os.environ.get("ETX_DSV2_NEXT", "alld4r0") == "alld4r0":
    _full.JOBS.clear()
    _full.add("build_alld4r0", "build alld4r0", after=("source",), timeout=1200)
    _full.add("pairs_alld4r0", "pairs alld4r0", lane="gpu",
              after=("preflight", "build_default", "build_all", "build_alld4r0"), timeout=1200)
JOBS += _full.JOBS
