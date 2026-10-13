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
# inputs reused from the box: without the prep jobs, preflight reads the files already there
JOBS = [dict(d, reads=[r for r in d["reads"] if r != "apt"]) for d in _full.JOBS
        if d["name"] in ("etx_env", "source", "build_default", "build_all", "preflight")]
_full.JOBS.clear()
import os
if os.environ.get("ETX_DSV2_NEXT", "alld4r0") == "alld4r0":   # run 4: the candidate (docs/DSV2-D1-D5.md, alld4r0)
    _full.build("alld4r0")
    _full.pairs("alld4r0")
else:
    _full.build("allrelaxed")
    _full.build("d4r0")
    _full.pairs("allrelaxed")
    _full.pairs("d4r0", timeout=900)
JOBS += _full.JOBS
