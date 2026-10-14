"""KM: K1 and M1 together (docs/DSV2-D1-D5.md, "KM": rule written before the code, commit bf92fd4).

GPU order: pairs_alld4r0l1kvsmw, five rounds of alld4r0l1, alld4r0l1mw, alld4r0l1kvsmw (15 invocations, ~90 s at
5.9 s each [E from M]). Decided against alld4r0l1mw; the gain vs alld4r0l1 is attribution only. The prep chain of
ds_inplace.py runs first and skips what the box already holds (the K1 session's model, reference and pack).
Run: ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=1800 tools/session.sh start <ip> examples/sessions/ds_km.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 1800
_full.JOBS = [dict(d) for d in _full.JOBS if d["name"] in
              ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "preflight")]
_full.pairs("alld4r0l1kvsmw")
_full.generate_builds()
JOBS = _full.JOBS
