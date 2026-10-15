"""Registered correctness-only queue. No provisioning. Hard cap 3600 s; no performance rounds."""
import copy
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("x8_prep", Path(__file__).with_name("ds_inplace.py"))
_p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p)
MAX_SESSION_SECONDS = 3600
PREP = ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_pack")
JOBS = [copy.deepcopy(j) for j in _p.JOBS if j["name"] in PREP]
for j in JOBS:
    if j["name"] == "prep_pack":
        j["reads"] = [f"{_p.MODEL}/config.json", "source.json"]
        j["cmd"] = 'mkdir -p "$ETX_DSV2_DATA"; ' + j["cmd"]
for arm in ("x1km", "x8", "alld4r0l1kvsmw"):
    JOBS.append(dict(name=f"build_{arm}", lane="cpu", gpu=None, env=dict(_p.ENV), reads=["source.json"],
                     writes=[f"b_{arm}/run"], cmd=f".venv/bin/python -u -m examples.dsv2lite.inplace_session build {arm}",
                     timeout=1200, quiet_s=90))
JOBS.append(dict(name="references", lane="gpu", gpu=0, env=dict(_p.ENV),
                 reads=["source.json", "prep_pack", "prep_env"], writes=["references_ready.json"],
                 cmd=f'{_p.PREP_PY} -u -m examples.dsv2lite.reference_prepare --model {_p.MODEL} --requests 8 --regenerate-missing',
                 timeout=1600, quiet_s=90))
JOBS.append(dict(name="diagnostics", lane="gpu", gpu=0, env=dict(_p.ENV),
                 reads=["references_ready.json", *(f"b_{arm}/run" for arm in ("x1km", "x8", "alld4r0l1kvsmw"))],
                 writes=["x8_correctness.json"], cmd=f"{_p.PREP_PY} -u -m examples.dsv2lite.x8_correctness",
                 timeout=1150, quiet_s=90))

JOBS.append(dict(name="summary", lane="final", gpu=None, env=dict(_p.ENV),
                 reads=["?references_ready.json", "?x8_correctness.json"], writes=["x8_summary.json"],
                 cmd=".venv/bin/python -u -m examples.dsv2lite.x8_correctness finalize", timeout=60, quiet_s=30))
