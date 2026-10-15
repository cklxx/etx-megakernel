"""One authorized continuation on the existing MI300X; hard cap includes trace analysis.

No rebuild, profiler, provisioning, concurrency or old attempt restart. Driver deployed
alongside this file; unchanged accepted imports come from PYTHONPATH=/root/etx.
Absolute execution/export stop 2026-10-15T17:22:10Z; >=300s export reserve.
"""
from pathlib import Path

MAX_SESSION_SECONDS = 900
LOCAL_ONLY = ()
HERE = Path(__file__).resolve().parent
BASE = "/root/sess/b4-live-20261015/20261015T160946Z-a7e517"
JOBS = [dict(name="multibatch_trace", lane="gpu", gpu=0, cwd=str(HERE),
             cmd=f"/root/etx-python/bin/python -u {HERE}/multibatch_trace.py --baseline {BASE}",
             env=dict(PYTHONPATH="/root/etx", PYTHONDONTWRITEBYTECODE="1",
                      ETX_DSV2_FLEET="/root/fleet-3fb67a6", ETX_DSV2_DATA=f"{BASE}/references"),
             timeout=850, quiet_s=90, writes=["bindings.json", "trace_report.json"],
             summary=[r"^TRACE COMPLETE "])]
