"""PREPARED, NOT RUN. Five paired rounds default / alld4r0 / alld4s, on one MI300X, batch 1, context 1024.

alld4s is all levers + D4 with 32-row slices; alld4r0 is the measured best arm with zero slice rows.
KEEP against alld4r0, device basis: every pair faster, median >= 30 us and > 2 x full-range spread.
Every invocation requires 32/32 HF-greedy tokens, 27/27 layers, and a 32/32 warm-up. Both device and wall
results are retained. The threshold was fixed before the implementation in docs/deepseek/DSV2-D1-D5.md.

Inputs and the pinned fleet export are reused from the box, verified before any timed invocation. No provisioning
or downloads. Thirty minutes is a hard session cap; builds run on CPU in parallel, then all 15 invocations use GPU 0.
Dry run: .venv/bin/python -m etx.tools.gpu_jobs examples/sessions/ds_d4s.py --gpus 0 --dry-run
Run: ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=1800 tools/session.sh start <ip> examples/sessions/ds_d4s.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 1800
# inputs reused from the box: without the prep jobs, preflight reads the files already there
JOBS = [dict(d, reads=[r for r in d["reads"] if r != "apt"]) for d in _full.JOBS
        if d["name"] in ("etx_env", "source", "build_default", "preflight")]
_full.JOBS.clear()
for arm in ("alld4r0", "alld4s"):
    _full.build(arm)
_full.pairs("alld4s")
JOBS += _full.JOBS
