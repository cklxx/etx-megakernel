"""The next paid hour (plan: docs/deepseek/DSV2-D1-D5.md, "The next paid hour"; every rule fixed in that file before its code).

GPU order (one pairs invocation 5.9 s, one traced token 7-8 s, from the 2026-10-13 runner records):
  pairs_l1         D2b rerun, 3-way default / relaxed / l1, D2's rule           15 inv, ~88 s
  pairs_alld4r0l1  best arm + D2b, 3-way default / alld4r0 / alld4r0l1, KEEP    15 inv, ~88 s
  pairs_alld3      D3, 2-way all / alld3, GO-or-DROP                            10 inv, ~59 s
  pairs_alld5      D5, 2-way all / alld5, KEEP-or-DROP (drop if neutral)        10 inv, ~59 s
  trace_default, trace_alld4r0, trace_alld4r0l1   stamped tokens: the per-stage record   3 inv, ~24 s
  pairs_alld4s     optional, last: Astra's batched D4 slices, 3-way default / alld4r0 / alld4s   15 inv, ~88 s
then the runner seals the session record ON THE BOX (MANIFEST + SEAL, etx/session/record.py) before any fetch.
The full prep chain of ds_inplace.py runs first (fresh droplet: 3 min 59 s to preflight on 2026-10-13); every build
runs in parallel during it. Every GPU job gates hard on preflight and on each binary it runs.
Run: examples/dsv2lite/reproduce.sh stage <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=1800 tools/session.sh start <ip> examples/sessions/ds_hour.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 1800
_full.JOBS = [dict(d) for d in _full.JOBS if d["name"] in
              ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "preflight")]
# the GPU lane runs strictly in this order (review 07, R07: a skip cannot let a later arm overtake an earlier one);
# each job reads the binaries its operation runs (ds_inplace.pairs / trace: from the arm registry), and inputs.json
GPU = [("pairs_l1", "pairs l1"), ("pairs_alld4r0l1", "pairs alld4r0l1"), ("pairs_alld3", "pairs alld3"),
       ("pairs_alld5", "pairs alld5"), ("trace_default", "trace default"), ("trace_alld4r0", "trace alld4r0"),
       ("trace_alld4r0l1", "trace alld4r0l1"), ("pairs_alld4s", "pairs alld4s")]
for name, op in GPU:
    kind, arm = op.split()
    _full.pairs(arm) if kind == "pairs" else _full.trace(arm, timeout=1200)
_full.generate_builds()
JOBS = _full.JOBS
