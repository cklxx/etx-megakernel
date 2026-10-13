"""The K1 hour (docs/DSV2-D1-D5.md, "K1": rules written before the code, commit 2ec2d29). Kept arm alld4r0l1 against:

GPU order (one pairs invocation 5.9 s, one traced token 7-8 s, from the 2026-10-13 runner records):
  pairs_alld4r0l1kvs   K1: kv_a computed once (1/8 per XCD) behind a global qkv event; KEEP or DROP   10 inv, ~59 s
  pairs_alld4r0l1noQ   ablation: the kept recipe without gemv_no_terminal_all; REMOVE or STAYS       10 inv, ~59 s
  pairs_alld4r0l1noA   ablation: the kept recipe without argmax_tree; REMOVE or STAYS                10 inv, ~59 s
  trace_alld4r0l1kvs   one stamped token of K1: attribution only (qkv vs attention), never a timing  1 inv, ~8 s
Deciding timings are unstamped, five interleaved rounds on one machine. The full prep chain of ds_inplace.py runs
first (fresh droplet: 3 min 59 s to preflight on 2026-10-13); every build runs in parallel during it; every GPU job gates
hard on preflight and on each binary it runs. The runner seals the record on the box before any fetch.
Run: examples/dsv2lite/reproduce.sh stage <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=1800 tools/session.sh start <ip> examples/sessions/ds_k1.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 1800
_full.JOBS = [dict(d) for d in _full.JOBS if d["name"] in
              ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "preflight")]
GPU = [("pairs_alld4r0l1kvs", "pairs alld4r0l1kvs"), ("pairs_alld4r0l1noQ", "pairs alld4r0l1noQ"),
       ("pairs_alld4r0l1noA", "pairs alld4r0l1noA"), ("trace_alld4r0l1kvs", "trace alld4r0l1kvs")]
for name, op in GPU:
    kind, arm = op.split()
    _full.pairs(arm) if kind == "pairs" else _full.trace(arm, timeout=600)
_full.generate_builds()
JOBS = _full.JOBS
