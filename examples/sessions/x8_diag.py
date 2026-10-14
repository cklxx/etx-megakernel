"""Which recipe is correct at batch 8? Build the four candidates side by side and run their gates.

The eight-request gate failed twice on 2026-10-14: first with K1 + M1 (`request 5: 3 layers within the gate`), then
with M1 alone (same symptom). B = 4 passes with K1 + M1 and B = 2 has not been gated yet, so the failure is specific to
eight requests and the candidate list is:

  x8p  plain replication recipe (the pre-section-10 arm: five levers + l1_loads, no D4, no M1, no K1)
  x8m  x8p + M1 (merge_uv_waves on four waves)
  x8k  x8p + K1 (shared kv_a)
  x8   x8p + M1 + K1 (section 10's recipe, the one the first failure came from)

Only the build and gate jobs run here; the winner's timing is measured by the batch study's own rounds afterwards
(docs/BATCH-2026-10-14.md section 10 says no timing is reported from an arm whose gate failed).

Run: examples/dsv2lite/reproduce.sh stage <ip> && ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=2700 \
     tools/session.sh start <ip> examples/sessions/x8_diag.py
"""
import copy
import importlib.util
from pathlib import Path

MAX_SESSION_SECONDS = 2700


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_batch = _load("ds_batch")
PREP = ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "preflight")
CANDIDATES = ("x8p", "x8m", "x8k", "x8")
INPLACE = ".venv/bin/python -u -m examples.dsv2lite.inplace_session"
BATCH = ".venv/bin/python -u -m examples.dsv2lite.batch_session"

_by_name = {j["name"]: j for j in _batch.JOBS}
_extra = []
for arm in CANDIDATES:
    b = copy.deepcopy(_by_name["build_x8"])       # the existing job carries the box env, timeout and summary lines
    b.update(name=f"build_{arm}", cmd=f"{INPLACE} build {arm}", writes=[f"b_{arm}/run"])
    g = copy.deepcopy(_by_name["bgate_x8"])
    g.update(name=f"bgate_{arm}", cmd=f"{BATCH} gate {arm}",
             reads=[f"b_{arm}/run" if p == "b_x8/run" else p for p in g["reads"]], writes=[f"bgate_{arm}.json"])
    _extra += [b, g]

JOBS = [_by_name[n] for n in PREP] + [_by_name["prep_goldens"]] + _extra
STUDIES = {"DeepSeek batch": [f"bgate_{a}" for a in CANDIDATES]}
