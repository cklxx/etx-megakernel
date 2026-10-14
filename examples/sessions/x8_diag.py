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
import importlib.util
from pathlib import Path

MAX_SESSION_SECONDS = 2700


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_batch = _load("ds_batch")
job, OP, GOLDENS, SUMMARY = _batch.job, _batch.OP, _batch.GOLDENS, _batch.SUMMARY

PREP = ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "preflight")
_prep = list(_batch.JOBS)          # ds_batch's own list, filtered by name below
_by_name = {j["name"]: j for j in _prep}
CANDIDATES = ("x8p", "x8m", "x8k", "x8")

for arm in CANDIDATES:
    job(f"build_{arm}", f"build {arm}", lane="cpu", reads=["source.json"], writes=[f"b_{arm}/run"], timeout=1200)
for arm in CANDIDATES:
    job(f"bgate_{arm}", f"{OP} gate {arm}", lane="gpu", reads=("inputs.json", GOLDENS, f"b_{arm}/run"),
        writes=[f"bgate_{arm}.json"], timeout=600, summary=SUMMARY)

_after = {j["name"]: j for j in _batch.JOBS}
JOBS = [_by_name[n] for n in PREP] + [_after[f"build_{a}"] for a in CANDIDATES] + [_after[f"bgate_{a}"] for a in CANDIDATES]
STUDIES = {"DeepSeek batch": [f"bgate_{a}" for a in CANDIDATES]}
