"""Batch-8 recipe diagnostic plus Qwen's four GPU jobs, on one box.

Two independent questions, one rental:
1. Which recipe is correct at batch 8? The eight-request gate failed with K1 + M1 and with M1 alone; B = 4 passes
   with both. examples/sessions/x8_diag.py's four candidates (x8p plain, x8m +M1, x8k +K1, x8 +M1+K1) are built and
   gated here; only the winner's timing is measured later, by the batch study's own rounds.
2. Qwen: the two arm builds, their two correctness certificates and the common-cache-mode pairs study
   (examples/sessions/qwen_g6.py). Its preflight needs the tree's own .venv (yaml + numpy inside it), which the
   staging step creates fresh, so the venv has to exist before this session starts.

Run: examples/dsv2lite/reproduce.sh stage <ip> &&
     ssh root@<ip> 'cd /root/etx && python3 -m venv --without-pip .venv && curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && .venv/bin/python /tmp/get-pip.py && .venv/bin/python -m pip install -e ".[dev]"' &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=5400 tools/session.sh start <ip> examples/sessions/x8_diag_qwen.py
"""
import importlib.util
import copy
from pathlib import Path

MAX_SESSION_SECONDS = 7200


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_batch, _q = _load("ds_batch"), _load("qwen_g6")
_by_name = {j["name"]: j for j in _batch.JOBS}
_qwen = {j["name"]: j for j in _q.jobs(prefix="qwen_", sub="qwen", shared_setup=True)}

PREP = ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "preflight")
CANDIDATES = ("x8p", "x8m", "x8k", "x8")
QWEN = ("qwen_build_hw_bench", "qwen_stage_env", "qwen_stage_vllm", "qwen_stage_dl", "qwen_stage_pack",
        "qwen_preflight", "qwen_build_kept", "qwen_build_kept_l1", "qwen_certify_kept", "qwen_certify_kept_l1",
        "qwen_pairs_kept_l1")
INPLACE = ".venv/bin/python -u -m examples.dsv2lite.inplace_session"
BATCH = ".venv/bin/python -u -m examples.dsv2lite.batch_session"

# The extra build/gate jobs copy the study's own jobs, so they inherit the box environment, timeouts and summary lines
_extra = []
for arm in CANDIDATES:
    b = copy.deepcopy(_by_name["build_x8"])
    b.update(name=f"build_{arm}", cmd=f"{INPLACE} build {arm}", writes=[f"b_{arm}/run"])
    g = copy.deepcopy(_by_name["bgate_x8"])
    g.update(name=f"bgate_{arm}", cmd=f"{BATCH} gate {arm}", writes=[f"bgate_{arm}.json"])
    _extra += [b, g]

CPU = [_by_name[n] for n in PREP]
CPU += [_qwen[n] for n in QWEN if _qwen[n].get("lane") == "cpu"]
CPU += [j for j in _extra if j["lane"] == "cpu"]
GPU = [_qwen[n] for n in QWEN if _qwen[n].get("lane") == "gpu"]
GPU += [j for j in _extra if j["lane"] == "gpu"]
JOBS = CPU + GPU
STUDIES = {"DeepSeek batch": [f"bgate_{a}" for a in CANDIDATES], "Qwen": ["qwen_pairs_kept_l1"]}
