"""Rerun of the jobs that the 2026-10-14 all_verify run skipped, on the same box, after the two input gaps were fixed.

The first run (session 20261014T112629Z-d09618) failed two input jobs and skipped everything downstream:
- prep_goldens failed because the prep job refused the reference set: requests 0 and 4 produced the same 32 greedy
  tokens. That is now a recorded collision instead of a refusal (docs/deepseek/BATCH-2026-10-14.md 4a). Nothing else about the
  Golden construction changed: request r is still fleet's prompt rotated by 97 r tokens.
- qwen_preflight failed because /root/rocm-headers/inc was empty on the box (hip/hip_runtime.h missing). 467 files
  including hip/hip_runtime.h were pushed from the laptop's ~/code/rocm-headers/inc before this run.

The runner creates a fresh record, so this session includes producers for the required record-local certificates
and binaries. The preparation jobs reuse the persistent model, packed weights and references when present; CPU
builds and preflight certificates populate the new record. Completed GPU measurements are not imported implicitly:
qwen_moerows and routing are not rerun, and the report has no optional routing projection in this record.

Budget [E]: DeepSeek goldens ~5 min [M] + batch gates and rounds ~40-60 min GPU [E]; Qwen preflight, two builds, two
certificates and the pairs study ~32 min [E]. Cap 7200 s = 2 h. The DeepSeek batch numbers come first by list order.

Run: examples/dsv2lite/reproduce.sh stage <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=7200 tools/session.sh start <ip> examples/sessions/rerun_skips.py
"""
import importlib.util
from pathlib import Path

MAX_SESSION_SECONDS = 7200


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_batch, _q = _load("ds_batch"), _load("qwen_g6")
_by_name = {j["name"]: dict(j) for j in _batch.JOBS}
_qwen = {j["name"]: dict(j) for j in _q.jobs(prefix="qwen_", sub="qwen", shared_setup=True)}

DS_GPU = ("prep_goldens", "bgate_x2", "bgate_x4", "bgate_x8", "vsel", "brounds")
DS_FINAL = ("breport",)
QWEN_JOBS = ("qwen_preflight", "qwen_build_kept", "qwen_build_kept_l1", "qwen_certify_kept",
             "qwen_certify_kept_l1", "qwen_pairs_kept_l1")

# A fresh record cannot see the prior record's relative artifacts. Include their producers as well as bare-job
# dependencies, before their consumers (in particular, GPU preparation must precede its GPU gates). Optional artifact
# reads may remain absent; pulling those producers in would repeat completed measurements such as routing.
_every = dict(_by_name, **_qwen)
_writers = {w.lstrip("?"): n for n, j in _every.items() for w in j.get("writes", [])}
_selected = []


def _include(n):
    if n in _selected:
        return
    for r in _every[n].get("reads", []):
        key = r.lstrip("?")
        producer = key if key in _every else None if r.startswith("?") else _writers.get(key)
        if producer is not None:
            _include(producer)
    _selected.append(n)


for n in QWEN_JOBS + DS_GPU + DS_FINAL:
    _include(n)

JOBS = [_every[n] for n in _selected]

STUDIES = {"DeepSeek batch": ["bgate_x2", "bgate_x4", "bgate_x8", "brounds"],
           "Qwen": ["qwen_pairs_kept_l1"]}
