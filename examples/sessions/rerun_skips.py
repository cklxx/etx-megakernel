"""Rerun of the jobs that the 2026-10-14 all_verify run skipped, on the same box, after the two input gaps were fixed.

The first run (session 20261014T112629Z-d09618) failed two input jobs and skipped everything downstream:
- prep_goldens failed because the prep job refused the reference set: requests 0 and 4 produced the same 32 greedy
  tokens. That is now a recorded collision instead of a refusal (docs/BATCH-2026-10-14.md 4a). Nothing else about the
  Golden construction changed: request r is still fleet's prompt rotated by 97 r tokens.
- qwen_preflight failed because /root/rocm-headers/inc was empty on the box (hip/hip_runtime.h missing). 467 files
  including hip/hip_runtime.h were pushed from the laptop's ~/code/rocm-headers/inc before this run.

Everything the first run completed is reused, not regenerated (AGENTS.md section 2.5): the arm builds
(build_x2/x4/x8, build_alld4r0l1kvsmw), prep_pack, prep_ref, routing and qwen_moerows stay as they are. This session
runs only the skipped set, in the same GPU order the combined session used.

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

# The runner refuses a read that names no job in this list, even an optional one (`?etx_env`), so pull in the
# transitive closure of the selected jobs' dependencies. Those jobs are cheap CPU steps that already ran on this box.
_every = dict(_by_name, **_qwen)
_order = list(QWEN_JOBS + DS_GPU + DS_FINAL)
_selected = []
for n in _order:
    if n in _selected:
        continue
    _selected.append(n)
    for r in _every[n].get("reads", []):
        r = r.lstrip("?")
        if r in _every and r not in _order:
            _order.append(r)

JOBS = [_every[n] for n in _selected]

STUDIES = {"DeepSeek batch": ["bgate_x2", "bgate_x4", "bgate_x8", "brounds"],
           "Qwen": ["qwen_pairs_kept_l1"]}
