"""DeepSeek's K1 queue and Qwen's prepared studies on ONE box, one runner: the two session files' own jobs, combined.

Seven experiments (docs/EXPERIMENTS-2026-10-14.md): DeepSeek K1, M1 + two kept-arm ablations (timed pairs) + one stamped
trace (attribution only); Qwen row ownership between MoE workgroups (hw_bench moerows; moegather is its cross-check, not a
study) + the common cache mode (pairs_kept_l1). certify_kept / certify_kept_l1 are correctness gates, not experiments.

Nothing is redefined here: examples/sessions/ds_k1.py's JOBS as they are, examples/sessions/qwen_g6.py's jobs() with
prefix "qwen_", record subdirectory qwen/ (model-separated raw outputs) and shared_setup (waits for DeepSeek's apt and
.venv, whatever their result, instead of racing apt). The models share no file: every Qwen read and write is under
qwen/ or in Qwen's own $HOME paths, so a failed Qwen input skips only Qwen's jobs, and the microbenchmark's outcome gates
nothing. One GPU lane, strictly in this order (gpu_jobs.py), so no two GPU jobs ever overlap:
  qwen_moerows (no model input, seconds [E]) -> DeepSeek prep_ref -> pairs_alld4r0l1kvs -> pairs_alld4r0l1mw ->
  pairs_alld4r0l1noQ ->
  pairs_alld4r0l1noA -> trace_alld4r0l1kvs -> qwen_stage_pack -> qwen_certify_kept -> qwen_certify_kept_l1 ->
  qwen_pairs_kept_l1
DeepSeek, the active priority, finishes first; Qwen's download and environment run on the CPU meanwhile.
Run: examples/dsv2lite/reproduce.sh stage <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=7200 tools/session.sh start <ip> examples/sessions/joint_ds_qwen.py
"""
import importlib.util
from pathlib import Path


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ds, _q = _load("ds_k1"), _load("qwen_g6")
MAX_SESSION_SECONDS = 7200          # DeepSeek ~7 min [M prep + E GPU]; Qwen staging [U] + 16 invocations <= 240 s each
_qwen = _q.jobs(prefix="qwen_", sub="qwen", shared_setup=True)
_first = {"qwen_build_hw_bench", "qwen_moerows", "qwen_stage_env", "qwen_stage_vllm", "qwen_stage_dl"}
JOBS = ([j for j in _qwen if j["name"] in _first] + [dict(j) for j in _ds.JOBS] +
        [j for j in _qwen if j["name"] not in _first])
STUDIES = {"DeepSeek": ["pairs_alld4r0l1kvs", "pairs_alld4r0l1mw", "pairs_alld4r0l1noQ", "pairs_alld4r0l1noA", "trace_alld4r0l1kvs"],
           "Qwen": ["qwen_moerows", "qwen_pairs_kept_l1"]}
