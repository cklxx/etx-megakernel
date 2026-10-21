"""Combined historical studies, on ONE box, one runner: DeepSeek's batch 2/4/8 study + Qwen's studies, combined.

kailun asked for one verification run instead of separate rentals. Nothing is redefined here: the two session files'
own jobs are composed, exactly as examples/sessions/joint_ds_qwen.py composes two of them.

Contents (docs/EXPERIMENTS-2026-10-14.md, docs/deepseek/BATCH-2026-10-14.md):
- DeepSeek batch (examples/sessions/ds_batch.py): B = 1 (the kept arm alld4r0l1kvsmw, BATCH-2026-10-14.md section 9),
  2, 4, 8 (arms x2, x4, x8) against the strongest legal vLLM configuration at the same B, total throughput and
  per-request latency decided separately, per-request goldens.
- Qwen (examples/sessions/qwen_g6.py): the row-ownership microbenchmark (moerows) and the common cache mode
  (pairs_kept_l1), with Qwen's staging (environment, vLLM v0.27.1, the pinned HF revision, packing on the GPU) inside
  the runner. Its two certificates are correctness gates, not studies.

Not in this run, with the reason:
- DeepSeek's K1, M1, KM, the two kept-arm ablations and the K1 trace: already measured, every run 32/32 and 27/27
  (examples/dsv2lite/results/2026-10-14-k1 and 2026-10-14-km). AGENTS.md section 2.5: no GPU run regenerates data that
  exists. The kept arm alld4r0l1kvsmw still runs five times here, as the batch study's B = 1 arm.
- The runtime-bundle epoch/static-only-reset form (P3) and the row-partitioned expert-combine (P4) have no testable arm
  until a local diagnostic and a stamp exist (docs/qwen/PROPOSALS-2026-10-14.md); queueing them would spend GPU minutes on
  nothing.

Isolation, from the artifact graph rather than from comments: the two models share no job name and no file (Qwen's are
prefixed `qwen_` and recorded under `qwen/`), a failed Qwen staging step skips only Qwen's jobs, and a failed
microbenchmark gates nothing else.

GPU lane, strictly in this order (the runner runs one GPU job at a time, in list order):
  prep_ref -> qwen_moerows (no model input, seconds) -> prep_goldens (7 rotated references) ->
  bgate_x2/x4/x8 (one gated invocation each) -> boxcontrol (same-session recipe difference) -> vsel (vLLM selection) -> brounds (rounds 1-5) -> routing ->
  qwen_stage_pack -> qwen_certify_kept -> qwen_certify_kept_l1 -> qwen_pairs_kept_l1
DeepSeek, the active priority, produces its numbers first; Qwen's download and environment build on the CPU meanwhile.

Budget [E]: DeepSeek prep 4 min [M] + batch ~40-60 min GPU [E] and a vLLM image pull [U, parallel]; Qwen staging [U]
+ 16 invocations about 32 min [E, worst 64]. Cap 10800 s = 3 h, about $6 at $1.99/h. No provisioning without the
owner's approval, and the box needs network access to Hugging Face, PyPI (torch for ROCm) and GitHub.

Run: examples/dsv2lite/reproduce.sh stage <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=10800 tools/session.sh start <ip> examples/sessions/all_verify.py
"""
import importlib.util
from pathlib import Path

MAX_SESSION_SECONDS = 10800


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_batch, _q = _load("ds_batch"), _load("qwen_g6")

PREP = ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "preflight")
BATCH_GPU = tuple(j["name"] for j in _batch.JOBS if j.get("lane") == "gpu" and j["name"] not in PREP)
BATCH_FINAL = ("breport",)

_by_name = {j["name"]: dict(j) for j in _batch.JOBS}
_qwen = _q.jobs(prefix="qwen_", sub="qwen", shared_setup=True)
QWEN_EARLY = ("qwen_build_hw_bench", "qwen_stage_env", "qwen_stage_vllm", "qwen_stage_dl")
QWEN_ARM = ("qwen_moerows",)                    # the one Qwen job that must precede DeepSeek's, so the GPU is not idle

CPU = [j for j in _qwen if j["name"] in QWEN_EARLY]
CPU += [_by_name[n] for n in PREP]
CPU += [_by_name["docker_pull"]]
CPU += [j for j in _by_name.values() if j["name"].startswith("build_")]
GPU = [j for j in _qwen if j["name"] in QWEN_ARM]
GPU += [_by_name[n] for n in BATCH_GPU]
# every remaining Qwen job in its own order: stage_pack, preflight, both builds, the two certificates, the pairs study
_seen = {j["name"] for j in CPU + GPU}
GPU += [j for j in _qwen if j["name"] not in _seen]
FINAL = [_by_name[n] for n in BATCH_FINAL]
JOBS = CPU + GPU + FINAL
assert {j["name"] for j in JOBS} >= set(_by_name), "a ds_batch.py job is missing from the combined session"

STUDIES = {"DeepSeek batch": ["bgate_x2", "bgate_x4", "bgate_x8", "brounds"],
           "Qwen": ["qwen_moerows", "qwen_pairs_kept_l1"]}
