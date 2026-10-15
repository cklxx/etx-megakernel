"""Historical batch study; current harness fixes await GPU revalidation: the batch 2 / 4 / 8 study on DeepSeek-V2-Lite (docs/BATCH-2026-10-14.md, rules committed before
the code in 0fc38c9). One MI300X, GPU 0. No provisioning without the owner's approval.

ETX at B = 1 (the kept arm alld4r0l1kvsmw, section 9), 2, 4, 8 (arms x2, x4, x8) against the strongest legal vLLM configuration at the
same B, on this box, in this session; total throughput and per-request latency decided separately (etx/session/batch.py).
Every request of every invocation is gated on its own HF-greedy golden (32/32 tokens, 27/27 layers, 32/32 warm-up); every
vLLM sequence on its own golden, on the warm-up and on every sample.

Order: the ds_inplace prep chain, then (CPU, in parallel) the vLLM image pull and the five builds; GPU, strictly in
order: the 7 rotated references (prep_goldens), one gated invocation per x arm, the two-invocation B = 1 recipe control (not a machine-offset measurement),
the vLLM selection round, rounds 1-5,
the measured expert union (routing), the report.
Budget [E]: ~40-60 min, about $2 at $1.99/h, plus ~1 GPU minute for boxcontrol (build excluded);
runner cap 5400 s (docs/BATCH-2026-10-14.md section 6).
Run: examples/dsv2lite/reproduce.sh stage <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
     ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=5400 tools/session.sh start <ip> examples/sessions/ds_batch.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ds_inplace", Path(__file__).with_name("ds_inplace.py"))
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

MAX_SESSION_SECONDS = 5400
_full.JOBS = [dict(d) for d in _full.JOBS if d["name"] in
              ("apt", "etx_env", "prep_env", "source", "prep_dl", "prep_ref", "prep_pack", "preflight")]
job, PREP_PY, MODEL = _full.job, _full.PREP_PY, _full.MODEL
from examples.dsv2lite.arms import BATCH_ARMS  # noqa: E402
from examples.dsv2lite.batch_session import BOXCONTROL_PREVIOUS_ARM  # noqa: E402
IMAGE = "vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7"
REQUESTS = 8
GOLDENS = "$ETX_DSV2_DATA/goldens.json"
OP = ".venv/bin/python -u -m examples.dsv2lite.batch_session"
SUMMARY = [r"^(BGATE|BOXCONTROL|VSEL|BROUNDS|BREPORT|BATCH-INCOMPLETE|GOLDENS) "]

# ---------------------------------------------------------------- per-request references (GPU) and the vLLM image (CPU)
job("prep_goldens", 'set -eu; for r in $(seq 0 ' + str(REQUESTS - 1) + '); do '
    'd="$ETX_DSV2_DATA"; [ "$r" = 0 ] || d="$d/r$r"; '
    f'if {PREP_PY} -m examples.dsv2lite.batch_goldens "$d" --verify-binding; '
    'then echo "request $r: certified reference present"; else mkdir -p "$d"; rm -f "$d/golden.binding.json"; '
    f'HIP_VISIBLE_DEVICES=$GPU {PREP_PY} examples/dsv2lite/reference_rotated.py --fleet "$ETX_DSV2_FLEET" '
    f'--rotate $((97 * r)) -- --model {MODEL} --out "$d/golden.npz"; '
    f'{PREP_PY} -m examples.dsv2lite.batch_goldens "$d" --certify; fi; done; '
    '.venv/bin/python -u -m examples.dsv2lite.inplace_session preflight; '
    f'{PREP_PY} -m examples.dsv2lite.batch_goldens "$ETX_DSV2_DATA" --requests {REQUESTS}',
    lane="gpu", reads=("inputs.json",), writes=[GOLDENS], timeout=1500, summary=SUMMARY + [r"prompt rotated|wrote "])
job("docker_pull", 'set -eu; S=; [ "$(id -u)" = 0 ] || S=sudo; $S docker --version; '
    f'$S docker pull {IMAGE}; $S docker image inspect --format "IMAGE {{{{.Id}}}}" {IMAGE}',
    reads=("apt",), writes=[], timeout=1500)

# ---------------------------------------------------------------- GPU: correctness first, then the timings, then captures
for arm in ("x2", "x4", "x8"):
    job(f"bgate_{arm}", f"{OP} gate {arm}", lane="gpu", reads=("inputs.json", GOLDENS, f"b_{arm}/run"),
        writes=[f"bgate_{arm}.json"], timeout=600, summary=SUMMARY)
job("boxcontrol", f"{OP} boxcontrol", lane="gpu",
    reads=("inputs.json", f"b_{BATCH_ARMS[1]}/run", f"b_{BOXCONTROL_PREVIOUS_ARM}/run"),
    writes=["boxcontrol.json"], timeout=360, summary=SUMMARY)
job("vsel", f"{OP} vsel", lane="gpu", reads=("docker_pull", GOLDENS, f"{MODEL}/config.json"), writes=["vsel.json"],
    timeout=1900, summary=SUMMARY)
job("brounds", f"{OP} rounds", lane="gpu",
    reads=("vsel.json", "boxcontrol.json", f"b_{BATCH_ARMS[1]}/run", *(f"?bgate_{a}.json" for a in ("x2", "x4", "x8"))),
    writes=["brounds.json"], timeout=3600, summary=SUMMARY)
job("routing", f"HIP_VISIBLE_DEVICES=$GPU {PREP_PY} examples/dsv2lite/routing_overlap.py --model {MODEL} "
    f'--fleet "$ETX_DSV2_FLEET" --batch {REQUESTS} --json "$ETX_SESS_LOGDIR/routing.json"',
    lane="gpu", reads=("?brounds.json",), writes=["routing.json"], timeout=900, summary=[r"^ROUTING "])
job("breport", f"{OP} report", lane="final", reads=("brounds.json", "vsel.json", "?routing.json"),
    writes=["batch_report.json"], timeout=120, summary=SUMMARY)
_full.generate_builds()
JOBS = _full.JOBS
