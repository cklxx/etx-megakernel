"""PREPARED, NOT RUN. One MI300X, GPU 0 only. No provisioning; the model download and reference run happen on the box.

Measures ETX's DeepSeek port against default ETX and shipped fleet (3fb67a6). Arm builds, invocation orders,
references and decision rules come from examples/dsv2lite/arms.py; build jobs are generated from measurement inputs.
The GPU lane runs baseline, the deciding combined arm, stage traces, then individual levers for attribution.
Every invocation is fresh and requires 32/32 HF-greedy tokens, 27/27 layers and a 32/32 warm-up. Scratch in a build or
any correctness failure rejects its job; an unfinished run is NOT COMPLETE. Expected local savings: 531.1 us
simulated / 533.4 us arithmetic (stage_table.md; BYTES rows are upper bounds).

Budget [planning cap]: 120 GPU minutes (MAX_SESSION_SECONDS), $5.98 at $2.99/h (Hot Aisle 1x; balance $6.29 on
2026-10-13) or $3.98 at $1.99/h. Expected use [E]: ~25 min environments + inputs (builds run in parallel), then
10 + 15 + 2 + 70 = 97 invocations x ~8 s (results_2026-09-27b runner: 7-8 s per 32-token invocation) = ~13 min,
plus builds' tail: ~45-60 min total.

Inputs are prepared ON THE BOX by the prep_* jobs (the ds_gap.py recipes): docs/core/DATA.md:59-64 says the fleet reference
outputs (~70 MB, ~2 GPU minutes) and the packed weights (31 GB, ~1 min download + ~1 min pack) were never kept. Both
come from the PINNED fleet export (ETX_DSV2_FLEET, default ~/fleet-3fb67a6, src hash verified by `source`) and
deepseek-ai/DeepSeek-Coder-V2-Lite-Base, into ETX_DSV2_DATA (default ~/dsv2-data); `preflight` compares every file
with the pinned measured hashes before the first timed run.

Run (after local validation, balance check and budget authorization):
  examples/dsv2lite/reproduce.sh stage <ip> && ETX_SESS_MAX_SECONDS=7200 tools/session.sh start <ip> examples/sessions/ds_inplace.py 0
Use --dry-run locally. Fetch the session directory into the tree before deleting the VM; delete it right after.
"""
import os

from examples.dsv2lite.arms import ARMS, SINGLE
from examples.dsv2lite.inplace_session import DATA_FILES

MAX_SESSION_SECONDS = 7200
JOBS = []

# The runner loads this file on the box, so these are the box's defaults; every job gets them, and the runner expands
# the $ETX_DSV2_DATA paths jobs read and write with the same values.
ENV = dict(ETX_DSV2_FLEET=os.environ.get("ETX_DSV2_FLEET", os.path.expanduser("~/fleet-3fb67a6")),
           ETX_DSV2_DATA=os.environ.get("ETX_DSV2_DATA", os.path.expanduser("~/dsv2-data")))
PREP_PY = "$HOME/dsv2-prep/.venv/bin/python"
MODEL = "$HOME/models/dsv2-lite-base"
DATA = {name: f"$ETX_DSV2_DATA/{name}" for name in DATA_FILES}


def job(name, cmd, *, lane="cpu", reads=(), writes=(), timeout=600, summary=()):
    JOBS.append(dict(name=name, lane=lane, gpu=0 if lane == "gpu" else None, reads=list(reads), writes=list(writes),
                     timeout=timeout, quiet_s=90, cmd=cmd, env=dict(ENV), summary=list(summary)))


def add(name, operation, **kw):
    job(name, f".venv/bin/python -u -m examples.dsv2lite.inplace_session {operation}",
        summary=[r"^(SOURCE|PREFLIGHT|BUILD|PAIRS|TRACE|D4TRACE|REPORT) "], **kw)


def build(arm, stamped=False):
    tag = "_stamps" if stamped else ""
    add(f"build_{arm}{tag}", f"build {arm}" + (" --stamped" if stamped else ""), reads=["source.json"],
        writes=[f"b_{arm}{tag}/run"], timeout=1200)


def pairs(arm, timeout=1200, name=None):
    """Reads exactly what `pairs <arm>` runs: the verified inputs and each build of the arm's order in the registry
    (an optional build, fleet for `all`, is context: measured only if it built)."""
    a = ARMS[arm]
    add(name or f"pairs_{arm}", f"pairs {arm}", lane="gpu", writes=[f"pairs_{arm}.json"], timeout=timeout,
        reads=["inputs.json", *(f"b_{b}/run" for b in a.order if b not in a.optional)])


def trace(arm, timeout=300):
    add(f"trace_{arm}", f"trace {arm}", lane="gpu", reads=["inputs.json", f"b_{arm}_stamps/run"],
        writes=[f"stages_{arm}.json"], timeout=timeout)


# ---------------------------------------------------------------- environments (cpu); recipes from ds_gap.py
job("apt", "python3 -c 'import ensurepip' 2>/dev/null || "
           "{ sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }")
job("etx_env", ".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
               ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; "
               ".venv/bin/python -c 'import etx; print(\"etx ok\")'", reads=("apt",))
job("prep_env", f"{PREP_PY} -c 1 2>/dev/null || {{ rm -rf $HOME/dsv2-prep; python3 -m venv $HOME/dsv2-prep/.venv; }}; "
                ". $HOME/dsv2-prep/.venv/bin/activate; "
                "pip install -q --upgrade pip; pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer; "
                "python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; "
                "pip install -q 'transformers>=4.39,<5' 'tokenizers<0.20' accelerate safetensors; "
                "SP=$(python3 -c 'import site;print(site.getsitepackages()[0])'); mkdir -p $SP/flash_attn; "
                "echo '# stub: satisfies transformers check_imports only' > $SP/flash_attn/__init__.py; "
                "python3 -c 'import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'",
    reads=("apt",), timeout=1800)
add("source", "source", reads=("etx_env",), writes=["source.json"], timeout=300)

# ---------------------------------------------------------------- inputs, produced on the box from the pinned export
job("prep_dl", f"HF_HUB_ENABLE_HF_TRANSFER=1 {PREP_PY} -c \"from huggingface_hub import snapshot_download; "
               f"snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', local_dir='{MODEL}', "
               "allow_patterns=['*.json', '*.safetensors', '*.py', 'tokenizer*'], max_workers=8); print('DL-DONE')\" "
               f"&& du -sh {MODEL} && test -f {MODEL}/config.json", reads=("prep_env",), writes=[f"{MODEL}/config.json"],
    timeout=1800)
job("prep_ref", 'mkdir -p "$ETX_DSV2_DATA" && cd "$ETX_DSV2_FLEET" && '
                '{ test -s "$ETX_DSV2_DATA/fleet_cache.bin" -a -s "$ETX_DSV2_DATA/golden_tokens.txt" '
                '-a -s "$ETX_DSV2_DATA/golden_hidden.bin" || '
                f'HIP_VISIBLE_DEVICES=$GPU {PREP_PY} src/host/reference_run.py --model {MODEL} '
                '--out "$ETX_DSV2_DATA/golden.npz"; } && ls -la "$ETX_DSV2_DATA"',
    lane="gpu", reads=(f"{MODEL}/config.json", "source.json"), timeout=1200,
    writes=[DATA[n] for n in ("fleet_cache.bin", "golden_tokens.txt", "golden_hidden.bin")],
    summary=[r"smallest top-2 margin|wrote |golden"])
job("prep_pack", 'cd "$ETX_DSV2_FLEET" && { test -s "$ETX_DSV2_DATA/weights.bin" -a -s "$ETX_DSV2_DATA/weights.bin.manifest" || '
                 f'{PREP_PY} src/host/pack_weights.py --model {MODEL} --out "$ETX_DSV2_DATA/weights.bin"; }} '
                 '&& ls -la "$ETX_DSV2_DATA"', reads=(f"{MODEL}/config.json", "prep_ref"), timeout=1200,
    writes=[DATA["weights.bin"], DATA["weights.bin.manifest"]], summary=[r"wrote |expert stride"])
# every GPU measurement reads inputs.json, which only a passed preflight writes, and each binary it runs: the
# 2026-10-13 d1d5 run started the single-lever pairs at +1 min, before prep_pack wrote the inputs at +3 min
add("preflight", "preflight", reads=list(DATA.values()), writes=["inputs.json"], timeout=600)

def generate_builds():
    """Materialize every measurement's build inputs, already derived from the registry's invocation order."""
    required = dict.fromkeys(r for j in JOBS for r in j["reads"] if r.startswith("b_") and r.endswith("/run"))
    first = next(i for i, j in enumerate(JOBS) if j["lane"] == "gpu" and j["name"] != "prep_ref")
    start = len(JOBS)
    for path in required:
        name = path[2:-4]
        build(name.removesuffix("_stamps"), stamped=name.endswith("_stamps"))
    generated = JOBS[start:]
    del JOBS[start:]
    JOBS[first:first] = generated


# ---------------------------------------------------------------- GPU, in order of importance (the list order)
pairs("baseline", timeout=900, name="baseline")
pairs("all")                             # the decision
trace("default")
trace("all")
for arm in ("coh", "d4"):                # new conditions before the attributions already measured on 2026-10-13
    pairs(arm)
add("d4trace", "d4trace", lane="gpu", reads=["inputs.json", "b_d4/run"], writes=["d4_trace.json"], timeout=300)
for arm in SINGLE:                       # attribution: each lever alone, after everything that decides
    pairs(arm, timeout=900)
add("report", "report", lane="final", reads=["pairs_all.json"], writes=["report.json"], timeout=120)

generate_builds()
