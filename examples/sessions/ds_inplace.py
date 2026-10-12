"""PREPARED, NOT RUN. One MI300X, GPU 0 only. No provisioning; the model download and reference run happen on the box.

Budget [planning cap, not a runtime prediction]: 120 allocated GPU minutes,
$3.98 at $1.99/h or $5.98 at $2.99/h, including source builds, hash verification,
31.4-GB weight uploads on each fresh invocation and all possible candidate arms.
Each decider additionally has a five warm-GPU-minute cap (startup/upload excluded).
Global deadline is 7200 s; every build/run has its own deadline and heartbeat.

Reject every arm on missing inputs, byte-different phase outputs, missing/partial
samples, deadline, nonfinite timing, any nonpositive weighted paired gain, median
<= twice full-range spread, or median <50 us/token. Full arms additionally require
32/32 HF greedy tokens AND 27/27 layers in EVERY invocation. Warmup must be 32/32.
Q: spec 0:qkv_5, 27*d_qkv >=50. gamma/tree: independent 0:qkv_5 and 0:router_5,
27*d_qkv+26*d_router >=50. Both uses that same weighted net, never summed standalone
screen medians. Five interleaved pairs alternate control/candidate order. Full
candidate rounds include default/candidate/fleet (reversed on alternate rounds),
so both comparisons are paired; the final checkpoint is >=70 us/token, every gain
positive and median >2*full-range spread, reported separately for wall and device.
A >=100-us
screen is labelled separately. A screen is diagnostic, never a token-speed claim.
Only the default/fleet baseline and final default stamps run without a screen gate.

Control/candidate phase graphs use the prior decider flags EV_STRIDE=32 and
ACQUIRE_DOMAIN_RELAXED. Full-token controls/candidates use today's default flags.
Native-prefix capture creates fixed input files ONCE from unmodified ETX; separate
fresh processes reload those same bytes before each single phase invocation.
Outputs are byte-compared before the screen can pass. No appended repeated fold.

Inputs are prepared ON THE BOX by the prep_* jobs below (the ds_gap.py recipes): docs/DATA.md:59-64
says the fleet reference outputs (~70 MB, ~2 GPU minutes) and the packed weights (31 GB, ~1 min download +
~1 min pack) were never kept, and 31 GB cannot be staged from the laptop. Both are produced from the PINNED
fleet export (ETX_DSV2_FLEET, default ~/fleet-3fb67a6, src hash verified by `source`) with
deepseek-ai/DeepSeek-Coder-V2-Lite-Base, into ETX_DSV2_DATA (default ~/dsv2-data), then SHA256SUMS is written
and `preflight` re-verifies every file before the first timed run. A prep step is skipped when its outputs
already exist (a re-run on the same box), never faked: the 32/32 gate checks the goldens against the model.

Stage the box from the laptop with examples/dsv2lite/stage_inplace.sh <ip> (clean tree only; pushes the
pinned export and SOURCE_IDENTITY), then start the runner.

All stdout/stderr and raw traces stay under gpu_jobs.py's ETX_SESS_LOGDIR.
Run (after local validation, balance check, and budget authorization):
examples/dsv2lite/stage_inplace.sh <ip> && ETX_SESS_MAX_SECONDS=7200 tools/session.sh start <ip> examples/sessions/ds_inplace.py 0
Use --dry-run locally. Delete the VM immediately after completion or deadline.
"""
MAX_SESSION_SECONDS = 7200
ARMS = ("Q", "gamma", "tree", "both")
JOBS = []


# The runner forwards only ETX_SESS_MAX_SECONDS, so the paths get their defaults in every job's shell.
PATHS = (': "${ETX_DSV2_FLEET:=$HOME/fleet-3fb67a6}"; : "${ETX_DSV2_DATA:=$HOME/dsv2-data}"; '
         "export ETX_DSV2_FLEET ETX_DSV2_DATA; ")
PREP_PY = "$HOME/dsv2-prep/.venv/bin/python"
MODEL = "$HOME/models/dsv2-lite-base"


def job(name, cmd, *, lane="cpu", after=(), gate=None, timeout=600, summary=()):
    JOBS.append(dict(name=name, lane=lane, gpu=0 if lane == "gpu" else None,
                     after=list(after), gate=gate, timeout=timeout, quiet_s=90,
                     cmd=PATHS + cmd, summary=list(summary)))


def add(name, operation, **kw):
    job(name, f".venv/bin/python -u -m examples.dsv2lite.inplace_session {operation}",
        summary=[r"^(SOURCE|PREFLIGHT|BUILD|BASELINE|SCREEN|FULL|TRACE|REPORT) "], **kw)


# ---------------------------------------------------------------- environments (cpu); recipes from ds_gap.py
job("apt", "python3 -c 'import ensurepip' 2>/dev/null || "
           "{ sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }")
job("etx_env", ".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
               ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; "
               ".venv/bin/python -c 'import etx; print(\"etx ok\")'", after=("apt",))
job("prep_env", f"{PREP_PY} -c 1 2>/dev/null || {{ rm -rf $HOME/dsv2-prep; python3 -m venv $HOME/dsv2-prep/.venv; }}; "
                ". $HOME/dsv2-prep/.venv/bin/activate; "
                "pip install -q --upgrade pip; pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer; "
                "python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; "
                "pip install -q 'transformers>=4.39,<5' 'tokenizers<0.20' accelerate safetensors; "
                "SP=$(python3 -c 'import site;print(site.getsitepackages()[0])'); mkdir -p $SP/flash_attn; "
                "echo '# stub: satisfies transformers check_imports only' > $SP/flash_attn/__init__.py; "
                "python3 -c 'import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'",
    after=("apt",), timeout=1800)
add("source", "source", after=("etx_env",), timeout=300)

# ---------------------------------------------------------------- inputs, produced on the box from the pinned export
job("prep_dl", f"HF_HUB_ENABLE_HF_TRANSFER=1 {PREP_PY} -c \"from huggingface_hub import snapshot_download; "
               f"snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', local_dir='{MODEL}', "
               "allow_patterns=['*.json', '*.safetensors', '*.py', 'tokenizer*'], max_workers=8); print('DL-DONE')\" "
               f"&& du -sh {MODEL} && test -f {MODEL}/config.json", after=("prep_env",), timeout=1800)
job("prep_ref", 'mkdir -p "$ETX_DSV2_DATA" && cd "$ETX_DSV2_FLEET" && '
                '{ test -s "$ETX_DSV2_DATA/fleet_cache.bin" -a -s "$ETX_DSV2_DATA/golden_tokens.txt" '
                '-a -s "$ETX_DSV2_DATA/golden_hidden.bin" || '
                f'HIP_VISIBLE_DEVICES=$GPU {PREP_PY} src/host/reference_run.py --model {MODEL} '
                '--out "$ETX_DSV2_DATA/golden.npz"; } && ls -la "$ETX_DSV2_DATA"',
    lane="gpu", after=("prep_dl", "source"), timeout=1200,
    summary=[r"smallest top-2 margin|wrote |golden"])
job("prep_pack", 'cd "$ETX_DSV2_FLEET" && { test -s "$ETX_DSV2_DATA/weights.bin" -a -s "$ETX_DSV2_DATA/weights.bin.manifest" || '
                 f'{PREP_PY} src/host/pack_weights.py --model {MODEL} --out "$ETX_DSV2_DATA/weights.bin"; }} '
                 '&& ls -la "$ETX_DSV2_DATA"', after=("prep_ref",), timeout=1200, summary=[r"wrote |expert stride"])
job("prep_sums", 'cd "$ETX_DSV2_DATA" && sha256sum weights.bin weights.bin.manifest fleet_cache.bin golden_tokens.txt '
                 'golden_hidden.bin > SHA256SUMS && cat SHA256SUMS', after=("prep_pack",), timeout=600)
add("preflight", "preflight", after=("prep_sums",), timeout=600)

# ---------------------------------------------------------------- builds (cpu, while the inputs are prepared)
for arm in ("default", "fleet", "stamps", *ARMS):
    add(f"build_{arm}", f"build {arm}", after=("source",), timeout=1200)
add("baseline", "baseline", lane="gpu", after=("preflight", "build_default", "build_fleet"), timeout=900)
add("capture", "capture", lane="gpu", after=("baseline",), timeout=600)
previous = "capture"
for arm in ARMS:
    add(f"screen_{arm}", f"screen {arm}", lane="gpu", after=(previous, f"build_{arm}"), timeout=1200)
    previous = f"screen_{arm}"
# All deciding microbenchmarks precede every candidate full-model run.
for arm in ARMS:
    add(f"full_{arm}", f"full {arm}", lane="gpu", after=(previous,),
        gate=(f"screen_{arm}", rf"^SCREEN arm={arm} GO=1(?: |$)"), timeout=900)
# A rejected screen skips its candidate but must not skip the final default trace.
add("trace", "trace", lane="gpu", after=(previous, "build_stamps"), timeout=300)
JOBS[-1]["after_any"] = [f"full_{arm}" for arm in ARMS]
add("report", "report", after=("baseline", "trace"), timeout=60)
JOBS[-1]["after_any"] = [f"full_{arm}" for arm in ARMS]
