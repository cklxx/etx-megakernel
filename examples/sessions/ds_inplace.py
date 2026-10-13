"""PREPARED, NOT RUN. One MI300X, GPU 0 only. No provisioning; the model download and reference run happen on the box.

What it measures (examples/dsv2lite/inplace_session.py): the ETX DeepSeek port with every lever that passed its local
gate (examples/dsv2lite/results/levers/*.json), as whole tokens, against default ETX and shipped fleet (3fb67a6).
  1. baseline        5 interleaved rounds default / fleet
  2. pairs all       5 interleaved rounds default / all / fleet: THE decision
  3. trace default, trace all   one stamped token each (STAMPS2 builds): the measured per-stage table
  4. pairs <lever>   5 rounds default / lever for Q, Qall, N, Nall, argmax, topk, prep: attribution, no verdict
  5. report          verdict + measured per-stage savings next to stage_table.json's expected ones
Every invocation is a fresh process and must show 32/32 HF-greedy tokens, 27/27 layers and a 32/32 warm-up.

Decision rule, fixed before the run (all vs default, device basis): every paired gain > 0, median >= 70 us/token,
median > 2 x full-range spread. Expected from the local account: 531.1 us simulated / 533.4 us arithmetic
(stage_table.md; BYTES rows are upper bounds). Reject `all` on any correctness gate, on scratch in its build, or on
the rule; then the single-lever pairs say which lever fails. A run that does not finish is NOT COMPLETE.

Budget [planning cap]: 120 GPU minutes (MAX_SESSION_SECONDS), $5.98 at $2.99/h (Hot Aisle 1x; balance $6.29 on
2026-10-13) or $3.98 at $1.99/h. Expected use [E]: ~25 min environments + inputs (builds run in parallel), then
10 + 15 + 2 + 70 = 97 invocations x ~8 s (results_2026-09-27b runner: 7-8 s per 32-token invocation) = ~13 min,
plus builds' tail: ~45-60 min total.

Inputs are prepared ON THE BOX by the prep_* jobs (the ds_gap.py recipes): docs/DATA.md:59-64 says the fleet reference
outputs (~70 MB, ~2 GPU minutes) and the packed weights (31 GB, ~1 min download + ~1 min pack) were never kept. Both
come from the PINNED fleet export (ETX_DSV2_FLEET, default ~/fleet-3fb67a6, src hash verified by `source`) and
deepseek-ai/DeepSeek-Coder-V2-Lite-Base, into ETX_DSV2_DATA (default ~/dsv2-data); SHA256SUMS is written and
`preflight` re-verifies every file before the first timed run.

Run (after local validation, balance check and budget authorization):
  examples/dsv2lite/stage_inplace.sh <ip> && ETX_SESS_MAX_SECONDS=7200 tools/session.sh start <ip> examples/sessions/ds_inplace.py 0
Use --dry-run locally. Fetch the session directory into the tree before deleting the VM; delete it right after.
"""
MAX_SESSION_SECONDS = 7200
SINGLE = ("Q", "Qall", "N", "Nall", "argmax", "topk", "prep", "relaxed")   # relaxed: D1, the sc0 acquire by name
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
        summary=[r"^(SOURCE|PREFLIGHT|BUILD|PAIRS|TRACE|REPORT) "], **kw)


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
DITEMS = ("coh", "d4")   # D2: three-way pairs default / relaxed / coh; D4: pairs default / d4 (docs/DSV2-D1-D5.md)
for arm in ("default", "fleet", "all", *SINGLE, *DITEMS):
    add(f"build_{arm}", f"build {arm}", after=("source",), timeout=1200)
for arm in ("default", "all"):
    add(f"build_{arm}_stamps", f"build {arm} --stamped", after=("source",), timeout=1200)

# ---------------------------------------------------------------- GPU, in order of importance
add("baseline", "pairs baseline", lane="gpu", after=("preflight", "build_default", "build_fleet"), timeout=900)
# A failed reference or decision must not cost the data that explain it: later GPU jobs follow "whatever the result".
add("pairs_all", "pairs all", lane="gpu", after=("preflight", "build_default", "build_all"), timeout=1200)
JOBS[-1]["after_any"] = ["baseline"]
add("trace_default", "trace default", lane="gpu", after=("build_default_stamps",), timeout=300)
JOBS[-1]["after_any"] = ["pairs_all"]
add("trace_all", "trace all", lane="gpu", after=("build_all_stamps",), timeout=300)
JOBS[-1]["after_any"] = ["trace_default"]
previous = "trace_all"
for arm in DITEMS:                       # new conditions before the attributions already measured on 2026-10-13
    needs = [f"build_{arm}"] + (["build_relaxed"] if arm == "coh" else [])
    add(f"pairs_{arm}", f"pairs {arm}", lane="gpu", after=(previous, *needs), timeout=1200)
    JOBS[-1]["after_any"] = [previous]
    JOBS[-1]["after"] = needs
    previous = f"pairs_{arm}"
add("d4trace", "d4trace", lane="gpu", after=(previous, "build_d4"), timeout=300)   # D4 criterion (a)
JOBS[-1]["after_any"] = [previous]
JOBS[-1]["after"] = ["build_d4"]
previous = "d4trace"
for arm in SINGLE:                       # attribution: each lever alone, after everything that decides
    add(f"pairs_{arm}", f"pairs {arm}", lane="gpu", after=(previous, f"build_{arm}"), timeout=900)
    JOBS[-1]["after_any"] = [previous]
    JOBS[-1]["after"] = [f"build_{arm}"]
    previous = f"pairs_{arm}"
add("report", "report", after=("pairs_all",), timeout=120)
JOBS[-1]["after_any"] = ["trace_default", "trace_all", *(f"pairs_{a}" for a in DITEMS), "d4trace",
                         *(f"pairs_{a}" for a in SINGLE)]
