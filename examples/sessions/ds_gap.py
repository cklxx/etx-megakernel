"""GPU session: why the ETX port of DeepSeek-V2-Lite is 0.15 ms/token behind fleet, what capping the megakernel at
256 registers per lane costs (the precondition of two-request interleaving), and how many experts concurrent requests
really share (the input of the batch2 estimate). Runs on one 2x MI300X VM in about 40 minutes:

  GPU 0  every timing A/B, pinned and alternated round by round (fleet itself, ETX sc0 / sc1 acquire, ETX capped at
         256 registers, params through global memory, no relay CU)
  GPU 1  the reference + weight packing first (everything waits on it), then the traces (ETX with prologue stamps,
         fleet) and the routing-overlap run
  cpu    environment, the DeepSeek download, the six builds, the analysis

Decisions (written before the session, docs/PLAN-v3.md section 8):
  * cap256 within 3% of the relaxed build -> the interleave keeps fleet's tiles; slower -> rework the GEMV first
  * measured 2-sequence expert union: fed to whatif.py; interleave is built if it still beats batch2 there
  * qkv prologue (ETX vs fleet) explains the +3.7 us/task -> fix it in the shim; otherwise look at the stream part

  python3 -m etx.tools.gpu_jobs examples/sessions/ds_gap.py --gpus 0,1 --logdir ~/sess
"""
from examples.dsv2lite.acquire import ACQUIRE, RELAXED   # D1: safe sc1 unless an arm names RELAXED
HOME = "$HOME"
FLEET = "~/fleet-mi300x"
RUN_ARGS = "--tokens 32 --context 1024 --repeat 2"
SUM_RUN = [r"^per-token", r"^tokens:", r"consecutive layers"]

JOBS = []


def job(**kw):
    JOBS.append(kw)


# ---------------------------------------------------------------- environment and data (cpu lane)
job(name="apt", lane="cpu", timeout=600,
    cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }")
job(name="etx_env", lane="cpu", reads="apt", timeout=600,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
        ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx; print(\"etx ok\")'")
job(name="fleet_env", lane="cpu", reads="apt", timeout=1800, cwd=FLEET,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv .venv; }; . .venv/bin/activate; "
        "pip install -q --upgrade pip; pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer; "
        "python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; "
        "pip install -q 'transformers>=4.39,<5' 'tokenizers<0.20' accelerate safetensors; "
        "SP=$(python3 -c 'import site;print(site.getsitepackages()[0])'); mkdir -p $SP/flash_attn; "
        "echo '# stub: satisfies transformers check_imports only' > $SP/flash_attn/__init__.py; "
        "python3 -c 'import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'")
job(name="dl_ds", lane="cpu", reads="fleet_env", timeout=1800, cwd=FLEET,
    cmd="HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', local_dir='$HOME/models/dsv2-lite-base', "
        "allow_patterns=['*.json', '*.safetensors', '*.py', 'tokenizer*'], max_workers=8); print('DL-DONE')\" && du -sh ~/models/dsv2-lite-base "
        "&& test -f ~/models/dsv2-lite-base/config.json")

# ---------------------------------------------------------------- builds (cpu lane, parallel)
BUILDS = {   # name -> (env, extra hipcc flags)
    "relaxed": ("", RELAXED),
    "sc1": ("", ""),
    "cap256": ("", f"{ACQUIRE} -DETX_MIN_WG_PER_CU=2"),
    "nocp": ("", f"{ACQUIRE} -DFLEET_CONST_PARAMS=0"),
    "relayoff": ("ETX_RELAY_PLAN=off", f"{ACQUIRE}"),
    "pro": ("", f"{ACQUIRE} -DETX_DS_PRO=1"),
}
for b, (env, flags) in BUILDS.items():
    job(name=f"build_{b}", lane="cpu", reads="etx_env", timeout=1200,
        cmd=f"{env} ETX_OUT=$HOME/sess/b_{b} bash examples/dsv2lite/build.sh ~/fleet-mi300x -Rpass-analysis=kernel-resource-usage {flags} "
            f"&& ls -la $HOME/sess/b_{b}/run",
        summary=[r"VGPRs: |AGPRs: |ScratchSize|Occupancy|built "])
job(name="build_fleet", lane="cpu", reads="fleet_env", timeout=1200, cwd=FLEET,
    cmd=". .venv/bin/activate && python3 src/host/taskgraph.py --kv-chunks 16 --emit build/taskgraph_d16.bin | tail -1 && "
        "hipcc --offload-arch=gfx942 -O3 -std=c++17 -DFLEET_NT_WEIGHTS=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip "
        "-o build/fleet_decode_nt && ls -la build/fleet_decode_nt")

# ---------------------------------------------------------------- reference and weights (GPU 1: everything below needs them)
job(name="ds_prep", gpu=1, reads=["fleet_env", "dl_ds"], timeout=2400, cwd=FLEET,
    cmd=". .venv/bin/activate && mkdir -p build && "
        "{ test -f build/fleet_cache.bin -a -f build/golden_tokens.txt -a -f build/golden_hidden.bin || "
        "python3 src/host/reference_run.py --model $HOME/models/dsv2-lite-base --out build/golden.npz; } && "
        "python3 src/host/pack_weights.py --model $HOME/models/dsv2-lite-base --out build/weights.bin",
    summary=[r"smallest top-2 margin|wrote build/|expert stride"])

# ---------------------------------------------------------------- timing A/B on GPU 0, alternated
VARIANTS = ["fleet", "relaxed", "sc1", "cap256", "nocp", "relayoff"]
prev = None
for rnd in (1, 2):
    for v in VARIANTS:
        run = ("./build/fleet_decode_nt --graph build/taskgraph_d16.bin --repeat 2" if v == "fleet"
               else f"$HOME/sess/b_{v}/run {RUN_ARGS}")
        deps = ["ds_prep", "build_fleet" if v == "fleet" else f"build_{v}"]
        job(name=f"t{rnd}_{v}", gpu=0, reads=deps, timeout=600, cwd=FLEET, cmd=run, summary=SUM_RUN)
        prev = f"t{rnd}_{v}"

# ---------------------------------------------------------------- traces and routing on GPU 1
job(name="trace_etx", gpu=1, reads=["ds_prep", "build_pro"], timeout=600, cwd=FLEET,
    cmd="ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/etx_trace.bin ETX_DS_PROLOGUE=$HOME/sess/etx_pro.bin "
        f"$HOME/sess/b_pro/run --tokens 8 --context 1024 && cp $HOME/sess/b_pro/plan.json $HOME/sess/etx_plan.json",
    summary=SUM_RUN + [r"wrote prologue"])
job(name="trace_fleet", gpu=1, reads=["ds_prep", "build_fleet", "trace_etx"], timeout=600, cwd=FLEET,
    cmd="./build/fleet_decode_nt --graph build/taskgraph_d16.bin --repeat 1 --trace $HOME/sess/fleet_trace.bin && "
        "cp build/taskgraph_d16.bin $HOME/sess/", summary=SUM_RUN)
job(name="routing", gpu=1, reads=["ds_prep", "trace_fleet"], timeout=1800,
    cmd="~/fleet-mi300x/.venv/bin/python examples/dsv2lite/routing_overlap.py --model $HOME/models/dsv2-lite-base "
        "--batch 4 --steps 32 --json $HOME/sess/routing.json", summary=[r"^ROUTING"])

# ---------------------------------------------------------------- analysis (cpu)
job(name="compare", lane="cpu", reads=["trace_etx", "trace_fleet"], timeout=600,
    cmd=".venv/bin/python -m examples.dsv2lite.compare_fleet --fleet-trace $HOME/sess/fleet_trace.bin "
        "--fleet-graph $HOME/sess/taskgraph_d16.bin --fleet-src ~/fleet-mi300x/src --etx-plan $HOME/sess/etx_plan.json "
        "--etx-trace $HOME/sess/etx_trace.bin --etx-prologue $HOME/sess/etx_pro.bin",
    summary=[r"^token:", r"QKV_FUSED|NORM_ROUTER|EXPERT_GATE_UP"])
job(name="whatif", lane="cpu", reads=["?routing"], timeout=300,
    cmd=".venv/bin/python -m examples.dsv2lite.whatif --overlap ~/etx/examples/vllm_llm/results/2026-09-27/overlap.log "
        "--routing $HOME/sess/routing.json", summary=[r"measured routed|interleave  f=1.0[0-4]|batch2|Reading"])
