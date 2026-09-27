"""GPU session: two DeepSeek-V2-Lite requests interleaved in one ETX megakernel (docs/PLAN-v3.md 4.4 / section 8).

Builds (cpu lane, parallel):
  x1          one request, fleet's tiles, sc0 domain acquire: today's ETX figure (3.73-3.75 ms)
  x1cap       one request, gemv_noinline patch + 2 workgroups/CU launch bounds: the real cost of <= 256 registers per
              lane (the 2026-09-27 "cap256" build never reached 2 waves/SIMD: 270 registers)
  x2_<c>      two requests, 2 workgroups per CU paired by CU (ETX_SLOT_BY_CU), no relay, coupling c = router | experts |
              none (examples/dsv2lite/graph.py ETX_DSV2_COUPLE)
  fleet       fleet itself
GPU 0: request 1's reference (rotated prompt), then every timing run, alternated over two rounds.
GPU 1: request 0's reference + weight packing, then a traced x2 run and its schedule analysis.

Pass / decide (written before the session):
  * correctness: every x2 run 32/32 tokens and 27/27 layers for BOTH requests, or the result does not count
  * x2 tokens/s against fleet at batch 1 (281) and vLLM at batch 2 (376, same prompts); the best coupling is kept
  * x1cap vs x1: the price of the register cap alone; x2's per-request latency vs x1 says what interleaving costs a request

  python3 -m etx.tools.gpu_jobs examples/sessions/ds_interleave.py --gpus 0,1 --logdir ~/sess
"""
FLEET = "~/fleet-mi300x"
RUN1 = "--tokens 32 --context 1024 --repeat 2"
RUN2 = RUN1 + " --r1-dir build_r1"
SUM1 = [r"^per-token", r"^tokens:", r"consecutive layers"]
SUM2 = [r"^r[01] per-token", r"^r[01] tokens:", r"consecutive layers", r"^RESULT"]

JOBS = []


def job(**kw):
    JOBS.append(kw)


# ---------------------------------------------------------------- environment and data (cpu lane)
job(name="apt", lane="cpu", timeout=600,
    cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }")
job(name="etx_env", lane="cpu", after="apt", timeout=600,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
        ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx; print(\"etx ok\")'")
job(name="fleet_env", lane="cpu", after="apt", timeout=1800, cwd=FLEET,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv .venv; }; . .venv/bin/activate && "
        "pip install -q --upgrade pip && pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer && "
        "{ python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; } && "
        "pip install -q 'transformers>=4.39,<5' 'tokenizers<0.20' accelerate safetensors && "
        "SP=$(python3 -c 'import site;print(site.getsitepackages()[0])') && mkdir -p $SP/flash_attn && "
        "echo '# stub: satisfies transformers check_imports only' > $SP/flash_attn/__init__.py && "
        "python3 -c 'import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'")
job(name="dl_ds", lane="cpu", after="fleet_env", timeout=1800, cwd=FLEET,
    cmd="HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', local_dir='$HOME/models/dsv2-lite-base', "
        "allow_patterns=['*.json', '*.safetensors', '*.py', 'tokenizer*'], max_workers=8); print('DL-DONE')\" && "
        "du -sh ~/models/dsv2-lite-base && test -f ~/models/dsv2-lite-base/config.json")

# ---------------------------------------------------------------- builds (cpu lane)
X2 = "ETX_DSV2_REQUESTS=2 ETX_RELAY_PLAN=off ETX_FLEET_PATCH=gemv_noinline"
X2F = "-DETX_ACQUIRE_DOMAIN_RELAXED -DETX_DS_REQUESTS=2 -DETX_SLOT_BY_CU"
BUILDS = {   # name -> (env, hipcc flags)
    "x1": ("", "-DETX_ACQUIRE_DOMAIN_RELAXED"),
    "x1cap": ("ETX_FLEET_PATCH=gemv_noinline", "-DETX_ACQUIRE_DOMAIN_RELAXED -DETX_MIN_WG_PER_CU=2"),
    "x2_router": (f"{X2} ETX_DSV2_COUPLE=router", X2F),
    "x2_experts": (f"{X2} ETX_DSV2_COUPLE=experts", X2F),
    "x2_none": (f"{X2} ETX_DSV2_COUPLE=none", X2F),
}
for b, (env, flags) in BUILDS.items():
    # the occupancy line is the check that matters for the paired builds: it must say 2 waves/SIMD
    job(name=f"build_{b}", lane="cpu", after="etx_env", timeout=1800,
        cmd=f"{env} ETX_OUT=$HOME/sess/b_{b} bash examples/dsv2lite/build.sh ~/fleet-mi300x -Rpass-analysis=kernel-resource-usage {flags} "
            f"2>&1 | tee $HOME/sess/b_{b}.buildlog | grep -A12 'Function Name: etx_megakernel_d0' | grep -E 'VGPRs:|AGPRs:|Scratch|Occupancy'; "
            f"test -x $HOME/sess/b_{b}/run",
        summary=[r"VGPRs:|AGPRs:|Scratch|Occupancy"])
job(name="build_fleet", lane="cpu", after="fleet_env", timeout=1200, cwd=FLEET,
    cmd=". .venv/bin/activate && python3 src/host/taskgraph.py --kv-chunks 16 --emit build/taskgraph_d16.bin | tail -1 && "
        "hipcc --offload-arch=gfx942 -O3 -std=c++17 -DFLEET_NT_WEIGHTS=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip "
        "-o build/fleet_decode_nt && ls -la build/fleet_decode_nt")

# ---------------------------------------------------------------- references: both GPUs at once
job(name="ds_prep", gpu=1, after=["fleet_env", "dl_ds"], timeout=2400, cwd=FLEET,
    cmd=". .venv/bin/activate && mkdir -p build && "
        "python3 src/host/reference_run.py --model $HOME/models/dsv2-lite-base --out build/golden.npz && "
        "python3 src/host/pack_weights.py --model $HOME/models/dsv2-lite-base --out build/weights.bin",
    summary=[r"smallest top-2 margin|wrote build/|expert stride"])
job(name="ref_r1", gpu=0, after=["fleet_env", "dl_ds"], timeout=2400,
    cmd="~/fleet-mi300x/.venv/bin/python examples/dsv2lite/reference_rotated.py --fleet ~/fleet-mi300x --rotate 97 -- "
        "--model $HOME/models/dsv2-lite-base --out $HOME/fleet-mi300x/build_r1/golden.npz && ls -la ~/fleet-mi300x/build_r1",
    summary=[r"prompt rotated|smallest top-2 margin|wrote .*golden_tokens"])

# ---------------------------------------------------------------- timing on GPU 0, alternated
VARIANTS = ["fleet", "x1", "x1cap", "x2_router", "x2_experts", "x2_none"]
prev = None
for rnd in (1, 2):
    for v in VARIANTS:
        if v == "fleet":
            cmd, summ, deps = "./build/fleet_decode_nt --graph build/taskgraph_d16.bin --repeat 2", SUM1, ["ds_prep", "build_fleet"]
        elif v.startswith("x2"):
            cmd, summ, deps = f"$HOME/sess/b_{v}/run {RUN2}", SUM2, ["ds_prep", "ref_r1", f"build_{v}"]
        else:
            cmd, summ, deps = f"$HOME/sess/b_{v}/run {RUN1}", SUM1, ["ds_prep", f"build_{v}"]
        job(name=f"t{rnd}_{v}", gpu=0, after=deps, after_any=[prev] if prev else [], timeout=600, cwd=FLEET, cmd=cmd, summary=summ)
        prev = f"t{rnd}_{v}"

# ---------------------------------------------------------------- trace of the interleave (GPU 1)
job(name="trace_x2", gpu=1, after=["ds_prep", "ref_r1", "build_x2_router"], timeout=600, cwd=FLEET,
    cmd=f"ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/x2_trace.bin $HOME/sess/b_x2_router/run --tokens 8 --context 1024 --r1-dir build_r1 && "
        "cp $HOME/sess/b_x2_router/plan.json $HOME/sess/x2_plan.json", summary=SUM2)
job(name="timeline_x2", lane="cpu", after="trace_x2", timeout=600,
    cmd=".venv/bin/python -m examples.dsv2lite.bytes > $HOME/sess/ds_bytes.json && "
        ".venv/bin/python -m etx.tools.timeline $HOME/sess/x2_plan.json $HOME/sess/x2_trace.bin --bytes $HOME/sess/ds_bytes.json "
        "--grids '_5$' --html $HOME/sess/x2_L5.html --json $HOME/sess/x2_L5.json",
    summary=[r"^window|critical path:|^memory:|below 25%"])
