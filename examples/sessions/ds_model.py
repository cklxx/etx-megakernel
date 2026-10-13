"""GPU session: calibrate the operation-level model of the DeepSeek-V2-Lite decode (docs/MODEL-dsv2.md).

GPU 0: the hardware primitives (bench/hw/hw_bench.hip: stream, shared, handoff, wg + clock, pair), then the in-situ
       phase benchmarks with the real tiles (ETX_DSV2_ISOLATE): every phase of layer 5 alone (one request), and the
       two-request pairs that explain the interleave's x1.4 (latency phase beside an expert stream, stream beside stream,
       latency beside latency).
GPU 1: request 0's reference + weights, request 1's reference, then hardware counters of whole tokens (one counter set per
       job, so an unsupported counter only loses its own job).
Each result feeds one row of docs/MODEL-dsv2.md section 5; the alternatives to check when a prediction misses are in
section 3 of the same file.

  python3 -m etx.tools.gpu_jobs examples/sessions/ds_model.py --gpus 0,1 --logdir ~/sess
"""
from examples.dsv2lite.acquire import ACQUIRE, RELAXED   # D1: safe sc1 unless an arm names RELAXED
FLEET = "~/fleet-mi300x"
JOBS = []


def job(**kw):
    JOBS.append(kw)


# ---------------------------------------------------------------- environment and data
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
job(name="ds_prep", gpu=1, after=["fleet_env", "dl_ds"], timeout=2400, cwd=FLEET,
    cmd=". .venv/bin/activate && mkdir -p build && "
        "{ test -f build/fleet_cache.bin -a -f build/golden_tokens.txt -a -f build/golden_hidden.bin || "
        "python3 src/host/reference_run.py --model $HOME/models/dsv2-lite-base --out build/golden.npz; } && "
        "python3 src/host/pack_weights.py --model $HOME/models/dsv2-lite-base --out build/weights.bin",
    summary=[r"smallest top-2 margin|wrote build/weights"])
job(name="ref_r1", gpu=1, after=["ds_prep"], timeout=2400,
    cmd="test -f ~/fleet-mi300x/build_r1/fleet_cache.bin || ~/fleet-mi300x/.venv/bin/python examples/dsv2lite/reference_rotated.py --fleet ~/fleet-mi300x --rotate 97 -- "
        "--model $HOME/models/dsv2-lite-base --out $HOME/fleet-mi300x/build_r1/golden.npz", summary=[r"wrote .*golden_tokens"])

# ---------------------------------------------------------------- hardware primitives (GPU 0, the quiet GPU, first)
job(name="build_hw", lane="cpu", timeout=600,
    cmd="hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/hw/hw_bench.hip -o $HOME/sess/hw_bench && ls -la $HOME/sess/hw_bench")
prev = None
for mode, t in (("stream", 300), ("shared", 300), ("handoff", 600), ("wg", 300), ("pair", 900)):
    job(name=f"hw_{mode}", gpu=0, after="build_hw", after_any=[prev] if prev else [], timeout=t,
        cmd=f"stdbuf -oL $HOME/sess/hw_bench {mode}", summary=[r"^RESULT"])
    prev = f"hw_{mode}"

# ---------------------------------------------------------------- in-situ phases with the real tiles
ONE = ";".join(f"0:{p}_5" for p in ("qkv", "qabs", "attn", "merge", "oproj", "router")) + ";0:gate_up_5+down_5;0:gate_up_5"
PAIRS = ";".join(["0:qkv_5", "0:attn_5", "0:router_5", "0:gate_up_5", "1:gate_up_5",
                  "0:qkv_5,1:gate_up_5", "0:attn_5,1:gate_up_5", "0:router_5,1:gate_up_5",
                  "0:gate_up_5,1:gate_up_5", "0:qkv_5,1:qkv_5", "0:attn_5,1:attn_5"])
job(name="build_iso1", lane="cpu", after="etx_env", timeout=1800,
    cmd=f"ETX_DSV2_ISOLATE='{ONE}' ETX_DSV2_ISOLATE_REPS=8 ETX_OUT=$HOME/sess/b_iso1 bash examples/dsv2lite/build.sh ~/fleet-mi300x "
        f"{ACQUIRE} > $HOME/sess/b_iso1.buildlog 2>&1; tail -2 $HOME/sess/b_iso1.buildlog; test -x $HOME/sess/b_iso1/run")
job(name="build_iso2", lane="cpu", after="etx_env", timeout=1800,
    cmd=f"ETX_DSV2_REQUESTS=2 ETX_DSV2_COUPLE=none ETX_RELAY_PLAN=off ETX_FLEET_PATCH=gemv_noinline ETX_DSV2_ISOLATE='{PAIRS}' "
        "ETX_DSV2_ISOLATE_REPS=8 ETX_OUT=$HOME/sess/b_iso2 bash examples/dsv2lite/build.sh ~/fleet-mi300x "
        f"{ACQUIRE} -DETX_DS_REQUESTS=2 -DETX_SLOT_BY_CU > $HOME/sess/b_iso2.buildlog 2>&1; "
        "tail -2 $HOME/sess/b_iso2.buildlog; test -x $HOME/sess/b_iso2/run")
job(name="iso1", gpu=0, after=["ds_prep", "build_iso1"], after_any=[prev], timeout=600, cwd=FLEET,
    cmd="ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/iso1_trace.bin $HOME/sess/b_iso1/run --tokens 2 --context 1024; "
        "cp $HOME/sess/b_iso1/plan.json $HOME/sess/iso1_plan.json; cd ~/etx && "
        ".venv/bin/python -m examples.dsv2lite.isolate_report $HOME/sess/iso1_plan.json $HOME/sess/iso1_trace.bin",
    summary=[r"^tokens:", r"^ *\d+ +\d+ +\w+_5 "])
job(name="iso2", gpu=0, after=["ds_prep", "ref_r1", "build_iso2", "iso1"], timeout=600, cwd=FLEET,
    cmd="ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/iso2_trace.bin $HOME/sess/b_iso2/run --tokens 2 --context 1024 --r1-dir build_r1; "
        "cp $HOME/sess/b_iso2/plan.json $HOME/sess/iso2_plan.json; cd ~/etx && "
        ".venv/bin/python -m examples.dsv2lite.isolate_report $HOME/sess/iso2_plan.json $HOME/sess/iso2_trace.bin",
    summary=[r"tokens:", r"^ *\d+ +\d+ +\w+_5 "])

# ---------------------------------------------------------------- whole-token builds for the counters (GPU 1)
job(name="build_x1", lane="cpu", after="etx_env", timeout=1800,
    cmd=f"ETX_OUT=$HOME/sess/b_x1 bash examples/dsv2lite/build.sh ~/fleet-mi300x {ACQUIRE} > $HOME/sess/b_x1.buildlog 2>&1; "
        "test -x $HOME/sess/b_x1/run")
job(name="build_x2", lane="cpu", after="etx_env", timeout=1800,
    cmd="ETX_DSV2_REQUESTS=2 ETX_DSV2_COUPLE=none ETX_RELAY_PLAN=off ETX_FLEET_PATCH=gemv_noinline ETX_OUT=$HOME/sess/b_x2 "
        f"bash examples/dsv2lite/build.sh ~/fleet-mi300x {ACQUIRE} -DETX_DS_REQUESTS=2 -DETX_SLOT_BY_CU "
        "> $HOME/sess/b_x2.buildlog 2>&1; test -x $HOME/sess/b_x2/run")
job(name="pmc_list", gpu=1, after="ref_r1", timeout=300, cmd="rocprofv3 --list-avail 2>&1 | head -400")
COUNTERS = {
    "insts": "SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM_RD SQ_INSTS_LDS",
    "busy": "GRBM_GUI_ACTIVE SQ_BUSY_CYCLES SQ_WAIT_INST_ANY SQ_ACTIVE_INST_VALU",
    "l2": "TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum",
    "icache": "SQC_ICACHE_HITS SQC_ICACHE_MISSES SQC_DCACHE_HITS SQC_DCACHE_MISSES",
    "tlb": "TCP_UTCL1_TRANSLATION_HIT_sum TCP_UTCL1_TRANSLATION_MISS_sum",
}
prev1 = "pmc_list"
for build, args in (("x1", ""), ("x2", " --r1-dir build_r1")):
    for cname, ctrs in COUNTERS.items():
        name = f"pmc_{build}_{cname}"
        job(name=name, gpu=1, after=[f"build_{build}", "ds_prep", "ref_r1"], after_any=[prev1], timeout=600, cwd=FLEET,
            cmd=f"rocprofv3 --pmc {ctrs} -d $HOME/sess/pmc_{build}_{cname} -o pmc --output-format csv -- "
                f"$HOME/sess/b_{build}/run --tokens 4 --context 1024{args} > $HOME/sess/pmc_{build}_{cname}.run.log 2>&1; "
                f"find $HOME/sess/pmc_{build}_{cname} -name '*counter_collection.csv' | head -1 | xargs -r cat | cut -c1-400 | head -20",
            summary=[r"etx_megakernel|Counter_Name|rror"])
        prev1 = name
