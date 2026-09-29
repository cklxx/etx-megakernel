"""GPU session: the first DSL build, one run (docs/MODEL-vllm-kernels.md section 13). Build: xcd + chain + fused q/k
norm + rope (grid) + the DSL streaming GEMVs (etx.ops.gemv, silu fused into gate_up) + the DSL top-k (etx.ops.topk). Gates: 32/32 tokens against HF
greedy and --check-dsl (the last layer's GEMVs within 2 bf16 ulps of a CPU reference). Numbers: the fused step, and
the same tiles launched one by one (--unfused). Pre-registered: simulated 4.42 ms; realistic ~4.9 (x1.11); today's best
fused 5.86, vLLM (same fusion) 4.67.

  tools/session.sh push <ip>; tools/session.sh start <ip> examples/sessions/dsl_gemv.py
"""
JOBS = []


def job(**kw):
    JOBS.append(kw)


LLM = "$HOME/llm"
PY = "~/llmenv/bin/python"
V = "VLLM=$HOME/vllm ETX_TORCH_HEADERS=$HOME/rocm-headers/pytorch"
RES = r"^RESULT|generated tokens match|per-token latency|CHECK-DSL|beyond 2 ulps|wrong \("
M, HF = "qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B"

job(name="apt", lane="cpu", timeout=600,
    cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }")
job(name="etx_env", lane="cpu", after="apt", timeout=600,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
        ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx; print(\"etx ok\")'")
job(name="llm_env", lane="cpu", after="apt", timeout=1800,
    cmd="[ -d ~/llmenv ] || python3 -m venv ~/llmenv; . ~/llmenv/bin/activate && pip install -q --upgrade pip && "
        "pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer safetensors && "
        "{ python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; } && "
        "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'")
job(name="dl", lane="cpu", after="llm_env", timeout=2400,
    cmd=f"HF_HUB_ENABLE_HF_TRANSFER=1 {PY} -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF}', local_dir='{LLM}/{M}/hf', allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'], "
        f"max_workers=16); print('DL-DONE')\" && test -f {LLM}/{M}/hf/config.json")
job(name="prep", gpu=0, after=["dl", "etx_env"], timeout=2400,
    cmd=f"{PY} examples/vllm_llm/prep.py --src {LLM}/{M}/hf --out {LLM}/{M} --maxlen 1072 && "
        f"cp examples/llm/results/golden_c1024_{M}.txt {LLM}/{M}/golden_c1024.txt && ls -la {LLM}/{M}/vl.bin", summary=[r"wrote .*vl.bin"])
B = "moe_dslgemv"
job(name=f"build_{B}", lane="cpu", after=["etx_env", "dl"], timeout=2400,
    cmd=f"{V} ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_QKNR=grid ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_OUT=$HOME/sess/b_{B} bash examples/vllm_llm/build.sh {LLM}/{M} "
        f"> $HOME/sess/b_{B}.buildlog 2>&1; grep -E 'tasks=|Function Name|VGPRs|error' $HOME/sess/b_{B}.buildlog | tail -4; "
        f"cp $HOME/sess/b_{B}.buildlog $HOME/sess/buildlog_{B}.txt; cp $HOME/sess/b_{B}/vl_plan.txt $HOME/sess/vlplan_{B}.txt; "
        f"cp $HOME/sess/b_{B}/plan.json $HOME/sess/plan_{B}.json; cp $HOME/sess/b_{B}/dsl_tiles.hip $HOME/sess/; test -x $HOME/sess/b_{B}/run",
    summary=[r"tasks=|megakernel|VGPRs Spill"])
RUN = f"$HOME/sess/b_{B}/run {LLM}/{M} $HOME/sess/b_{B}/vl_plan.txt"
# one run each: the fused step with the correctness gates, then the same tiles one launch at a time
job(name="run_fused", gpu=0, after=[f"build_{B}", "prep"], timeout=900, cmd=f"stdbuf -oL {RUN} --repeat 1 --check-dsl", summary=[RES])
job(name="run_unfused", gpu=0, after=[f"build_{B}", "prep"], after_any=["run_fused"], timeout=900, cmd=f"stdbuf -oL {RUN} --repeat 1 --unfused", summary=[RES])
job(name="trace_fused", gpu=0, after=[f"build_{B}", "prep"], after_any=["run_unfused"], timeout=900,
    cmd=f"ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/trace_{B}.bin stdbuf -oL {RUN} --max-steps 8 --repeat 1", summary=[RES])
