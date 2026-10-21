"""GPU session: the DSL builds, one run each (docs/qwen/MODEL-vllm-kernels.md sections 13, 16). Build 3 (the best by the
configuration search): xcd + the DSL streaming GEMVs (etx.ops.gemv, silu fused into gate_up) + the attention block by
kv head (etx.ops.attnprep on XCD h, vLLM's attention and reduce on XCD h) + the folds (in_norm / post_norm / final_norm
into the qkv / router / lm_head prologues, the top-k into the expert tasks; no moe_sum). Build 2 (no folds) as the
fallback and the measurement of the fold alone. Gates: 32/32 tokens against HF greedy and --check-dsl (the last layer's
DSL ops within 2 bf16 ulps of a CPU reference). Numbers: the fused step, the same tiles launched one by one (--unfused),
two interleaved requests. Build 3 carries the touch hook (ETX_VL_TOUCH=1, section 18: the dense GEMVs' first weight
trip pulled into L2 by the idle waves during the event wait). Pre-registered (sim x 1.11): build 3 + touch 3.20 ->
~3.56 ms (vLLM 4.67; without the hook 3.54 -> ~3.93), two requests 2.18 -> ~2.42 ms per token (413 tok/s; vLLM
batch 2: 360; without the hook ~2.53); build 2 (no folds, no touch) 4.01 -> ~4.45.

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
job(name="etx_env", lane="cpu", reads="apt", timeout=600,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
        ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx; print(\"etx ok\")'")
job(name="llm_env", lane="cpu", reads="apt", timeout=1800,
    cmd="[ -d ~/llmenv ] || python3 -m venv ~/llmenv; . ~/llmenv/bin/activate && pip install -q --upgrade pip && "
        "pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer safetensors && "
        "{ python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; } && "
        "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'")
job(name="dl", lane="cpu", reads="llm_env", timeout=2400,
    cmd=f"HF_HUB_ENABLE_HF_TRANSFER=1 {PY} -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF}', local_dir='{LLM}/{M}/hf', allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'], "
        f"max_workers=16); print('DL-DONE')\" && test -f {LLM}/{M}/hf/config.json")
job(name="prep", gpu=0, reads=["dl", "etx_env"], timeout=2400,
    cmd=f"{PY} examples/vllm_llm/prep.py --src {LLM}/{M}/hf --out {LLM}/{M} --maxlen 1072 && "
        f"cp examples/llm/results/golden_c1024_{M}.txt {LLM}/{M}/golden_c1024.txt && ls -la {LLM}/{M}/vl.bin", summary=[r"wrote .*vl.bin"])
B = "moe_dsl3"
DSL = "ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head"
job(name=f"build_{B}", lane="cpu", reads=["etx_env", "dl"], timeout=2400,
    cmd=f"{V} {DSL} ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_OUT=$HOME/sess/b_{B} bash examples/vllm_llm/build.sh {LLM}/{M} "
        f"> $HOME/sess/b_{B}.buildlog 2>&1; grep -E 'tasks=|Function Name|VGPRs|error' $HOME/sess/b_{B}.buildlog | tail -4; "
        f"cp $HOME/sess/b_{B}.buildlog $HOME/sess/buildlog_{B}.txt; cp $HOME/sess/b_{B}/vl_plan.txt $HOME/sess/vlplan_{B}.txt; "
        f"cp $HOME/sess/b_{B}/plan.json $HOME/sess/plan_{B}.json; cp $HOME/sess/b_{B}/dsl_tiles.hip $HOME/sess/; test -x $HOME/sess/b_{B}/run",
    summary=[r"tasks=|megakernel|VGPRs Spill"])
RUN = f"$HOME/sess/b_{B}/run {LLM}/{M} $HOME/sess/b_{B}/vl_plan.txt"
# one run each: the fused step with the correctness gates, then the same tiles one launch at a time
job(name="run_fused", gpu=0, reads=[f"build_{B}", "prep"], timeout=900, cmd=f"stdbuf -oL {RUN} --repeat 1 --check-dsl", summary=[RES])
job(name="run_unfused", gpu=0, reads=[f"build_{B}", "prep"], timeout=900, cmd=f"stdbuf -oL {RUN} --repeat 1 --unfused", summary=[RES])
job(name="trace_fused", gpu=0, reads=[f"build_{B}", "prep"], timeout=900,
    cmd=f"ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/trace_{B}.bin stdbuf -oL {RUN} --max-steps 8 --repeat 1", summary=[RES])
# build 2 (no folds): the fallback if build 3 fails a gate, and the fold's own measurement
BF = "moe_dsl2"
job(name=f"build_{BF}", lane="cpu", reads=["etx_env", "dl"], timeout=2400,
    cmd=f"{V} {DSL} ETX_OUT=$HOME/sess/b_{BF} bash examples/vllm_llm/build.sh {LLM}/{M} > $HOME/sess/b_{BF}.buildlog 2>&1; "
        f"grep -E 'tasks=|VGPRs Spill|error' $HOME/sess/b_{BF}.buildlog | tail -3; cp $HOME/sess/b_{BF}/vl_plan.txt $HOME/sess/vlplan_{BF}.txt; "
        f"cp $HOME/sess/b_{BF}/plan.json $HOME/sess/plan_{BF}.json; test -x $HOME/sess/b_{BF}/run", summary=[r"tasks=|VGPRs Spill"])
job(name="run_fused_b2", gpu=0, reads=[f"build_{BF}", "prep"], timeout=900,
    cmd=f"stdbuf -oL $HOME/sess/b_{BF}/run {LLM}/{M} $HOME/sess/b_{BF}/vl_plan.txt --repeat 1 --check-dsl", summary=[RES])
# two interleaved requests on the same structure (predicted 4.84 ms per step of two tokens: 2.42 ms per token, 413 tok/s;
# vLLM batch 2: 360): request 1 = the prompt rotated by 97 tokens, its golden from HF greedy
B2 = "moe_dsl3x2"
job(name="llm_env_tf", lane="cpu", reads="llm_env", timeout=1800,
    cmd=". ~/llmenv/bin/activate && pip install -q 'transformers>=4.51,<5' accelerate && python3 -c 'import transformers; print(transformers.__version__)'")
job(name="golden_r97", gpu=0, reads=["llm_env_tf", "dl"], timeout=2400,
    cmd=f"{PY} examples/llm/prep.py --model {HF} --src {LLM}/{M}/hf --out {LLM}/{M} --context 1024 --rotate 97 --golden-only && "
        f"head -c 200 {LLM}/{M}/golden_c1024_r97.txt", summary=[r"HF greedy|gen:"])
job(name=f"build_{B2}", lane="cpu", reads=["etx_env", "dl"], timeout=2400,
    cmd=f"{V} {DSL} ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1 ETX_VL_REQUESTS=2 ETX_OUT=$HOME/sess/b_{B2} "
        f"bash examples/vllm_llm/build.sh {LLM}/{M} > $HOME/sess/b_{B2}.buildlog 2>&1; grep -E 'tasks=|Function Name|VGPRs|error' $HOME/sess/b_{B2}.buildlog | tail -4; "
        f"cp $HOME/sess/b_{B2}.buildlog $HOME/sess/buildlog_{B2}.txt; cp $HOME/sess/b_{B2}/vl_plan.txt $HOME/sess/vlplan_{B2}.txt; "
        f"cp $HOME/sess/b_{B2}/plan.json $HOME/sess/plan_{B2}.json; test -x $HOME/sess/b_{B2}/run", summary=[r"tasks=|megakernel|VGPRs Spill"])
job(name="run_two", gpu=0, reads=[f"build_{B2}", "prep", "golden_r97"], timeout=900,
    cmd=f"stdbuf -oL $HOME/sess/b_{B2}/run {LLM}/{M} $HOME/sess/b_{B2}/vl_plan.txt --repeat 1 --golden1 golden_c1024_r97.txt", summary=[RES])
# after the experiment, one 20 s probe (does not touch the builds): the streaming bandwidth of 1 / 2 / 4 / 8 XCDs
# (hw_bench lfx), the number that decides whether layer pipelining across XCDs is worth building (added 2026-09-29)
job(name="build_bench", lane="cpu", reads="apt", timeout=900,
    cmd="mkdir -p $HOME/sess && hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/hw/hw_bench.hip -o $HOME/sess/hw_bench && ls -la $HOME/sess/hw_bench")
job(name="hw_lfx", gpu=0, reads=["build_bench"], timeout=300,
    cmd="stdbuf -oL $HOME/sess/hw_bench lfx", summary=[r"^RESULT"])
# and the hand-off with the data as the signal (hw_bench sdat: coherent stores / plain + wbl2 polled by coherent loads,
# evx's device shapes): the number that decides whether the sentinel hand-off (docs/core/SURVEY-2026-09-29.md section 2) is
# worth building against the measured evx / trace hand-offs
job(name="hw_sdat", gpu=0, reads=["build_bench"], timeout=300,
    cmd="stdbuf -oL $HOME/sess/hw_bench sdat", summary=[r"^RESULT"])
