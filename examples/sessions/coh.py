"""GPU session: the software phase boundary rebuilt, measured (docs/MODEL-vllm-kernels.md section 20).
Microbenchmarks FIRST -- they are the diagnosis whatever the builds do: hw_bench restart (what one buffer_inv sc1
costs a warm phase, the megakernel's per-phase cold start), cohload (inv + plain loads against coherent sc0 sc1
loads for a staged activation), stream (the fixed width sweep: gate_up at 8/16/37/74/96/148/192 workgroups decides
whether narrow phases hold aggregate bandwidth -- the 2.22 ms question), lfx. The primary build is build 5 =
build 3 + ETX_VL_COH (coherent activation access, no acquire inv: weights survive in L2 and the touch finally
works) + ETX_VL_ARGPF (wave 1 prefetches the next task's argument block during the arrive) + ETX_VL_BAR2 (two
__syncthreads per task instead of four), one run each with the gates: 32/32 tokens against HF greedy and
--check-dsl. Build 4 (COH alone) is the fallback and the layer attribution: one fused run after build 5's runs.
Then two interleaved requests on build 5. Pre-registered readings (section 20, anchored to build 3's 5.895):
~5.9 the boundary was never the cost; 5.25-5.7 the overhead share moved, touch still dead; 4.6-5.2 both landed
(floor 4.6, no promise below); build 5 vs build 4 isolates ARGPF+BAR2 (model bound 0.10-0.35 ms: the measured
0.24 ms argument-path spread plus one to two dropped barriers x 0.106-0.282 us x 384 phases); fused < unfused
flips the 0.235 ms fusion penalty, the claim ETX is built on. vLLM 4.67. Two requests: 7.1-8.7 ms per step
(build 3 measured 8.988).

  tools/session.sh push <ip>; tools/session.sh start <ip> examples/sessions/coh.py
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

# ---------------------------------------------------------------- microbenchmarks first (GPU 0, while dl runs)
job(name="build_bench", lane="cpu", after="apt", timeout=900,
    cmd="mkdir -p $HOME/sess && hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/hw/hw_bench.hip -o $HOME/sess/hw_bench && ls -la $HOME/sess/hw_bench")
MB_PREV = None
for mode in ("restart", "cohload", "stream", "lfx"):
    job(name=f"hw_{mode}", gpu=0, after="build_bench", after_any=[MB_PREV] if MB_PREV else [], timeout=600,
        cmd=f"stdbuf -oL $HOME/sess/hw_bench {mode}", summary=[r"^RESULT"])
    MB_PREV = f"hw_{mode}"

job(name="prep", gpu=0, after=["dl", "etx_env"], after_any=[MB_PREV], timeout=2400,
    cmd=f"{PY} examples/vllm_llm/prep.py --src {LLM}/{M}/hf --out {LLM}/{M} --maxlen 1072 && "
        f"cp examples/llm/results/golden_c1024_{M}.txt {LLM}/{M}/golden_c1024.txt && ls -la {LLM}/{M}/vl.bin", summary=[r"wrote .*vl.bin"])

# ---------------------------------------------------------------- build 5: COH + ARGPF + BAR2 (primary, one run each)
B = "moe_all"
DSL = "ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_DSL_GEMV=1 ETX_VL_DSL_TOPK=1 ETX_VL_DSL_ATTN=head ETX_VL_DSL_FOLD=1 ETX_VL_TOUCH=1"
job(name=f"build_{B}", lane="cpu", after=["etx_env", "dl"], timeout=2400,
    cmd=f"{V} {DSL} ETX_VL_COH=1 ETX_VL_ARGPF=1 ETX_VL_BAR2=1 ETX_OUT=$HOME/sess/b_{B} bash examples/vllm_llm/build.sh {LLM}/{M} "
        f"> $HOME/sess/b_{B}.buildlog 2>&1; grep -E 'tasks=|Function Name|VGPRs|error' $HOME/sess/b_{B}.buildlog | tail -4; "
        f"cp $HOME/sess/b_{B}.buildlog $HOME/sess/buildlog_{B}.txt; cp $HOME/sess/b_{B}/vl_plan.txt $HOME/sess/vlplan_{B}.txt; "
        f"cp $HOME/sess/b_{B}/plan.json $HOME/sess/plan_{B}.json; cp $HOME/sess/b_{B}/dsl_tiles.hip $HOME/sess/; test -x $HOME/sess/b_{B}/run",
    summary=[r"tasks=|megakernel|VGPRs Spill"])
RUN = f"$HOME/sess/b_{B}/run {LLM}/{M} $HOME/sess/b_{B}/vl_plan.txt"
# the timed fused number, with the per-task execution counter OFF (it is ~67k device atomics per token on the fused
# path only -- see etx/runtime/include/etx/host.h; correctness here rests on --check-dsl and 32/32 tokens)
job(name="run_fused", gpu=0, after=[f"build_{B}", "prep"], timeout=900, cmd=f"stdbuf -oL {RUN} --repeat 1 --check-dsl", summary=[RES])
# the same build with the counter ON: the difference is exactly what our own instrumentation cost every fused
# number we have ever measured (build 3 5.895 carried it; every unfused number did not)
job(name="run_fused_cnt", gpu=0, after=[f"build_{B}", "prep"], after_any=["run_fused"], timeout=900,
    cmd=f"ETX_EXEC_COUNT=1 stdbuf -oL {RUN} --repeat 1", summary=[RES])
job(name="run_unfused", gpu=0, after=[f"build_{B}", "prep"], after_any=["run_fused_cnt"], timeout=900, cmd=f"stdbuf -oL {RUN} --repeat 1 --unfused", summary=[RES])
job(name="trace_fused", gpu=0, after=[f"build_{B}", "prep"], after_any=["run_unfused"], timeout=900,
    cmd=f"ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/trace_{B}.bin stdbuf -oL {RUN} --max-steps 8 --repeat 1", summary=[RES])

# ---------------------------------------------------------------- build 4 (COH alone): the fallback and the ARGPF+BAR2 attribution
BF = "moe_coh"
job(name=f"build_{BF}", lane="cpu", after=["etx_env", "dl"], timeout=2400,
    cmd=f"{V} {DSL} ETX_VL_COH=1 ETX_OUT=$HOME/sess/b_{BF} bash examples/vllm_llm/build.sh {LLM}/{M} > $HOME/sess/b_{BF}.buildlog 2>&1; "
        f"grep -E 'tasks=|VGPRs Spill|error' $HOME/sess/b_{BF}.buildlog | tail -3; cp $HOME/sess/b_{BF}/vl_plan.txt $HOME/sess/vlplan_{BF}.txt; "
        f"cp $HOME/sess/b_{BF}/plan.json $HOME/sess/plan_{BF}.json; test -x $HOME/sess/b_{BF}/run", summary=[r"tasks=|VGPRs Spill"])
job(name="run_fused_b4", gpu=0, after=[f"build_{BF}", "prep"], after_any=["trace_fused"], timeout=900,
    cmd=f"stdbuf -oL $HOME/sess/b_{BF}/run {LLM}/{M} $HOME/sess/b_{BF}/vl_plan.txt --repeat 1 --check-dsl", summary=[RES])

# ---------------------------------------------------------------- two interleaved requests on build 5
B2 = "moe_allx2"
job(name="llm_env_tf", lane="cpu", after="llm_env", timeout=1800,
    cmd=". ~/llmenv/bin/activate && pip install -q 'transformers>=4.51,<5' accelerate && python3 -c 'import transformers; print(transformers.__version__)'")
job(name="golden_r97", gpu=0, after=["llm_env_tf", "dl"], after_any=["run_fused_b4"], timeout=2400,
    cmd=f"{PY} examples/llm/prep.py --model {HF} --src {LLM}/{M}/hf --out {LLM}/{M} --context 1024 --rotate 97 --golden-only && "
        f"head -c 200 {LLM}/{M}/golden_c1024_r97.txt", summary=[r"HF greedy|gen:"])
job(name=f"build_{B2}", lane="cpu", after=["etx_env", "dl"], timeout=2400,
    cmd=f"{V} {DSL} ETX_VL_COH=1 ETX_VL_ARGPF=1 ETX_VL_BAR2=1 ETX_VL_REQUESTS=2 ETX_OUT=$HOME/sess/b_{B2} "
        f"bash examples/vllm_llm/build.sh {LLM}/{M} > $HOME/sess/b_{B2}.buildlog 2>&1; grep -E 'tasks=|Function Name|VGPRs|error' $HOME/sess/b_{B2}.buildlog | tail -4; "
        f"cp $HOME/sess/b_{B2}.buildlog $HOME/sess/buildlog_{B2}.txt; cp $HOME/sess/b_{B2}/vl_plan.txt $HOME/sess/vlplan_{B2}.txt; "
        f"cp $HOME/sess/b_{B2}/plan.json $HOME/sess/plan_{B2}.json; test -x $HOME/sess/b_{B2}/run", summary=[r"tasks=|megakernel|VGPRs Spill"])
job(name="run_two", gpu=0, after=[f"build_{B2}", "prep", "golden_r97"], timeout=900,
    cmd=f"stdbuf -oL $HOME/sess/b_{B2}/run {LLM}/{M} $HOME/sess/b_{B2}/vl_plan.txt --repeat 1 --golden1 golden_c1024_r97.txt", summary=[RES])
