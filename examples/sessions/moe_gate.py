"""GPU session: the dense gate, then Qwen3-30B-A3B with vLLM's kernels fused (docs/MODEL-moe.md sections 3-4).

Question: is ETX's hand-off (h) plus what the imported form adds per task (p) now small enough that fusing vLLM's own
kernels beats vLLM (MoE break-even: 12 h + 15 p < 32.4 us per layer), and does the MoE build do it?

Two GPUs, each busy with its own model family from the start (on a one-GPU VM everything folds onto GPU 0):
GPU 0 = dense: packs Qwen2.5-1.5B / Qwen3-8B, then G1, then the 1.5B trace and kernel trace.
GPU 1 = MoE: packs Qwen3-30B-A3B, the --check, the MoE timing, then its trace and kernel trace.
Each A/B comparison runs on one GPU, alternated over two rounds; every run 32/32 tokens or it does not count.
  G1  dense, stride 32 (the gfx942 default) vs -DETX_EV_STRIDE=1: Qwen2.5-1.5B base / xcd_s1, Qwen3-8B base / xcd, and the
      unfused builds. Pre-registered: 1.5B xcd_s1 1.95-2.25 ms (was 2.248), 8B xcd 5.35-5.86 (was 5.864), base builds
      gain 0.3-0.7 ms; if stride 32 gains nothing, H1 (false sharing of consecutive event words) is wrong.
  MoE fused base / xcd and unfused, against vLLM 4.898 ms (measured 2026-09-27, not re-run). Gate for "ETX wins":
      below 4.65 ms (5% under vLLM).
GPU 1 (correctness and traces):
  MoE --check: every imported launch of layers 0-1 byte-identical to vLLM's kernel (the Triton ones from vLLM's own
      code object) or no MoE number counts; per-kernel original vs imported time (p per kernel, P2-P4)
  G2  ETX_TRACE of 1.5B xcd_s1 and MoE xcd (stride 32): per hand-off, last producer end -> first consumer start (h, H2-H4)
  G3  rocprofv3 kernel trace of the 1.5B and MoE unfused builds: per launch, against vLLM's trace of the same kernel (p)
Nothing measured before is re-run (docs/DATA.md): vLLM's numbers and kernel traces, the hardware microbenchmarks.

  tools/session.sh push <ip>; tools/session.sh start <ip> examples/sessions/moe_gate.py
"""
JOBS = []


def job(**kw):
    JOBS.append(kw)


MODELS = {"qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", "qwen3-8b": "Qwen/Qwen3-8B", "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B"}
LLM = "$HOME/llm"
PY = "~/llmenv/bin/python"
V = "VLLM=$HOME/vllm ETX_TORCH_HEADERS=$HOME/rocm-headers/pytorch"
RES = r"^RESULT|generated tokens match|per-token latency"

# ---------------------------------------------------------------- environment and data (cpu lane)
job(name="apt", lane="cpu", timeout=600,
    cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }")
job(name="etx_env", lane="cpu", reads="apt", timeout=600,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
        ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx; print(\"etx ok\")'")
# prep.py builds vLLM's rotary cache with torch on the GPU, as vLLM does: the ROCm wheel, not the CPU one
job(name="llm_env", lane="cpu", reads="apt", timeout=1800,
    cmd="[ -d ~/llmenv ] || python3 -m venv ~/llmenv; . ~/llmenv/bin/activate && pip install -q --upgrade pip && "
        "pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer safetensors && "
        "{ python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; } && "
        "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'")
for n, hf in MODELS.items():
    job(name=f"dl_{n}", lane="cpu", reads="llm_env", timeout=2400,
        cmd=f"HF_HUB_ENABLE_HF_TRANSFER=1 {PY} -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{hf}', local_dir='{LLM}/{n}/hf', allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'], "
            f"max_workers=16); print('DL-DONE')\" && du -sh {LLM}/{n}/hf && test -f {LLM}/{n}/hf/config.json")
    # the goldens are kept in the repo (HF greedy, 1024-token context): no reference run
    job(name=f"prep_{n}", gpu=1 if n == "qwen3-30b-a3b" else 0, reads=[f"dl_{n}", "etx_env"], timeout=2400,
        cmd=f"{PY} examples/vllm_llm/prep.py --src {LLM}/{n}/hf --out {LLM}/{n} --maxlen 1072 && "
            f"cp examples/llm/results/golden_c1024_{n}.txt {LLM}/{n}/golden_c1024.txt && ls -la {LLM}/{n}/vl.bin",
        summary=[r"wrote .*vl.bin"])

# ---------------------------------------------------------------- builds (cpu lane, parallel)
# name -> (model, env, extra hipcc flags); stride 32 is the default, s1 the A/B
SLOTS1 = "attn_mfma4_g4=1,attn_mfma16_g6=1,attn_reduce=1,rms_norm_3d=1,reshape_and_cache=1"
BUILDS = {
    "q15_base": ("qwen2.5-1.5b", "", ""),
    "q15_base_s1": ("qwen2.5-1.5b", "", "-DETX_EV_STRIDE=1"),
    "q15_xs1": ("qwen2.5-1.5b", f"ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_SLOTS={SLOTS1}", ""),
    "q15_xs1_s1": ("qwen2.5-1.5b", f"ETX_VL_XCD=1 ETX_VL_CHAIN=1 ETX_VL_SLOTS={SLOTS1}", "-DETX_EV_STRIDE=1"),
    "q8_base": ("qwen3-8b", "", ""),
    "q8_base_s1": ("qwen3-8b", "", "-DETX_EV_STRIDE=1"),
    "q8_xcd": ("qwen3-8b", "ETX_VL_XCD=1 ETX_VL_CHAIN=1", ""),
    "q8_xcd_s1": ("qwen3-8b", "ETX_VL_XCD=1 ETX_VL_CHAIN=1", "-DETX_EV_STRIDE=1"),
    "moe_base": ("qwen3-30b-a3b", "ETX_VL_CHECK=1", ""),          # also the --check build and the unfused baseline
    "moe_xcd": ("qwen3-30b-a3b", "ETX_VL_XCD=1 ETX_VL_CHAIN=1", ""),
}
for b, (m, env, flags) in BUILDS.items():
    job(name=f"build_{b}", lane="cpu", reads=["etx_env", f"dl_{m}"], timeout=2400,
        cmd=f"{V} {env} ETX_HIPCC_FLAGS='{flags}' ETX_OUT=$HOME/sess/b_{b} bash examples/vllm_llm/build.sh {LLM}/{m} "
            f"> $HOME/sess/b_{b}.buildlog 2>&1; grep -E 'tasks=|Function Name|VGPRs|error' $HOME/sess/b_{b}.buildlog | tail -4; "
            f"cp $HOME/sess/b_{b}.buildlog $HOME/sess/buildlog_{b}.txt; test -x $HOME/sess/b_{b}/run",
        summary=[r"tasks=|megakernel|VGPRs Spill"])


def run(b, extra=""):
    m = BUILDS[b][0]
    return f"$HOME/sess/b_{b}/run {LLM}/{m} $HOME/sess/b_{b}/vl_plan.txt {extra}"


# ---------------------------------------------------------------- MoE correctness first (GPU 1)
job(name="moe_check", gpu=1, reads=["build_moe_base", "prep_qwen3-30b-a3b"], timeout=1200,
    cmd=f"stdbuf -oL {run('moe_base', '--check 2')}", summary=[r"^CHECK|DIFFERS|HANG|  total|fused_moe|topk|moe_sum"])

# ---------------------------------------------------------------- timing, alternated: dense on GPU 0, MoE on GPU 1
DENSE = [("q15_base", ""), ("q15_base_s1", ""), ("q15_xs1", ""), ("q15_xs1_s1", ""), ("q15_base", "--unfused"),
         ("q8_base", ""), ("q8_base_s1", ""), ("q8_xcd", ""), ("q8_xcd_s1", ""), ("q8_base", "--unfused")]
MOE = [("moe_base", ""), ("moe_xcd", ""), ("moe_base", "--unfused")]
LAST = {}
for gpu, T in ((0, DENSE), (1, MOE)):
    prev = "moe_check" if gpu == 1 else None
    for rnd in (1, 2):
        for b, extra in T:
            m = BUILDS[b][0]
            name = f"t{rnd}_{b}{'_unf' if extra else ''}"
            deps = [f"build_{b}", f"prep_{m}"] + (["moe_check"] if m == "qwen3-30b-a3b" else [])
            job(name=name, gpu=gpu, reads=deps, timeout=900,
                cmd=f"stdbuf -oL {run(b, extra)}", summary=[RES])
            prev = name
    LAST[gpu] = prev

# ---------------------------------------------------------------- traces, after each GPU's timing (analysed off the machine)
for b in ("q15_xs1", "moe_xcd"):
    m = BUILDS[b][0]
    g = 1 if m == "qwen3-30b-a3b" else 0
    job(name=f"trace_{b}", gpu=g, reads=[f"build_{b}", f"prep_{m}"] + (["moe_check"] if g == 1 else []), timeout=900,
        cmd=f"ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/trace_{b}.bin stdbuf -oL {run(b, '--max-steps 8 --repeat 1')}; "
            f"cp $HOME/sess/b_{b}/plan.json $HOME/sess/plan_{b}.json; cp $HOME/sess/b_{b}/vl_plan.txt $HOME/sess/vlplan_{b}.txt",
        summary=[RES])
for b in ("q15_base", "moe_base"):
    m = BUILDS[b][0]
    g = 1 if m == "qwen3-30b-a3b" else 0
    job(name=f"rocprof_{b}", gpu=g, reads=[f"build_{b}", f"prep_{m}"], timeout=1200,
        cmd=f"rocprofv3 --kernel-trace --output-format csv -d $HOME/sess/rocprof_{b} -o kt -- {run(b, '--unfused --max-steps 8 --repeat 1')} "
            f"> $HOME/sess/rocprof_{b}.run.log 2>&1; tail -3 $HOME/sess/rocprof_{b}.run.log; cp $HOME/sess/b_{b}/vl_plan.txt $HOME/sess/vlplan_{b}.txt; "
            f"find $HOME/sess/rocprof_{b} -name '*kernel_trace.csv' | head -1 | xargs -r wc -l",
        summary=[RES, r"kernel_trace"])
