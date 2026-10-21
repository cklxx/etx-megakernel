"""GPU session: the two fixes the kernel model ranks first, each alone and together, and two interleaved requests
(docs/qwen/MODEL-vllm-kernels.md sections 6-8). One variant per switch, everything else equal:

  -DETX_IMP_SW_BARRIER    imported barriers as the old software slot barrier (default now: hardware s_barrier in rounds)
  -DETX_IMP_VECTOR_ARGS   imported kernels' parameters through the old vector path (default now: wave-uniform, scalar)
  -DETX_NO_PRE_WAIT       no argument-block touch before the event wait (default now: wave 0 touches it)

Pre-registered (xcd schedule, ms per token; the simulation x 1.135, its bias against today's 6.63 measured):
  old (all three switches)            6.63 (today's build; if it is not within 3% the VM differs: compare ratios only)
  hardware barriers only              6.03
  argument path only                  5.05 / 5.67 / 6.10 if the per-kernel front falls from 2.1 to 0.3 / 1.0 / 1.5 us
  both                                4.45 / 5.07 / 5.51
  two requests, both                  347 / 304 / 278 tokens/s  (vLLM batch 2: 349; batch 1: 204)
  fused_moe gate_up task body         21.9 -> ~15 us with hardware barriers (trace)
Correctness first: the base build's --check (every launch of layers 0-1 byte-identical, the new barrier protocol in the
imported kernels) and every timed run 32/32 (both requests for the two-request build), or the number does not count.
Microbenchmarks first, on GPU 0 while the models download (bench/hw/hw_bench.hip lat / lf / launch / evx,
bench/hw/imp_bench.hip in both protocols); their predictions are in docs/qwen/MODEL-vllm-kernels.md section 10.
Also: instruction-cache counters per launch during --check (vLLM's original next to the imported form: the hypothesis for
the unexplained unfused excess), and one ETX_TRACE of the new xcd build (its front and bodies against the model).

  tools/session.sh push <ip>; tools/session.sh start <ip> examples/sessions/moe_fix.py
"""
JOBS = []


def job(**kw):
    JOBS.append(kw)


MODELS = {"qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B"}
LLM = "$HOME/llm"
PY = "~/llmenv/bin/python"
V = "VLLM=$HOME/vllm ETX_TORCH_HEADERS=$HOME/rocm-headers/pytorch"
RES = r"^RESULT|generated tokens match|per-token latency"
OLD = "-DETX_IMP_SW_BARRIER -DETX_IMP_VECTOR_ARGS -DETX_NO_PRE_WAIT"

# ---------------------------------------------------------------- environment and data
job(name="apt", lane="cpu", timeout=600,
    cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }")
job(name="etx_env", lane="cpu", reads="apt", timeout=600,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
        ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx; print(\"etx ok\")'")
# prep.py builds vLLM's rotary cache with torch on the GPU (the ROCm wheel); the second request's golden needs transformers
job(name="llm_env", lane="cpu", reads="apt", timeout=1800,
    cmd="[ -d ~/llmenv ] || python3 -m venv ~/llmenv; . ~/llmenv/bin/activate && pip install -q --upgrade pip && "
        "pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer safetensors && "
        "{ python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; } && "
        "pip install -q 'transformers>=4.51,<5' accelerate && "
        "python3 -c 'import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'")
for n, hf in MODELS.items():
    job(name=f"dl_{n}", lane="cpu", reads="llm_env", timeout=2400,
        cmd=f"HF_HUB_ENABLE_HF_TRANSFER=1 {PY} -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{hf}', local_dir='{LLM}/{n}/hf', allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'], "
            f"max_workers=16); print('DL-DONE')\" && du -sh {LLM}/{n}/hf && test -f {LLM}/{n}/hf/config.json")
    job(name=f"prep_{n}", gpu=0, reads=[f"dl_{n}", "etx_env"], timeout=2400,
        cmd=f"{PY} examples/vllm_llm/prep.py --src {LLM}/{n}/hf --out {LLM}/{n} --maxlen 1072 && "
            f"cp examples/llm/results/golden_c1024_{n}.txt {LLM}/{n}/golden_c1024.txt && ls -la {LLM}/{n}/vl.bin",
        summary=[r"wrote .*vl.bin"])

# ---------------------------------------------------------------- microbenchmarks first (GPU 0, while the models download)
# every parameter the XCD simulation used to fit or estimate (docs/qwen/MODEL-vllm-kernels.md section 10), measured directly:
# dependent round trips by cache level and after ETX's invalidates (t_v, t_s, the argument path), bytes in flight -> rate
# (the bandwidth), the kernel boundary with dependent loads (F), ETX's own event protocol in the plan's shapes (hand-offs),
# and the imported barrier protocols from etx/import.h itself (t_swbar, the round barrier)
HB = "hipcc -O3 -std=c++17 --offload-arch=gfx942"
job(name="build_bench", lane="cpu", reads="apt", timeout=900,
    cmd=f"mkdir -p $HOME/sess && {HB} bench/hw/hw_bench.hip -o $HOME/sess/hw_bench && "
        f"{HB} -Ietx/runtime/include bench/hw/imp_bench.hip -o $HOME/sess/imp_bench_hw && "
        f"{HB} -Ietx/runtime/include -DETX_IMP_SW_BARRIER bench/hw/imp_bench.hip -o $HOME/sess/imp_bench_sw && ls -la $HOME/sess/hw_bench $HOME/sess/imp_bench_*")
MB_PREV = None
for mode in ("lat", "lf", "launch", "evx"):
    job(name=f"hw_{mode}", gpu=0, reads="build_bench", timeout=300,
        cmd=f"stdbuf -oL $HOME/sess/hw_bench {mode}", summary=[r"^RESULT"])
    MB_PREV = f"hw_{mode}"
for v in ("hw", "sw"):
    job(name=f"impbar_{v}", gpu=0, reads="build_bench", timeout=180,
        cmd=f"stdbuf -oL $HOME/sess/imp_bench_{v}", summary=[r"^RESULT"])
    MB_PREV = f"impbar_{v}"

# ---------------------------------------------------------------- builds (cpu lane, parallel)
S1 = "ETX_VL_SLOTS=attn_mfma16_g8=1,attn_mfma16_g6=1,attn_reduce=1,rms_norm_3d=1,reshape_and_cache=1"
XCD = "ETX_VL_XCD=1 ETX_VL_CHAIN=1"
BUILDS = {   # name -> (model, env, hipcc flags)
    "moe_base_new": ("qwen3-30b-a3b", "ETX_VL_CHECK=1", ""),
    "moe_xcd_old": ("qwen3-30b-a3b", XCD, OLD),
    "moe_xcd_hw": ("qwen3-30b-a3b", XCD, "-DETX_IMP_VECTOR_ARGS -DETX_NO_PRE_WAIT"),
    "moe_xcd_args": ("qwen3-30b-a3b", XCD, "-DETX_IMP_SW_BARRIER"),
    "moe_xcd_nopre": ("qwen3-30b-a3b", XCD, "-DETX_NO_PRE_WAIT"),
    "moe_xcd_new": ("qwen3-30b-a3b", XCD, ""),
    "moe_xcds1_new": ("qwen3-30b-a3b", f"{XCD} {S1}", ""),
    "moe_xcd2_new": ("qwen3-30b-a3b", f"{XCD} ETX_VL_REQUESTS=2", ""),
    # vLLM's fused_qk_norm_rope in place of rms_norm_3d x2 + rotary (docs/qwen/MODEL-vllm-kernels.md section 11)
    "moe_base_qknr": ("qwen3-30b-a3b", "ETX_VL_CHECK=1 ETX_VL_QKNR=1", ""),
    "moe_xcd_qknr_new": ("qwen3-30b-a3b", f"{XCD} ETX_VL_QKNR=grid", ""),
    "moe_xcd_qknrc_new": ("qwen3-30b-a3b", f"{XCD} ETX_VL_QKNR=1", ""),
    "q15_base_new": ("qwen2.5-1.5b", "ETX_VL_CHECK=1", ""),
    "q15_xs1_old": ("qwen2.5-1.5b", f"{XCD} {S1}", OLD),
    "q15_xs1_new": ("qwen2.5-1.5b", f"{XCD} {S1}", ""),
}
for b, (m, env, flags) in BUILDS.items():
    job(name=f"build_{b}", lane="cpu", reads=["etx_env", f"dl_{m}"], timeout=2400,
        cmd=f"{V} {env} ETX_HIPCC_FLAGS='{flags}' ETX_OUT=$HOME/sess/b_{b} bash examples/vllm_llm/build.sh {LLM}/{m} "
            f"> $HOME/sess/b_{b}.buildlog 2>&1; grep -E 'tasks=|Function Name|VGPRs|error' $HOME/sess/b_{b}.buildlog | tail -4; "
            f"cp $HOME/sess/b_{b}.buildlog $HOME/sess/buildlog_{b}.txt; cp $HOME/sess/b_{b}/vl_plan.txt $HOME/sess/vlplan_{b}.txt; "
            f"cp $HOME/sess/b_{b}/plan.json $HOME/sess/plan_{b}.json; test -x $HOME/sess/b_{b}/run",
        summary=[r"tasks=|megakernel|VGPRs Spill"])


def run(b, extra=""):
    m = BUILDS[b][0]
    return f"$HOME/sess/b_{b}/run {LLM}/{m} $HOME/sess/b_{b}/vl_plan.txt {extra}"


# ---------------------------------------------------------------- correctness first: the new barrier protocol, byte for byte
for b, m in (("q15_base_new", "qwen2.5-1.5b"), ("moe_base_new", "qwen3-30b-a3b"), ("moe_base_qknr", "qwen3-30b-a3b")):
    job(name=f"check_{b}", gpu=0, reads=[f"build_{b}", f"prep_{m}"], timeout=600,
        cmd=f"stdbuf -oL {run(b, '--check 2')}", summary=[r"^CHECK|DIFFERS|HANG|  total"])

# ---------------------------------------------------------------- timing, alternated over two rounds (GPU 0)
T = [("q15_xs1_old", ""), ("q15_xs1_new", ""), ("q15_base_new", "--unfused"),
     ("moe_xcd_old", ""), ("moe_xcd_hw", ""), ("moe_xcd_args", ""), ("moe_xcd_nopre", ""), ("moe_xcd_new", ""),
     ("moe_xcds1_new", ""), ("moe_base_new", ""), ("moe_base_new", "--unfused"),
     ("moe_xcd_qknr_new", ""), ("moe_xcd_qknrc_new", ""), ("moe_base_qknr", "--unfused")]
prev = "check_moe_base_new"
for rnd in (1, 2):
    for b, extra in T:
        m = BUILDS[b][0]
        name = f"t{rnd}_{b}{'_unf' if extra else ''}"
        chk = "check_moe_base_new" if m == "qwen3-30b-a3b" else "check_q15_base_new"
        chk = "check_moe_base_qknr" if "qknr" in b else chk
        job(name=name, gpu=0, reads=[f"build_{b}", f"prep_{m}", chk], timeout=900,
            cmd=f"stdbuf -oL {run(b, extra)}", summary=[RES])
        prev = name

# ---------------------------------------------------------------- two requests: request 1 = the prompt rotated by 97 tokens
job(name="golden_r97", gpu=0, reads=["llm_env", "dl_qwen3-30b-a3b"], timeout=2400,
    cmd=f"{PY} examples/llm/prep.py --model Qwen/Qwen3-30B-A3B --src {LLM}/qwen3-30b-a3b/hf --out {LLM}/qwen3-30b-a3b "
        f"--context 1024 --rotate 97 --golden-only && head -c 300 {LLM}/qwen3-30b-a3b/golden_c1024_r97.txt",
    summary=[r"HF greedy|gen:"])
prev2 = "golden_r97"
for rnd in (1, 2):
    for b in ("moe_xcd2_new", "moe_xcd_new"):
        name = f"t2req{rnd}_{b}"
        extra = "--golden1 golden_c1024_r97.txt" if b == "moe_xcd2_new" else ""
        job(name=name, gpu=0, reads=[f"build_{b}", "prep_qwen3-30b-a3b", "golden_r97", "check_moe_base_new"],
            timeout=900, cmd=f"stdbuf -oL {run(b, extra)}", summary=[RES])
        prev2 = name

# ---------------------------------------------------------------- vLLM, same session, with and without its q/k norm + rope fusion
# the baseline for the ETX_VL_QKNR builds is vLLM running the same kernel (pass_config enable_qk_norm_rope_fusion, which
# also turns on the +rotary_embedding custom op); the default is re-measured beside it (same image, same VM). The trace
# checks the fused kernel actually ran (a pass that does not match leaves the unfused kernels).
IMG = "rocm/vllm:latest"
DOCKER = ("sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g "
          "--security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v $HOME:$HOME -w $HOME/etx")
QKNR_PC = """'{"enable_qk_norm_rope_fusion": true}'"""
QKNR_PC_IN = """'{\\"enable_qk_norm_rope_fusion\\": true}'"""          # inside bash -c "..."
job(name="docker_pull", lane="cpu", reads="apt", timeout=2400,
    cmd=f"sudo docker pull -q {IMG} && sudo docker run --rm {IMG} python3 -c 'import vllm; print(\"vllm\", vllm.__version__)'",
    summary=[r"^vllm "])
GOLD = f"{LLM}/qwen3-30b-a3b/golden_c1024.txt"
prev3 = prev2
for rnd in (1, 2):
    for B in (1, 2):
        for tag, pc in (("default", ""), ("qknr", f"--pass-config {QKNR_PC}")):
            name = f"vllm{rnd}_b{B}_{tag}"
            job(name=name, gpu=0, reads=["docker_pull", "prep_qwen3-30b-a3b"], timeout=1200,
                cmd=f"{DOCKER} {IMG} python3 examples/llm/vllm_bench.py --model {LLM}/qwen3-30b-a3b/hf --golden {GOLD} "
                    f"--label qwen3-30b-a3b_{tag} --batch {B} {pc} --json $HOME/sess/{name}.json",
                summary=[r"^VLLM|median_ms|rror"])
            prev3 = name
job(name="vllm_prof_qknr", gpu=0, reads=["docker_pull", "prep_qwen3-30b-a3b"], timeout=1500,
    cmd=f"mkdir -p $HOME/sess/vprof_qknr && chmod 777 $HOME/sess/vprof_qknr && {DOCKER} {IMG} bash -c \"rocprofv3 --kernel-trace "
        f"--output-format csv -d $HOME/sess/vprof_qknr -- python3 examples/vllm_llm/vllm_profile.py run --model {LLM}/qwen3-30b-a3b/hf "
        f"--golden {GOLD} --pass-config {QKNR_PC_IN} > $HOME/sess/vprof_qknr/run.log 2>&1\"; sudo chown -R $USER $HOME/sess/vprof_qknr; "
        f"tail -2 $HOME/sess/vprof_qknr/run.log; f=$(find $HOME/sess/vprof_qknr -name '*kernel_trace.csv' | head -1); "
        f"echo \"fusedQKNormRope launches: $(grep -c fusedQKNormRope $f)  rms_norm launches: $(grep -c rms_norm_kernel $f)  "
        f"rotary launches: $(grep -c rotary_embedding_kernel $f)\"", summary=[r"launches"])
prev2 = "vllm_prof_qknr"

# ---------------------------------------------------------------- traces and counters, last (analysed off the machine)
job(name="trace_moe_xcd_qknr_new", gpu=0, reads=["build_moe_xcd_qknr_new", "prep_qwen3-30b-a3b", "check_moe_base_qknr"],
    timeout=900, cmd=f"ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/trace_moe_xcd_qknr_new.bin stdbuf -oL {run('moe_xcd_qknr_new', '--max-steps 8 --repeat 1')}",
    summary=[RES])
job(name="trace_moe_xcd_new", gpu=0, reads=["build_moe_xcd_new", "prep_qwen3-30b-a3b", "check_moe_base_new"],
    timeout=900, cmd=f"ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/trace_moe_xcd_new.bin stdbuf -oL {run('moe_xcd_new', '--max-steps 8 --repeat 1')}",
    summary=[RES])
job(name="trace_moe_xcd2_new", gpu=0, reads=["build_moe_xcd2_new", "golden_r97"], timeout=900,
    cmd=f"ETX_TRACE=1 ETX_TRACE_RAW=$HOME/sess/trace_moe_xcd2_new.bin stdbuf -oL "
        f"{run('moe_xcd2_new', '--golden1 golden_c1024_r97.txt --max-steps 8 --repeat 1')}", summary=[RES])
job(name="rocprof_moe_base_new", gpu=0, reads=["build_moe_base_new", "prep_qwen3-30b-a3b"], timeout=1200,
    cmd=f"rocprofv3 --kernel-trace --output-format csv -d $HOME/sess/rocprof_moe_base_new -o kt -- {run('moe_base_new', '--unfused --max-steps 8 --repeat 1')} "
        f"> $HOME/sess/rocprof_moe_base_new.run.log 2>&1; tail -2 $HOME/sess/rocprof_moe_base_new.run.log; "
        f"find $HOME/sess/rocprof_moe_base_new -name '*kernel_trace.csv' | head -1 | xargs -r wc -l", summary=[r"kernel_trace"])
# instruction cache per launch: the --check run launches vLLM's original kernel and the imported form one after the other
for cname, ctrs in (("icache", "SQC_ICACHE_HITS SQC_ICACHE_MISSES"), ("insts", "SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_SMEM")):
    job(name=f"pmc_check_{cname}", gpu=0, reads=["build_moe_base_new", "prep_qwen3-30b-a3b"], timeout=1200,
        cmd=f"rocprofv3 --pmc {ctrs} -d $HOME/sess/pmc_check_{cname} -o pmc --output-format csv -- {run('moe_base_new', '--check 2')} "
            f"> $HOME/sess/pmc_check_{cname}.run.log 2>&1; tail -2 $HOME/sess/pmc_check_{cname}.run.log; "
            f"find $HOME/sess/pmc_check_{cname} -name '*counter_collection.csv' | head -1 | xargs -r wc -l", summary=[r"CHECK|counter_collection"])
