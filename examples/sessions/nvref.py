"""One MI300X reference session; docs/SPEC-nv-reference-2026-10-08.md.

Host wall, batch 1, context 1024; five interleaved rounds. No NVIDIA claim.
Estimate: 35 minutes / $1.74, ceiling 60 minutes / $3.00. Operator must delete
at 55 minutes of VM life, dropping nice-to-have jobs first (spec section 9).
"""
JOBS = []


def job(**kw):
    JOBS.append(kw)


LLM = "$HOME/llm"
PY = "~/llmenv/bin/python"
V = "VLLM=$HOME/vllm ETX_TORCH_HEADERS=$HOME/rocm-headers/pytorch ETX_ROCM_HEADERS=$HOME/rocm-headers/inc"
RES = r"^RESULT|generated tokens match|per-token latency|CHECK-DSL|beyond 2 ulps|GATE|BITCMP"
M, HF = "qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B"
S = "$HOME/sess"

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
job(name="dl", lane="cpu", after="llm_env", timeout=2400, quiet_s=900,
    cmd=f"HF_HUB_ENABLE_HF_TRANSFER=1 {PY} -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF}', local_dir='{LLM}/{M}/hf', allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'], "
        f"max_workers=16); print('DL-DONE')\" && test -f {LLM}/{M}/hf/config.json")
job(name="prep", gpu=0, after=["dl", "etx_env"], timeout=2400, quiet_s=600,
    cmd=f"{PY} examples/vllm_llm/prep.py --src {LLM}/{M}/hf --out {LLM}/{M} --maxlen 1072 && "
        f"cp examples/llm/results/golden_c1024_{M}.txt {LLM}/{M}/golden_c1024.txt && ls -la {LLM}/{M}/vl.bin", summary=[r"wrote .*vl.bin"])

# Keep dl verbatim as requested; the separately recorded revision is queried
# after the download, not asserted to be a pinned snapshot revision.
job(name="record", lane="cpu", after="dl", timeout=120,
    cmd="set -e; rocm-smi --showproductname; cat /opt/rocm/.info/version; uname -r; nproc; "
        "lspci | grep -i mi300; "
        f"{PY} -c \"from huggingface_hub import model_info; print('HF_REV', model_info('{HF}').sha)\"")
job(name="golden_sha", lane="cpu", after="prep", timeout=120,
    cmd=f'echo "bab2f1fbe12f70179711ec40a215c74710845b73169b1cc6a3a6ca50e0a19484  {LLM}/{M}/golden_c1024.txt" | sha256sum -c')
IMAGE = "vllm/vllm-openai-rocm:v0.27.1"
job(name="docker_pull", lane="cpu", timeout=1200, quiet_s=900,
    cmd=f"sudo docker pull {IMAGE} && printf 'IMAGE ' && sudo docker inspect --format '{{{{index .RepoDigests 0}}}}' {IMAGE}")

# Two CPU builds are implicit in section 4; keep the tiles.py build shape.
ALL = "-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH"
for b in ("all", "chk"):
    job(name=f"build_{b}", lane="cpu", after=["etx_env", "dl"], timeout=2400, quiet_s=900,
        cmd=f"set -o pipefail; {V} ETX_VL_CHECK={int(b == 'chk')} ETX_HIPCC_FLAGS='{ALL}' "
            f"ETX_OUT={S}/b_{b} bash examples/vllm_llm/build.sh {LLM}/{M} 2>&1 | tee {S}/b_{b}.buildlog && "
            f"cp {S}/b_{b}/vl_plan.txt {S}/vlplan_{b}.txt && "
            f"cp {S}/b_{b}/dsl_tiles.hip {S}/dsl_tiles_{b}.hip && "
            f"grep -m1 'etx_megakernel_d0' {S}/b_{b}.buildlog || true; "
            f"if grep -Eq 'ScratchSize.*40.*VGPRs Spill: 9' {S}/b_{b}.buildlog; then "
            "echo 'SPILL matches the previous session (9 / 40)'; else "
            "echo 'SPILL differs from the previous session (9 / 40): informational, reported not asserted'; fi; "
            f"test -x {S}/b_{b}/run", summary=[r"tasks=|megakernel|SPILL"])


def run(b="all", extra=""):
    return f"{S}/b_{b}/run {LLM}/{M} {S}/b_{b}/vl_plan.txt {extra}"


job(name="gate_all", gpu=0, after=["build_all", "prep", "golden_sha"], timeout=600,
    cmd=f"set -o pipefail; rm -rf {S}/dump_all; mkdir -p {S}/dump_all; ETX_DUMP={S}/dump_all ETX_DUMP_STEP=1040 "
        f"ETX_TOKTRACE={S}/tok_all.txt ETX_VL_PROGRESS=256 stdbuf -oL {run(extra='--repeat 1 --check-dsl')} 2>&1 | tee {S}/gate_all.txt; "
        f"grep -q '^RESULT .*matched=32 gen=32' {S}/gate_all.txt && grep -q '^CHECK-DSL OK' {S}/gate_all.txt && "
        f"test -s {S}/dump_all/logits.bin && echo GATE-OK", summary=[RES])
job(name="gate_unf", gpu=0, after=["build_all", "prep", "golden_sha"], after_any=["gate_all"], timeout=600,
    cmd=f"set -o pipefail; ETX_VL_PROGRESS=256 stdbuf -oL {run(extra='--unfused --repeat 1')} 2>&1 | tee {S}/gate_unf.txt; "
        f"grep -q '^RESULT .*matched=32 gen=32' {S}/gate_unf.txt && echo GATE-OK", summary=[RES])
job(name="check_imp", gpu=0, after=["build_chk", "prep", "golden_sha"], after_any=["gate_unf"], timeout=600,
    cmd=f"set -o pipefail; stdbuf -oL {run('chk', '--check 2')} 2>&1 | tee {S}/check_imp.txt; "
        f"! grep -q '^CHECK FAILED' {S}/check_imp.txt && "
        f"{{ grep -qx 'CHECK OK: 0 launch(es) differ' {S}/check_imp.txt || "
        f"grep -Fq -- '--check needs a plan without per-XCD replicas (build without ETX_VL_XCD)' {S}/check_imp.txt; }}",
    summary=[r"CHECK|--check needs"])

ENV = 'set -o pipefail; echo "ETX_ENV ETX_UNFUSED_SYNC=${ETX_UNFUSED_SYNC:-} ETX_UNFUSED_NOGRAPH=${ETX_UNFUSED_NOGRAPH:-}"; '
DEPS = ["golden_sha", "gate_all", "gate_unf", "check_imp"]


def vllm(name, aiter=False):
    return ("sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g "
            "--security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$GPU "
            + ("-e VLLM_ROCM_USE_AITER=1 " if aiter else "")
            + f"-v $HOME:$HOME -w $HOME/etx --entrypoint python3 {IMAGE} examples/llm/vllm_bench.py "
            f"--model {LLM}/{M}/hf --golden {LLM}/{M}/golden_c1024.txt --label default --batch 1 --tokens 32 "
            f"--repeats 1 --json {S}/{name}.json")


prev = "check_imp"
pair_logs = []
for rnd in range(1, 6):
    arms = ["vllm", "etx", "etxu"] if rnd % 2 else ["etxu", "etx", "vllm"]
    for arm in arms:
        name = f"{arm}_r{rnd}"
        is_vllm = arm == "vllm"
        job(name=name, gpu=0, after=DEPS + (["docker_pull"] if is_vllm else []), after_any=[prev],
            timeout=420 if is_vllm else 300, quiet_s=240 if is_vllm else 180,
            cmd=vllm(name) if is_vllm else ENV + f"ETX_VL_PROGRESS=1 stdbuf -oL {run(extra='--unfused' if arm == 'etxu' else '')}",
            summary=[r"^VLLM|^RESULT"])
        if not is_vllm:
            pair_logs.append(f"{S}/{name}.log")
        prev = name

for name, env in (("hostprof", "ETX_HOST_PROF=1 ETX_VL_PROGRESS=1 "), ("etx_noprog", "")):
    job(name=name, gpu=0, after=DEPS, after_any=[prev], timeout=300,
        cmd=ENV + env + f"stdbuf -oL {run()}", summary=[r"^RESULT|HOSTPROF"])
    prev = name
for rnd in range(1, 4):
    name = f"vllm_aiter_r{rnd}"
    job(name=name, gpu=0, after=DEPS + ["docker_pull"], after_any=[prev], timeout=420, quiet_s=240,
        cmd=vllm(name, aiter=True), summary=[r"^VLLM"])
    prev = name
# pairs' separate verdict is not this experiment's verdict; retain only its table.
job(name="report", lane="cpu", after_any=[prev], timeout=120,
    cmd=f".venv/bin/python -m examples.vllm_llm.nvref_report {S} && "
        f".venv/bin/python -m examples.vllm_llm.pairs {' '.join(pair_logs)} --pairs 5 --basis wall "
        "| sed '/^VERDICT/,$d'", summary=[r"."])
