"""G6 on Qwen3-30B-A3B: the compiler's `coherent_l1` mode (ETX_COHERENCE=l1) against the kept build, under G6's rule
(docs/INVENTORY-SYSTEM-2026-10-13.md section 4.1, written before the code). DeepSeek's half was measured in the
2026-10-13 paid hour (examples/dsv2lite/results/2026-10-13-hour, pairs_l1: +330 us vs default, spread 13 us).

Order: preflight, both builds and the hw_bench build (CPU, in parallel) -> moerows, the Q2 decision microbenchmark
(GPU minutes, no model input: row ownership contiguous vs interleaved on Qwen's moe_up phase, cold, checksummed; plus
moegather as its contiguous cross-check) -> certify kept -> certify kept_l1 -> pairs_kept_l1 (10 invocations). Every GPU job reads what it runs, so a failed build or certificate skips what depends on it, never
runs it. The runner seals the record on the box before any fetch.

The box needs the prepared Qwen inputs (model with golden_c1024.txt, HF config) named by the JSON at
$ETX_QWEN_G6_CONFIG: {"model": ..., "hf": ..., "golden_sha256": ...}; the builds (examples/vllm_llm/build.sh) also need
the vLLM source tree at $VLLM (default ~/vllm) and, on an image without HIP headers, ETX_ROCM_HEADERS.
Run: tools/session.sh stage <ip> && ETX_QWEN_G6_CONFIG=... tools/session.sh start <ip> examples/sessions/qwen_g6.py
"""
import os

MAX_SESSION_SECONDS = 7200          # staging (download, pack: [U], nvref allowed 2400 s each) + 2 builds + 16 invocations
OP = ".venv/bin/python -u -m examples.vllm_llm.g6_session"
SUMMARY = [r"^(PREFLIGHT|BUILD|CERTIFIED|VERDICT|PAIRS) "]
# the box's Qwen inputs (examples/vllm_llm/stage_qwen.py writes it); session.sh start does not forward the variable, so
# every job carries it explicitly
CONFIG = os.environ.get("ETX_QWEN_G6_CONFIG", os.path.expanduser("~/qwen-g6.json"))
# staging, pinned BEFORE anything is fetched: the HF revision every Qwen session recorded (docs/NVREF-2026-10-09.md:26,
# docs/NVREF2-2026-10-09.md:59), the vLLM runtime's release tag (vm_preflight.VLLM_TAG), the repo's golden (its sha256 is
# vm_preflight.GOLDEN_SHA256). Paths are the box's: what vm_preflight checks by default.
HF_REPO = "Qwen/Qwen3-30B-A3B"
HF_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
VLLM_TAG = "v0.27.1"
MODEL = "$HOME/llm/qwen3-30b-a3b"
VLLM = "$HOME/vllm"
LLMENV = "$HOME/qwen-llmenv"
APT = "python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }"


def jobs(prefix: str = "", sub: str = "", shared_setup: bool = False) -> list:
    """The session's jobs. The joint run (examples/sessions/joint_ds_qwen.py) passes prefix "qwen_" and sub "qwen", so
    names and files cannot collide with DeepSeek's, and shared_setup: DeepSeek's `apt` and `etx_env` jobs install the
    venv module and the repo's .venv, so this session waits for them (whatever their result) instead of racing apt."""
    d = f"{sub}/" if sub else ""
    rec = f'"$ETX_SESS_LOGDIR/{d}'                          # the record (sub)directory, quoted, as a shell prefix
    env = {"ETX_QWEN_G6_CONFIG": CONFIG, "ETX_QWEN_G6_RECORD": sub}
    out = []

    def job(name, operation, *, lane="cpu", reads=(), writes=(), timeout=900, cmd=None, summary=SUMMARY, waits=()):
        out.append(dict(name=prefix + name, lane=lane, gpu=0 if lane == "gpu" else None,
                        reads=[*waits, *(d + r for r in reads)], writes=[d + w for w in writes], env=dict(env),
                        timeout=timeout, quiet_s=180, cmd=cmd or f"{OP} {operation}", summary=summary))

    # the Q2 microbenchmark first: it needs no model input, so it uses the GPU while staging runs
    job("build_hw_bench", None, writes=["b_hw/hw_bench"], timeout=600,
        cmd=f'mkdir -p {rec}b_hw" && hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/hw/hw_bench.hip -o {rec}b_hw/hw_bench"')
    # exit 4 = INCOMPLETE (a checksum mismatch: no decision), 3 = a hang; the DECISION line carries PASS / NO-GO
    job("moerows", None, lane="gpu", reads=["b_hw/hw_bench"], writes=["moerows.txt"], timeout=600,
        summary=[r"^(RESULT|DECISION) moe"],
        cmd=f'set -o pipefail; {rec}b_hw/hw_bench" moerows | tee {rec}moerows.txt" && {rec}b_hw/hw_bench" moegather')
    # ---- staging: idempotent (a staged box passes each step in seconds); every step verifies what it produced
    setup_waits = ("?apt",) if shared_setup else ()
    if not shared_setup:
        job("apt", None, timeout=600, cmd=APT, summary=())
        job("etx_env", None, reads=(), timeout=600, summary=(), waits=(prefix + "apt",),
            cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
                ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx'")
        setup_waits = (prefix + "apt",)
    job("stage_env", None, timeout=1800, summary=(), waits=setup_waits, writes=[],
        cmd=f"[ -d {LLMENV} ] || python3 -m venv {LLMENV}; . {LLMENV}/bin/activate && pip install -q --upgrade pip && "
            "pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer safetensors && "
            "{ python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; } && "
            "python3 -c 'import torch, safetensors, huggingface_hub; print(\"QWEN-ENV OK\", torch.__version__)'")
    out[-1]["writes"] = [f"{LLMENV}/bin/python"]
    job("stage_vllm", None, timeout=900, summary=[r"^VLLM "],
        cmd=f"[ -d {VLLM}/.git ] || git clone --depth 1 --branch {VLLM_TAG} https://github.com/vllm-project/vllm.git {VLLM}; "
            f'test "$(git -C {VLLM} rev-parse HEAD)" = "$(git -C {VLLM} rev-parse {VLLM_TAG}^{{commit}})" && '
            f'echo "VLLM $(git -C {VLLM} rev-parse HEAD) = {VLLM_TAG}" || {{ echo "VLLM {VLLM} is not at {VLLM_TAG}: move it aside"; exit 1; }}')
    out[-1]["writes"] = [f"{VLLM}/csrc/rocm/attention.cu"]
    # the revision is pinned in the request itself; an existing hf/ without the same REVISION file is an unpinned (or
    # different) download and is refused, never reused
    job("stage_dl", None, timeout=2400, summary=[r"^HF "],
        cmd=f'R={MODEL}/hf/REVISION; if [ -e {MODEL}/hf/config.json ] && [ "$(cat $R 2>/dev/null)" != {HF_REVISION} ]; then '
            f'echo "HF {MODEL}/hf exists without revision {HF_REVISION}: refusing to reuse it; move it aside"; exit 1; fi; '
            f'HF_HUB_ENABLE_HF_TRANSFER=1 {LLMENV}/bin/python -c "from huggingface_hub import snapshot_download; '
            f"snapshot_download('{HF_REPO}', revision='{HF_REVISION}', local_dir='{MODEL}/hf', "
            f"allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'], max_workers=16)\" && "
            f'echo {HF_REVISION} > $R && echo "HF {HF_REPO}@{HF_REVISION}"')
    out[-1]["reads"] = [f"{LLMENV}/bin/python"]
    out[-1]["writes"] = [f"{MODEL}/hf/REVISION"]
    # packing builds vLLM's rotary cache with torch on the GPU (as every earlier Qwen session did): the GPU lane
    job("stage_pack", None, lane="gpu", timeout=2400, summary=[r"wrote .*vl.bin", r"^QWEN-STAGED "],
        cmd=f"{{ [ -s {MODEL}/vl.manifest ] || {LLMENV}/bin/python examples/vllm_llm/prep.py --src {MODEL}/hf --out {MODEL} --maxlen 1072; }} && "
            f"cp examples/llm/results/golden_c1024_qwen3-30b-a3b.txt {MODEL}/golden_c1024.txt && "
            f".venv/bin/python -c \"import json, os; from examples.vllm_llm.vm_preflight import GOLDEN_SHA256 as G; "
            f"m = os.path.expandvars('{MODEL}'); json.dump(dict(model=m, hf=m + '/hf', golden_sha256=G), open('{CONFIG}', 'w')); "
            f"print('QWEN-STAGED', m)\"")
    out[-1]["reads"] = [f"{MODEL}/hf/REVISION", f"{VLLM}/csrc/rocm/attention.cu", *(() if shared_setup else (prefix + "etx_env",))]
    out[-1]["writes"] = [f"{MODEL}/vl.manifest", CONFIG]
    job("preflight", "preflight", writes=["inputs.json"], timeout=300,
        waits=(("?etx_env",) if shared_setup else ()) + (CONFIG,))
    for arm in ("kept", "kept_l1"):
        job(f"build_{arm}", f"build {arm}", reads=["inputs.json"], writes=[f"b_{arm}/arm.json"], timeout=1500)   # never on an unstaged box
    for arm in ("kept", "kept_l1"):
        job(f"certify_{arm}", f"certify {arm}", lane="gpu", reads=["inputs.json", f"b_{arm}/arm.json"],
            writes=[f"certificate_{arm}/gate.log"], timeout=1200)
    job("pairs_kept_l1", "pairs kept_l1", lane="gpu", writes=["pairs_kept_l1.json"], timeout=3000,
        reads=["inputs.json", "certificate_kept/gate.log", "certificate_kept_l1/gate.log"])
    return out


JOBS = jobs()
