"""Pre-registered import coverage and vendor-clock diagnostics; no timing claim.

Machine A: one MI300X VF, gfx942, a vLLM image with the device passed
through, and dev headers at ~/rocm-headers/inc. A runtime-only image has
no hip/hip_runtime.h. On a box you did not provision, ETX_SESS_USER=root.
Budget: at most 20 GPU-minutes, about $0.70 at the published $1.99/GPU-hour;
expected about 12 minutes when model, image and environment are warm.
Operator hard stop: 20 minutes of GPU allocation, including setup; delete
immediately on completion. Job timeouts are not an aggregate budget limiter.

Warm prerequisites retained from nvref2: ~/sess/b_all (run, vl_plan.txt,
plan.json), ~/sess/dump_all, and the packed ~/llm/qwen3-30b-a3b model plus
its golden_c1024.txt. This fixed job list does not rebuild/prep those inputs.
A cold machine is outside the 12-minute estimate; stage inputs beforehand.

Go/no-go, before the run: check_full passing with all 97 imported instances
means the import gate covers every instance in the NR=1 build only, not the
timed replicated build. Fewer instances keeps coverage partial; report the
shortfall. The binary prints per-instance results and NOT-IMPORTED counts,
not a checked-import total: CHECK-COVERAGE below counts its result lines.
BOUNDARY-SUMMARY crossings=0 means vendor attribution can proceed in a later
session (only for a nonempty trace); crossings above zero mean the two
timestamp domains disagree. The per-kernel question stays open, with measured
overhang as evidence. No crossing-count pass/fail threshold is applied.

The vendor arm may stop at its first parser refusal, before later libraries.
Report each trace actually produced, never claim missing arms were measured.
Boundary mode emits worst-first JSON plus BOUNDARY-SUMMARY; the vendor parser
emits BOUNDARY-CROSS. No per-kernel attribution is inferred from either.

Local inspection only: python -m examples.sessions.nvcov --dry-run
Uses the same etx.tools.gpu_jobs dry-run as the other session files.
"""
JOBS = []


def job(**kw):
    JOBS.append(kw)


LLM = "$HOME/llm"
PY = "~/llmenv/bin/python"
V = "VLLM=$HOME/vllm ETX_TORCH_HEADERS=$HOME/rocm-headers/pytorch ETX_ROCM_HEADERS=$HOME/rocm-headers/inc"
M, HF = "qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B"
S = "$HOME/sess"
LOGS = "${ETX_SESS_LOGDIR:-$HOME/sess}"
IMAGE = "vllm/vllm-openai-rocm:v0.27.1"

job(name="apt", lane="cpu", timeout=600,
    cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }")

job(name="docker_pull", lane="cpu", timeout=1200, quiet_s=900,
    cmd=f"sudo docker pull {IMAGE} && printf 'IMAGE ' && sudo docker inspect --format '{{{{index .RepoDigests 0}}}}' {IMAGE}")

job(name="llm_env", lane="cpu", after="apt", timeout=1800,
    cmd="[ -d ~/llmenv ] || python3 -m venv ~/llmenv; . ~/llmenv/bin/activate && pip install -q --upgrade pip && "
        "pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer safetensors && "
        "{ python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; } && "
        "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'")

job(name="dl", lane="cpu", after="llm_env", timeout=2400, quiet_s=900,
    cmd=f"HF_HUB_ENABLE_HF_TRANSFER=1 {PY} -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF}', local_dir='{LLM}/{M}/hf', allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'], "
        f"max_workers=16); print('DL-DONE')\" && test -f {LLM}/{M}/hf/config.json")

job(name="etx_env", lane="cpu", after="apt", timeout=600,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
        ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx; print(\"etx ok\")'")

job(name="record", lane="cpu", after="dl", timeout=120,
    cmd="set -e; rocm-smi --showproductname; cat /opt/rocm/.info/version; uname -r; nproc; "
        "(lspci -nn | grep -iE 'mi300|74b5' || echo 'lspci: no MI300 line (0x74b5 is the VF device id)'); "
        f"{PY} -c \"from huggingface_hub import model_info; print('HF_REV', model_info('{HF}').sha)\" || echo 'HF_REV unknown (the hub lookup failed: no network or no local copy)'")

ALL = "-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH"
for b in ("chk1",):
    job(name=f"build_{b}", lane="cpu", after=["etx_env", "dl"], timeout=2400, quiet_s=900,
        cmd=f"set -o pipefail; {V} ETX_VL_CHECK={int(b != 'all')} "
            f"{'ETX_VL_XCD=0 ' if b == 'chk1' else ''}ETX_HIPCC_FLAGS='{ALL}' "
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


ENV = 'set -o pipefail; echo "ETX_ENV ETX_UNFUSED_SYNC=${ETX_UNFUSED_SYNC:-} ETX_UNFUSED_NOGRAPH=${ETX_UNFUSED_NOGRAPH:-}"; '
DEPS = {
    "check": ["build_chk1", "record"],
    "vendor": ["docker_pull", "etx_env", "record"],
}

# Success dependencies gate usable inputs; completion barriers preserve order
# even when correctness fails. A diagnostic is not an endorsement of a gate.
job(name="check_full", gpu=0, after=DEPS["check"], timeout=600, quiet_s=180,
    cmd=ENV + f"stdbuf -oL {run('chk1', '--check 48')} 2>&1 | tee {LOGS}/check_full.txt; "
        "check_rc=${PIPESTATUS[0]}; "
        f"count=$(grep -Ec '^[[:space:]]+[^[:space:]]+[[:space:]]+L-?[0-9]+[[:space:]].* (identical|DIFFERS).* original ' {LOGS}/check_full.txt); "
        'echo "CHECK-COVERAGE imported_instances=$count expected=97 shortfall=$((97-count)) basis=binary-instance-lines"; '
        f"test \"$check_rc\" -eq 0 && grep -qx 'CHECK OK: 0 launch(es) differ' {LOGS}/check_full.txt && "
        'test "$count" -eq 97',
    summary=[r"^CHECK|instance\(s\)|^\s+\S+\s+L-?\d+\s+.*(?:identical|DIFFERS)"])

# Keep kcmp's vendor invocation intact, changing only the output directory.
# Always inspect traces after parser refusal; retain the vendor exit code as
# evidence, not as a crossing-count threshold. Missing traces are incomplete.
name = "kcmp_boundary"
MODEL = f"{LLM}/{M}"
cmd = ('sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host '
       '--security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 '
       f'-v $HOME:$HOME -w $HOME/etx --entrypoint python3 {IMAGE} '
       'examples/vllm_llm/vendor_gemv.py --lib hipblaslt,rocblas,tunable '
       f'--copies 48 --warmup 10 --iters 30 --check {S}/dump_all --model {MODEL} --out {LOGS}/{name}')
job(name=name, gpu=0, lane="gpu", after=DEPS["vendor"], after_any=["check_full"],
    timeout=600, quiet_s=60,
    cmd=ENV + f'test ! -e {LOGS}/{name} || {{ echo REFUSED: existing trace directory; exit 1; }}; ' + cmd + '; vendor_rc=$?; echo "VENDOR-EXIT rc=$vendor_rc"; seen=0; report_rc=0; '
        f'for lib in hipblaslt rocblas tunable; do trace="{LOGS}/{name}/$lib/trace"; '
        'if [ -d "$trace" ] && compgen -G "$trace/*kernel_trace.csv" >/dev/null && '
        'compgen -G "$trace/*marker_api_trace.csv" >/dev/null; then '
        'seen=$((seen+1)); echo "BOUNDARY-TRACE lib=$lib dir=$trace"; '
        '.venv/bin/python -m etx.tools.ktrace --trace-dir "$trace" --boundaries || report_rc=1; '
        'else echo "BOUNDARY-MISSING lib=$lib (no complete trace pair; not zero crossings)"; fi; done; '
        'test "$seen" -gt 0 && test "$report_rc" -eq 0',
    summary=[r"^BOUNDARY-CROSS|^BOUNDARY-SUMMARY|^BOUNDARY-TRACE|^BOUNDARY-MISSING|^VENDOR-EXIT"])

# Commands and definitions copied verbatim; rewire trace1 below because the
# requested final report cannot also be its prerequisite (a dependency cycle).
job(name="trace1", gpu=0, after="report", timeout=300,
    cmd=f"ETX_TRACE=1 ETX_TRACE_RAW={S}/tr_nvref2.bin stdbuf -oL {run(extra='--repeat 1')}")
job(name="floors_trace", lane="cpu", after="trace1", timeout=300,
    cmd=f".venv/bin/python -m etx.tools.floors --trace {S}/b_all/plan.json {S}/tr_nvref2.bin "
        f"> {S}/floors_nvref2.txt && cat {S}/floors_nvref2.txt")
job(name="evx", gpu=0, after="floors_trace", timeout=300,
    cmd=f"hipcc -O3 -std=c++17 --offload-arch=gfx942 bench/hw/hw_bench.hip -o {S}/hw_bench && "
        f"stdbuf -oL {S}/hw_bench evx")

for entry in JOBS:
    entry["machine"] = "A"  # The reference sessions use the runner's A default.
    if entry["name"] == "trace1":
        entry["after"] = ["etx_env", "record"]
        entry["after_any"] = ["check_full", "kcmp_boundary"]

job(name="report", lane="cpu", machine="A", after_any=[j["name"] for j in JOBS], timeout=120,
    cmd=f'logs="{LOGS}"; echo "NVCOV EVIDENCE BEGIN $logs"; '
        'for name in record build_chk1 check_full kcmp_boundary trace1 floors_trace evx; do '
        'echo "--- $name ---"; '
        'if [ -f "$logs/$name.log" ]; then '
        "grep -E 'CHECK|instance|identical|DIFFERS|BOUNDARY-|VENDOR-EXIT|overhang_us|kernel_end_us|marker_end_us|label|in_range|kernel|RESULT|TRACE|FLOOR|evx|EVX|GB/s|TB/s|us|ms|ERROR|Error|error|FAILED|REFUSED|SPILL|gfx942|MI300|HF_REV' "
        '"$logs/$name.log" || echo "NO MATCHING EVIDENCE"; '
        'else echo "MISSING LOG: $name (not complete)"; fi; done; '
        'echo "--- runner status ---"; cat "$logs/status.json"; echo "NVCOV EVIDENCE END"',
    summary=[r"."])


if __name__ == "__main__":
    import argparse
    from etx.tools.gpu_jobs import main

    parser = argparse.ArgumentParser(description="Inspect the prepared session; use gpu_jobs to execute it.")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.parse_args()
    raise SystemExit(main([__file__, "--dry-run"]))
