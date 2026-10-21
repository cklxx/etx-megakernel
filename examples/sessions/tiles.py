"""GPU session: four flag-gated DSL tile changes, each alone and together, against the shipped control build
(docs/qwen/TILES-2026-10-07.md). Qwen3-30B-A3B, batch 1, context 1024, one MI300X. Every flag claims bit-identical output:

  -DETX_DSL_WSUM     the flush's wave sum: 2 LDS-pipe steps + 4 DPP steps instead of 6 ds_bpermute round trips
  -DETX_DSL_SCALAR   argument block through scalar loads: the stream loop's row / segment control on the SALU
  -DETX_DSL_GU1      moe_up's gate and up rows as one stream pass (one drain and one flush sequence fewer per task)
  -DETX_DSL_RBATCH   the folded norm issues all its residual rows before the first wait (1-2 round trips, not R+1)

Order: correctness first (32/32 + CHECK-DSL per build, stage buffers at decode step 1040 and every step's argmax
compared byte for byte with the control, the control run twice to prove it deterministic), then timing (three
rounds of adjacent control / flag pairs, order reversed in round 2), then one ETX_TRACE per build, then counters.
Pre-registered (doc section 6; device median, ms per token, paired flag - control):
  rbatch -0.06..-0.17   scalar -0.02..-0.20   gu1 -0.04..-0.12   wsum -0.02..-0.12   all -0.15..-0.45
KEEP: bit-identical, 32/32, CHECK-DSL OK, every pair faster, mean <= -0.030 (examples/vllm_llm/tile_ab.py).
Budget: ~20 min GPU, ~50 min wall with the download; stop at 1.5 h ($4.50 at $2.99/h) whatever is left.

  tools/session.sh push <ip>; tools/session.sh start <ip> examples/sessions/tiles.py
  tools/session.sh fetch <ip> results_tiles   # then: .venv/bin/python -m examples.vllm_llm.tile_ab results_tiles
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
job(name="etx_env", lane="cpu", reads="apt", timeout=600,
    cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
        ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; .venv/bin/python -c 'import etx; print(\"etx ok\")'")
job(name="llm_env", lane="cpu", reads="apt", timeout=1800,
    cmd="[ -d ~/llmenv ] || python3 -m venv ~/llmenv; . ~/llmenv/bin/activate && pip install -q --upgrade pip && "
        "pip install -q numpy 'huggingface_hub[hf_transfer]' hf_transfer safetensors && "
        "{ python3 -c 'import torch' 2>/dev/null || pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch; } && "
        "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'")
job(name="dl", lane="cpu", reads="llm_env", timeout=2400, quiet_s=900,
    cmd=f"HF_HUB_ENABLE_HF_TRANSFER=1 {PY} -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF}', local_dir='{LLM}/{M}/hf', allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'], "
        f"max_workers=16); print('DL-DONE')\" && test -f {LLM}/{M}/hf/config.json")
job(name="prep", gpu=0, reads=["dl", "etx_env"], timeout=2400, quiet_s=600,
    cmd=f"{PY} examples/vllm_llm/prep.py --src {LLM}/{M}/hf --out {LLM}/{M} --maxlen 1072 && "
        f"cp examples/llm/results/golden_c1024_{M}.txt {LLM}/{M}/golden_c1024.txt && ls -la {LLM}/{M}/vl.bin", summary=[r"wrote .*vl.bin"])

# ---------------------------------------------------------------- builds (cpu lane, parallel); the shipped defaults + one flag set
ALL = "-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH"
BUILDS = {"ctl": "", "all": ALL, "wsum": "-DETX_DSL_WSUM", "scalar": "-DETX_DSL_SCALAR", "gu1": "-DETX_DSL_GU1",
          "rbatch": "-DETX_DSL_RBATCH"}
# megakernel VGPRs spilled / scratch bytes per lane from the ROCm 7.2.4 container (rocm_build_check.sh): the VM's
# compiler should print the same; a different number means a different compiler, not a failed build
SPILL = {"ctl": (8, 36), "all": (9, 40), "wsum": (8, 36), "scalar": (9, 40), "gu1": (14, 44), "rbatch": (7, 32)}
FLAGS = [f for f in BUILDS if f != "ctl"]
for b, flags in BUILDS.items():
    sp, sc = SPILL[b]
    job(name=f"build_{b}", lane="cpu", reads=["etx_env", "dl"], timeout=2400, quiet_s=900,
        cmd=f"{V} ETX_HIPCC_FLAGS='{flags}' ETX_OUT={S}/b_{b} bash examples/vllm_llm/build.sh {LLM}/{M} > {S}/b_{b}.buildlog 2>&1; "
            f"grep -E 'tasks=|Function Name|VGPRs|error' {S}/b_{b}.buildlog | tail -4; cp {S}/b_{b}.buildlog {S}/buildlog_{b}.txt; "
            f"cp {S}/b_{b}/vl_plan.txt {S}/vlplan_{b}.txt; cp {S}/b_{b}/dsl_tiles.hip {S}/dsl_tiles_{b}.hip; "
            f"grep -q 'megakernel_d0.*ScratchSize \\[bytes/lane\\]: {sc}\\b.*VGPRs Spill: {sp}\\b' {S}/b_{b}.buildlog "
            f"&& echo 'SPILL as predicted ({sp} / {sc})' || echo 'SPILL differs from the container ({sp} / {sc})'; test -x {S}/b_{b}/run",
        summary=[r"tasks=|megakernel|SPILL"])


def run(b, extra=""):
    return f"{S}/b_{b}/run {LLM}/{M} {S}/b_{b}/vl_plan.txt {extra}"


# ---------------------------------------------------------------- correctness first (GPU 0): the host exits 0 on a mismatch, so grep
# Stage buffers (the last layer's resid, hn, qkv, attn_out, o, rlog, topk, mc1-3, dn, logits) after decode step 1040 of
# 1055, and every step's argmax. The control runs twice: if ctl and ctl2 differ, bit identity cannot be tested this way.
prev = "prep"
for b in ["ctl", "all"] + FLAGS[1:] + ["ctl2"]:
    bb = "ctl" if b == "ctl2" else b
    job(name=f"gate_{b}", gpu=0, reads=[f"build_{bb}", "prep"], timeout=600,
        cmd=f"set -o pipefail; rm -rf {S}/dump_{b}; mkdir -p {S}/dump_{b}; ETX_DUMP={S}/dump_{b} ETX_DUMP_STEP=1040 "
            f"ETX_TOKTRACE={S}/tok_{b}.txt ETX_VL_PROGRESS=256 stdbuf -oL {run(bb, '--repeat 1 --check-dsl')} 2>&1 | tee {S}/gate_{b}.txt; "
            f"grep -q '^RESULT .*matched=32 gen=32' {S}/gate_{b}.txt && grep -q '^CHECK-DSL OK' {S}/gate_{b}.txt && "
            f"test -s {S}/dump_{b}/logits.bin && echo GATE-OK", summary=[RES])
    prev = f"gate_{b}"
job(name="bitcmp", lane="cpu", reads=["gate_ctl", *(f"?gate_{b}" for b in FLAGS + ["ctl2"])], timeout=300,
    cmd=f"cd {S}; ls dump_ctl/*.bin >/dev/null || {{ echo 'BITCMP-ERROR no control dump'; exit 1; }}; ls -la dump_ctl; "
        f"for b in ctl2 {' '.join(FLAGS)}; do d=''; [ -s dump_$b/logits.bin ] || d=' no-dump'; "
        f"for f in dump_ctl/*.bin; do n=$(basename $f); cmp -s $f dump_$b/$n || d=\"$d $n\"; done; "
        f"cmp -s tok_ctl.txt tok_$b.txt || d=\"$d argmax-trace\"; "
        f"if [ -z \"$d\" ]; then echo \"BITCMP $b identical\"; else echo \"BITCMP $b DIFFERS$d\"; fi; done", summary=[r"BITCMP"])

# ---------------------------------------------------------------- timing (GPU 0): adjacent control / flag pairs, three rounds
# a pair needs both builds' gates; the default --repeat 2 reports the second (warm) run
prev = "bitcmp"
for rnd in (1, 2, 3):
    order = FLAGS if rnd != 2 else FLAGS[::-1]
    for f in order:
        arms = [("ctl", "ctl"), ("x", f)] if rnd != 2 else [("x", f), ("ctl", "ctl")]
        for arm, b in arms:
            name = f"ab{rnd}_{f}_{arm}"
            job(name=name, gpu=0, reads=[f"gate_{f}", "gate_ctl", f"?{prev}"], timeout=300,
                cmd=f"stdbuf -oL {run(b)}", summary=[RES])
            prev = name

# ---------------------------------------------------------------- per-type bodies (one traced full run per build, last step)
TR = ["ctl"] + FLAGS
for b in TR:
    job(name=f"tr_{b}", gpu=0, reads=[f"gate_{b}"], timeout=300,
        cmd=f"ETX_TRACE=1 ETX_TRACE_RAW={S}/tr_{b}.bin stdbuf -oL {run(b, '--repeat 1')}", summary=[RES, r"total span"])
    prev = f"tr_{b}"
job(name="verdict", lane="cpu", reads=[f"?{prev}"], timeout=120,
    cmd=f".venv/bin/python -m examples.vllm_llm.tile_ab {S}", summary=[r"."])

# ---------------------------------------------------------------- nice to have, last: instruction counters, control and all
PMC = "SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_SMEM"
SUM = ("import csv,glob,sys,collections; c=collections.Counter(); "
       "[c.update({r['Counter_Name']: float(r['Counter_Value'])}) for f in glob.glob(sys.argv[1]+'/**/*counter_collection.csv', recursive=True) "
       "for r in csv.DictReader(open(f)) if 'megakernel' in r.get('Kernel_Name', '')]; "
       "print('PMC', sys.argv[1], ' '.join(f'{k}={v:.4g}' for k, v in sorted(c.items())))")
for b in ("ctl", "all"):
    job(name=f"pmc_{b}", gpu=0, reads=[f"gate_{b}", "?verdict"], timeout=900,
        cmd=f"rocprofv3 --pmc {PMC} -d {S}/pmc_{b} -o pmc --output-format csv -- {run(b, '--max-steps 64 --repeat 1')} "
            f"> {S}/pmc_{b}.run.log 2>&1; tail -2 {S}/pmc_{b}.run.log; python3 -c \"{SUM}\" {S}/pmc_{b}", summary=[r"^PMC"])
    prev = f"pmc_{b}"
