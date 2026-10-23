"""GLM-5.3 Phase 1 (/tmp/o55-lead/O55-GOAL.md): single stream, batch 1, one replica on 8x MI325X.

Arms, in order: EP8 (parallel=2), headTP+EP8 (parallel=2, attention_tp=1), TP8 (parallel=1), all
devices=8. Pipeline parallelism is not built or run at batch 1.

Order: environment and capacity checks, the pinned checkpoint, staging, prompts and the HF reference
(host Python with torch), then per arm: build (refused unless GLM-RESIDENCY PASS) and its certificate
run, which is the full-model HF greedy 32/32 gate with all 32 logits. No timing from an arm is quoted
unless its certificate shows CORRECTNESS 32/32.

Paired T10 timing (examples.glm53.pairs) runs only when GLM_REFERENCE names the comparator arm: the
goal does not name a batch-1 comparator, so none is assumed. Without it the session ends after the
certificates.

Every build, qualification and bench runs inside the ROCm 7.2.4 image (examples/glm53/rocm724.sh);
the same three arms compiled, linked and passed residency 608/608 on the desktop with that toolchain.
Every timed run is serial; nothing overlaps a timed run.

Environment:
  GLM_MODEL_ROOT   checkpoint directory (downloaded here if absent; ~305 GiB)
  GLM_ARMS         comma list, run in this order (default ep8,head-tp-ep8,tp8)
  GLM_REFERENCE    comparator arm for T10 pairs (default unset: certificates only)
  GLM_MAX_MEMORY   HF reference device map (default 230 GiB on each of 8 GPUs + 400 GiB CPU)
  GLM_PAIRS        interleaved pairs per candidate arm (default 5)
  GLM_GPU_GIB      per-GPU capacity for the plan check (default 256, MI325X)
  ETX_ROCM_BASE    base image for rocm724.sh (default rocm/dev-ubuntu-22.04:7.2.4)

Estimated GPU-VM time (not measured): image 0.2 h, download 1 h, stage 1-2 h, HF reference 1-4 h,
three builds 0.5 h, three certificates 0.5 h; with GLM_REFERENCE, 2 x 10 benches at about 5 min
each 1.7 h: 4-10 h. MAX_SESSION_SECONDS caps the session at 12 h. Delete the VM when the runner ends.
"""
import json
import os
import shlex

ARMS = {  # name -> Options fields (one 8-card replica, B1)
    'ep8': dict(devices=8, parallel=2),
    'head-tp-ep8': dict(devices=8, parallel=2, attention_tp=1),
    'tp8': dict(devices=8, parallel=1),
}
arms = [a for a in os.environ.get('GLM_ARMS', 'ep8,head-tp-ep8,tp8').split(',') if a]
reference_arm = os.environ.get('GLM_REFERENCE', '')
unknown = [a for a in arms + ([reference_arm] if reference_arm else []) if a not in ARMS]
if unknown:
    raise ValueError(f'unknown GLM arms {unknown}; known: {sorted(ARMS)}')
if reference_arm and reference_arm not in arms:
    raise ValueError(f'GLM_REFERENCE {reference_arm} must be one of GLM_ARMS {arms}')
PAIRS = int(os.environ.get('GLM_PAIRS', '5'))
if PAIRS < 1:
    raise ValueError('GLM_PAIRS must be positive')
MAX_SESSION_SECONDS = 12 * 3600
gpu_gib = float(os.environ.get('GLM_GPU_GIB', '256'))
model = shlex.quote(os.environ.get('GLM_MODEL_ROOT', os.path.expanduser('~/glm53-model')))
memory = shlex.quote(os.environ.get('GLM_MAX_MEMORY', json.dumps({**{str(i): '230GiB' for i in range(8)}, 'cpu': '400GiB'})))
PY = '.venv/bin/python -u'
LOG = '"$ETX_SESS_LOGDIR"'
STAGE = f'{LOG}/stage'
PROMPTS = f'{LOG}/prompts.json'
REFERENCE = f'{LOG}/reference.json'
EIGHT = {'HIP_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}
IMAGE = f'bash examples/glm53/rocm724.sh {LOG} --'          # build/run inside the ROCm 7.2.4 image


def bench(name, build):
    return (f"{IMAGE} python3 -u -m examples.glm53.bench --stage {STAGE} --build {build} --reference {REFERENCE} "
            f"--out {LOG}/{name} --seconds 3600")


JOBS = [
    dict(name='apt', lane='cpu', timeout=600,
         cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3-venv; }"),
    dict(name='etx_env', lane='cpu', reads='apt', timeout=1800,
         cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
             ".venv/bin/pip install -q -e '.[reference]' accelerate huggingface_hub; "
             ".venv/bin/python -c 'import etx, numpy, torch, transformers, safetensors, accelerate; "
             "print(\"torch\", torch.__version__, \"gpus\", torch.cuda.device_count())'"),
    # The build/run image: the only networked step besides the download. Must report roc-7.2.4.
    dict(name='image', lane='cpu', reads='apt', timeout=1800, quiet_s=600,
         cmd=f"mkdir -p {LOG} && {IMAGE} hipcc --version | tee {LOG}/image-hipcc.txt && grep -q 'roc-7.2.4' {LOG}/image-hipcc.txt",
         writes=['image-hipcc.txt'], summary=[r'clang version']),
    dict(name='preflight', lane='cpu', reads=['etx_env', 'image'], timeout=300, env=EIGHT,
         cmd="rocm-smi --showproductname --showmeminfo vram | grep -E 'Card series|VRAM Total'; nproc; free -g | head -2; "
             "df -h \"$HOME\" | tail -1; "
             ".venv/bin/python -c 'import torch; n=torch.cuda.device_count(); assert n==8, n; "
             "[print(torch.cuda.get_device_properties(i).name, torch.cuda.get_device_properties(i).multi_processor_count) for i in range(n)]' && "
             "test $(df --output=avail -BG \"$HOME\" | tail -1 | tr -dc 0-9) -ge 750 || { echo 'NEED >=750 GiB free disk'; exit 1; }",
         summary=[r'MI3', r'NEED']),
    dict(name='download', lane='cpu', reads='preflight', timeout=4 * 3600, quiet_s=900,
         cmd=f"test -f {model}/REVISION || {{ REV=$({PY} -c 'from examples.glm53.prepare import metadata; print(metadata()[1][\"sha\"])'); "
             f".venv/bin/hf download zai-org/GLM-5.3-Flash --revision $REV --local-dir {model} && echo $REV > {model}/REVISION; }}"),
    dict(name='stage', lane='cpu', reads='download', timeout=3 * 3600, quiet_s=900,
         cmd=f"{PY} -m examples.glm53.stage --src {model} --out {STAGE} --maxlen 1072 --gate production --trust-verified",
         writes=['stage/stage.json']),
    dict(name='capacity', lane='cpu', reads='stage', timeout=600,
         cmd=f"{PY} -m examples.glm53.prepare --require-model --stage {STAGE} --gate production --batch 1 --context 1024 "
             f"--gpu-gib {gpu_gib} --out {LOG}/prepare.json", writes=['prepare.json']),
    dict(name='prompts', lane='cpu', reads='capacity', timeout=300,
         cmd=f"{PY} -c 'import sys; from pathlib import Path; from examples.glm53 import run_all; "
             f"from examples.glm53.prepare import metadata; "
             f"run_all.prompts(Path(sys.argv[1]), 1, run_all.SEED, metadata()[0][\"text_config\"][\"vocab_size\"])' {PROMPTS}",
         writes=['prompts.json']),
    dict(name='reference', lane='cpu', reads='prompts', timeout=6 * 3600, quiet_s=900, env=EIGHT,
         cmd=f"mkdir -p {LOG}/offload && {PY} -m examples.glm53.reference --src {model} --prompts {PROMPTS} "
             f"--out {REFERENCE} --offload {LOG}/offload --max-memory {memory} --gate production",
         writes=['reference.json']),
]

# Builds run in goal order; each certificate (the 32/32 gate) waits for its build and the previous
# certificate, whatever that certificate's result, so one failing arm does not hide the next.
previous = 'reference'
for arm in arms:
    options = json.dumps(ARMS[arm])
    build = f'{LOG}/{arm}/build'
    JOBS.append(dict(
        name=f'{arm}_build', lane='cpu', reads=['stage', 'image'], timeout=2400, quiet_s=300, env={'ETX_GLM_BATCH': '1'},
        cmd=f"mkdir -p {LOG}/{arm} && {PY} -c 'import json,sys; from examples.glm53.options import Options; "
            f"print(json.dumps(Options(**json.loads(sys.argv[1])).to_dict()))' {shlex.quote(options)} > {LOG}/{arm}/options.json && "
            f"{IMAGE} bash examples/glm53/build_model.sh {STAGE} {build} --options {LOG}/{arm}/options.json",
        writes=[f'{arm}/build/run', f'{arm}/build/residency.json'], summary=[r'GLM-RESIDENCY (PASS|FAIL)']))
    JOBS.append(dict(
        name=f'{arm}_certificate', lane='cpu', reads=[f'{arm}_build', 'reference'] + ([f'?{previous}'] if previous != 'reference' else []),
        timeout=2 * 3600, quiet_s=600, env=EIGHT, cmd=bench(f'{arm}/certificate', build),
        writes=[f'{arm}/certificate/bench_result.json'], summary=[r'^BENCH', r'^DEVICES', r'^CORRECTNESS', r'^TIMING']))
    previous = f'{arm}_certificate'

# T10 pairs against the named comparator; a pair job needs both certificates (both arms 32/32).
if reference_arm:
    for arm in [a for a in arms if a != reference_arm]:
        certificates = []
        first = True
        for i in range(PAIRS):
            order = ('ref', 'cand') if i % 2 == 0 else ('cand', 'ref')    # counterbalance the order across pairs
            for side in order:
                name = f'{arm}_p{i}_{side}'
                # The first bench needs both 32/32 certificates; every bench runs after the previous one.
                reads = [f'{reference_arm}_certificate', f'{arm}_certificate', f'?{previous}'] if first else [previous]
                first = False
                JOBS.append(dict(
                    name=name, lane='cpu', reads=reads, timeout=2 * 3600, quiet_s=600, env=EIGHT,
                    cmd=bench(f'{arm}/{name}', f'{LOG}/{reference_arm}/build' if side == 'ref' else f'{LOG}/{arm}/build'),
                    writes=[f'{arm}/{name}/bench_result.json'], summary=[r'^BENCH', r'^DEVICES', r'^CORRECTNESS']))
                certificates.append(f'{arm}/{name}/bench_result.json')
                previous = name
        JOBS.append(dict(
            name=f'{arm}_pairs', lane='final', reads=[f'?{c}' for c in certificates], timeout=300,
            cmd=f"{PY} -m examples.glm53.pairs --pairs {PAIRS} --out {LOG}/{arm}/pairs.json "
                + ' '.join(f'{LOG}/{c}' for c in certificates),
            writes=[f'{arm}/pairs.json'], summary=[r'^GLM-PAIR']))
