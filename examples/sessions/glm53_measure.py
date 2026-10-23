"""GLM-5.3 launchable build and T10 measurement on one 2x MI300X/MI325X VM (world 2, batch 1).

Order (docs/glm/PARALLEL-MEASUREMENT-PLAN-2026-10-22.md): environment and capacity checks, the
pinned checkpoint, staging, prompts and the HF reference (host Python with torch), then the PP2
reference build and its certificate, then each candidate arm, then five interleaved,
counterbalanced reference/candidate bench pairs per arm, judged by examples.glm53.pairs with the
frozen keep/drop rule.

Every build, qualification and bench runs inside the ROCm 7.2.4 image (examples/glm53/rocm724.sh),
so the compiler that keeps 2 waves/SIMD and the HIP runtime the binary links (libamdhip64.so.7) are
the same whatever ROCm the host has; build_model.sh refuses a non-resident binary
(examples.glm53.residency). This path was rehearsed without a GPU on the module-gate stage: build,
link, GLM-RESIDENCY PASS 608/608, GPU-LINKED identity, and bench up to hipGetDeviceCount.
Every timed run is serial (each bench reads the previous one); nothing overlaps a timed run.

Environment:
  GLM_MODEL_ROOT   checkpoint directory (downloaded here if absent; ~305 GiB)
  GLM_ARMS         comma list of candidate arms (default ffn-tp2,ep2,pp2-up-balanced)
  GLM_MAX_MEMORY   HF reference device map (default 170 GiB per MI300X GPU + 400 GiB CPU)
  GLM_PAIRS        interleaved pairs per arm (default 5)
  GLM_GPU_GIB      per-GPU capacity for the plan check (default 192, MI300X; 256 on MI325X)
  ETX_ROCM_BASE    base image for rocm724.sh (default rocm/dev-ubuntu-22.04:7.2.4)

Estimated GPU-VM time (not measured): image 0.2 h, download 1 h, stage 1-2 h, HF reference 2-6 h,
PP2 build and certificate 0.5 h, three arm builds 0.5 h, 3 x 10 benches at about 5 min each 2.5 h:
8-13 h. MAX_SESSION_SECONDS caps the whole session at 14 h. Delete the VM when the runner ends.
"""
import json
import os
import shlex

ARMS = {  # name -> Options fields (world 2, B1); the PP2 reference uses the measured default.
    'ffn-tp2': dict(devices=2, parallel=1),
    'ep2': dict(devices=2, parallel=2),
    'pp2-up-balanced': dict(devices=2, up_tail_balance=1),
}
arms = [a for a in os.environ.get('GLM_ARMS', 'ffn-tp2,ep2,pp2-up-balanced').split(',') if a]
unknown = [a for a in arms if a not in ARMS]
if unknown:
    raise ValueError(f'unknown GLM arms {unknown}; known: {sorted(ARMS)}')
PAIRS = int(os.environ.get('GLM_PAIRS', '5'))
if PAIRS < 1:
    raise ValueError('GLM_PAIRS must be positive')
MAX_SESSION_SECONDS = 14 * 3600
gpu_gib = float(os.environ.get('GLM_GPU_GIB', '192'))
model = shlex.quote(os.environ.get('GLM_MODEL_ROOT', os.path.expanduser('~/glm53-model')))
memory = shlex.quote(os.environ.get('GLM_MAX_MEMORY', '{"0":"170GiB","1":"170GiB","cpu":"400GiB"}'))
PY = '.venv/bin/python -u'
LOG = '"$ETX_SESS_LOGDIR"'
STAGE = f'{LOG}/stage'
PROMPTS = f'{LOG}/prompts.json'
REFERENCE = f'{LOG}/reference.json'
TWO = {'HIP_VISIBLE_DEVICES': '0,1'}
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
    dict(name='preflight', lane='cpu', reads=['etx_env', 'image'], timeout=300, env=TWO,
         cmd="rocm-smi --showproductname --showmeminfo vram | grep -E 'Card series|VRAM Total'; nproc; free -g | head -2; "
             "df -h \"$HOME\" | tail -1; "
             ".venv/bin/python -c 'import torch; n=torch.cuda.device_count(); assert n==2, n; "
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
    dict(name='reference', lane='cpu', reads='prompts', timeout=6 * 3600, quiet_s=900, env=TWO,
         cmd=f"mkdir -p {LOG}/offload && {PY} -m examples.glm53.reference --src {model} --prompts {PROMPTS} "
             f"--out {REFERENCE} --offload {LOG}/offload --max-memory {memory} --gate production",
         writes=['reference.json']),
    # PP2 reference build (measured default options) and its own certificate: the launchable build.
    dict(name='pp2_build', lane='cpu', reads=['stage', 'image'], timeout=2400, quiet_s=300, env={'ETX_GLM_BATCH': '1'},
         cmd=f"{IMAGE} bash examples/glm53/build_model.sh {STAGE} {LOG}/pp2/build",
         writes=['pp2/build/run', 'pp2/build/residency.json'], summary=[r'GLM-RESIDENCY (PASS|FAIL)']),
    dict(name='pp2_certificate', lane='cpu', reads=['pp2_build', 'reference'], timeout=2 * 3600, quiet_s=600, env=TWO,
         cmd=bench('pp2/certificate', f'{LOG}/pp2/build'),
         writes=['pp2/certificate/bench_result.json'], summary=[r'^BENCH', r'^DEVICES', r'^CORRECTNESS', r'^TIMING']),
]

previous = 'pp2_certificate'
for arm in arms:
    options = json.dumps(dict(ARMS[arm]))
    build = f'{LOG}/{arm}/build'
    # Each arm needs the PP2 certificate; it only waits for the previous arm, whatever that arm's result.
    reads = ['pp2_certificate'] + ([f'?{previous}'] if previous != 'pp2_certificate' else [])
    JOBS.append(dict(
        name=f'{arm}_build', lane='cpu', reads=reads, timeout=2400, quiet_s=300, env={'ETX_GLM_BATCH': '1'},
        cmd=f"mkdir -p {LOG}/{arm} && {PY} -c 'import json,sys; from examples.glm53.options import Options; "
            f"print(json.dumps(Options(**json.loads(sys.argv[1])).to_dict()))' {shlex.quote(options)} > {LOG}/{arm}/options.json && "
            f"{IMAGE} bash examples/glm53/build_model.sh {STAGE} {build} --options {LOG}/{arm}/options.json",
        writes=[f'{arm}/build/run', f'{arm}/build/residency.json'], summary=[r'GLM-RESIDENCY (PASS|FAIL)']))
    previous = f'{arm}_build'
    certificates = []
    for i in range(PAIRS):
        order = ('ref', 'cand') if i % 2 == 0 else ('cand', 'ref')    # counterbalance the order across pairs
        for side in order:
            name = f'{arm}_p{i}_{side}'
            JOBS.append(dict(
                name=name, lane='cpu', reads=previous, timeout=2 * 3600, quiet_s=600, env=TWO,
                cmd=bench(f'{arm}/{name}', f'{LOG}/pp2/build' if side == 'ref' else build),
                writes=[f'{arm}/{name}/bench_result.json'], summary=[r'^BENCH', r'^DEVICES', r'^CORRECTNESS']))
            certificates.append(f'{arm}/{name}/bench_result.json')
            previous = name
    JOBS.append(dict(
        name=f'{arm}_pairs', lane='final', reads=[f'?{c}' for c in certificates], timeout=300,
        cmd=f"{PY} -m examples.glm53.pairs --pairs {PAIRS} --out {LOG}/{arm}/pairs.json "
            + ' '.join(f'{LOG}/{c}' for c in certificates),
        writes=[f'{arm}/pairs.json'], summary=[r'^GLM-PAIR']))
