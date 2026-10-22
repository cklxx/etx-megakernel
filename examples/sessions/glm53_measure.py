"""GLM-5.3 launchable build and T10 measurement on one 2x MI300X/MI325X VM (world 2, batch 1).

Order (docs/glm/PARALLEL-MEASUREMENT-PLAN-2026-10-22.md): environment and capacity checks, the
pinned checkpoint, then examples.glm53.run_all for the PP2 reference (stage, HF reference, build with
the residency gate, two-run qualification with HF greedy 32/32 and all logits, oracle, bench
certificate). Only after PP2 passes: each candidate arm is built from the same stage and qualified
by its own bench, then five interleaved, counterbalanced reference/candidate bench pairs, judged by
examples.glm53.pairs with the frozen keep/drop rule. Every timed run is serial (each bench reads the
previous one); nothing overlaps a timed run.

Environment:
  GLM_MODEL_ROOT   checkpoint directory (downloaded here if absent; ~305 GiB)
  GLM_ARMS         comma list of candidate arms (default ffn-tp2,ep2,pp2-up-balanced)
  GLM_MAX_MEMORY   HF reference device map (default 170 GiB per MI300X GPU + 400 GiB CPU)
  GLM_PAIRS        interleaved pairs per arm (default 5)
  GLM_GPU_GIB      per-GPU capacity for the plan check (default 192, MI300X; 256 on MI325X)

The builds need a compiler that keeps 2 waves/SIMD (ROCm 7.2.4 does; 6.4.1 does not):
examples/glm53/build_model.sh refuses a non-resident binary through examples.glm53.residency.
Estimated GPU-VM time (not measured): download 1 h, stage 1-2 h, HF reference 2-6 h, PP2 build and
qualification 0.5 h, three arm builds 0.5 h, 3 x 10 benches at about 5 min each 2.5 h:
8-12 h. MAX_SESSION_SECONDS caps the whole session at 13 h. Delete the VM when the runner ends.
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
MAX_SESSION_SECONDS = 13 * 3600
model = shlex.quote(os.environ.get('GLM_MODEL_ROOT', os.path.expanduser('~/glm53-model')))
gpu_gib = float(os.environ.get('GLM_GPU_GIB', '192'))   # per-GPU capacity the plan must fit (MI300X 192, MI325X 256)
memory = shlex.quote(os.environ.get('GLM_MAX_MEMORY', '{"0":"170GiB","1":"170GiB","cpu":"400GiB"}'))
PY = '.venv/bin/python -u'
LOG = '"$ETX_SESS_LOGDIR"'
PP2 = f'{LOG}/pp2'
STAGE = f'{PP2}/stage'
REFERENCE = f'{PP2}/reference.json'
TWO = {'HIP_VISIBLE_DEVICES': '0,1'}

JOBS = [
    dict(name='apt', lane='cpu', timeout=600,
         cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3-venv; }"),
    dict(name='etx_env', lane='cpu', reads='apt', timeout=1800,
         cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
             ".venv/bin/pip install -q -e '.[reference]' accelerate huggingface_hub; "
             ".venv/bin/python -c 'import etx, numpy, torch, transformers, safetensors, accelerate; "
             "print(\"torch\", torch.__version__, \"gpus\", torch.cuda.device_count())'"),
    # Record the toolchain and capacity facts the session depends on; refuse early instead of hours in.
    dict(name='preflight', lane='cpu', reads='etx_env', timeout=300, env=TWO,
         cmd="hipcc --version | head -2; rocm-smi --showproductname --showmeminfo vram | grep -E 'Card series|VRAM Total'; "
             "nproc; free -g | head -2; df -h \"$HOME\" | tail -1; "
             ".venv/bin/python -c 'import torch; n=torch.cuda.device_count(); assert n==2, n; "
             "[print(torch.cuda.get_device_properties(i).name, torch.cuda.get_device_properties(i).multi_processor_count) for i in range(n)]' && "
             "test $(df --output=avail -BG \"$HOME\" | tail -1 | tr -dc 0-9) -ge 750 || { echo 'NEED >=750 GiB free disk'; exit 1; }",
         summary=[r'clang version', r'MI3', r'NEED']),
    dict(name='download', lane='cpu', reads='preflight', timeout=4 * 3600, quiet_s=900,
         cmd=f"test -f {model}/REVISION || {{ REV=$({PY} -c 'from examples.glm53.prepare import metadata; print(metadata()[1][\"sha\"])'); "
             f".venv/bin/hf download zai-org/GLM-5.3-Flash --revision $REV --local-dir {model} && echo $REV > {model}/REVISION; }}"),
    # PP2 reference end to end: stage, HF reference, build (residency-gated), qualify x2, oracle, bench.
    dict(name='pp2_run_all', lane='cpu', reads='download', timeout=10 * 3600, quiet_s=900, env=TWO,
         cmd=f"{PY} -m examples.glm53.run_all --model-root {model} --out {PP2} --batch 1 --gpus 2 "
             f"--max-memory {memory} --bandwidth-tbps 4.3 --gpu-gib {gpu_gib}",
         writes=['pp2/certificate/bench_result.json'],
         summary=[r'^RESULT', r'^BANDWIDTH', r'^FAILED', r'GLM-RESIDENCY (PASS|FAIL)', r'TOKENS']),
]

previous = 'pp2_run_all'
for arm in arms:
    options = json.dumps(dict(ARMS[arm]))
    build = f'{LOG}/{arm}/build'
    # Each arm needs the PP2 reference; it only waits for the previous arm, whatever that arm's result.
    reads = ['pp2_run_all'] + ([f'?{previous}'] if previous != 'pp2_run_all' else [])
    JOBS.append(dict(
        name=f'{arm}_build', lane='cpu', reads=reads, timeout=2400, quiet_s=300,
        cmd=f"mkdir -p {LOG}/{arm} && {PY} -c 'import json,sys; from examples.glm53.options import Options; "
            f"print(json.dumps(Options(**json.loads(sys.argv[1])).to_dict()))' {shlex.quote(options)} > {LOG}/{arm}/options.json && "
            f"ETX_GLM_BATCH=1 ETX_PYTHON=.venv/bin/python bash examples/glm53/build_model.sh {STAGE} {build} --options {LOG}/{arm}/options.json",
        writes=[f'{arm}/build/run', f'{arm}/build/residency.json'], summary=[r'GLM-RESIDENCY (PASS|FAIL)']))
    previous = f'{arm}_build'
    certificates = []
    for i in range(PAIRS):
        order = ('ref', 'cand') if i % 2 == 0 else ('cand', 'ref')    # counterbalance the order across pairs
        for side in order:
            target = f'{PP2}/build' if side == 'ref' else build
            name = f'{arm}_p{i}_{side}'
            JOBS.append(dict(
                name=name, lane='cpu', reads=previous, timeout=2 * 3600, quiet_s=600, env=TWO,
                cmd=f"{PY} -m examples.glm53.bench --stage {STAGE} --build {target} --reference {REFERENCE} "
                    f"--out {LOG}/{arm}/{name} --seconds 3600",
                writes=[f'{arm}/{name}/bench_result.json'], summary=[r'^BENCH', r'^DEVICES', r'^CORRECTNESS']))
            certificates.append(f'{LOG}/{arm}/{name}/bench_result.json')
            previous = name
    JOBS.append(dict(
        name=f'{arm}_pairs', lane='final', reads=[f'?{arm}/{c.rsplit("/", 2)[1]}/bench_result.json' for c in certificates],
        timeout=300, cmd=f"{PY} -m examples.glm53.pairs --pairs {PAIRS} --out {LOG}/{arm}/pairs.json {' '.join(certificates)}",
        writes=[f'{arm}/pairs.json'], summary=[r'^GLM-PAIR']))
