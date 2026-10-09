"""Strongest-tested vLLM search on one MI300X VF, Qwen3-30B-A3B bf16.

VM life is capped at 150 minutes, including provisioning, push and ssh.
The runner cap is 140 minutes, derived by reserving a 10-minute setup allowance.
The operator must delete the VM at its life deadline or immediately on completion;
MAX_SESSION_SECONDS bounds the runner and does not enforce VM deletion.
At $1.99/GPU-hour the VM-life ceiling costs about $4.98 (estimate).

Host wall, batch 1, context 1024, 32 greedy tokens. Before timing, certify all
imported instances at positions 0 and 1040 in the same CHECK-enabled build.
Screen S1–S9 twice between S0 brackets; exclude arms without lever proof or
32/32 on every sample. Bracket drift must be <=0.050 ms; both deltas must be
negative and their mean <=-0.050 ms. Screen one compatible combination twice.
Confirm S0, the finalist, ETX fused and same-kernel HIP graph in five alternating
rounds. Select the faster median vLLM reference from these confirmation rounds.
The report keeps nvref's verdict rule and emits SPREAD-BOUND independently.
No provisioning, image pull or GPU access occurs on import or --dry-run.
"""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from examples.sessions import nvchk
from examples.vllm_llm import vlbest_report as report

VM_LIFE_SECONDS = 150 * 60
SETUP_ALLOWANCE_SECONDS = 10 * 60
MAX_SESSION_SECONDS = VM_LIFE_SECONDS - SETUP_ALLOWANCE_SECONDS
BUDGET = f'BUDGET vm-life={VM_LIFE_SECONDS // 60}m runner-cap={MAX_SESSION_SECONDS // 60}m'
LOGS = nvchk.LOGS
IMAGE = report.IMAGE
JOBS = []


def job(**kw):
    kw.setdefault('machine', 'A')
    kw.setdefault('quiet_s', 180)
    JOBS.append(kw)


job(name='budget', lane='cpu', timeout=10, cmd=f"printf '%s\\n' '{BUDGET}'")
for original in nvchk.JOBS:
    if original['name'] == 'etx_r1':
        break
    entry = deepcopy(original)
    deps = entry.get('after', [])
    entry['after'] = ([deps] if isinstance(deps, str) else deps) + ['budget']
    job(**entry)

job(name='docker_pull', lane='cpu', after='budget', timeout=1200,
    cmd=f'set -euo pipefail; sudo docker pull {IMAGE}; '
        f"printf 'IMAGE '; sudo docker inspect --format '{{{{index .RepoDigests 0}}}}' {IMAGE}")


def container_prefix():
    # timeout lives inside the daemon-owned container, not just its client.
    return ('sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video '
            '--ipc=host --shm-size 16g --security-opt seccomp=unconfined '
            '-e HIP_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 '
            '-v "$HOME:$HOME" -w "$HOME/etx" --entrypoint timeout ')


def arm_command(cfg, name):
    env = ''.join(f'-e {shlex.quote(k)}="{v}" ' for k, v in cfg['env'].items())
    flags = ' '.join(shlex.quote(f) for f in cfg['flags'])
    return (container_prefix() + env + f'{IMAGE} -k 5 360 python3 examples/llm/vllm_bench.py '
            f'--model {nvchk.MODEL}/hf --golden {nvchk.MODEL}/golden_c1024.txt '
            f'--label {shlex.quote(cfg["id"])} --batch 1 --tokens 32 --repeats 5 --echo-config '
            f'--engine-kw {shlex.quote(json.dumps(cfg["kw"]))} {flags} --json {LOGS}/{name}.json')


# Configuration creation is bounded and archived before the search. The tuning
# script is the pinned upstream file; no ETX kernel tuning is performed here.
job(name='moe_configs', gpu=0, after=['docker_pull', 'check_certificate'], timeout=180,
    cmd=container_prefix() + f'{IMAGE} -k 5 150 python3 -m examples.sessions.vlbest --prepare-moe')
job(name='moe_tune', gpu=0, after=['moe_configs', 'dl'], timeout=1230,
    cmd=container_prefix() + f'{IMAGE} -k 5 1200 python3 examples/vllm_llm/vendor/benchmark_moe.py '
        f'--model {nvchk.MODEL}/hf --batch-size 1 --tune --save-dir "$HOME/moecfg/tuned"')

finished = ['moe_tune']
for rnd in (1, 2):
    arms = ['S0_open'] + [f'S{i}' for i in (range(1, 10) if rnd == 1 else range(9, 0, -1))] + ['S0_close']
    for arm in arms:
        name = f'a_r{rnd}_{arm}'
        deps = ['docker_pull', 'golden_sha', 'check_certificate']
        if arm == 'S1':
            deps += ['moe_configs']
        if arm == 'S9':
            deps += ['moe_tune']
        job(name=name, gpu=0, after=deps, after_any=list(finished), timeout=390,
            cmd=arm_command(report.config(arm.split('_')[0]), name))
        finished.append(name)
job(name='select_screen', lane='cpu', after=['etx_env'], after_any=list(finished), timeout=60,
    cmd=f'.venv/bin/python -m examples.vllm_llm.vlbest_report {LOGS} --select screen')
finished.append('select_screen')
for rnd in (1, 2):
    for arm in ('S0_open', 'C', 'S0_close'):
        name = f'b_r{rnd}_{arm}'
        job(name=name, gpu=0, after=['select_screen', 'check_certificate'], after_any=list(finished), timeout=390,
            cmd=f'.venv/bin/python -m examples.sessions.vlbest --run-selected combination --arm {arm} --name {name} --directory {LOGS}')
        finished.append(name)
job(name='select_final', lane='cpu', after='select_screen', after_any=list(finished), timeout=60,
    cmd=f'.venv/bin/python -m examples.vllm_llm.vlbest_report {LOGS} --select final')
finished.append('select_final')
for rnd in range(1, 6):
    for arm in (('S0', 'F', 'etx', 'etxu') if rnd % 2 else ('etxu', 'etx', 'F', 'S0')):
        name = f'{arm}_r{rnd}'
        cmd = (nvchk.capture('--unfused' if arm == 'etxu' else '') if arm.startswith('etx') else
               arm_command(report.config('S0'), name) if arm == 'S0' else
               f'.venv/bin/python -m examples.sessions.vlbest --run-selected final --arm F --name {name} --directory {LOGS}')
        job(name=name, gpu=0, after=['select_final', 'check_certificate'], after_any=list(finished), timeout=390, cmd=cmd)
        finished.append(name)
job(name='report', lane='cpu', after_any=[j['name'] for j in JOBS], timeout=120,
    cmd=f'.venv/bin/python -m examples.vllm_llm.vlbest_report {LOGS}')


def prepare_moe(root=None, target=None):
    import shutil
    if root is None:
        import vllm
        root = Path(vllm.__file__).parent
    candidates = list(root.rglob('E=128,N=768,device_name=AMD_Instinct_MI308*.json'))
    if len(candidates) != 1:
        raise ValueError(f'expected exactly one MI308 donor config, found {len(candidates)}')
    target = target if target is not None else Path.home() / 'moecfg'
    (target / 'mi308').mkdir(parents=True, exist_ok=True)
    (target / 'tuned').mkdir(parents=True, exist_ok=True)
    dest = target / 'mi308' / 'E=128,N=768,device_name=AMD_Instinct_MI300X.json'
    shutil.copyfile(candidates[0], dest)
    print(f'MOE-DONOR {candidates[0]} -> {dest}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--prepare-moe', action='store_true')
    mode.add_argument('--run-selected', choices=['combination', 'final'])
    ap.add_argument('--arm')
    ap.add_argument('--name')
    ap.add_argument('--directory', type=Path)
    a = ap.parse_args(argv)
    if a.dry_run:
        from etx.tools.gpu_jobs import main as runner
        print(BUDGET, flush=True)
        return runner([__file__, '--dry-run'])
    if a.prepare_moe:
        prepare_moe()
        return 0
    data = json.loads((a.directory / ('screen.json' if a.run_selected == 'combination' else 'final.json')).read_text())
    cfg = data[a.run_selected]
    if cfg is None:
        print('NOT-APPLIED C: fewer than two compatible winning axes', flush=True)
        return 0
    if a.arm.startswith('S0'):
        cfg = report.config('S0')
    env = dict(os.environ, ETX_SESS_LOGDIR=str(a.directory))
    return subprocess.run(['bash', '-c', arm_command(cfg, a.name)], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
