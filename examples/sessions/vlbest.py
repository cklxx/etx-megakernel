"""Strongest-tested vLLM search on one MI300X VF, Qwen3-30B-A3B bf16.

VM life is capped at 150 minutes, including provisioning, push and ssh.
The runner cap is 140 minutes, derived by reserving a 10-minute setup allowance.
The operator must delete the VM at its life deadline or immediately on completion;
MAX_SESSION_SECONDS bounds the runner and does not enforce VM deletion.
At $1.99/GPU-hour, the plan is 65–75 minutes / $2.16–2.49, with a
155-minute / $5.14 estimated overrun ceiling; the hard VM stop is 150 minutes.
Do not override the runner cap below 140 minutes.
S9 is absent because ray is missing in the pinned image (session-iv3fdc3m/docker_pull.log:8).
Drop order: triplets, then C2, then S4f, then S5f, then the S3 re-screen; never S1, never the finals.

Host wall, batch 1, context 1024, 32 greedy tokens. Before timing, require the rebuilt CHECK
build identity to equal the prior 97/97 certificate at positions 0 and 1040.
Screen S1, S3, S4f, S5f and S6f twice between S0 brackets; exclude arms without lever proof or
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
import shutil

from examples.vllm_llm import import_certificate as cert
from examples.vllm_llm.pairs import Refusal

from etx.tools.gpu_jobs import session_limit
from examples.sessions import nvchk
from examples.vllm_llm import vlbest_report as report

VM_LIFE_SECONDS = 150 * 60
SETUP_ALLOWANCE_SECONDS = 10 * 60
MAX_SESSION_SECONDS = VM_LIFE_SECONDS - SETUP_ALLOWANCE_SECONDS
BUDGET = f'BUDGET vm-life={VM_LIFE_SECONDS // 60}m runner-cap={MAX_SESSION_SECONDS // 60}m'
if 'ETX_SESS_MAX_SECONDS' in os.environ:
    BUDGET = (f'BUDGET vm-life={VM_LIFE_SECONDS // 60}m '
              f'runner-cap={session_limit(MAX_SESSION_SECONDS) / 60:g}m override=ETX_SESS_MAX_SECONDS')
LOGS = nvchk.LOGS
IMAGE = report.IMAGE
JOBS = []


def job(**kw):
    kw.setdefault('machine', 'A')
    kw.setdefault('quiet_s', 180)
    JOBS.append(kw)


job(name='docker_pull', lane='cpu', timeout=1200,
    cmd=f'set -euo pipefail; sudo docker pull {IMAGE}; '
        f"printf 'IMAGE '; sudo docker inspect --format '{{{{index .RepoDigests 0}}}}' {IMAGE}; "
        f'sudo docker run --rm -v "$HOME:$HOME" -w "$HOME/etx" --entrypoint timeout '
        f'{IMAGE} -k 5 30 python3 -m examples.sessions.vlbest --probe-ray')


job(name='budget', lane='cpu', timeout=10, cmd=f"printf '%s\\n' '{BUDGET}'")
for original in nvchk.JOBS:
    if original['name'] == 'etx_r1':
        break
    if original['name'] in ('check_timed', 'check_certificate'):
        continue
    entry = deepcopy(original)
    if entry['name'] == 'gate_chkx':
        entry['after'] = ['build_identity', 'record']
    deps = entry.get('after', [])
    entry['after'] = ([deps] if isinstance(deps, str) else deps) + ['budget']
    job(**entry)
    if entry['name'] == 'build_chkx':
        job(name='build_identity', lane='cpu', after='build_chkx', timeout=120,
            cmd=f'.venv/bin/python -m examples.sessions.vlbest --verify-build --directory {LOGS}')


def container_prefix():
    # timeout lives inside the daemon-owned container, not just its client.
    return ('sudo docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video '
            '--ipc=host --shm-size 16g --security-opt seccomp=unconfined '
            '-e HIP_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 '
            '-v "$HOME:$HOME" -w "$HOME/etx" --entrypoint timeout ')


def arm_command(cfg, name, timeout=360):
    env = ''.join(f'-e {shlex.quote(k)}="{v}" ' for k, v in cfg['env'].items())
    flags = ' '.join(shlex.quote(f) for f in cfg['flags'])
    return (container_prefix() + env + f'{IMAGE} -k 5 {timeout} python3 examples/llm/vllm_bench.py '
            f'--model {nvchk.MODEL}/hf --golden {nvchk.MODEL}/golden_c1024.txt '
            f'--label {shlex.quote(cfg["id"])} --batch 1 --tokens 32 --repeats 5 --echo-config '
            f'--engine-kw {shlex.quote(json.dumps(cfg["kw"]))} {flags} --json {LOGS}/{name}.json')


# Donor preparation has no budget admission and needs no GPU.
job(name='moe_configs', lane='cpu', after=['docker_pull', 'dl'], timeout=180,
    cmd=f'sudo docker run --rm -v "$HOME:$HOME" -w "$HOME/etx" --entrypoint timeout '
        f'{IMAGE} -k 5 150 python3 -m examples.sessions.vlbest --prepare-moe')

# Admit optional work in reverse drop priority; reserve both screenings and
# the unchanged finals before admitting any lower-priority diagnostic.
job(name='screen_budget', lane='cpu', after=['gate_unf', 'moe_configs'], timeout=10,
    cmd='python3 -m examples.sessions.vlbest --budget-guard screen')
finished = ['screen_budget']
for rnd in (1, 2):
    arms = ['S0_open'] + list(report.SCREEN_ARMS if rnd == 1 else reversed(report.SCREEN_ARMS)) + ['S0_close']
    for arm in arms:
        name = f'a_r{rnd}_{arm}'
        deps = ['docker_pull', 'golden_sha', 'build_identity', 'gate_unf']
        if arm == 'S1':
            deps += ['moe_configs']
        job(name=name, gpu=0, after=deps, after_any=list(finished), timeout=390,
            gate=('screen_budget', f'^SCREEN {arm} GO$') if arm in ('S3', 'S4f', 'S5f') else None,
            cmd=f'.venv/bin/python -m examples.sessions.vlbest --run-arm {arm.split("_")[0]} --name {name} --directory {LOGS}')
        finished.append(name)
job(name='select_screen', lane='cpu', after=['etx_env'], after_any=list(finished), timeout=60,
    cmd=f'.venv/bin/python -m examples.vllm_llm.vlbest_report {LOGS} --select screen')
finished.append('select_screen')
job(name='b_budget', lane='cpu', after='select_screen', timeout=10,
    cmd='python3 -m examples.sessions.vlbest --budget-guard b')
finished.append('b_budget')
for rnd in (1, 2):
    for arm in ('S0_open', 'C2', 'S0_close'):
        name = f'b_r{rnd}_{arm}'
        job(name=name, gpu=0, after=['select_screen', 'build_identity', 'gate_unf'], after_any=list(finished), timeout=390,
            gate=('b_budget', r'^STAGE-B GO$'),
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
        job(name=name, gpu=0, after=['select_final', 'build_identity', 'gate_unf'], after_any=list(finished), timeout=390, cmd=cmd)
        finished.append(name)
job(name='report', lane='cpu', after_any=[j['name'] for j in JOBS], timeout=120,
    cmd=f'.venv/bin/python -m examples.vllm_llm.vlbest_report {LOGS}')


job(name='triplet_budget', lane='cpu', after_any=['report'], timeout=10,
    cmd='python3 -m examples.sessions.vlbest --budget-guard triplets')
finished = ['report', 'triplet_budget']
for original in nvchk.JOBS:
    if original['name'].startswith(('poll_r', 'block_r', 'graph_r')):
        entry = deepcopy(original)
        entry.update(after=['build_identity', 'gate_unf'], after_any=list(finished),
                     gate=('triplet_budget', r'^TRIPLETS GO$'))
        job(**entry)
        finished.append(entry['name'])
job(name='sync_report', lane='cpu', after_any=list(finished), timeout=120,
    cmd=f'.venv/bin/python -m examples.sessions.nvchk --report-sync {LOGS}')


CERTIFICATE = Path(__file__).resolve().parents[1] / 'vllm_llm/certificates/vlbest-iv3fdc3m'
CERTIFIED_SHA256 = '517b6920cde0904c1e17e0dc059bded765cd21de42a285473be54c1c4a1246ef'


def verify_build(directory, build=None, model=None):
    """Transfer only import checks, after verifying actual rebuilt bytes."""
    build = build or Path.home() / 'sess/b_all'
    model = model or Path(nvchk.MODEL.replace('$HOME', str(Path.home())).replace('~', str(Path.home())))
    identity = cert.verified_build(build, model)
    if identity.get('check_enabled') is not True or cert.digest(identity) != CERTIFIED_SHA256:
        raise Refusal('build identity mismatch: certificate does not transfer; stop session')
    prior = cert.validate(CERTIFICATE, timing=False)
    if prior['build_sha256'] != CERTIFIED_SHA256:
        raise Refusal('recorded certificate identity mismatch')
    cert.export_build(build, model, directory)
    for pos in (0, 1040):
        shutil.copyfile(CERTIFICATE / f'check_p{pos}.txt', directory / f'check_p{pos}.txt')
    result = cert.validate(directory, timing=False)
    print('BUILD-IDENTITY TRANSFER ' + json.dumps(result, sort_keys=True), flush=True)
    return 0


def probe_ray():
    import importlib
    try:
        importlib.import_module('ray')
    except ImportError:
        available = False
    else:
        available = True
    print('MOE-TUNE ray=' + ('yes' if available else 'no'), flush=True)


def budget_guard(stage):
    remaining = int(os.environ.get('ETX_SESS_REMAINING_SECONDS', '0'))
    if stage == 'screen':
        # S0 brackets (4), S1/S6f (4), finals (20), report reserve.
        reserve = 8 * 180 + 19 * 60 + 120
        for arm, cost in [('S3', 480), ('S5f', 480), ('S4f', 480), ('C2', 540), ('triplets', 270)]:
            reserve += cost
            print(f'SCREEN {arm} ' + ('GO' if remaining >= reserve else 'SKIPPED budget'), flush=True)
        return 0
    cost, reserve = (9 * 60, 19 * 60) if stage == 'b' else (9 * 30, 120)
    label = 'STAGE-B' if stage == 'b' else 'TRIPLETS'
    admission = Path(os.environ.get('ETX_SESS_LOGDIR', str(Path.home() / 'sess'))) / 'screen_budget.log'
    arm = 'C2' if stage == 'b' else 'triplets'
    admitted = admission.exists() and f'SCREEN {arm} GO' in admission.read_text().splitlines()
    if not admitted or remaining < cost + reserve:
        print(f'{label} SKIPPED budget remaining={remaining} reserve={reserve}', flush=True)
        return 0
    print(f'{label} GO', flush=True)
    return 0


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
    dest = target / 'mi308' / 'E=128,N=768,device_name=AMD_Instinct_MI300X.json'
    shutil.copyfile(candidates[0], dest)
    print(f'MOE-DONOR {candidates[0]} -> {dest}', flush=True)


def partition_failed(text):
    return ('Disabling fuse_qk_norm_rope_kvcache' in text or
            ('ValueError' in text and 'use_inductor_graph_partition' in text and
             ('2.9' in text or 'torch' in text.lower())))


def run_arm(cfg, name, directory):
    """Keep failed attempts in full separate logs; only selected evidence is scored."""
    import time
    cfg = deepcopy(cfg)
    env = dict(os.environ, ETX_SESS_LOGDIR=str(directory))
    if not any(c in ('S4f', 'S5f', 'S6f') for c in cfg['components']):
        return subprocess.run(['bash', '-c', arm_command(cfg, name)], env=env).returncode
    resolved = directory / f"{cfg['id']}.resolved.json"
    if resolved.exists():
        cfg = json.loads(resolved.read_text())
    deadline = time.monotonic() + 365
    for attempt in ('primary', 'fallback'):
        path = directory / f'{name}.{attempt}.log'
        with path.open('w') as output:
            timeout = max(1, min(360, int(deadline - time.monotonic()) - 5))
            proc = subprocess.Popen(['bash', '-c', arm_command(cfg, name, timeout)], env=env,
                                    stdout=output, stderr=subprocess.STDOUT)
            while True:
                try:
                    rc = proc.wait(timeout=min(30, max(.01, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    print(f'ARM-PROGRESS {name} {attempt}: full output in {path}', flush=True)
                    if time.monotonic() >= deadline:
                        proc.kill()
                        proc.wait()
                        return 124
        text = path.read_text()
        cc = cfg['kw'].get('compilation_config', {})
        if attempt == 'primary' and cc.get('use_inductor_graph_partition') and partition_failed(text):
            print(f'ARM-FALLBACK {name}: rejected attempt archived at {path}', flush=True)
            cfg['kw']['compilation_config'] = {'splitting_ops': []}
            continue
        print(text, end='', flush=True)
        if rc == 0:
            resolved.write_text(json.dumps(cfg))
            (directory / f'{name}.config.json').write_text(json.dumps(cfg))
        return rc
    return 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--verify-build', action='store_true')
    mode.add_argument('--run-arm')
    mode.add_argument('--prepare-moe', action='store_true')
    mode.add_argument('--probe-ray', action='store_true')
    mode.add_argument('--budget-guard', choices=['screen', 'b', 'triplets'])
    mode.add_argument('--run-selected', choices=['combination', 'final'])
    ap.add_argument('--arm')
    ap.add_argument('--name')
    ap.add_argument('--directory', type=Path)
    a = ap.parse_args(argv)
    if a.dry_run:
        from etx.tools.gpu_jobs import main as runner
        print(BUDGET, flush=True)
        return runner([__file__, '--dry-run'])
    if a.verify_build:
        try:
            return verify_build(a.directory)
        except (Refusal, OSError, ValueError, KeyError) as exc:
            print(f'BUILD-IDENTITY REFUSED: {exc}', flush=True)
            return 2
    if a.run_arm:
        return run_arm(report.config(a.run_arm), a.name, a.directory)
    if a.probe_ray:
        probe_ray()
        return 0
    if a.budget_guard:
        return budget_guard(a.budget_guard)
    if a.prepare_moe:
        prepare_moe()
        return 0
    data = json.loads((a.directory / ('screen.json' if a.run_selected == 'combination' else 'final.json')).read_text())
    cfg = data[a.run_selected]
    if cfg is None:
        print('NOT-APPLIED C2: fewer than two compatible winning axes', flush=True)
        return 0
    if a.arm.startswith('S0'):
        cfg = report.config('S0')
    return run_arm(cfg, a.name, a.directory)


if __name__ == '__main__':
    raise SystemExit(main())
