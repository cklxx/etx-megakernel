"""S3 vs S3 + S9 tuned Triton MoE, bs1/ctx1024, one MI300X VF.

150 minute VM life / 140 minute runner cap, at most $4.98 at $1.99/hour.
No budget-dropped arms: eight screening runs and twenty fresh finals.
No provisioning on import or dry-run. Operator deletes VM when done.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
from statistics import median
import subprocess
import time

from examples.sessions import nvchk, vlbest as base
from examples.vllm_llm import vlbest_report as report
from examples.vllm_llm.nvref_report import Session, correctness
from examples.vllm_llm.pairs import Refusal

MAX_SESSION_SECONDS = base.MAX_SESSION_SECONDS
BUDGET = base.BUDGET
LOGS, IMAGE = nvchk.LOGS, report.IMAGE
CERTIFIED_SHA256 = base.CERTIFIED_SHA256
verify_build = base.verify_build
arm_command = base.arm_command
SCREEN_ARMS = ('S3', 'M1')
JOBS = []


def job(**kw):
    kw.setdefault('machine', 'A')
    kw.setdefault('quiet_s', 180)
    JOBS.append(kw)


def command(mode):
    return f'.venv/bin/python -m examples.sessions.vlbest3 {mode} --directory {LOGS}'


def g0_command():
    return ('sudo docker run --rm --pull=never --network=none '
            '-e VLLM_TUNED_CONFIG_FOLDER="$HOME/moecfg/tuned" '
            '-e HOME="$HOME" -e PYTHONUNBUFFERED=1 -v "$HOME:$HOME" -w "$HOME/etx" '
            f'--entrypoint timeout {IMAGE} -k 5 60 python3 -m examples.vllm_llm.vlbest3_g0')


job(name='g0', lane='cpu', timeout=75, cmd=g0_command())
job(name='docker_pull', lane='cpu', reads='g0', timeout=30,
    cmd=f"printf 'IMAGE '; sudo docker inspect --format '{{{{index .RepoDigests 0}}}}' {IMAGE}")
job(name='budget', lane='cpu', timeout=10, cmd=f"printf '%s\\n' '{BUDGET}'")
for original in nvchk.JOBS:
    if original['name'] == 'etx_r1':
        break
    if original['name'] in ('check_timed', 'check_certificate'):
        continue
    entry = deepcopy(original)
    if entry['name'] == 'gate_chkx':
        entry['reads'] = ['build_identity', 'record']
    deps = entry.get('reads', [])
    entry['reads'] = ([deps] if isinstance(deps, str) else deps) + ['budget', 'g0']
    job(**entry)
    if entry['name'] == 'build_chkx':
        job(name='build_identity', lane='cpu', reads='build_chkx', timeout=120,
            cmd=command('--verify-build'))

job(name='moe_tune', gpu=0, reads=['gate_unf', 'build_identity', 'g0'], timeout=2700,
    cmd=command('--tune'))
# Always runs, including when the runner killed the tuner client. Removes any
# daemon-owned process by PID before checking the receipt; no orphan tuning.
job(name='moe_gate', lane='cpu', reads=['?moe_tune'], timeout=45,
    cmd=command('--tune-gate'))
previous = 'moe_gate'
for rnd in (1, 2):
    arms = ['S0_open'] + list(SCREEN_ARMS if rnd == 1 else reversed(SCREEN_ARMS)) + ['S0_close']
    for arm in arms:
        name = f'a_r{rnd}_{arm}'
        job(name=name, gpu=0, reads=[previous, 'moe_tune', 'golden_sha', 'build_identity'], timeout=390,
            cmd=command(f'--run-arm {arm.split("_")[0]} --name {name}'))
        previous = name
job(name='select_final', lane='cpu', reads=[previous, 'etx_env'], timeout=60,
    cmd=command('--select'))
previous = 'select_final'
for rnd in range(1, 6):
    for arm in (('S0', 'F', 'etx', 'etxu') if rnd % 2 else ('etxu', 'etx', 'F', 'S0')):
        name = f'{arm}_r{rnd}'
        job(name=name, gpu=0, reads=[previous, 'gate_unf', 'build_identity'], timeout=390,
            cmd=command(f'--run-arm {arm} --name {name}'))
        previous = name
job(name='report', lane='final', timeout=120,
    cmd=command('--report'))


def container_name(directory):
    return 'etx-vlbest3-tune-' + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]


def tuner_command(directory):
    return (base.container_prefix().replace('--rm ', '--rm --pull=never --network=none ')
            .replace('--entrypoint timeout ', '-e HOME="$HOME" -e HF_HUB_OFFLINE=1 '
                     f'--name {container_name(directory)} --entrypoint timeout ')
            + f'{IMAGE} -k 5 2670 python3 -m examples.vllm_llm.vlbest3_tuner '
            f'--model {nvchk.MODEL}/hf --tp-size 1 --batch-size 1 --save-dir "$HOME/moecfg/tuned"')


def cleanup_tuner(directory):
    name = container_name(directory)
    # Match the module argument as an exact field; never pkill/pgrep -f over ssh.
    script = "ps -eo pid,args | awk '$0 !~ /awk/ {for(i=2;i<=NF;i++) if($i==\"examples.vllm_llm.vlbest3_tuner\") print $1}' | xargs -r kill -KILL"
    inspect = subprocess.run(['sudo', 'docker', 'inspect', name], capture_output=True, timeout=10)
    if inspect.returncode == 0:
        subprocess.run(['sudo', 'docker', 'exec', name, 'sh', '-c', script], timeout=10, check=False)
        subprocess.run(['sudo', 'docker', 'rm', '-f', name], timeout=10, check=True)
        print('TUNER cleanup by PID and container removal', flush=True)


def target_table():
    return Path.home() / 'moecfg/tuned' / report.MOE_FILENAME


def validate_table(target):
    table = json.loads(target.read_text())
    if not isinstance(table.get('1'), dict) or not table['1']:
        raise Refusal('NOT-COMPLETE: table missing key "1" or empty winner')
    return table


def tune(directory):
    target = target_table()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    target.with_suffix('.progress.json').unlink(missing_ok=True)
    receipt = directory / 'moe_tuned.json'
    receipt.unlink(missing_ok=True)
    started = time.monotonic()
    proc = subprocess.Popen(['bash', '-c', tuner_command(directory)])
    try:
        while True:
            try:
                rc = proc.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                print(f'TUNER-PROGRESS elapsed={elapsed:.0f}s job-cap=2700s hard-deadline=2700s', flush=True)
                if elapsed >= 2700:
                    cleanup_tuner(directory)
                    proc.kill()
                    proc.wait()
                    raise Refusal('NOT-COMPLETE: tuning exceeded 45 minutes')
        if rc:
            raise Refusal(f'NOT-COMPLETE: tuner exit {rc}; no retry')
        validate_table(target)
        receipt.write_text(json.dumps({'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))
    finally:
        cleanup_tuner(directory)
    return 0


def tune_gate(directory):
    cleanup_tuner(directory)
    target = target_table()
    has_key = False
    try:
        has_key = isinstance(json.loads(target.read_text()).get('1'), dict)
    except (OSError, ValueError, AttributeError):
        pass
    count = 'unknown'
    try:
        count = json.loads(target.with_suffix('.progress.json').read_text())['configs_done']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    print(f'TABLE path={target.resolve()} key_1={has_key} configs_tuned={count}', flush=True)
    try:
        if type(count) is not int or count <= 0:
            raise Refusal('missing/invalid completed config count')
        validate_table(target)
        receipt = json.loads((directory / 'moe_tuned.json').read_text())
        if receipt['sha256'] != hashlib.sha256(target.read_bytes()).hexdigest():
            raise Refusal('table changed after tuning')
    except (OSError, ValueError, KeyError, TypeError, Refusal) as exc:
        print(f'NOT-APPLIED S9 table admission: {exc}', flush=True)
        raise Refusal('NOT-COMPLETE: missing/invalid table or successful tuner receipt') from exc
    print('GO', flush=True)
    return 0


def final_config(won):
    cfg = report.config('M1' if won else 'S3')
    if won:
        cfg['id'] = 'S9'
    return cfg


def select(directory, write=True):
    session = Session(directory)
    brackets = [[report.observation(session.read(f'a_r{r}_S0_{end}'), report.config('S0'))
                 for end in ('open', 'close')] for r in (1, 2)]
    values = {arm: [report.observation(session.read(f'a_r{r}_{arm}'), report.config(arm))
                    for r in (1, 2)] for arm in SCREEN_ARMS}
    if any(abs(a-b) > .050 + 1e-12 for a, b in brackets):
        raise Refusal('NOT-COMPLETE: S0 bracket drift > 0.050 ms')
    deltas = [m-s for m, s in zip(values['M1'], values['S3'])]
    won = all(d < 0 for d in deltas) and median(deltas) <= -.050 + 1e-12
    cfg = final_config(won)
    search = dict(tested=['S0', 'S3', 'S9'], brackets=brackets, deltas=deltas,
                  winners={cfg['id']: dict(config=cfg, gain=median(deltas) if won else 0)},
                  excluded={} if won else {'S9': 'NO-WINNER; NOT-APPLIED; tested-and-not-won'})
    data = dict(design='vlbest3', search=search, final=cfg, combination=None)
    if write:
        for filename in ('screen.json', 'final.json'):
            (directory / filename).write_text(json.dumps(data, indent=2))
        print(f'S9 {"GO" if won else "NO-WINNER; NOT-APPLIED; tested-and-not-won"} deltas={deltas} R={cfg["id"]}; five fresh final rounds required', flush=True)
    return data


def run_arm(arm, name, directory):
    cfg = (json.loads((directory / 'final.json').read_text())['final'] if arm == 'F'
           else report.config(arm))
    cmd = (nvchk.capture('--unfused' if arm == 'etxu' else '') if arm.startswith('etx')
           else arm_command(cfg, name))
    # Tee full subprocess output to the job's log and a local evidence file;
    # validate immediately, before allowing the next GPU job to start.
    path = directory / f'{name}.raw.log'
    rc = subprocess.run(['bash', '-o', 'pipefail', '-c', cmd + ' 2>&1 | tee ' + shlex.quote(str(path))]).returncode
    if rc:
        raise Refusal(f'NOT-COMPLETE: {name} exit {rc}')
    text = path.read_text()
    if arm.startswith('etx'):
        correctness(text, name)
    else:
        report.observation(text, cfg)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('dry-run', 'g0-command', 'tuner-command', 'verify-build', 'tune', 'tune-gate', 'select', 'report'):
        mode.add_argument('--' + flag, action='store_true')
    mode.add_argument('--run-arm')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--name')
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            from etx.tools.gpu_jobs import main as runner
            print(BUDGET, flush=True)
            return runner([__file__, '--dry-run'])
        if args.g0_command:
            print(g0_command()); return 0
        if args.tuner_command:
            print(tuner_command(args.directory)); return 0
        if args.verify_build:
            return verify_build(args.directory)
        if args.tune:
            return tune(args.directory)
        if args.tune_gate:
            return tune_gate(args.directory)
        if args.select:
            select(args.directory); return 0
        if args.report:
            status = {j['name']: j['status'] for j in json.loads((args.directory / 'status.json').read_text())}
            bad = {n: s for n, s in status.items() if n != 'report' and s != 'ok'}
            if bad:
                raise Refusal(f'NOT-COMPLETE: session jobs {bad}; see g0.log, moe_tune.log, moe_gate.log and arm logs')
            report.report(args.directory); return 0
        return run_arm(args.run_arm, args.name, args.directory)
    except (Refusal, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(f'NOT-COMPLETE: {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
