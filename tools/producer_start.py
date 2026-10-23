"""Bound push, interpreter attachment and start by one supplied allocation deadline.

Only run after GPU authorization. This command does not provision a machine.
The venv must already exist outside any checkout. No dependency download occurs.
"""
import argparse
import math
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def launch(ip, venv, config, allocation_deadline):
    now = time.time()
    if not math.isfinite(allocation_deadline) or not 0 < allocation_deadline-now <= 3600:
        raise ValueError('allocation deadline must be in the next 3600 seconds')
    if not Path(venv).is_absolute() or not Path(config).is_absolute():
        raise ValueError('remote venv and config paths must be absolute')
    staging_deadline = min(now+300, allocation_deadline-300)
    def run(argv, deadline, env=None):
        seconds = deadline-time.time()
        if seconds <= 0:
            raise ValueError('allocation/staging deadline expired; stop and fetch')
        print('STAGE command '+shlex.join(map(str, argv))+f' timeout_s={seconds:.1f}', flush=True)
        # A process group deadline also stops ssh/rsync children. Keep all output.
        from etx.session import proc
        rc = proc.run(list(map(str, argv)), sink=sys.stdout, deadline_s=seconds, poll_s=1, env=env, cwd=ROOT)
        if rc:
            raise RuntimeError(f'staging/start failed exit={rc}; stop and fetch')
    # The session transport through its external translator, under this interpreter.
    transport = [sys.executable, '-m', 'etx.tools.session', '--root', ROOT]
    run([*transport, 'push', ip], staging_deadline)
    user = os.environ.get('ETX_SESS_USER', 'hotaisle')
    remote = 'cd ~/etx && python3 -B -m examples.vllm_llm.producer_stage '+shlex.quote(venv)
    run(['ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new',
         f'{user}@{ip}', remote], staging_deadline)
    # Leave margin for SSH startup. The existing persistent runner lease keeps
    # the first deadline on retry; this cap cannot increase it.
    cap = min(3300, int(allocation_deadline-time.time()-15))
    if cap <= 300:
        raise ValueError('no compute budget after evidence reserve; stop and fetch')
    run([*transport, 'start', ip, 'examples/sessions/qwen_producer.py', '0', 'A'],
        min(time.time()+15, allocation_deadline),
        dict(os.environ, ETX_QWEN_G6_CONFIG=config, ETX_SESS_MAX_SECONDS=str(cap)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ip')
    ap.add_argument('--venv', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--allocation-deadline', required=True, type=float, help='original allocation end, Unix seconds')
    args = ap.parse_args()
    launch(args.ip, args.venv, args.config, args.allocation_deadline)


if __name__ == '__main__':
    main()
