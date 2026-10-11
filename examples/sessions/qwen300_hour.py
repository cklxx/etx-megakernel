"""One pinned, offline, 60-minute deciding session; importing is CPU only.

python -m examples.sessions.qwen300_hour --template prepared.json
python -m examples.sessions.qwen300_hour --dry-run
python -m etx.tools.gpu_jobs examples/sessions/qwen300_hour.py --gpus 0 --machine A --logdir OUT

The VM must have prepared.json at ~/etx/qwen300-hour.json, or set
ETX_QWEN300_CONFIG for a direct gpu_jobs invocation. No download/build/debug
occurs during the measurement budget. Fresh checks are never transferred.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics as st
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('qwen300_hour_arms.json')
MAX_SESSION_SECONDS = 3600
if not 0 < float(os.environ.get('ETX_SESS_MAX_SECONDS', 3600)) <= 3600:
    raise ValueError('qwen300-hour cap must be positive and <=3600 seconds')
JOBS = [dict(name='qwen300_hour', gpu=0, machine='A', timeout=3600, quiet_s=180,
             cmd='stdbuf -oL -eL python3 -u -m examples.sessions.qwen300_hour --run')]


def manifest(path=MANIFEST):
    return json.loads(Path(path).read_text())


def template():
    return dict(model='/absolute/prepared/qwen3-30b-a3b',
                builds={name: '/absolute/prepared/'+name for name in ('kept', 'stamps', 'stages', 'C1', 'C5')},
                build_commits={}, combinations=[], cpu_gates='/absolute/prepared/cpu-gates.json',
                hf='/absolute/prepared/qwen3-30b-a3b/hf',
                image='vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7',
                manifest=str(MANIFEST), golden_sha256='bab2f1fbe12f70179711ec40a215c74710845b73169b1cc6a3a6ca50e0a19484')


class Session:
    def __init__(self, config, logdir, seconds=3600):
        self.c = config
        self.out = Path(logdir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        self.end = self.start + min(3590, seconds-10)  # reserve report/outer-runner margin
        self.deadline = min(self.start+900, self.end)
        self.rows = {}
        self.arms = {}
        self.certified = set()
        self.m = manifest(config.get('manifest', MANIFEST))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('ETX_', 'DSL_ABLATE_'))}
        self.env.update(ETX_VL_PROGRESS='1', PYTHONUNBUFFERED='1')
        self.env['HIP_VISIBLE_DEVICES'] = os.environ.get('GPU', '0')

    def row(self, name, value, status, rule):
        self.rows[name] = dict(value=value, status=status, rule=rule)
        print(f'THRESHOLD {name} {status} value={json.dumps(value)} rule={rule}', flush=True)

    def probe(self, stream):
        print('STUCK-STATE probe: ps and rocm-smi (full output in command log)', flush=True)
        for cmd in (['ps', '-eo', 'pid,ppid,etime,stat,args'], ['rocm-smi']):
            try:
                subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=3, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                stream.write(f'PROBE {exc}\n'); stream.flush()

    def step(self, name, cmd, extra=None, timeout=180, expected=120):
        remaining = min(timeout, self.deadline-time.monotonic(), self.end-time.monotonic())
        if remaining <= 0:
            raise TimeoutError(f'{name}: NOT-COMPLETE deadline')
        path = self.out/(name+'.log')
        print(f'COMMAND {name} limit={remaining:.1f}s expected={expected}s log={path}', flush=True)
        with path.open('x') as out:
            out.write(f'# job {name} gpu={os.environ.get("GPU", "0")} machine=A\n')
            out.write('# command '+json.dumps(list(map(str, cmd)))+'\n'); out.flush()
            p = subprocess.Popen(list(map(str, cmd)), stdout=out, stderr=subprocess.STDOUT,
                                 env=dict(self.env, **(extra or {})), start_new_session=True)
            begin = time.monotonic(); size = path.stat().st_size; last = begin; probed = False
            try:
                while p.poll() is None:
                    now = time.monotonic()
                    if now-begin >= remaining:
                        self.probe(out)
                        raise TimeoutError(f'{name}: NOT-COMPLETE substep deadline pid={p.pid}')
                    try:
                        p.wait(timeout=min(5, remaining-(now-begin)))
                    except subprocess.TimeoutExpired:
                        new = path.stat().st_size
                        if new != size: last = time.monotonic(); size = new
                        elapsed = time.monotonic()-begin
                        print(f'PROGRESS {name} elapsed={elapsed:.0f}s pid={p.pid} bytes={size}', flush=True)
                        # During decode each step is printed. A missing step for 10s
                        # is a stuck step, even if the HIP call itself is blocked.
                        tail = path.read_text(errors='replace')[-4096:]
                        if '  step ' in tail and time.monotonic()-last > 10:
                            self.probe(out)
                            raise TimeoutError(f'{name}: NOT-COMPLETE step watchdog (10s)')
                        if elapsed > expected+180 and time.monotonic()-last > 180 and not probed:
                            self.probe(out); probed = True
                if p.returncode:
                    raise RuntimeError(f'{name}: NOT-COMPLETE rc={p.returncode}, full log={path}')
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
        return path.read_text(errors='replace')

    def attempt(self, name, fn):
        try:
            return fn()
        except (OSError, ValueError, RuntimeError, TimeoutError, KeyError) as exc:
            print(f'ARM {name} NOT-COMPLETE: {type(exc).__name__}: {exc}', flush=True)
            self.arms[name] = dict(status='NOT-COMPLETE', reason=str(exc))
            return None

    def finish(self):
        for name, rule in REGISTERED.items():
            self.rows.setdefault(name, dict(value=None, status='NOT-COMPLETE', rule=rule))
        result = dict(thresholds=self.rows, arms=self.arms, elapsed_s=time.monotonic()-self.start,
                      conditions=dict(precision='bf16', batch=1, context=1024, tokens=32, machine='A',
                                      gpu=os.environ.get('GPU', '0')), attribution='No removable-time credit from stamps')
        (self.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        print('REGISTERED THRESHOLDS | measured | pass/fail | rule', flush=True)
        for name, r in self.rows.items():
            print(f'{name} | {json.dumps(r["value"])} | {r["status"]} | {r["rule"]}', flush=True)


REGISTERED = {
    'A1.busy': 'real >=0.100 ms; dead <0.050 ms (frozen task registration)',
    'A1.free': 'real >=0.050 ms (frozen task registration)',
    'A1.floor': '1.00 us/edge; retain below-floor count/min, never clip raw gaps',
    'A2.route': 'available body headroom >=0.142 ms / 2.96 us/layer; estimate, no removal proof',
    'A3.host': 'name stages accounting for unexplained 0.086 ms beyond priced 0.049 ms',
    'A4.reset': 'pursue only >0.030 ms/step, host and device separately',
    'A.stop': 'at <=15 minutes: no exclusive >=0.324 ms outside booked rows => close target route',
    'B.keep': '3 pairs, all faster, median >=0.030 ms, median >2*spread',
    'C.fusion': 'net >=1.042 us/layer; VGPR <=128; LDS <=29752 B',
    'C.body': '3 pairs, all faster, median >=0.030 ms, median >2*spread',
    'D.target': '5 complete rounds, host-wall step <=3.3333 ms, all fresh correctness gates',
    'D.fusion': '5 adjacent pinned pairs; every gain >0; median >2%; median >2*spread',
    'D.system': '5 adjacent pinned pairs; LEAD all deltas <0, median <=-0.050 ms',
}


def dry_run():
    """Exercise CPU logic, shell syntax and runner registration; never spawn GPU commands."""
    for j in JOBS:
        subprocess.run(['bash', '-n', '-c', j['cmd']], check=True)
    assert sum((900, 900, 1200, 600)) == MAX_SESSION_SECONDS
    print(json.dumps(dict(budgets_s=dict(A=900, B=900, C=1200, D=600),
                          manifest=manifest(), thresholds=REGISTERED), indent=2))
    from etx.tools.gpu_jobs import main as runner
    return runner([str(Path(__file__)), '--dry-run'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--template', type=Path)
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--run', action='store_true')
    a = ap.parse_args(argv)
    if a.template: a.template.write_text(json.dumps(template(), indent=2)+'\n'); return 0
    if a.dry_run: return dry_run()
    c = json.loads(Path(os.environ.get('ETX_QWEN300_CONFIG', ROOT/'qwen300-hour.json')).read_text())
    s = Session(c, os.environ['ETX_SESS_LOGDIR'], float(os.environ.get('ETX_SESS_REMAINING_SECONDS', 3600)))
    try:
        raise RuntimeError('Harness skeleton: tranches not installed yet')
    finally:
        s.finish()


if __name__ == '__main__':
    raise SystemExit(main())
