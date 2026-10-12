"""One standalone session: is moe_up's grid too narrow? Four arms, interleaved.

python -m etx.tools.gpu_jobs examples/sessions/moe_rpt.py --dry-run
ETX_MOE_CONFIG=prepared.json tools/session.sh start IP examples/sessions/moe_rpt.py 0 A

The question this job answers: at MI=768 the up grid is TOPK * (MI / RPT)
workgroups, 192 at RPT=32 and 384 at RPT=16, while the machine has 304 CUs and
the build keeps 256 workers. Down already fills it (TOPK * (H / 64) = 256).
Does widening the up grid buy more than the split costs?

The arm set, all built from one tree and one compile configuration:
  kept        RPT 32, CU 256  -- the measured build, the baseline for every pair
  MOE24       RPT 24, CU 256  -- 256 workgroups: the one exact cover, one round
  MOE16       RPT 16, CU 256  -- 384 workgroups
  MOE16CU304  RPT 16, CU 304  -- 384 workgroups, more workers resident

Method: correctness first (fresh import check and 32/32 against the golden, per
arm), then interleaved rounds, baseline and every candidate in the same round,
and the wall delta per round. A candidate keeps only if every round is faster,
the median gain is at least 0.030 ms, and the median is more than twice the
spread. Missing numbers are NOT-COMPLETE, never zero.

The template records the rule, the arm list, the budget and the model path.
Builds, the full weight pack, the golden and the portable snapshots must exist.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

MAX_SESSION_SECONDS = 3600
JOBS = [dict(name='moe_rpt', gpu=0, machine='A', timeout=MAX_SESSION_SECONDS, quiet_s=300,
             cmd='stdbuf -oL -eL python3 -u -m examples.sessions.moe_rpt --run')]
BASE = 'kept'
CANDIDATES = tuple(a for a in (os.environ.get('ETX_MOE_ARMS') or 'MOE24,MOE16,MOE16CU304').split(',') if a and a != BASE)
ROUNDS = 5
KEEP_MEDIAN_MS = 0.030        # the registered KEEP bar for a body pair


def template():
    """What the runner needs: one model, one golden, four prepared builds."""
    return dict(model='/root/llm/qwen3-30b-a3b', golden='examples/llm/results/golden_c1024_qwen3-30b-a3b.txt',
                builds={a: f'/root/etx/build/moe_rpt/{a}' for a in (BASE, *CANDIDATES)},
                budget_s=MAX_SESSION_SECONDS, keep=dict(median_ms_ge=KEEP_MEDIAN_MS, rounds=ROUNDS),
                arms=list((BASE, *CANDIDATES)))


RESULT = re.compile(r'RESULT\b.*?(?P<body>matched=(?P<matched>\d+) gen=(?P<gen>\d+) '
                    r'median_ms=(?P<median>[0-9.]+) wall_ms=(?P<wall>[0-9.]+) tokens_per_s=(?P<tps>[0-9.]+))')


def samples(text, arm):
    """Every RESULT line must be a full 32/32 run; take its wall_ms."""
    rows = [m.groupdict() for m in RESULT.finditer(text)]
    if not rows:
        raise ValueError(f'{arm}: no RESULT line; the run did not report')
    for r in rows:
        if int(r['matched']) != 32 or int(r['gen']) != 32 or float(r['wall']) <= 0:
            raise ValueError(f'{arm}: incomplete run {r["body"]}')
    walls = [float(r['wall']) for r in rows]
    return dict(wall_ms=statistics.median(walls), device_ms=statistics.median(float(r['median']) for r in rows),
                runs=len(rows), spread_ms=max(walls) - min(walls))


def decide(per_round):
    """per_round: list of (base_wall, cand_wall) in round order."""
    deltas = [b - c for b, c in per_round]
    spread = max(max(b for b, _ in per_round) - min(b for b, _ in per_round),
                 max(c for _, c in per_round) - min(c for _, c in per_round))
    median = statistics.median(deltas)
    return dict(rounds=len(deltas), deltas_ms=deltas, median_ms=median, spread_ms=spread,
                every_round_faster=all(d > 0 for d in deltas),
                median_ge_bar=median >= KEEP_MEDIAN_MS,
                median_gt_2spread=median > 2 * spread,
                keep=all(d > 0 for d in deltas) and median >= KEEP_MEDIAN_MS and median > 2 * spread)


def run(config, logdir):
    c = json.loads(Path(config).read_text())
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + MAX_SESSION_SECONDS
    arms = list((BASE, *CANDIDATES))
    def build_dir(arm):
        entry = c['builds'][arm]
        return Path(entry['path'] if isinstance(entry, dict) else entry)
    for arm in arms:
        b = build_dir(arm)
        for f in ('run', 'plan.json', 'vl_plan.txt'):
            if not (b / f).exists():
                raise ValueError(f'prepared build missing {f}: {arm}')
    # One compile configuration: the difference between the arms is the lever.
    plans = {a: (build_dir(a) / 'plan.json').read_bytes() for a in arms}
    if plans[BASE] == plans[CANDIDATES[0]]:
        raise ValueError('candidate plan is identical to the baseline: the lever did not reach the build')
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(('ETX_TRACE', 'ETX_EXEC_COUNT', 'ETX_INBODY_RAW', 'ETX_VL_MOE_RPT', 'ETX_VL_CU'))}
    clean.update(ETX_VL_PROGRESS='64')

    def step(name, cmd, extra=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('NOT-COMPLETE session deadline')
        path = logdir / (name + '.log')
        if path.exists():
            raise ValueError('fresh session logdir required: ' + str(path))
        print(f'COMMAND {name} deadline={remaining:.1f}s log={path}', flush=True)
        with path.open('w') as out:
            out.write('# command ' + json.dumps(list(map(str, cmd))) + '\n')
            out.flush()
            p = subprocess.Popen(list(map(str, cmd)), stdout=out, stderr=subprocess.STDOUT,
                                 env=dict(clean, **(extra or {})), start_new_session=True)
            start = time.monotonic()
            try:
                while p.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('NOT-COMPLETE ' + name)
                    try:
                        p.wait(timeout=min(15, remaining))
                    except subprocess.TimeoutExpired:
                        print(f'PROGRESS {name} elapsed={time.monotonic()-start:.0f}s pid={p.pid}', flush=True)
                        if time.time() - path.stat().st_mtime > 180:
                            for probe in (['ps', '-eo', 'pid,ppid,etime,stat,args'], ['rocm-smi']):
                                try:
                                    subprocess.run(probe, timeout=min(5, max(.01, deadline - time.monotonic())),
                                                   check=False, stdout=out, stderr=subprocess.STDOUT)
                                except (OSError, subprocess.TimeoutExpired):
                                    pass
                if p.returncode:
                    raise RuntimeError(f'NOT-COMPLETE {name} rc={p.returncode}; full log={path}')
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        return path.read_text()

    def host(arm, args, extra=None):
        b = build_dir(arm)
        cmd = [b / 'run', c['model'], b / 'vl_plan.txt', '--golden', c['golden'], '--gen', '32', *args]
        return cmd, extra

    report: dict = dict(status='NOT-COMPLETE', gpu=os.environ.get('GPU', '0'), arms=arms,
                  rounds=ROUNDS, pairs={}, correctness={}, role_body_us={})
    try:
        # Correctness first, in full, per arm: the golden must reproduce 32/32.
        for arm in arms:
            text = step(f'{arm}_gen32', *host(arm, []))
            report['correctness'][arm] = samples(text, arm)
        for rnd in range(ROUNDS):
            walls = {}
            for arm in arms:
                text = step(f'r{rnd}_{arm}', *host(arm, []))
                walls[arm] = samples(text, arm)['wall_ms']
            for cand in CANDIDATES:
                report['pairs'].setdefault(cand, []).append((walls[BASE], walls[cand]))
        for cand in CANDIDATES:
            report['pairs'][cand] = decide(report['pairs'][cand])
        report['keep'] = [c for c in CANDIDATES if report['pairs'][c]['keep']] or None
        report['status'] = 'MEASUREMENT-COMPLETE'
    finally:
        (logdir / 'moe_rpt_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 1 if report['status'] != 'MEASUREMENT-COMPLETE' else 0


def main():
    if float(os.environ.get('ETX_SESS_MAX_SECONDS', MAX_SESSION_SECONDS)) > MAX_SESSION_SECONDS:
        raise ValueError('moe_rpt measurement hard stop cannot be extended')
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run', action='store_true')
    p.add_argument('--template', type=Path)
    a = p.parse_args()
    if a.template:
        a.template.write_text(json.dumps(template(), indent=2) + '\n')
        return
    if a.run:
        raise SystemExit(run(os.environ['ETX_MOE_CONFIG'], os.environ['ETX_SESS_LOGDIR']))
    p.error('use gpu_jobs.py --dry-run or --template FILE')


if __name__ == '__main__':
    main()
