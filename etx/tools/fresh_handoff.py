"""Plan, run and independently check the experimental gfx942 handoff probe.

No GPU activity on import, plan, or analyze. Synthetic pressure is not model evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

MIN_WORDS = 10_000_000      # pass definition: checked words per reader, in every mode
MIN_HANDOFFS = 1_000_000    # count requirement: handoffs per variant (case), in every mode
MODES = ('cached', 'bypass', 'forced_old', 'primed_stale')


def matrix(repeats: int = 3) -> dict:
    if repeats < 3:
        raise ValueError('at least three paired repetitions required')
    cases = []
    for size, topology, load in itertools.product(
            (64, 8192, 65536), ('same', 'cross', 'fanout'), ('idle', 'synthetic_busy')):
        for repeat in range(repeats + 1):
            modes = (('cached', 'bypass') if repeat % 2 == 0 else ('bypass', 'cached')) if repeat < repeats else MODES[2:]
            for mode in modes:
                rounds = MIN_HANDOFFS
                cases.append(dict(id=f'{size}-{topology}-{load}-{mode}-{repeat}', bytes=size,
                                  topology=topology, load=load, mode=mode, repeat=repeat, handoffs=rounds))
    return dict(schema_version=1, evidence='planned, not measured', cases=cases,
                model_followup=[dict(model=m, status='blocked on hardware probe and model integration',
                                     required=['full matrix under actual decode load', '32/32 HF greedy',
                                               'same-operator bypass comparison', 'full decode latency'])
                                for m in ('Qwen3-30B-A3B', 'Mixtral-8x7B')])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def quantile(hist: Counter, q: float) -> int:
    target = math.ceil(sum(hist.values()) * q)
    n = 0
    for value, count in sorted(hist.items()):
        n += count
        if n >= target:
            return value
    raise ValueError('empty timing samples')


def validate_case(directory: Path, case: dict) -> list[dict]:
    records = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
    require(len(records) == 2, 'two completed kernel launches required')
    require([r['epoch'] for r in records] == [1, 2], 'epochs must be 1 then 2')
    require([r['launch_id'] for r in records] == [1, 2], 'distinct launch IDs required')
    require(records[0]['payload_address'] == records[1]['payload_address'], 'relaunch must reuse payload address')
    summaries = []
    for r in records:
        for key in ('schema_version', 'epoch', 'launch_id', 'payload_address', 'bytes', 'handoffs',
                    'readers', 'words_checked', 'wrong_words', 'slot_stride_bytes', 'wall_clock_khz',
                    'timer_min_ticks', 'ncu', 'cu_coverage', 'busy_bytes'):
            require(type(r[key]) is int, f'non-integer metadata: {key}')
        require(r['schema_version'] == 1, 'unsupported schema')
        for k in ('bytes', 'topology', 'load', 'mode', 'handoffs'):
            require(r[k] == case[k], f'case mismatch: {k}')
        size, rounds, mode = r['bytes'], r['handoffs'], r['mode']
        require(size in (64, 8192, 65536) and mode in MODES, 'unsupported payload or mode')
        require(r['topology'] in ('same', 'cross', 'fanout'), 'unknown topology')
        require(r['load'] in ('idle', 'synthetic_busy'), 'unknown load')
        readers = 7 if r['topology'] == 'fanout' else 1
        require(r['readers'] == readers, 'wrong reader count')
        words = size // 4
        negative = mode in MODES[2:]
        require(rounds >= MIN_HANDOFFS and rounds * words >= MIN_WORDS,
                'insufficient handoffs or checked words per reader')
        require(r['words_checked'] == rounds * words * readers, 'incorrect checked-word total')
        require(r['complete'] is True and r['placement_valid'] is True, 'incomplete run or invalid placement')
        require(r['payload_address'] > 0 and r['payload_address'] % 128 == 0, 'unaligned address')
        require(r['slot_stride_bytes'] == math.ceil(size / 128) * 128, 'incorrect cache-line stride')
        require(r['wall_clock_khz'] > 0 and r['timer_min_ticks'] >= 0, 'invalid clock metadata')
        require(r['gpu'].startswith('gfx942') and r['load_kind'] == 'synthetic_only', 'unsupported provenance')
        require(0 < r['ncu'] <= 1024 and r['cu_coverage'] == r['ncu'], 'incomplete CU coverage')
        require(r['load'] == 'idle' or r['busy_bytes'] > 0, 'busy load did no work')
        with (directory / f"placement-{r['epoch']}.csv").open() as f:
            placements = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
        require(len(placements) == r['ncu'], 'missing placement records')
        require({p['block'] for p in placements} == set(range(r['ncu'])), 'duplicate/missing blocks')
        require(len({p['cu_key'] for p in placements}) == r['ncu'], 'duplicate CU assignment')
        require(all(0 <= p['cu_key'] < 2048 and p['invalidations'] == 1 for p in placements), 'CU invalidation coverage')
        require(all(p['role'] in range(-1, readers + 1) for p in placements), 'invalid role')
        active = [p for p in placements if p['role'] >= 0]
        require(sorted(p['role'] for p in active) == list(range(readers + 1)), 'missing/duplicate roles')
        for p in active:
            expected_x = 0 if r['topology'] == 'same' else p['role']
            require(p['cu_key'] >> 8 == expected_x, 'wrong XCD placement')
        hist = {k: Counter() for k in ('first_load_ticks', 'bulk_checked_ticks', 'handoff_ack_ticks')}
        wrong = count = 0
        # File is reader-major; enforce every (reader, handoff) exactly once without
        # retaining millions of rows in memory. Full duration counted once only.
        full_digest = [hashlib.sha256() for _ in range(readers)]
        with (directory / f"samples-{r['epoch']}.csv").open() as f:
            for row in csv.DictReader(f):
                v = {k: int(value) for k, value in row.items()}
                require(v['reader'] == count // rounds + 1 and v['handoff'] == count % rounds + 1,
                        'missing, duplicate or unordered sample')
                require(count < rounds * readers, 'extra samples')
                require(0 <= v['wrong_words'] <= words, 'invalid error count')
                wrong += v['wrong_words']
                for k in hist:
                    require(v[k] > 0, 'invalid timing')
                    if k != 'handoff_ack_ticks' or v['reader'] == 1:
                        hist[k][v[k]] += 1
                full_digest[v['reader'] - 1].update(f"{v['handoff_ack_ticks']}\n".encode())
                count += 1
        require(count == rounds * readers, 'truncated samples')
        require(len({h.digest() for h in full_digest}) == 1, 'inconsistent writer timings across readers')
        require(wrong == r['wrong_words'], 'error total disagrees with samples')
        if mode == 'forced_old':
            require(wrong == rounds * words * readers, 'forced checker fault was not fully detected')
        elif not negative:
            require(wrong == 0, 'stale or incorrect payload')
        expected_valid = wrong > 0 if negative else wrong == 0
        require(r['valid'] is expected_valid, 'valid flag disagrees with observations')
        state = 'inconclusive' if mode == 'primed_stale' and wrong == 0 else 'pass'
        summary = dict(epoch=r['epoch'], state=state, wrong_words=wrong, timer_min_ticks=r['timer_min_ticks'])
        if not negative:
            summary['timings_us'] = {k: {name: quantile(h, q) * 1000 / r['wall_clock_khz']
                                         for name, q in (('p50', .5), ('p95', .95), ('p99', .99))}
                                    for k, h in hist.items()}
            summary['load_target_met'] = summary['timings_us']['first_load_ticks']['p50'] <= .2
        summaries.append(summary)
    return summaries


def analyze(root: Path) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['plan'] == matrix(), 'manifest differs from frozen experiment matrix')
    report = dict(hardware_scope='synthetic microbenchmark only; models unvalidated', cases=[], paired=[])
    successful = {}
    for case in manifest['plan']['cases']:
        entry = dict(id=case['id'])
        directory = root / case['id']
        try:
            status = json.loads((root / f"{case['id']}.status.json").read_text())
            require(status['returncode'] in (0, 1), 'process failure/timeout')
            expected_files = {'results.jsonl', 'samples-1.csv', 'samples-2.csv', 'placement-1.csv', 'placement-2.csv'}
            require(set(status['sha256']) == expected_files, 'missing evidence inventory')
            for name, sha in status['sha256'].items():
                require(digest(directory / name) == sha, f'checksum mismatch: {name}')
            entry['epochs'] = validate_case(directory, case)
            state = 'pass' if all(e['state'] == 'pass' for e in entry['epochs']) else 'inconclusive'
            require(status['returncode'] == (0 if state == 'pass' else 1), 'unexpected exit status')
            entry['state'] = state
            if state == 'pass':
                successful[case['id']] = entry['epochs']
        except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
            entry.update(state='fail', reason=str(e))
        report['cases'].append(entry)
    for case in manifest['plan']['cases']:
        if case['mode'] != 'cached':
            continue
        baseline_id = case['id'].replace('-cached-', '-bypass-')
        if case['id'] in successful and baseline_id in successful:
            for cached, bypass in zip(successful[case['id']], successful[baseline_id]):
                a = cached['timings_us']['handoff_ack_ticks']['p50']
                b = bypass['timings_us']['handoff_ack_ticks']['p50']
                report['paired'].append(dict(cached=case['id'], bypass=baseline_id, epoch=cached['epoch'],
                                             cached_ack_us=a, bypass_ack_us=b, speedup=b/a))
    report['mechanism_gate'] = ('pass' if len(report['cases']) == len(matrix()['cases']) and
                               all(e['state'] == 'pass' for e in report['cases']) else 'not_passed')
    # The gate does not promote this protocol into the model runtime.
    return report


def run(binary: Path, root: Path, budget: float, case_seconds: float) -> None:
    require(budget > 0 and case_seconds > 0, 'positive deadlines required')
    binary = binary.resolve(strict=True)
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    sources = ['bench/sync/sync_bench.hip', 'bench/sync/fresh_handoff.h', 'etx/tools/fresh_handoff.py']
    manifest = dict(plan=matrix(), binary=str(binary), binary_sha256=digest(binary),
                    sources_sha256={s: digest(repo / s) for s in sources},
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                    worktree=subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True),
                    case_seconds=case_seconds, budget_seconds=budget)
    machine = {}
    for name, command in (
        ('uname', ['uname', '-a']),
        ('hipconfig', ['hipconfig', '--full']),
        ('rocm-smi', ['rocm-smi', '--showuniqueid', '--showdriverversion', '--showclocks',
                      '--showpower', '--showuse', '--showpids']),
    ):
        path = root / f'{name}.log'
        try:
            with path.open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=15)
            machine[name] = dict(returncode=completed.returncode, sha256=digest(path))
        except (OSError, subprocess.TimeoutExpired) as e:
            machine[name] = dict(error=str(e))
    manifest['machine_capture'] = machine
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    deadline = time.monotonic() + budget
    for case in manifest['plan']['cases']:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        command = [str(binary), 'fresh', str(case['bytes']), str(case['handoffs']),
                   case['topology'], case['mode'], case['load'], str((root / case['id']).resolve())]
        print('START', case['id'], flush=True)
        with (root / f"{case['id']}.log").open('w') as log:
            log.write(json.dumps(command) + '\n'); log.flush()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            end = time.monotonic() + min(left, case_seconds)
            code = None
            try:
                while code is None:
                    try:
                        code = process.wait(timeout=min(10, max(.01, end-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        print('WAIT', case['id'], flush=True)
                        if time.monotonic() >= end:
                            process.kill(); process.wait(); code = 124
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
        directory = root / case['id']
        inventory = {p.name: digest(p) for p in directory.glob('*') if p.is_file()}
        (root / f"{case['id']}.status.json").write_text(json.dumps(dict(returncode=code, sha256=inventory)) + '\n')
        print('DONE', case['id'], code, flush=True)
        # A timed-out/driver-failed kernel may still own the device; do not queue
        # subsequent cases onto an unhealthy GPU. Retain incomplete evidence.
        if code not in (0, 1):
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('plan')
    execute = commands.add_parser('run')
    execute.add_argument('--binary', type=Path, required=True)
    execute.add_argument('--out', type=Path, required=True)
    execute.add_argument('--budget-seconds', type=float, default=7200)
    execute.add_argument('--case-seconds', type=float, default=420)
    inspect = commands.add_parser('analyze')
    inspect.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print(json.dumps(matrix(), indent=2))
    elif args.command == 'run':
        run(args.binary, args.out, args.budget_seconds, args.case_seconds)
        result = analyze(args.out)
        (args.out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        raise SystemExit(0 if result['mechanism_gate'] == 'pass' else 1)
    else:
        result = analyze(args.directory)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['mechanism_gate'] == 'pass' else 1)


if __name__ == '__main__':
    main()
