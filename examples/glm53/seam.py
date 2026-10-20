"""Opt-in production-shaped synthetic B-row handoff; NOT full GLM layer cost.

Build: ETX_GLM_BATCH=1 bash examples/glm53/build_seam.sh /tmp/seam-b1
Run (authorized two-GPU host only): python -m examples.glm53.seam run
  --build-root /tmp/seam-build --logdir /tmp/seam-logs
Offline: python -m examples.glm53.seam summary /tmp/seam-logs/*.log

Each step's buffers are poisoned and events host-reset before timing, so this
is a controlled synthetic condition, not the model's warm-cache steady-state.
Rank 0 directly generates its BF16 bit-pattern output; rank 1 does peer reads
plus local observation stores (not production layer arithmetic).

A payload row is 4*4096 BF16 bit patterns (32768 bytes). The control arm
produces/reads only the two endpoint words per row, with identical row events
and returned-token dependency. Payload-minus-control is an incremental synthetic
fill/read/control-work estimate, NOT pure interconnect latency or a full-layer prediction.
Trace runs are separate instrumented runs. Each difference uses one device's
s_memrealtime clock (gfx942: 10 ns/tick); clocks of different GPUs are never
subtracted. Runtime taken->ready includes setup, wait, acquire and barrier;
ready->done includes body, store drain and barrier, not just memcpy.

Ceiling: the local bandwidth-only floor is 32768/4.3e12 = 0.00762 us/row
per copy (estimate, not a fabric floor). Baseline gaps/fabric bandwidth are
unknown until measured. No performance go/no-go threshold is invented here:
qualification requires every element/token of every row and all requested
rounds/iterations, otherwise FAIL/INCOMPLETE. No inference/fusion verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from etx.frontends import hip_link
from etx.ir import Graph, Resource

BATCHES = (1, 2, 4, 8)
ROW_ELEMENTS = 4 * 4096
POISON = 0xDEAD
TILES = 'examples/glm53/seam_tiles.hip'
ROOT = Path(__file__).resolve().parents[2]
LABEL = 'production-shaped synthetic handoff; not full layer cost'
METRICS = ('wall_us', 'rank0_span_us', 'rank1_span_us')
TRACE_METRICS = ('rank0_row_roundtrip_us', 'send_body_drain_us',
                 'consumer_wait_acquire_us', 'consumer_body_drain_us', 'feedback_wait_acquire_us')


def build():
    g = Graph('glm53_seam_rows')
    g.symbols = ['B']
    for name in ('mailbox', 'received'):
        g.tensor(name, ('B', 4, 4096), dtype='bf16', bytes_per_elem=2)
    g.tensor('epoch', (1,), dtype='int32', bytes_per_elem=4)
    for name in ('token', 'returned'):
        g.tensor(name, ('B',), dtype='int32', bytes_per_elem=4)
    g.tensor('control', (1,), dtype='int32', bytes_per_elem=4)
    g.etensor('ready', ('B',), wait_count=1)
    g.etensor('feedback', ('B',), wait_count=1)
    r = Resource(threads=256, vgpr=64)  # model.py's whole-row production resource
    g.call_device('seam_send', ('B',), hip_link(TILES, 'glm53_seam_send'), device=0,
                  resource=r, args=['epoch', 'mailbox', 'control'], reads=['epoch', 'control'],
                  writes=['mailbox'], out_edges={'ready': 'i->i'})
    g.call_device('seam_consume', ('B',), hip_link(TILES, 'glm53_seam_consume'), device=1,
                  resource=r, args=['mailbox', 'received', 'token', 'control'], reads=['mailbox', 'control'],
                  writes=['received', 'token'], in_edges={'ready': 'i->i'}, out_edges={'feedback': 'i->i'})
    g.call_device('seam_feedback', ('B',), hip_link(TILES, 'glm53_seam_feedback'), device=0,
                  resource=r, args=['token', 'returned'], reads=['token'], writes=['returned'],
                  in_edges={'feedback': 'i->i'})
    return g


def bindings(**overrides):
    b = dict(B=1)
    b.update(overrides)
    if b['B'] not in BATCHES:
        raise ValueError('seam batch must be 1, 2, 4 or 8')
    return b


def runtime(bindings, rng=None):
    return {}


def oracle(batch, epoch, control=False):
    bindings(B=batch)
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    values = [(i + 17 * epoch) % 0x7F00 for i in range(batch * ROW_ELEMENTS)]
    received = [POISON if control and i % ROW_ELEMENTS not in (0, ROW_ELEMENTS-1) else x
                for i, x in enumerate(values)]
    tokens = [values[r*ROW_ELEMENTS] + values[(r+1)*ROW_ELEMENTS-1] for r in range(batch)]
    return values, received, tokens


def _integer(record, key, low, high):
    x = record.get(key)
    if type(x) is not int or not low <= x <= high:
        raise ValueError(f'invalid {key}: {x!r}')
    return x


def _positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def parse_log(path):
    """Fail closed on truncated, duplicate, malformed or failed run receipts."""
    config = result = None
    steps = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith('SEAM-'):
            continue  # Full compiler/runtime chatter is retained, never grep-filtered.
        prefix, _, text = line.partition(' ')
        if prefix not in ('SEAM-CONFIG', 'SEAM-STEP', 'SEAM-RESULT'):
            if prefix == 'SEAM-FAIL':
                raise ValueError(f'{path}: correctness/runtime failure')
            continue
        try:
            rec = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}: malformed {prefix}') from exc
        if not isinstance(rec, dict):
            raise ValueError(f'{path}: invalid record')
        if prefix == 'SEAM-CONFIG':
            if config is not None or steps or result is not None:
                raise ValueError('duplicate/out-of-order config')
            if rec.get('schema') != 1 or rec.get('label') != LABEL:
                raise ValueError('wrong schema/measurement label')
            bindings(B=_integer(rec, 'batch', 1, 8))
            _integer(rec, 'iterations', 1, 1000000)
            _integer(rec, 'rounds', 1, 100)
            _integer(rec, 'warmup', 1, 10000)
            if type(rec.get('trace')) is not bool or rec.get('row_bytes') != 32768:
                raise ValueError('invalid trace/row shape')
            config = rec
        elif prefix == 'SEAM-RESULT':
            if config is None or result is not None or rec.get('status') != 'PASS':
                raise ValueError('missing config/duplicate result/failed run')
            result = rec
        else:
            if config is None or result is not None:
                raise ValueError('step outside run')
            rnd = _integer(rec, 'round', 0, config['rounds']-1)
            phase, arm = rec.get('phase'), rec.get('arm')
            if phase not in ('warmup', 'sample') or arm not in ('payload', 'control'):
                raise ValueError('invalid phase/arm')
            count = config['iterations'] if phase == 'sample' else config['warmup']
            step = _integer(rec, 'iteration', 0, count-1)
            key = rnd, arm, phase, step
            if key in steps:
                raise ValueError('duplicate step')
            if rec.get('rows_checked') != config['batch'] or rec.get('elements_checked') != config['batch']*ROW_ELEMENTS or rec.get('correct') is not True:
                raise ValueError('incomplete row correctness')
            if any(not _positive(rec.get(k)) for k in METRICS):
                raise ValueError('invalid wall/device span')
            if config['trace']:
                rows = rec.get('trace_rows')
                if not isinstance(rows, list) or len(rows) != config['batch']:
                    raise ValueError('missing per-row trace')
                for r, row in enumerate(rows):
                    if not isinstance(row, dict) or row.get('row') != r:
                        raise ValueError('invalid trace row')
                    for name in ('send', 'consume', 'feedback'):
                        stamps = row.get(name)
                        if (not isinstance(stamps, list) or len(stamps) != 3 or
                                any(type(t) is not int or t <= 0 for t in stamps) or
                                stamps != sorted(stamps)):
                            raise ValueError('invalid same-device timestamps')
                    # Only send and feedback are on rank 0. Never compare consume to them.
                    if row['feedback'][1] < row['send'][1]:
                        raise ValueError('invalid rank0 roundtrip')
            elif 'trace_rows' in rec:
                raise ValueError('trace attached to uninstrumented run')
            steps[key] = rec
    if config is None or result is None:
        raise ValueError(f'{path}: INCOMPLETE run (config/result missing)')
    expected = config['rounds'] * 2 * (config['iterations'] + config['warmup'])
    if len(steps) != expected or result.get('steps_checked') != expected:
        raise ValueError(f'{path}: INCOMPLETE run ({len(steps)}/{expected} steps)')
    return config, steps


def _trace_metrics(row):
    s, c, f = (row[k] for k in ('send', 'consume', 'feedback'))
    return dict(zip(TRACE_METRICS, ((f[1]-s[1])*0.01, (s[2]-s[1])*0.01,
                                  (c[1]-c[0])*0.01, (c[2]-c[1])*0.01, (f[1]-f[0])*0.01)))


def summarize(paths):
    runs, seen = [], set()
    for path in paths:
        config, steps = parse_log(path)
        manifest_verified = _check_manifest(path, config)
        identity = config['batch'], config['trace']
        if identity in seen:
            raise ValueError('duplicate batch/trace run; summarize independent sessions separately')
        seen.add(identity)
        metrics = METRICS + (TRACE_METRICS if config['trace'] else ())
        per_round = []
        for rnd in range(config['rounds']):
            arms = {}
            for arm in ('payload', 'control'):
                samples = [steps[rnd, arm, 'sample', i] for i in range(config['iterations'])]
                arms[arm] = {k: statistics.median(s[k] for s in samples) for k in METRICS}
                if config['trace']:
                    # Median across all per-row observations; never aggregate absolute device clocks.
                    rows = [_trace_metrics(row) for s in samples for row in s['trace_rows']]
                    arms[arm].update({k: statistics.median(r[k] for r in rows) for k in TRACE_METRICS})
            per_round.append(dict(round=rnd, **arms,
                                  payload_minus_control={k: arms['payload'][k]-arms['control'][k] for k in metrics}))
        aggregate = {arm: {k: dict(median=statistics.median(r[arm][k] for r in per_round),
                                  spread=max(r[arm][k] for r in per_round)-min(r[arm][k] for r in per_round))
                           for k in metrics} for arm in ('payload', 'control', 'payload_minus_control')}
        runs.append(dict(config=config, source=str(path), runner_manifest_verified=manifest_verified,
                         rounds=per_round, round_medians_us=aggregate))
    if not runs:
        raise ValueError('no input logs')
    return dict(schema=1, status='COMPLETE', label=LABEL, runs=runs,
                cross_gpu_clock_subtraction=False,
                attribution='payload minus two-endpoint-word control: fill/read/control work, not pure wait or fabric latency',
                setup_condition='host poisons mailbox/output and resets events before each timed step; not model steady-state',
                producer='direct cooperative BF16 bit-pattern fill; consumer does peer reads plus observation stores',
                trace_tick_ns=10,
                host_wall='two launches through both stream retirements; excludes setup/reset/H2D/D2H/checks',
                device_spans='separate HIP event elapsed times on each rank; no cross-GPU clock assumption',
                trace='instrumented only: taken->ready includes setup/wait/acquire/barrier; ready->done includes body/drain/barrier')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts(build_dir):
    return {name: dict(path=str((build_dir/name).resolve()), sha256=_sha256(build_dir/name))
            for name in ('run', 'plan.json', 'plan_data.h')}


def _check_contract(config, requested):
    if any(config.get(k) != v or type(config.get(k)) is not type(v) for k, v in requested.items()):
        raise ValueError(f'run receipt does not match requested contract: requested={requested}, actual={config}')


def _check_manifest(path, config):
    """Honor an adjacent runner manifest if present, including requested workload."""
    manifest_path = Path(path).parent/'manifest.json'
    if not manifest_path.exists():
        return False  # Standalone raw native logs carry no external request contract.
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get('status') != 'COMPLETE':
        raise ValueError('runner manifest is not COMPLETE')
    entries = [entry for entry in manifest.get('runs', [])
               if isinstance(entry, dict) and Path(entry.get('log', '')).name == Path(path).name]
    if len(entries) != 1 or entries[0].get('status') != 'PASS':
        raise ValueError('missing/ambiguous/failed manifest entry')
    entry = entries[0]
    request = entry.get('requested')
    if not isinstance(request, dict) or set(request) != {'batch', 'trace', 'iterations', 'rounds', 'warmup'}:
        raise ValueError('missing manifest workload contract')
    _check_contract(config, request)
    if entry.get('log_sha256') != _sha256(path):
        raise ValueError('raw log hash differs from manifest')
    return True


def run(args):
    """Explicit GPU command. Never called by imports, summary or local compilation."""
    args.logdir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.logdir/'manifest.json'
    manifest = dict(schema=1, label=LABEL, model_certification=False, status='RUNNING', runs=[],
                    source_snapshot_sha256={name: _sha256(ROOT/name) for name in (
                        'examples/glm53/seam.py', 'examples/glm53/seam_host.hip',
                        'examples/glm53/seam_tiles.hip', 'examples/glm53/build_seam.sh')},
                    provenance_limit='artifact identities, not a model or reproducible-build certificate')

    def save():
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')

    save()
    logs = []
    for batch in args.batches:
        for trace in (False, True):
            log = args.logdir / f'b{batch}-trace{int(trace)}.log'
            build_dir = args.build_root/f'b{batch}'
            command = [str((build_dir/'run').resolve()), str(args.iterations), str(args.rounds),
                       str(args.warmup), str(int(trace))]
            requested = dict(batch=batch, trace=trace, iterations=args.iterations,
                             rounds=args.rounds, warmup=args.warmup)
            entry = dict(requested=requested, command=command, log=str(log.resolve()), status='RUNNING')
            manifest['runs'].append(entry)
            save()
            env = dict(os.environ, HIP_VISIBLE_DEVICES='0,1', ETX_SYNC='poll', ETX_DEV_RESET='0')
            print(f'SEAM-RUN batch={batch} trace={trace} log={log}', flush=True)
            try:
                with log.open('w') as output:
                    entry['artifacts'] = _artifacts(build_dir)
                    save()
                    # Native watchdog diagnoses stalls without invoking HIP. This separate
                    # deadline also bounds setup/D2H/driver hangs; full logs are unfiltered.
                    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                          stderr=subprocess.STDOUT) as proc:
                        start = time.monotonic()
                        while True:
                            remaining = args.timeout - (time.monotonic()-start)
                            if remaining <= 0:
                                proc.kill()
                                proc.wait()
                                raise subprocess.TimeoutExpired(command, args.timeout)
                            try:
                                code = proc.wait(timeout=min(5, remaining))
                                entry['returncode'] = code
                                if code:
                                    raise subprocess.CalledProcessError(code, command)
                                break
                            except subprocess.TimeoutExpired:
                                print(f'SEAM-RUN progress batch={batch} trace={trace} '
                                      f'elapsed_s={time.monotonic()-start:.1f} full_log={log}', flush=True)
                config, _ = parse_log(log)
                _check_contract(config, requested)
                if _artifacts(build_dir) != entry['artifacts']:
                    raise ValueError('build artifacts changed while measuring')
                entry['status'] = 'PASS'
                logs.append(log)
                print(f'SEAM-RUN OK batch={batch} trace={trace}', flush=True)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                entry.update(status='FAILED', error=str(exc))
                manifest['status'] = 'FAILED'
                raise
            finally:
                if log.exists():
                    entry['log_sha256'] = _sha256(log)
                save()
    manifest['status'] = 'COMPLETE'
    save()
    print(json.dumps(summarize(logs), sort_keys=True), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('summary', help='offline, fail-closed parser; no GPU imports/calls')
    s.add_argument('logs', nargs='+', type=Path)
    r = sub.add_parser('run', help='explicitly execute prebuilt probes on an authorized two-GPU host')
    r.add_argument('--build-root', type=Path, required=True)
    r.add_argument('--logdir', type=Path, required=True)
    r.add_argument('--batches', nargs='+', type=int, choices=BATCHES, default=list(BATCHES))
    r.add_argument('--iterations', type=int, default=1000)
    r.add_argument('--rounds', type=int, default=5)
    r.add_argument('--warmup', type=int, default=20)
    r.add_argument('--timeout', type=float, default=240, help='hard deadline per batch/trace process, seconds')
    args = p.parse_args(argv)
    try:
        if args.command == 'summary':
            print(json.dumps(summarize(args.logs), sort_keys=True))
        else:
            if not (1 <= args.iterations <= 1000000 and 1 <= args.rounds <= 100 and
                    1 <= args.warmup <= 10000 and _positive(args.timeout)):
                raise ValueError('invalid iteration/round/warmup/deadline bounds')
            if len(args.batches) != len(set(args.batches)):
                raise ValueError('duplicate batches')
            run(args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'SEAM-ERROR {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
