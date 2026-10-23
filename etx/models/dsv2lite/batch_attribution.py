"""Bound B1/B2/B4 follow-up. Raw counters and instrumented spans never supply a timing verdict."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

from etx.session import batch
from etx.models.declarations import declaration as _declaration
from . import paths
BATCH_ARMS = _declaration(paths.ROOT, "dsv2lite.arms").BATCH_ARMS
from .batch_goldens import sha
from .batch_session import BatchSession
from etx.models.dsv2lite.session import write_json
from .reference_prepare import verify

BATCHES = (1, 2, 4)
COUNTER = "FETCH_SIZE"  # rocprof derived metric, KiB read by each dispatch; retain definition/version on target.


def counter_rows(path: Path) -> list[dict]:
    rows = []
    for file in sorted(path.rglob('*counter_collection.csv')):
        with file.open() as f:
            rows.extend(dict(r, source=str(file)) for r in csv.DictReader(f))
    if not rows:
        raise ValueError('no counter CSV rows; counter support is a device-only gate')
    return rows


def traffic(rows: list[dict], *, kernel=None, expected=None, allow_zero=False) -> dict:
    """Actual counter values, no shape-byte substitution or profiler-time correction."""
    selected = [r for r in rows if r.get('Counter_Name') == COUNTER and
                (kernel is None or kernel in r.get('Kernel_Name', ''))]
    keys = [(r.get('Agent_Id'), r.get('Process_Id'), r['Dispatch_Id']) for r in selected]
    if not selected or len(set(keys)) != len(keys):
        raise ValueError('missing or duplicate per-dispatch FETCH_SIZE values')
    if expected is not None and len(selected) != expected:
        raise ValueError(f'{len(selected)} dispatches, expected {expected}')
    vals = [float(r['Counter_Value']) for r in selected]
    if any(not math.isfinite(x) or x < 0 for x in vals) or (sum(vals) == 0 and not allow_zero):
        raise ValueError('nonpositive or nonfinite traffic')
    spans = [(int(r['Start_Timestamp']), int(r['End_Timestamp'])) for r in selected]
    if any(hi <= lo for lo, hi in spans):
        raise ValueError('invalid dispatch span')
    return dict(counter=COUNTER, unit='rocprof FETCH_SIZE KB (retain target metric definition)', dispatches=len(vals), total_fetch_kb=sum(vals),
                median_fetch_kb=statistics.median(vals), device_span_us=(max(hi for lo, hi in spans)-min(lo for lo, hi in spans))/1000,
                dispatch_duration_sum_us=sum(hi-lo for lo, hi in spans)/1000,
                scope='selected GPU kernel dispatches; excludes host DMA; not useful-payload bytes')


def dispatch_key(row: dict) -> tuple[str, str, str]:
    fields = ('Process_Id', 'Agent_Id', 'Dispatch_Id')
    if any(row.get(k) is None or not str(row.get(k, '')).strip() for k in fields):
        raise ValueError('dispatch coverage requires Process_Id, Agent_Id and Dispatch_Id')
    return tuple(str(row[k]).strip() for k in fields)


def dispatch_span(row: dict) -> tuple[int, int]:
    try:
        lo, hi = int(row['Start_Timestamp']), int(row['End_Timestamp'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid profiler timestamp fields') from exc
    if lo < 0 or hi <= lo:
        raise ValueError('invalid profiler span')
    return lo, hi


def ranged_traffic(path: Path, b: int) -> dict:
    rows = [r for r in counter_rows(path) if r.get('Counter_Name') == COUNTER]
    markers, kernels = [], []
    for file in sorted(path.rglob('*marker_api_trace.csv')):
        with file.open() as f:
            markers.extend(csv.DictReader(f))
    for file in sorted(path.rglob('*kernel_trace.csv')):
        with file.open() as f:
            kernels.extend(csv.DictReader(f))
    if not kernels:
        raise ValueError('dispatch coverage requires a retained kernel-dispatch trace')
    # A GPU worker may be a child of the process emitting ROCTx. Match trace and counter
    # identities across the complete capture, rather than assuming the marker PID owns GPU work.
    def inside(records, lo, hi, label):
        out = {}
        for row in records:
            key, (start, end) = dispatch_key(row), dispatch_span(row)
            if start < hi and end > lo:
                if start < lo or end > hi:
                    raise ValueError(f'dispatch crosses {label}; traffic attribution refused')
                if key in out:
                    raise ValueError(f'duplicate dispatch coverage record in {label}: {key}')
                out[key] = row
        return out

    result = {}
    ranges = []
    for phase in ['warmup', 'warmup_short', *(p for k in range(batch.SAMPLES) for p in (f'sample{k}', f'sample{k}_short'))]:
        label = f'ds_b4:b{b}:{phase}'
        match = [r for r in markers if r.get('Function', r.get('Name')) == label]
        if len(match) != 1:
            raise ValueError(f'need exactly one profiler marker {label}')
        lo, hi = dispatch_span(match[0])
        if any(lo < end and hi > start for start, end in ranges):
            raise ValueError('overlapping profiler call ranges; traffic attribution refused')
        ranges.append((lo, hi))
        traced, measured = inside(kernels, lo, hi, label), inside(rows, lo, hi, label)
        missing, extra = traced.keys() - measured.keys(), measured.keys() - traced.keys()
        if not traced or missing or extra:
            raise ValueError(f'dispatch coverage mismatch in {label}: traced={len(traced)} '
                             f'missing={sorted(missing)} extra={sorted(extra)}')
        if any(dispatch_span(traced[k]) != dispatch_span(measured[k]) for k in traced):
            raise ValueError(f'dispatch coverage timestamp mismatch in {label}')
        result[phase] = dict(traffic(list(measured.values()), allow_zero=True),
                             dispatch_coverage=dict(status='COMPLETE', traced=len(traced), measured=len(measured),
                                                    identity_fields=['Process_Id', 'Agent_Id', 'Dispatch_Id']))
    # This subtraction is an estimate from measured call traffic. Prefill is not independently isolated.
    net = [(result[f'sample{k}']['total_fetch_kb'] - result[f'sample{k}_short']['total_fetch_kb']) / 31
           for k in range(batch.SAMPLES)]
    if any(x <= 0 for x in net):
        raise ValueError('nonpositive full-minus-short traffic')
    return dict(calls=result, decode_fetch_kb_estimate=net,
                limitation='full-minus-short counter estimate; prefill/cache cancellation is not independently proved')


def graph_eligibility(plan: dict) -> dict:
    unsafe = {'fleet_expert_down_cached', 'fleet_d4_routed_down', 'fleet_d4_shared_down', 'fleet_expert_down_k'}
    found = {name: value.get('symbol') for name,value in plan.get('types', {}).items()
             if value.get('symbol') in unsafe}
    reason = ('gate_up/down routing is handed off through LDS; grid-by-grid launches end that LDS lifetime'
              if found else 'no complete replay-safety certificate for this plan; unknown tasks are refused')
    return dict(status='UNAVAILABLE', reason=reason, unsafe_types=found,
                evidence=['examples/dsv2lite/fleet_shim.hip: FLAG_ROUTING_CACHED',
                          'etx/codegen/kernel.py: _emit_unfused allocates per-launch LDS',
                          'docs/core/ETX_Technical_Design.md section 15.7: retained wrong-layer unfused result'])


class AttributionSession(BatchSession):
    batches = BATCHES

    def __init__(self):
        super().__init__()
        ready = json.loads((self.logs / 'references_ready.json').read_text())
        if ready['status'] != 'READY' or len(ready['requests']) != 4:
            raise ValueError('four producer-bound reference tuples required')
        self.data = self.logs / 'references'
        for r in ready['requests']:
            d = self.data if r['request'] == 0 else self.data / f"r{r['request']}"
            verify(d, r['receipt']['recipe'])
        self.capture_batch = None

    def run(self, name, command, **kw):
        """Reuse the pinned Docker command and supervision; add profiling only to capture jobs."""
        if self.capture_batch is not None and 'examples/llm/vllm_bench.py' in command:
            command = list(map(str, command))
            at = command.index('--batches')
            if command[at+1] != str(self.capture_batch):
                raise ValueError('profiled vLLM call must invoke only its selected batch')
            command += ['--profile-ranges']
            at = command.index('python3', command.index('--entrypoint'))
            command[at:at] = ['rocprofv3', '--kernel-trace', '--marker-trace', '--pmc', COUNTER,
                              '--output-format', 'csv', '-d', str(self.logs / name), '--']
        return super().run(name, command, **kw)

    def identity(self, b):
        arm = BATCH_ARMS[b]
        return dict(arm=arm, batch=b, files={f: sha(self.out(arm)/f) for f in
                    ('run', 'plan.json', 'megakernel_d0.hip', 'resources.json')},
                    references_sha256=sha(self.logs / 'references_ready.json'),
                    source_sha256=sha(self.logs / 'source.json'))

    def etx(self, b, name, arm=None):
        before = self.identity(b)
        result = super().etx(b, name, arm)
        if before != self.identity(b):
            return dict(error='binary/plan/input identity changed during invocation')
        return dict(result, identity=before)

    def gate(self, arm):
        super().gate(arm)
        path = self.logs / f'bgate_{arm}.json'
        value = json.loads(path.read_text())
        value['identity'] = self.identity(value['batch'])
        write_json(path, value)

    def graph(self, b, name):
        before = self.identity(b)
        eligibility = graph_eligibility(json.loads((self.out(BATCH_ARMS[b])/'plan.json').read_text()))
        if eligibility['status'] != 'ELIGIBLE':
            return dict(error=eligibility['reason'], status='UNAVAILABLE', eligibility=eligibility, identity=before)
        try:
            result = batch.etx_result(self.run(name, [self.out(BATCH_ARMS[b])/'run', *self.etx_args(b), '--unfused']), b)
            if before != self.identity(b):
                raise ValueError('same-kernel graph identity changed')
            return dict(result, identity=before, mode='same_tiles_one_launch_per_grid_HIP_graph', clock='host_wall')
        except (OSError, ValueError, RuntimeError) as exc:
            return dict(error=str(exc), identity=before)

    def graph_gate(self, arm):
        b = next(b for b in BATCHES if BATCH_ARMS[b] == arm)
        path = self.fresh(self.logs / f'graph_gate_{arm}.json')
        result = self.graph(b, f'graph_gate_{arm}')
        write_json(path, dict(passed='error' not in result, result=result, identity=self.identity(b)))
        print(f"GRAPH-BASELINE arm={arm} {result.get('status', 'PASS' if 'error' not in result else 'INCOMPLETE')}", flush=True)

    def require_gates(self):
        for b in BATCHES:
            gate = json.loads((self.logs / f'bgate_{BATCH_ARMS[b]}.json').read_text())
            if not gate['passed'] or gate.get('identity') != self.identity(b):
                raise ValueError(f'B{b} correctness/identity gate failed; no timing or captures')


    def vsel(self):
        self.require_gates()
        return super().vsel()

    def rounds(self):
        self.require_gates()
        out = self.fresh(self.logs/'brounds.json')
        selection = json.loads((self.logs/'vsel.json').read_text())['selection']
        if any(not selection[str(b)]['config'] for b in BATCHES):
            raise ValueError('no legal corrected vLLM baseline; timing refused')
        configs = list(dict.fromkeys(selection[str(b)]['config'] for b in BATCHES))
        assigned = {c: tuple(b for b in BATCHES if selection[str(b)]['config'] == c) for c in configs}
        available_graphs = {b for b in BATCHES if json.loads((self.logs/f'graph_gate_{BATCH_ARMS[b]}.json').read_text())['passed']}
        order = [(kind,b) for b in BATCHES for kind in (('etx','graph') if b in available_graphs else ('etx',))] + [('vllm',c) for c in configs]
        rounds = []
        failed = False
        for k in range(1, batch.ROUNDS+1):
            r = dict(etx={}, graph={}, vllm={})
            for kind, x in (order if k % 2 else order[::-1]):
                if kind == 'vllm':
                    r[kind][x] = self.vllm(x, f'b{k}_vllm_{x}', batches=assigned[x])
                else:
                    r[kind][str(x)] = getattr(self, kind)(x, f'b{k}_{kind}_{BATCH_ARMS[x]}')
                value = r[kind][x if kind == 'vllm' else str(x)]
                if 'error' in value or (kind == 'vllm' and
                        (set(value) != set(assigned[x]) or any(not value.get(b, {}).get('legal') for b in assigned[x]))):
                    failed = True
                    break
            rounds.append(r)
            write_json(out.with_suffix('.partial.json'), dict(rounds=rounds))
            if failed:
                break
        graph_verdicts = {}
        for b in BATCHES:
            if b not in available_graphs:
                graph_verdicts[str(b)] = dict(status='UNAVAILABLE',
                    gate=json.loads((self.logs/f'graph_gate_{BATCH_ARMS[b]}.json').read_text()))
                continue
            pairs = [(r['etx'].get(str(b), {'error':'not run'}), r['graph'].get(str(b), {'error':'not run'})) for r in rounds]
            graph_verdicts[str(b)] = (dict(status='COMPLETE', clock='host_wall',
                throughput=batch.verdict([batch.gain(g['step_ms'],e['step_ms']) for e,g in pairs]))
                if len(pairs) == batch.ROUNDS and all('error' not in x for pair in pairs for x in pair) else dict(status='INCOMPLETE'))
        write_json(out, dict(status='INCOMPLETE' if failed else 'COMPLETE', order=order, skipped=[], rounds=rounds, same_kernel_graph=graph_verdicts))
        if failed:
            raise RuntimeError('a timed correctness/identity gate failed; remaining timings and captures stopped')
        print('BROUNDS five corrected-vLLM pairs at B1/B2/B4; same-kernel graph eligibility recorded separately', flush=True)

    def check_rounds(self):
        value = json.loads((self.logs/'brounds.json').read_text())
        raw = value['rounds']
        selection = {int(b): s for b,s in json.loads((self.logs/'vsel.json').read_text())['selection'].items()}
        rounds = [dict(etx={int(b):e for b,e in r['etx'].items()},
                       vllm={c:(v if 'error' in v else {int(b):x for b,x in v.items()}) for c,v in r['vllm'].items()}) for r in raw]
        results = batch.study(rounds, selection, BATCHES)
        if (value.get('status') != 'COMPLETE' or len(raw) != batch.ROUNDS or
                any(x.get('status') != 'COMPLETE' for x in results.values())):
            raise ValueError('all five corrected-vLLM pairs must complete before attribution')

    def captures(self):
        out = self.fresh(self.logs / 'attribution.json')
        result = dict(status='INCOMPLETE', batches={}, instrumented_timings_for_verdict=False)
        try:
            self.require_gates()
            self.check_rounds()
            self.run('rocprof_version', ['rocprofv3', '--version'], timeout=60)
            self.run('rocprof_metrics', ['rocprofv3', '--list-avail'], timeout=60)
            selection = json.loads((self.logs / 'vsel.json').read_text())['selection']
            # A failed gate stops all captures. Timed calls remain independently gated by the existing parser.
            for b in BATCHES:
                gate = json.loads((self.logs / f'bgate_{BATCH_ARMS[b]}.json').read_text())
                if not gate['passed'] or not selection[str(b)]['config']:
                    raise ValueError(f'B{b} gate or legal baseline missing')
            for b in BATCHES:
                arm, config = BATCH_ARMS[b], selection[str(b)]['config']
                rec = result['batches'][str(b)] = dict(identity=self.identity(b), config=config)
                name = f'body_{arm}'
                raw = self.logs / f'{name}.trace.bin'
                batch.etx_result(self.run(name, [self.out(arm)/'run', *self.etx_args(b)],
                    env=dict(ETX_TRACE='1', ETX_TRACE_RAW=str(raw))), b)
                from etx.tools.timeline import load_plan, read_raw, build_trace, latency_report
                rec['body_wait'] = latency_report(build_trace(load_plan(self.out(arm)/'plan.json'), read_raw(raw)))
                name = f'traffic_{arm}'
                cmd = ['rocprofv3', '--pmc', COUNTER, '--output-format', 'csv', '-d', str(self.logs/name), '--',
                       self.out(arm)/'run', *self.etx_args(b)]
                batch.etx_result(self.run(name, cmd, timeout=600), b)
                rec['etx_traffic'] = traffic(counter_rows(self.logs/name), kernel='etx_megakernel', expected=64)
                name = f'traffic_vllm_b{b}_{config}'
                self.capture_batch = b
                try:
                    parsed = self.vllm(config, name, batches=(b,))
                finally:
                    self.capture_batch = None
                if b not in parsed or not parsed[b]['legal']:
                    raise ValueError(f'profiled vLLM gate failed: {parsed}')
                rec['vllm_traffic'] = ranged_traffic(self.logs/name, b)
                if rec['identity'] != self.identity(b):
                    raise ValueError('capture identity changed during profiling')
                write_json(out.with_suffix('.partial.json'), result)
            result['status'] = 'COMPLETE'
        except (OSError, ValueError, RuntimeError, SystemExit) as exc:
            result['error'] = str(exc)
            raise
        finally:
            write_json(out, result)

    def routing(self):
        self.require_gates()
        self.check_rounds()
        out = self.fresh(self.logs/'routing.json')
        result = dict(status='INCOMPLETE', bound_prefix=False, executed_prefix={}, per_batch={},
                      native_etx_routing_measured=False,
                      limitation='HF routing is measured separately at each B; equality to ETX picks is unmeasured')
        try:
            from .batch_session import MODEL
            for b in BATCHES:
                path = self.logs/f'routing_b{b}.json'
                self.run(f'routing_b{b}', [sys.executable, '-u', '-m', 'examples.dsv2lite.routing_overlap',
                    '--model', str(MODEL), '--fleet', str(self.fleet), '--batch', str(b), '--steps','32',
                    '--context','1024', '--reference-data', str(self.data), '--json', str(path)], timeout=300)
                value = json.loads(path.read_text())
                if not value.get('bound_prefix') or value['batch'] != b:
                    raise ValueError(f'B{b} bound routing missing')
                result['executed_prefix'][str(b)] = value['executed_prefix'][str(b)]
                result['per_batch'][str(b)] = dict(path=path.name, sha256=sha(path),
                    identity=self.identity(b), reference_sha256=value['reference_sha256'])
            result.update(status='COMPLETE', bound_prefix=True)
        except (OSError, ValueError, RuntimeError) as exc:
            result['error'] = str(exc)
            raise
        finally:
            write_json(out, result)

    def report(self):
        super().report()
        path = self.logs/'batch_report.json'
        value = json.loads(path.read_text())
        value['projection_scope'] = 'conditional estimate from per-B HF expert union; native ETX picks are unmeasured'
        write_json(path, value)

    def final(self):
        out = dict(status='INCOMPLETE', performance=None, attribution=None, routing=None,
                   same_kernel_hip_graph_baseline=dict(status='INCOMPLETE'), attribution_status='INCOMPLETE',
                   performance_claim_status='INCOMPLETE: requires both legal baselines', corrected_vllm_status='INCOMPLETE')
        for key, file in [('performance','batch_report.json'), ('attribution','attribution.json'), ('routing','routing.json')]:
            try:
                out[key] = json.loads((self.logs/file).read_text())
            except (OSError, ValueError) as exc:
                out[key] = dict(error=str(exc))
        try:
            out['same_kernel_hip_graph_baseline'] = json.loads((self.logs/'brounds.json').read_text())['same_kernel_graph']
        except (OSError, ValueError, KeyError):
            pass
        if set(out['performance'].get('batches', {})) == {'1','2','4'} and all(x.get('status') == 'COMPLETE' for x in out['performance']['batches'].values()):
            out['corrected_vllm_status'] = 'COMPLETE'
        if out['attribution'].get('status') == 'COMPLETE' and out['routing'].get('bound_prefix'):
            out['attribution_status'] = 'COMPLETE'
        if (set(out['same_kernel_hip_graph_baseline']) == {'1','2','4'} and
                all(x.get('status') == 'COMPLETE' for x in out['same_kernel_hip_graph_baseline'].values()) and
                out['attribution'].get('status') == 'COMPLETE' and out['routing'].get('bound_prefix') and
                all(x.get('status') == 'COMPLETE' for x in out['performance'].get('batches', {}).values()) and
                set(out['performance'].get('batches', {})) == {'1','2','4'}):
            out['status'] = 'COMPLETE'
            out['performance_claim_status'] = 'COMPLETE'
        write_json(self.logs/'b4_result.json', out)
        print('B4-RESULT', out['status'], flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation', choices=('graph-eligibility','gate','graph-gate','vsel','rounds','report','check-rounds','captures','routing','final'))
    ap.add_argument('arm', nargs='?')
    ap.add_argument('--output', type=Path)
    a = ap.parse_args(argv)
    if a.operation == 'graph-eligibility':
        if a.arm is None or a.output is None:
            ap.error('graph-eligibility requires plan path and --output')
        plan = Path(a.arm)
        result = graph_eligibility(json.loads(plan.read_text()))
        write_json(a.output, dict(passed=False, result=result, plan_sha256=sha(plan), executable_sha256=sha(plan.parent/'run') if (plan.parent/'run').is_file() else None))
        print('GRAPH-BASELINE', result['status'], result['reason'], flush=True)
        return
    if a.output is not None:
        ap.error('--output is only valid for graph-eligibility')
    if (a.operation in ('gate','graph-gate')) != (a.arm is not None):
        ap.error('only gate requires an arm')
    if a.operation == 'final':
        s = AttributionSession.__new__(AttributionSession)
        BatchSession.__init__(s)
    else:
        s = AttributionSession()
    if a.operation in ('gate','graph-gate'):
        if a.arm not in [BATCH_ARMS[b] for b in BATCHES]:
            ap.error('only B1/B2/B4 arms are allowed')
        getattr(s, a.operation.replace('-', '_'))(a.arm)
    else:
        getattr(s, a.operation.replace('-', '_'))()


if __name__ == '__main__':
    main()
