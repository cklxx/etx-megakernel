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
import re
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


def schedule_signature(plan):
    """Shapes/queues/boundaries fixed; body symbols may change for a shape arm."""
    return {k: plan[k] for k in ('domains', 'workers_per_domain', 'events', 'ev_inst', 'task_in', 'descs')}


def edge_prices(plan, raw):
    from etx.tools import inbody as reader
    report = reader.analyze(plan, raw)
    scale = raw['tick_ns']/1000
    words = reader.event_words(plan)
    by_worker = {}
    for t in report['tasks']:
        by_worker.setdefault(t['record'][3], []).append(t)
    previous = {}
    for ts in by_worker.values():
        ts.sort(key=lambda t: t['record'][0])
        for a, b in zip(ts, ts[1:]): previous[b['task']] = a
    edges = []
    critical = set(report['critical_task_ids'])
    for t in report['tasks']:
        i = t['task']; taken, body, _, worker, ready_hi, poll = t['record']
        prev = previous.get(i)
        for e in plan['task_in'][i]:
            lo, hi, owner = raw['events'][words[e]]
            # E precedes the previous task's releases/arrives/dequeue. That
            # interval is not observable as previous-task busy vs worker idle.
            if prev is None:
                kind = 'free'
            elif prev['record'][0] <= lo and hi <= prev['record'][2]:
                kind = 'busy'
            elif lo >= taken:
                kind = 'free'
            else:
                kind = 'ambiguous'
            selected = i in critical and hi == ready_hi
            edges.append(dict(consumer=i, event=e, producer=owner,
                all_producers=plan['ev_inst'][e][2], worker=worker,
                previous_task=prev['task'] if prev else None, edge_class=kind,
                arrive_bounds_ticks=[lo, hi], gap_bounds_us=[max(0, body-hi)*scale, (body-lo)*scale],
                poll_gap_bounds_us=[max(0, poll-hi)*scale, (poll-lo)*scale],
                legacy_critical_gap_us=t.get('gap_us'), critical_readiness=selected,
                nonfinal_producer_arrivals='UNOBSERVED: event row stamps only final producer'))
    # One event-readiness edge per critical consumer, never sum tied fan-in
    # observations twice. Ties differing in class conservatively stay unknown.
    projection = []
    for i in report['critical_task_ids']:
        tied = [e for e in edges if e['consumer'] == i and e['critical_readiness']]
        if not tied: continue
        row = dict(tied[0]); row['tied_events'] = [e['event'] for e in tied]
        if len({e['edge_class'] for e in tied}) > 1: row['edge_class'] = 'ambiguous'
        projection.append(row)
    def distribution(rows):
        out = {}
        for kind in ('busy', 'free', 'ambiguous'):
            xs = [e['gap_bounds_us'][0] for e in rows if e['edge_class'] == kind]
            upper = [e['gap_bounds_us'][1] for e in rows if e['edge_class'] == kind]
            out[kind] = dict(count=len(xs), gaps_us=xs, upper_gaps_us=upper,
                median_us=st.median(xs) if xs else None, min_us=min(xs) if xs else None,
                below_floor_count=sum(x < 1 for x in xs),
                above_floor_lower_ms=sum(max(0, x-1) for x in xs)/1000,
                above_floor_upper_ms=sum(max(0, x-1) for x in upper)/1000,
                signed_excess_ms=sum(x-1 for x in xs)/1000)
        return out
    return dict(edges=edges, critical_edges=projection, all_edges=distribution(edges),
                critical_pools=distribution(projection), floor_us=1., inbody=report,
                basis='B minus final-arrival bracket; critical-consumer projection, not a removable pool')


def attention_price(report):
    rows = []
    for layer in range(48):
        ps = {p['phase']: p['body_us'] for p in report['phases'] if p['layer'] == layer}
        if 'attn' not in ps or 'reduce' not in ps:
            raise ValueError(f'A2 incomplete critical attention/reduce layer {layer}')
        rows.append(dict(layer=layer, attention_us=ps['attn'], reduce_us=ps['reduce'],
                         body_us=ps['attn']+ps['reduce']))
    total = sum(r['body_us'] for r in rows)/1000
    # Registered historical serial floors: not an attainable candidate body.
    floor = 48*(2.92+2.81)/1000
    return dict(layers=rows, body_ms=total, floor_estimate_ms=floor,
                available_cut_estimate_ms=total-floor, required_cut_ms=.142,
                candidate_required_mean_us=total*1000/48-2.96,
                attainable_cut_ms=None, status='HEADROOM-ONLY: no legal replacement measured')


def stage_prices(text):
    rows = [json.loads(l.split(' ', 1)[1]) for l in text.splitlines() if l.startswith('HOUR-STAGE ')]
    if [(r['run'], r['step']) for r in rows] != [(0, t) for t in range(1023, 1055)]:
        raise ValueError('A3/A4 NOT-COMPLETE: all 32 decode steps required')
    if any(r['device_tail_reset'] for r in rows):
        raise ValueError('A4 device reset tail uninstrumented; cannot price it as host reset')
    keys = ('upload_us', 'reset_submit_us', 'residency_us', 'launch_us', 'wait_us',
            'retire_readback_us', 'instrument_other_us', 'reset_device_us', 'kernel_device_us', 'wall_us')
    if any(not math.isfinite(r[k]) for r in rows for k in keys): raise ValueError('nonfinite stage')
    avg = {k: st.mean(r[k] for r in rows) for k in keys}
    # wait includes kernel execution; subtract it ONCE. Event envelope differs
    # from token device span and reset overlaps host submission: not additive cuts.
    named = {k: avg[k]/1000 for k in ('upload_us', 'reset_submit_us', 'residency_us', 'launch_us', 'retire_readback_us')}
    named['wait_minus_kernel_ms'] = (avg['wait_us']-avg['kernel_device_us'])/1000
    named['instrument_other_ms'] = avg['instrument_other_us']/1000
    return dict(rows=rows, mean_us=avg, named_host_ms=named,
                host_excess_ms=(avg['wall_us']-avg['kernel_device_us'])/1000,
                accounted_beyond_priced_ms=sum(v for k, v in named.items() if k != 'instrument_other_ms')-.049,
                epoch='NOT-PRESENT in kept host-reset build; no epoch-removal claim')


def host(s, arm, name, args=(), extra=None):
    from examples.vllm_llm import import_certificate as cert
    b = Path(s.c['builds'][arm]); m = Path(s.c['model'])
    text = s.step(name, [sys.executable, '-u', '-m', 'examples.vllm_llm.import_certificate',
                        '--run-build', b, '--model', m, '--', *args], extra)
    cert.stamped(text, cert.verified_build(b, m))
    return text


def token_gate(text):
    from examples.vllm_llm.nvref_report import correctness
    correctness(text, 'qwen300-hour')
    matches = re.findall(r'^run (\d+): (\d+)/(\d+) generated tokens match HF greedy', text, re.M)
    if matches != [('0', '32', '32')]:
        raise ValueError('NOT-COMPLETE requires exactly one complete free-running 32/32 repeat')
    return text


def certify(s, arm):
    from examples.vllm_llm import import_certificate as cert
    if arm in s.certified: return
    b = Path(s.c['builds'][arm]); m = Path(s.c['model'])
    archive = s.out/('certificate_'+arm); archive.mkdir()
    cert.export_build(b, m, archive)
    for pos in (0, 1040):
        text = host(s, arm, f'{arm}_check_p{pos}', ['--check', '48', '--check-pos', str(pos)])
        (archive/f'check_p{pos}.txt').write_text(text)
    result = cert.validate(archive, timing=False)
    text = token_gate(host(s, arm, arm+'_token_gate', ['--repeat', '1', '--check-dsl']))
    if 'CHECK-DSL OK' not in text: raise ValueError('CHECK-DSL missing')
    (archive/'gate_chkx.log').write_text(text)
    s.certified.add(arm)
    s.arms[arm] = dict(status='CERTIFIED', fresh_certificate=result)


def preflight(s):
    from examples.vllm_llm import import_certificate as cert
    gates = json.loads(Path(s.c['cpu_gates']).read_text())
    for k in ('pytest', 'gfx942', 'dry_run', 'shell', 'local_matrix_16', 'rocm_build_check'):
        if gates.get(k) is not True: raise ValueError('staging CPU gate missing/failed: '+k)
    m = Path(s.c['model'])
    if cert.sha(m/'golden_c1024.txt') != s.c['golden_sha256']: raise ValueError('independent HF golden changed')
    from examples.llm.vllm_bench import read_golden
    prompt, golden = read_golden(m/'golden_c1024.txt')
    if len(prompt) != 1024 or len(golden) != 32: raise ValueError('frozen context/tokens changed')
    (s.out/'prepared.json').write_text(json.dumps(s.c, indent=2)+'\n')
    (s.out/'manifest.json').write_text(json.dumps(s.m, indent=2)+'\n')
    (s.out/'cpu-gates.json').write_text(json.dumps(gates, indent=2)+'\n')
    kept = json.loads((Path(s.c['builds']['kept'])/'plan.json').read_text())
    for arm in ('kept', 'stamps', 'stages'):
        b = Path(s.c['builds'][arm]); ident = cert.verified_build(b, m)
        if not ident['check_enabled']: raise ValueError('CHECK build required: '+arm)
        if schedule_signature(json.loads((b/'plan.json').read_text())) != schedule_signature(kept):
            raise ValueError('worker/task/event boundary change: '+arm)
        if '-DETX_INBODY_REPORT' not in ident['hipcc_flags']: raise ValueError('report-enabled build required')
        if arm == 'stages' and '-DETX_HOUR_PROFILE' not in ident['hipcc_flags']: raise ValueError('profile build missing')
        if arm != 'stages' and '-DETX_HOUR_PROFILE' in ident['hipcc_flags']: raise ValueError('profile contaminates timing')
    s.step('hardware', ['rocm-smi', '--showproductname'], timeout=15, expected=5)


def tranche_a(s):
    """900s including fresh correctness, cold loads and all deciding probes."""
    for arm in ('kept', 'stamps', 'stages'):
        if s.attempt('A.gate.'+arm, lambda arm=arm: certify(s, arm)) is None and arm not in s.certified:
            continue
    if 'stamps' in s.certified:
        def stamped():
            raw = s.out/'inbody.bin'
            token_gate(host(s, 'stamps', 'A_stamp', ['--repeat', '1'], {'ETX_INBODY_RAW': str(raw)}))
            from etx.tools import inbody
            p = json.loads((Path(s.c['builds']['stamps'])/'plan.json').read_text())
            result = edge_prices(p, inbody.read_raw(raw))
            (s.out/'A_edges.json').write_text(json.dumps(result, indent=2)+'\n')
            pools = result['critical_pools']; busy = pools['busy']['above_floor_lower_ms']; free = pools['free']['above_floor_lower_ms']
            s.row('A1.busy', pools['busy'], 'PASS' if busy >= .100 else 'DEAD' if pools['busy']['above_floor_upper_ms'] < .050 and not pools['ambiguous']['count'] else 'UNRESOLVED', REGISTERED['A1.busy'])
            s.row('A1.free', pools['free'], 'PASS' if free >= .050 else 'UNRESOLVED', REGISTERED['A1.free'])
            s.row('A1.floor', dict(floor_us=1, distributions=pools, ambiguity=pools['ambiguous']), 'RECORDED', REGISTERED['A1.floor'])
            a2 = attention_price(result['inbody'])
            (s.out/'A_attention.json').write_text(json.dumps(a2, indent=2)+'\n')
            s.row('A2.route', a2, 'HEADROOM-PASS' if a2['available_cut_estimate_ms'] >= .142 else 'FAIL', REGISTERED['A2.route'])
            return result
        s.attempt('A1/A2', stamped)
    if 'stages' in s.certified:
        def stages():
            text = token_gate(host(s, 'stages', 'A_stages', ['--repeat', '1']))
            result = stage_prices(text)
            (s.out/'A_stages.json').write_text(json.dumps(result, indent=2)+'\n')
            explains = result['accounted_beyond_priced_ms'] >= .086 and abs(result['mean_us']['instrument_other_us']) <= 5
            s.row('A3.host', result, 'PASS' if explains else 'DOES-NOT-ACCOUNT', REGISTERED['A3.host'])
            reset = {k: result['mean_us'][k]/1000 for k in ('reset_submit_us', 'reset_device_us')}
            s.row('A4.reset', reset, 'PURSUE' if max(reset.values()) > .030 else 'CLOSED', REGISTERED['A4.reset'])
        s.attempt('A3/A4', stages)
    # Attributions fail closed. B/D may still price existing passenger arms;
    # they cannot reopen the target campaign on a residual or gap alone.
    s.row('A.stop', dict(exclusive_new_opportunity_ms=None,
                        decision='CLOSE-TARGET-ROUTE; continue only registered intermediate candidates/baselines'),
          'NOT-ESTABLISHED', REGISTERED['A.stop'])
    print(f'FIRST-STOP elapsed={time.monotonic()-s.start:.1f}s (<=900s); stamps price intervals, not removals', flush=True)


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
