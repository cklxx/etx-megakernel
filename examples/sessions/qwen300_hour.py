"""One pinned, offline, 140-minute deciding session; importing is CPU only.

python -m examples.sessions.qwen300_hour --template prepared.json
python -m examples.sessions.qwen300_hour --dry-run
python -m examples.sessions.qwen300_hour --prewarm prepared.json
python -m etx.tools.gpu_jobs examples/sessions/qwen300_hour.py --gpus 0 --machine A --logdir OUT

The VM must have prepared.json at ~/etx/qwen300-hour.json, or set
ETX_QWEN300_CONFIG for a direct gpu_jobs invocation. No download/build/debug
occurs during the measurement budget. Fresh checks are never transferred.

Deadlines (including each tranche's fresh correctness work): A=900s (885s
commands + 15s first-stop decision), B=1320s, C=4020s (NO_TOUCH=1620s,
ISOLATE_POLLER=1200s, C6=1200s), guard=240s, D>=1910s of remaining time, report=10s.
240 + 900 + 1320 + 4020 + 1910 + 10 = 8400s; no head arm or pair is shortened.
C estimates: 1575 + 1140 + 1085 = 3800s, with 220s margin. C6's estimate
is 9 expected 120s host calls plus 5s ancestry; adapter eligibility remains
mandatory. D's warm-call upper estimate is 933s + 3*28s = 1017s, including
fresh certification of a new final combination (PREFLIGHT Q4); it has 893s
margin. Prewarm/staging finish before the clock. These are estimates, not
runtime guarantees: missed gates/deadlines stay NOT-COMPLETE with full logs.
The runner stops at 8400s, 600s below the 150-minute machine life.

KEY32 acceptance is reused, never a new timing tranche:
results_qwen300_2026-10-11/job2_run4.log:138,144 records 0.231ms paired
saving (spread 0.016ms) and 4.180ms host wall = 239.23 tok/s, with checks
at :47,60 and 32/32 in all timed logs (docs/KEY32-MEASURED-2026-10-11.md).
This scheduling-only change leaves the KEY32 build recipe/identity unchanged.
A new candidate/diagnostic/combination identity still buys fresh correctness
inside its existing tranche before timing; it never adds KEY32 off/on pairs.
An unchanged historical KEY32 arm buys no separate re-run.
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
from examples.vllm_llm.pairs import Refusal

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('qwen300_hour_arms.json')
# Registration: main 9f91f3a, docs/PLAN-300TOKS-2026-10-11.md:297,304,310,318,320.
MAX_SESSION_SECONDS = 8400
ENVIRONMENT_GUARD_SECONDS = 240
TRANCHE_SECONDS = dict(A=900, B=1320, C=4020, D=1910)
REPORT_SECONDS = 10
if not 0 < float(os.environ.get('ETX_SESS_MAX_SECONDS', MAX_SESSION_SECONDS)) <= MAX_SESSION_SECONDS:
    raise ValueError('qwen300-hour cap must be positive and <=8400 seconds')
JOBS = [dict(name='qwen300_hour', gpu=0, machine='A', timeout=MAX_SESSION_SECONDS, quiet_s=180,
             cmd='stdbuf -oL -eL python3 -u -m examples.sessions.qwen300_hour --run')]


def manifest(path=MANIFEST):
    return json.loads(Path(path).read_text())


def template():
    return dict(model='/absolute/prepared/qwen3-30b-a3b',
                builds={name: '/absolute/prepared/'+name for name in
                        ('kept', 'reset', 'stamps', 'stages', 'C1', 'C5', 'C1_C5', 'NO_TOUCH',
                         'NO_TOUCH_stamps', 'ISOLATE_POLLER', 'C6')},
                environment_guard=dict(
                    trees={name: dict(build='/absolute/prepared/guard-'+commit,
                        source_tree='/absolute/prepared/source-'+commit,
                        normalized_device_ir='/absolute/prepared/guard-'+commit+'/normalized.ll',
                        cpu_gates='/absolute/prepared/guard-'+commit+'/cpu-gates.json',
                        identity_sha256='REQUIRED', seal_sha256='REQUIRED')
                        for name, commit in (('old', '7c85efb'), ('hour', 'e068e47'))},
                    weight_sha256={}),
                x5_builds=['kept'],
                isolation_probe=None,  # sealed offline probe: {argv: [...], sha256: executable SHA256}
                source_commit='REQUIRED full commit of pushed tree', build_commits={}, build_seals={},
                combinations=[dict(id='C1_C5', build='C1_C5', members=['C1', 'C5'], enabled=True,
                                   flags={'ETX_VL_ROUTER_NARROW': '1'}, defines=['ETX_DSL_NORM_HOIST'])],
                cpu_gates='/absolute/prepared/cpu-gates.json',
                hf='/absolute/prepared/qwen3-30b-a3b/hf',
                image='vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7',
                manifest=str(MANIFEST), golden_sha256='bab2f1fbe12f70179711ec40a215c74710845b73169b1cc6a3a6ca50e0a19484')


SOURCE_FILES = ('examples/sessions/qwen300_hour.py', 'examples/sessions/qwen300_hour_arms.json',
                'examples/vllm_llm/host.hip', 'etx/runtime/include/etx/host.h')
BASE_FLAGS = '-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH -DETX_INBODY_REPORT'


def source_hashes():
    from examples.vllm_llm.import_certificate import sha
    paths = {ROOT/p for p in SOURCE_FILES}
    paths.update(ROOT/p for p in ('examples/sessions/qwen300.py', 'examples/llm/vllm_bench.py',
                                  'examples/llm/configs/qwen3-30b-a3b.json'))
    for directory in ('etx', 'examples/vllm_llm'):
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.hip', '.h', '.sh') and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def prepare(config_path):
    """Explicit offline CPU staging command, OUTSIDE the paid measurement job.

    Existing model/HF/golden/venv/native compiler must be staged. This only
    invokes the repository builder; it never executes a host or provisions.
    No invented correctness seal: fresh GPU checks remain tranche work.
    """
    from examples.vllm_llm import import_certificate as cert
    c = json.loads(config_path.read_text())
    if not re.fullmatch('[a-f0-9]{40}', c['source_commit']): raise ValueError('full source commit required')
    gates = json.loads(Path(c['cpu_gates']).read_text())
    if gates.get('source_hashes') != source_hashes(): raise ValueError('CPU gates belong to different source')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ETX_VL_', 'ETX_DSL_', 'ETX_LLM_', 'DSL_ABLATE_', 'ETX_INBODY_', 'ETX_BF16_'))
           and k != 'ETX_PREFETCH_ISOLATE_POLLER'}
    env.update(ETX_VL_CHECK='1', ETX_TOPK_KEY32='1', ETX_DEV_RESET='0', ETX_VL_REQUESTS='1',
               ETX_VL_CU='256', ETX_VL_TOUCH='1', ETX_PREFETCH_ISOLATE_POLLER='0')
    # Optional C prototypes are NOT built here: integrate and prove before use.
    variants = dict(kept={}, reset={'ETX_DEV_RESET': '1'}, stamps={'ETX_INBODY_BUILD': '1'}, stages={'profile': True},
                    C1={'ETX_VL_ROUTER_NARROW': '1'}, C5={'hoist': True},
                    C1_C5={'ETX_VL_ROUTER_NARROW': '1', 'hoist': True},
                    NO_TOUCH={'no_touch': True},
                    NO_TOUCH_stamps={'no_touch': True, 'ETX_INBODY_BUILD': '1'},
                    ISOLATE_POLLER={'ETX_PREFETCH_ISOLATE_POLLER': '1'})
    for arm, toggles in variants.items():
        if arm not in c['builds']: continue
        b = Path(c['builds'][arm]).resolve()
        if b.exists(): raise ValueError('fresh staging output required: '+str(b))
        b.mkdir(parents=True)
        flags = (BASE_FLAGS+(' -DETX_HOUR_PROFILE' if toggles.get('profile') else '')
                 +(' -DETX_DSL_NORM_HOIST' if toggles.get('hoist') else '')
                 +(' -DETX_NO_TOUCH' if toggles.get('no_touch') else ''))
        use = dict(env, ETX_OUT=str(b), ETX_HIPCC_FLAGS=flags,
                   **{k: v for k, v in toggles.items() if k.startswith('ETX_')})
        print(f'CPU-STAGE {arm} build only; no GPU invocation', flush=True)
        with (b/'staging.log').open('x') as log:
            log.write('# frozen build environment '+json.dumps({k: v for k, v in use.items() if k.startswith('ETX_')})+'\n'); log.flush()
            subprocess.run(['bash', str(ROOT/'examples/vllm_llm/build.sh'), c['model']], env=use,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        identity = cert.verified_build(b, Path(c['model']))
        record = dict(source_commit=c['source_commit'], source_hashes=source_hashes(),
                      condition=dict(precision='bf16', batch=1, context=1024, tokens=32, key32=True),
                      flags={k: v for k, v in use.items() if k.startswith('ETX_')},
                      build_sha256=cert.digest(identity))
        (b/'hour_build.json').write_text(json.dumps(record, indent=2)+'\n')
        c['build_seals'][arm] = cert.sha(b/'hour_build.json')
        c['build_commits'][arm] = c['source_commit']
    config_path.write_text(json.dumps(c, indent=2)+'\n')
    print('CPU-STAGE COMPLETE: GPU correctness still UNCHECKED; no timing eligibility borrowed', flush=True)


def verify_seal(s, arm):
    from examples.vllm_llm import import_certificate as cert
    b = Path(s.c['builds'][arm]); record = b/'hour_build.json'
    if cert.sha(record) != s.c['build_seals'][arm]: raise ValueError('unsealed/stale prepared build: '+arm)
    seal = json.loads(record.read_text())
    if seal['source_hashes'] != source_hashes(): raise ValueError('pushed tree differs from CPU-tested build')
    if seal['source_commit'] != s.c['build_commits'][arm]: raise ValueError('sealed source commit changed')
    if seal['condition'] != dict(precision='bf16', batch=1, context=1024, tokens=32, key32=True):
        raise ValueError('frozen precision/shape/KEY32 changed')
    if seal['flags'].get('ETX_TOPK_KEY32') != '1': raise ValueError('KEY32 must remain on')
    if seal['build_sha256'] != cert.digest(cert.verified_build(b, Path(s.c['model']))):
        raise ValueError('build changed after staging seal')
    return seal


def prewarm(config_path):
    """CPU-only setup before starting gpu_jobs and its 140-minute clock."""
    c = json.loads(config_path.read_text())
    hf_weights = sorted(Path(c['hf']).glob('*.safetensors'))
    if not hf_weights: raise ValueError('staged HF weight shards missing')
    files = [Path(c['model'])/'vl.bin', *hf_weights]
    for path in files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError('staged weight file missing/empty: '+str(path))
    print('CPU-PREWARM begin; separate staging budget, timeout=600s; files='+
          json.dumps(list(map(str, files))), flush=True)
    begin = time.monotonic()
    with open(os.devnull, 'wb') as sink:
        p = subprocess.Popen(['cat', *map(str, files)], stdout=sink)
        try:
            while p.poll() is None:
                remaining = 600-(time.monotonic()-begin)
                if remaining <= 0: raise TimeoutError('CPU-PREWARM staging deadline')
                try:
                    p.wait(timeout=min(5, remaining))
                except subprocess.TimeoutExpired:
                    print(f'CPU-PREWARM progress elapsed={time.monotonic()-begin:.0f}s pid={p.pid}', flush=True)
            if p.returncode: raise RuntimeError(f'CPU-PREWARM rc={p.returncode}')
        finally:
            if p.poll() is None: p.kill(); p.wait()
    if 'environment_guard' in c:
        from examples.vllm_llm.import_certificate import sha
        c['environment_guard']['weight_sha256'] = {str(path): sha(path) for path in files}
        config_path.write_text(json.dumps(c, indent=2)+'\n')
    print(f'CPU-PREWARM complete elapsed={time.monotonic()-begin:.1f}s; start runner next', flush=True)


class Session:
    def __init__(self, config, logdir, seconds=MAX_SESSION_SECONDS):
        self.c = config
        self.out = Path(logdir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        self.end = self.start + min(MAX_SESSION_SECONDS-REPORT_SECONDS, seconds-REPORT_SECONDS)  # reserve report/outer-runner margin
        self.deadline = min(self.start+TRANCHE_SECONDS['A']-15, self.end)  # leave room for the first-stop table
        self.rows = {}
        self.arms = {}
        self.certified = set()
        self.m = manifest(config.get('manifest', MANIFEST))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('ETX_', 'DSL_ABLATE_', 'VLLM_', 'BF16_'))}
        self.env.update(ETX_VL_PROGRESS='1', PYTHONUNBUFFERED='1')
        self.env['HIP_VISIBLE_DEVICES'] = os.environ.get('GPU', '0')

    def row(self, name, value, status, rule):
        arm = next((a for a in self.m['B']+self.m['C'] if name == 'C.'+a['id']), {})
        source = THRESHOLD_SOURCES.get(name, arm.get('source', THRESHOLD_SOURCES.get(name.split('.')[0]+'.keep', 'manifest/prototype registration')))
        self.rows[name] = dict(value=value, status=status, rule=rule, source=source)
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
                        if ('  step ' in tail or 'HOUR-STEP-BEGIN' in tail) and time.monotonic()-last > 10:
                            self.probe(out)
                            raise TimeoutError(f'{name}: NOT-COMPLETE step watchdog (10s)')
                        begin_at = tail.rfind('VLLM-GENERATE-BEGIN')
                        end_at = tail.rfind('VLLM-GENERATE-END')
                        if begin_at > end_at and time.monotonic()-last > 10:
                            self.probe(out)
                            raise TimeoutError(f'{name}: NOT-COMPLETE vLLM burst watchdog (10s)')
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
        except (OSError, ValueError, RuntimeError, TimeoutError, KeyError, Refusal, subprocess.SubprocessError) as exc:
            print(f'ARM {name} NOT-COMPLETE: {type(exc).__name__}: {exc}', flush=True)
            self.arms[name] = dict(status='NOT-COMPLETE', reason=str(exc))
            return None

    def finish(self):
        for name, rule in REGISTERED.items():
            self.rows.setdefault(name, dict(value=None, status='NOT-COMPLETE', rule=rule, source=THRESHOLD_SOURCES[name]))
        result = dict(thresholds=self.rows, arms=self.arms, elapsed_s=time.monotonic()-self.start,
                      conditions=dict(precision='bf16', batch=1, context=1024, tokens=32, machine='A',
                                      gpu=os.environ.get('GPU', '0')), attribution='No removable-time credit from stamps')
        (self.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        print('REGISTERED THRESHOLDS | measured | pass/fail | rule', flush=True)
        for name, r in self.rows.items():
            # Omit raw populations, never truncate the actual threshold values.
            display = json.dumps(metric_summary(r['value']))
            print(f'{name} | {display} | {r["status"]} | {r["rule"]} [{r["source"]}]', flush=True)


REGISTERED = {
    'GUARD': 'two pinned old/hour kept pairs abs(delta)<0.030 ms; original attn p0 <=7.9 us; kept anchor <=4.210 ms; sealed weights and IR',
    'A1.busy': 'instrument classification only: real >=0.100 ms; dead <0.050 ms',
    'A1.free': 'instrument classification only: real >=0.050 ms',
    'A1.floor': '1.00 us/edge; retain below-floor count/min, never clip raw gaps',
    'A.loads': 'logical load-phase rate >=4.0 TB/s: non-byte work; <=3.3: memory path; requires perturbation gate, no extra run',
    'A2.route': 'available body headroom >=0.142 ms / 2.96 us/layer; estimate, no removal proof',
    'A3.host': 'name stages accounting for unexplained 0.086 ms beyond priced 0.049 ms',
    'RESET.keep': '3 adjacent pinned pairs, all faster, median >=0.030 ms, median >2*spread; only ETX_DEV_RESET changes',
    'A4.reset': 'pursue only >0.030 ms/step, host and device separately',
    'A.stop': 'S1 at <=15 minutes: G >=0.088 ms (1.833 us/layer) in named above-floor gap atoms, else close target route',
    'X5': '3 x 100 warmed isolated replays/role/layer, ctx1024; legal pair <=7.69 us/layer; incomplete inventory stays [U]',
    'B.stop': 'S2 after X5, before body pairs: G + R2 + H >=0.324 ms (whole-target 6.743 us/layer), else close target route; unknowns count zero',
    'B.keep': '3 pairs, all faster, median >=0.030 ms, median >2*spread',
    'C.fusion': 'net >=1.042 us/layer; VGPR <=128; LDS <=29752 B',
    'C.body': '3 pairs, all faster, median >=0.030 ms, median >2*spread',
    'C.NO_TOUCH': '3 complete adjacent pinned pairs; every wall gain >0; median >=0.030 ms; median >2*(max-min); no additional KEEP threshold',
    'C.ISOLATE_POLLER': '3 complete adjacent pinned pairs; every wall gain >0; median >=0.030 ms; median >2*(max-min); fresh measured round trip <2.0 us',
    'D.target': '5 complete rounds, host-wall step <=3.3333 ms, all fresh correctness gates',
    'D.fusion': '5 adjacent pinned pairs; every gain >0; median >2%; median >2*spread',
    'D.system': '5 adjacent pinned pairs; LEAD all deltas <0, median <=-0.050 ms',
}
THRESHOLD_SOURCES = {
    'GUARD': 'docs/LEVERS-MORE-2026-10-12.md:section 1; PLAN-300-CERTAIN-2026-10-12.md F0',
    'A1.busy': 'docs/qwen300-hour-2026-10-11.md:39',
    'A1.free': 'docs/qwen300-hour-2026-10-11.md:40',
    'A1.floor': 'docs/X7-SCHEDULE-WHATIF-2026-10-11.md:164',
    'A.loads': 'docs/VERIFY-PLAN-2026-10-12.md; MASTER-TABLE-2026-10-11.md QK6/OP3/M.J',
    'A2.route': 'docs/qwen300-hour-2026-10-11.md:42',
    'A3.host': 'docs/MASTER-TABLE-2026-10-11.md:60',
    'RESET.keep': 'docs/VERIFY-PLAN-2026-10-12.md; KEEP from PLAN-300TOKS-2026-10-11.md:322',
    'A4.reset': 'docs/PLAN-generic-decoder-2026-10-08.md:154',
    'A.stop': 'docs/PLAN-300TOKS-2026-10-11.md:301-302 (main 9f91f3a)',
    'X5': 'docs/PLAN-300TOKS-2026-10-11.md:307; REVIEW-POOL-ANALYSIS-2026-10-11.md:123',
    'B.stop': 'docs/PLAN-300TOKS-2026-10-11.md:306-308 (main 9f91f3a)',
    'B.keep': 'docs/PLAN-300TOKS-2026-10-11.md:314,322 (main 9f91f3a)',
    'C.fusion': 'docs/PLAN-300TOKS-2026-10-11.md:315,331 (main 9f91f3a)',
    'C.body': 'docs/PLAN-300TOKS-2026-10-11.md:314,322 (main 9f91f3a)',
    'C.NO_TOUCH': 'docs/PLAN-300TOKS-2026-10-11.md:310,314; docs/XCD-PREFETCH-2026-10-11.md:76,125',
    'C.ISOLATE_POLLER': 'docs/VERIFY-GAPS-2026-10-11.md:141-193; examples/vllm_llm/prefetch_isolation_cpu.py',
    'D.target': 'docs/PLAN-300TOKS-2026-10-11.md:318 (main 9f91f3a)',
    'D.fusion': 'examples/vllm_llm/pairs.py:33-34,138-143',
    'D.system': 'examples/vllm_llm/nvref_report.py:115-125',
}


def metric_summary(value):
    omitted = {'rows', 'layers', 'gaps_us', 'upper_gaps_us', 'rounds', 'samples', 'gate_wall'}
    if isinstance(value, dict):
        return {k: metric_summary(v) for k, v in value.items() if k not in omitted}
    if isinstance(value, list): return [metric_summary(v) for v in value]
    return value


def schedule_signature(plan):
    """Shapes/queues/boundaries fixed; body symbols may change for a shape arm."""
    return {k: plan[k] for k in ('domains', 'workers_per_domain', 'events', 'ev_inst', 'task_in', 'descs')}


def edge_prices(plan, raw):
    from etx.tools import inbody as reader
    report = reader.analyze(plan, raw)
    scale = raw['tick_ns']/1000
    words = reader.event_words(plan)
    known_busy_until = {}
    for e, word in enumerate(words):
        lo, hi, owner = raw['events'][word]
        if 0 < lo <= hi and owner in plan['ev_inst'][e][2]:
            # An owned final-arrival after stamp is still inside that task's
            # original arrival call/diagnostic overlay, before its dequeue.
            known_busy_until[owner] = max(known_busy_until.get(owner, 0), hi)
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
        prev_busy_end = max(prev['record'][2], known_busy_until.get(prev['task'], 0)) if prev else None
        for e in plan['task_in'][i]:
            lo, hi, owner = raw['events'][words[e]]
            # E precedes the previous task's releases/arrives/dequeue. That
            # interval is not observable as previous-task busy vs worker idle.
            if prev is None:
                kind = 'free'
            elif prev['record'][0] <= lo and hi <= prev_busy_end:
                kind = 'busy'
            elif lo >= taken:
                kind = 'free'
            else:
                kind = 'ambiguous'
            selected = i in critical and hi == ready_hi
            edges.append(dict(consumer=i, event=e, producer=owner,
                all_producers=plan['ev_inst'][e][2], worker=worker,
                previous_task=prev['task'] if prev else None, edge_class=kind,
                previous_last_known_busy_tick=prev_busy_end,
                arrive_bounds_ticks=[lo, hi], gap_bounds_us=[max(0, body-hi)*scale, (body-lo)*scale],
                poll_gap_bounds_us=[max(0, poll-hi)*scale, (poll-lo)*scale],
                legacy_critical_gap_us=t.get('gap_us'), critical_readiness=selected,
                priced_gap_us=max(0, body-max(raw['legacy'][p][2] for p in plan['ev_inst'][e][2]))*scale,
                nonfinal_producer_arrivals='UNOBSERVED: event row stamps only final producer'))
    # One event-readiness edge per critical consumer, never sum tied fan-in
    # observations twice. Ties differing in class conservatively stay unknown.
    projection = []
    for i in report['critical_task_ids']:
        tied = [e for e in edges if e['consumer'] == i and e['critical_readiness']]
        if not tied: continue
        row = dict(tied[0]); row['tied_events'] = [e['event'] for e in tied]
        row['priced_gap_us'] = row['legacy_critical_gap_us']
        if len({e['edge_class'] for e in tied}) > 1: row['edge_class'] = 'ambiguous'
        projection.append(row)
    def distribution(rows):
        out = {}
        for kind in ('busy', 'free', 'ambiguous'):
            xs = [e['priced_gap_us'] for e in rows if e['edge_class'] == kind]
            upper = list(xs)  # legacy E/B price is exact accounting; classification can remain ambiguous
            out[kind] = dict(count=len(xs), gaps_us=xs, upper_gaps_us=upper,
                median_us=st.median(xs) if xs else None, min_us=min(xs) if xs else None,
                below_floor_count=sum(x < 1 for x in xs),
                above_floor_lower_ms=sum(max(0, x-1) for x in xs)/1000,
                above_floor_upper_ms=sum(max(0, x-1) for x in upper)/1000,
                signed_excess_ms=sum(x-1 for x in xs)/1000)
        return out
    return dict(edges=edges, critical_edges=projection, all_edges=distribution(edges),
                critical_pools=distribution(projection), floor_us=1., gap_atoms=gap_atom_prices(report), inbody=report,
                basis='legacy critical B-D gap (same basis as registered pools/floor); classify at final-arrival bracket; readiness B-A bounds separate; not a removable pool')


def gap_atom_prices(report):
    """Stamped G, not removable savings: plan section 7, lines 301-302 (main 9f91f3a).

    Intersect each stamp with its critical gap to avoid crediting parallel/body
    time twice. P1 fetch and P7 alone are not stamped; they get no invented time.
    Observation keeps the conservative upper idle price, 0.951 us.
    Arrive/setup use the owner's preflight fix 6 pricing decision: 0.30/0.15 us
    from the 1.00 us floor decomposition in
    docs/PREFLIGHT-QWEN300-HOUR-2026-10-11.md:20,55 (pre 0.05, arrive 0.30,
    observation 0.50, setup 0.15). These are pricing estimates, not new measured
    idle latencies; the observation price remains gfx942.yaml:69-72's 0.951 us.
    """
    scale = report['raw']['tick_ns']/1000
    atoms = {'P10.arrive': dict(idle_us=.30, intervals_us=[]),
             'P5/P11.observation': dict(idle_us=.951, intervals_us=[]),
             'P4/P6/P7.setup': dict(idle_us=.15, intervals_us=[])}
    for t in report['tasks']:
        if not t.get('critical') or t['arrive_is_synthetic']: continue
        # The two booked attention boundaries are outside gap rows 9-10.
        if t['phase'] in ('attn', 'reduce'): continue
        _, body, _, _, hi, poll = t['record']
        lo = t['arrive_bounds_ticks'][0]
        gap_start = body-t['gap_us']/scale
        for name, begin, end in (('P10.arrive', lo, hi),
                                 ('P5/P11.observation', hi, max(hi, poll)),
                                 ('P4/P6/P7.setup', max(hi, poll), body)):
            us = max(0., min(body, end)-max(gap_start, begin))*scale
            if us: atoms[name]['intervals_us'].append(us)
    for atom in atoms.values():
        atom['above_floor_ms'] = sum(max(0., us-atom['idle_us']) for us in atom['intervals_us'])/1000
        atom['us_layer'] = atom['above_floor_ms']*1000/48
        atom['counted'] = atom['us_layer'] >= .625 or math.isclose(atom['us_layer'], .625, abs_tol=1e-12)
    return dict(atoms=atoms, G_ms=sum(a['above_floor_ms'] for a in atoms.values() if a['counted']),
                basis='stamped critical-gap atom excess; upper opportunity estimate, not certified savings',
                unobserved=['P1.fetch', 'isolated P7.barrier', 'nonfinal producer arrivals'])


def first_stop(s, gap_atoms):
    # S1: plan section 7, line 302 (main 9f91f3a). Pool labels never decide S1.
    g = gap_atoms['G_ms'] if gap_atoms is not None else None
    passes = g is not None and math.isfinite(g) and (g >= .088 or math.isclose(g, .088, rel_tol=0., abs_tol=1e-12))
    passes = passes and 'kept' in s.certified
    incomplete = g is None or not math.isfinite(g) or 'kept' not in s.certified
    s.row('A.stop', dict(G_ms=g, gap_atoms=gap_atoms,
                        decision='END SESSION: NOT-COMPLETE' if incomplete else
                                 'CONTINUE to S2' if passes else 'STOP-SPEND: close target route'),
          'NOT-COMPLETE' if incomplete else 'PASS' if passes else 'STOP', REGISTERED['A.stop'])
    print(f'FIRST-STOP elapsed={time.monotonic()-s.start:.1f}s (guard + A <=1140s)', flush=True)
    return passes


def second_stop(s):
    # S2: plan section 7, lines 306-308 (main 9f91f3a). Unknown R2/H count zero.
    g = s.rows['A.stop']['value']['G_ms']
    attention = s.rows.get('A2.route', {}).get('value') or {}
    r2 = attention.get('attainable_cut_ms') or 0.  # never substitute the body-floor ceiling
    stages = s.rows.get('A3.host', {}).get('value') or {}
    named = dict(stages.get('named_host_ms', {}))
    # Pending reset work can be inside wait but outside the kernel envelope.
    # Do not call that queued device time an attributable host round trip.
    named['wait_minus_kernel_ms'] = max(0., named.get('wait_minus_kernel_ms', 0.)
                                       -stages.get('mean_us', {}).get('reset_device_us', 0.)/1000)
    # H prices the attributable remainder, not the host pursuit classification.
    # Subtract the booked reset portion once; exclude device time/overhead.
    h = min(.086, max(0., named.get('reset_submit_us', 0.)-.049)
            +sum(max(0., named.get(k, 0.)) for k in
                 ('upload_us', 'residency_us', 'launch_us', 'retire_readback_us', 'wait_minus_kernel_ms')))
    if any(not math.isfinite(v) or v < 0 for v in (g, r2, h)):
        raise ValueError('S2 nonfinite/negative attribution')
    total = g+r2+h
    passes = total >= .324 or math.isclose(total, .324, rel_tol=0., abs_tol=1e-12)
    s.row('B.stop', dict(G_ms=g, R2_ms=r2, H_ms=h, total_ms=total,
                        decision='CONTINUE to B/C/D' if passes else 'STOP-SPEND: close target route'),
          'PASS' if passes else 'STOP', REGISTERED['B.stop'])
    return passes


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
                candidate_required_mean_us=(total-.142)*1000/48,
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
    device = [json.loads(l.split(' ', 1)[1]) for l in text.splitlines() if l.startswith('HOUR-DEVICE ')]
    if [(r['run'], r['step']) for r in device] != [(0, t) for t in range(1023, 1055)]:
        raise ValueError('A3 missing token-device envelope')
    token_us = st.mean(r['ms']*1000 for r in device)
    # wait includes kernel execution; subtract it ONCE. Event envelope differs
    # from token device span and reset overlaps host submission: not additive cuts.
    named = {k: avg[k]/1000 for k in ('upload_us', 'reset_submit_us', 'residency_us', 'launch_us', 'retire_readback_us')}
    named['wait_minus_kernel_ms'] = (avg['wait_us']-avg['kernel_device_us'])/1000
    named['kernel_outside_token_timers_ms'] = (avg['kernel_device_us']-token_us)/1000
    named['instrument_other_ms'] = avg['instrument_other_us']/1000
    return dict(rows=rows, mean_us=avg, named_host_ms=named,
                host_excess_ms=(avg['wall_us']-token_us)/1000,
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
    extra = {'ETX_INBODY_RAW': str(s.out/'stamps_load_off.bin'), 'ETX_INBODY_LOADS': '0',
             'ETX_HOUR_TRACE_STEP': '1024'} if arm == 'stamps' else None
    text = token_gate(host(s, arm, arm+'_token_gate', ['--repeat', '1', '--check-dsl'], extra))
    if 'CHECK-DSL OK' not in text: raise ValueError('CHECK-DSL missing')
    (archive/'gate_chkx.log').write_text(text)
    s.certified.add(arm)
    s.arms[arm] = dict(status='CERTIFIED', fresh_certificate=result, gate_wall=timing(text))


def preflight(s):
    from examples.vllm_llm import import_certificate as cert
    gates = json.loads(Path(s.c['cpu_gates']).read_text())
    for k in ('pytest', 'gfx942', 'dry_run', 'shell', 'local_matrix_16', 'rocm_build_check'):
        if gates.get(k) is not True: raise ValueError('staging CPU gate missing/failed: '+k)
    if gates.get('source_hashes') != source_hashes(): raise ValueError('CPU gates belong to another source')
    m = Path(s.c['model'])
    hf = json.loads((Path(s.c['hf'])/'config.json').read_text())
    if any(hf.get(k) != v for k, v in dict(model_type='qwen3_moe', num_hidden_layers=48,
            hidden_size=2048, num_key_value_heads=4, num_experts=128, num_experts_per_tok=8).items()):
        raise ValueError('frozen Qwen3-30B-A3B shape changed')
    if hf.get('torch_dtype', hf.get('dtype')) != 'bfloat16': raise ValueError('bf16 weights required; no FP8')
    if hf.get('quantization_config') not in (None, {}): raise ValueError('quantized/FP8 checkpoint forbidden')
    from examples.sessions.qwen300 import IMAGE
    if s.c['image'] != IMAGE: raise ValueError('strongest legal reference requires pinned vLLM image')
    if cert.sha(m/'golden_c1024.txt') != s.c['golden_sha256']: raise ValueError('independent HF golden changed')
    from examples.llm.vllm_bench import read_golden
    prompt, golden = read_golden(m/'golden_c1024.txt')
    if len(prompt) != 1024 or len(golden) != 32: raise ValueError('frozen context/tokens changed')
    (s.out/'prepared.json').write_text(json.dumps(s.c, indent=2)+'\n')
    (s.out/'manifest.json').write_text(json.dumps(s.m, indent=2)+'\n')
    (s.out/'cpu-gates.json').write_text(json.dumps(gates, indent=2)+'\n')
    kept = json.loads((Path(s.c['builds']['kept'])/'plan.json').read_text())
    for arm in ('kept', 'stamps', 'stages'):
        seal = verify_seal(s, arm)
        b = Path(s.c['builds'][arm]); ident = cert.verified_build(b, m)
        if not ident['check_enabled']: raise ValueError('CHECK build required: '+arm)
        if schedule_signature(json.loads((b/'plan.json').read_text())) != schedule_signature(kept):
            raise ValueError('worker/task/event boundary change: '+arm)
        if '-DETX_INBODY_REPORT' not in ident['hipcc_flags']: raise ValueError('report-enabled build required')
        if arm == 'stages' and '-DETX_HOUR_PROFILE' not in ident['hipcc_flags']: raise ValueError('profile build missing')
        if arm != 'stages' and '-DETX_HOUR_PROFILE' in ident['hipcc_flags']: raise ValueError('profile contaminates timing')
        for name in ('hour_build.json', 'hipcc.log', 'compiler_version.txt', 'compile.log', 'staging.log'):
            src = b/name
            if src.exists():
                (s.out/(arm+'_'+name)).write_bytes(src.read_bytes())
        if arm == 'stamps' and seal['flags'].get('ETX_INBODY_BUILD') != '1': raise ValueError('stamp build missing')
        if arm != 'stamps' and seal['flags'].get('ETX_INBODY_BUILD', '0') != '0': raise ValueError('stamps contaminate timing')


def tranche_a(s):
    """900s including fresh correctness and all deciding probes, after cache setup."""
    gap_atoms = None
    for arm in ('kept', 'stamps', 'stages'):
        if s.attempt('A.gate.'+arm, lambda arm=arm: certify(s, arm)) is None and arm not in s.certified:
            continue
    if 'stamps' in s.certified:
        def stamped():
            raw = s.out/'inbody.bin'
            text = token_gate(host(s, 'stamps', 'A_stamp', ['--repeat', '1'],
                                  {'ETX_INBODY_RAW': str(raw), 'ETX_HOUR_TRACE_STEP': '1024'}))
            if 'HOUR-SNAPSHOT run=0 step=1024' not in text: raise ValueError('registered ctx1024 snapshot missing')
            output = s.out/'A_edges.json'
            s.step('A_analysis', [sys.executable, '-u', '-m', 'examples.sessions.qwen300_hour', '--analyze',
                                 Path(s.c['builds']['stamps'])/'plan.json', raw, output], timeout=45, expected=15)
            summary = output.with_suffix('.summary.json')
            result = json.loads(summary.read_text())
            # Control comes from the already-budgeted stamps correctness gate.
            # Analyze under the same bounded subprocess, not a parent rewrite.
            result['diagnostic_control'] = dict(
                on_wall=timing(text), kept_off_wall=s.arms.get('kept', {}).get('gate_wall'),
                stamps_null_wall=s.arms.get('stamps', {}).get('gate_wall'),
                basis='single noninterleaved complete controls; report movement, no correction/subtraction or perturbation pass',
                warning='ON allocations/stores/spills perturb the schedule; pools are ON observations, not certified OFF savings')
            summary.write_text(json.dumps(result, indent=2)+'\n')
            s.row('A.loads', result.get('load_phases'), 'RECORDED-UNVERIFIED' if result.get('load_phases', {}).get('totals') else 'NOT-COMPLETE', REGISTERED['A.loads'])
            pools = result['critical_pools']; busy = pools['busy']['above_floor_lower_ms']; free = pools['free']['above_floor_lower_ms']
            s.row('A1.busy', pools['busy'], 'PASS' if busy >= .100 else 'DEAD' if pools['busy']['above_floor_upper_ms'] < .050 and not pools['ambiguous']['count'] else 'UNRESOLVED', REGISTERED['A1.busy'])
            s.row('A1.free', pools['free'], 'PASS' if free >= .050 else 'UNRESOLVED', REGISTERED['A1.free'])
            s.row('A1.floor', dict(floor_us=1, distributions=pools, ambiguity=pools['ambiguous']), 'RECORDED', REGISTERED['A1.floor'])
            return result
        result = s.attempt('A1', stamped)
        if result is not None:
            gap_atoms = result['gap_atoms']
            def attention():
                a2 = attention_price(result['inbody'])
                (s.out/'A_attention.json').write_text(json.dumps(a2, indent=2)+'\n')
                s.row('A2.route', a2, 'HEADROOM-PASS' if a2['available_cut_estimate_ms'] >= .142 else 'FAIL', REGISTERED['A2.route'])
                return a2
            if s.attempt('A2', attention) is None:
                s.row('A2.route', None, 'NOT-COMPLETE', REGISTERED['A2.route'])
    if 'stages' in s.certified:
        def stages():
            text = token_gate(host(s, 'stages', 'A_stages', ['--repeat', '1']))
            result = stage_prices(text)
            (s.out/'A_stages.json').write_text(json.dumps(result, indent=2)+'\n')
            explained = result['accounted_beyond_priced_ms']
            explains = explained >= .086 or math.isclose(explained, .086, abs_tol=1e-12)
            s.row('A3.host', result, 'PASS' if explains else 'DOES-NOT-ACCOUNT', REGISTERED['A3.host'])
            reset = {k: result['mean_us'][k]/1000 for k in ('reset_submit_us', 'reset_device_us')}
            s.row('A4.reset', reset, 'PURSUE' if max(reset.values()) > .030 else 'CLOSED', REGISTERED['A4.reset'])
        s.attempt('A3/A4', stages)
    if 'kept' in s.certified:
        registered_reset(s)  # requested price is collected even if S1/S2 later stops
    return first_stop(s, gap_atoms)


def timing(text, mode='fused'):
    token_gate(text)
    from examples.vllm_llm.pairs import parse
    runs = parse(text.splitlines()).runs
    if len(runs) != 1 or runs[0].arm != mode: raise ValueError('wrong timing mode')
    rows = [json.loads(l.split(' ', 1)[1]) for l in text.splitlines() if l.startswith('HOUR-WALL ')]
    if [(r['run'], r['step']) for r in rows] != [(0, t) for t in range(1024, 1055)]:
        raise ValueError('NOT-COMPLETE: 31 post-first-token steps required')
    if any(not math.isfinite(r['ms']) or r['ms'] <= 0 for r in rows): raise ValueError('invalid timing')
    return dict(wall_ms=st.mean(r['ms'] for r in rows), samples=rows,
                result_wall_ms=runs[0].wall_ms, device_ms=runs[0].device_ms,
                kernel_event_ms=kernel_mean(text))


def kernel_mean(text):
    rows = [json.loads(l.split(' ', 1)[1]) for l in text.splitlines() if l.startswith('HOUR-KERNEL ')]
    if not rows: return None  # historical hosts; never synthesize a tail price
    if [(r['run'], r['step']) for r in rows] != [(0, t) for t in range(1024, 1055)]:
        raise ValueError('NOT-COMPLETE kernel-event population')
    if any(not math.isfinite(r['ms']) or r['ms'] <= 0 for r in rows): raise ValueError('invalid kernel event')
    return st.mean(r['ms'] for r in rows)


def keep(gains, count=3, threshold=.030):
    if len(gains) != count: raise ValueError('NOT-COMPLETE pair population')
    if any(not math.isfinite(g) for g in gains): raise ValueError('nonfinite paired gain')
    med = st.median(gains); spread = max(gains)-min(gains)
    return dict(gains_ms=gains, median_ms=med, spread_ms=spread,
                keep=all(g > 0 for g in gains) and (med >= threshold or math.isclose(med, threshold, abs_tol=1e-12)) and med > 2*spread)


def eligible_arm(s, a):
    """No builder is called here: a candidate must be sealed before the clock."""
    from examples.vllm_llm import import_certificate as cert
    if not a.get('enabled'): raise ValueError(a.get('reason', 'disabled in manifest'))
    seal = verify_seal(s, a['build'])
    if seal['flags'].get('ETX_INBODY_BUILD', '0') != '0': raise ValueError('stamps contaminate candidate')
    b = Path(s.c['builds'][a['build']]); identity = cert.verified_build(b, Path(s.c['model']))
    if not identity['check_enabled']: raise ValueError('candidate needs CHECK build')
    flags = identity['hipcc_flags']
    if '-DETX_INBODY_REPORT' not in flags or '-DETX_HOUR_PROFILE' in flags or '-DETX_INBODY=' in flags:
        raise ValueError('candidate timing must be report-enabled, stamps/profile OFF')
    for k, v in a.get('flags', {}).items():
        if identity['overrides'].get(k) != v: raise ValueError('candidate flag missing: '+k)
    for define in a.get('defines', []):
        if '-D'+define not in flags: raise ValueError('candidate define missing: '+define)
    p = json.loads((b/'plan.json').read_text())
    kept = json.loads((Path(s.c['builds']['kept'])/'plan.json').read_text())
    if schedule_signature(p) != schedule_signature(kept): raise ValueError('frozen workers/tasks/boundaries changed')
    for name in ('hour_build.json', 'hipcc.log', 'compiler_version.txt', 'compile.log', 'staging.log'):
        src = b/name
        if src.exists() and not (s.out/(a['build']+'_'+name)).exists():
            (s.out/(a['build']+'_'+name)).write_bytes(src.read_bytes())
    return a


def candidate_pairs(s, a, tranche, publish=True):
    arm = a['build']; ident = a['id']
    certify(s, arm)
    rounds = []
    for rnd in range(3):
        row = {}
        for who in (('kept', arm) if rnd % 2 == 0 else (arm, 'kept')):
            text = host(s, who, f'{tranche}_{ident}_r{rnd}_{who}', ['--repeat', '1'])
            if who == 'reset' and 'HOUR-RESET enabled=1 checked=1 valid=1' not in text:
                raise ValueError('reset tail not confirmed; fallback cannot be a KEEP arm')
            row[who] = timing(text)
        rounds.append(row)
    result = keep([r['kept']['wall_ms']-r[arm]['wall_ms'] for r in rounds])
    result.update(status='KEEP' if result['keep'] else 'REJECT', rounds=rounds, overlap=a.get('overlap'),
                  registered_keep_threshold=a.get('keep_threshold'))
    s.arms[ident] = result
    if publish:
        s.row(tranche+'.'+ident, result, result['status'],
              REGISTERED.get(tranche+'.'+ident, REGISTERED['B.keep' if tranche == 'B' else 'C.body']))
    return result


def registered_reset(s):
    from examples.vllm_llm import reset_pair
    end = s.deadline
    s.deadline = min(time.monotonic()+210, end)
    try:
        result = s.attempt('RESET.keep', lambda: reset_pair.run(s))
        if result is None:
            s.row('RESET.keep', None, 'NOT-COMPLETE', REGISTERED['RESET.keep'])
        return result
    finally:
        s.deadline = end


def tranche_b(s):
    # Tranche B: plan section 7, line 304 (main 9f91f3a).
    s.deadline = min(time.monotonic()+TRANCHE_SECONDS['B']-120, s.end)
    if 'kept' not in s.certified:
        print('B SKIP: kept fresh correctness gate failed', flush=True); return []
    survivors = []
    for a in s.m['B']:
        if not a.get('enabled'):
            s.arms[a['id']] = dict(status='SKIP', reason=a.get('reason', 'disabled'))
            print(f'B ARM {a["id"]} SKIP: disabled in manifest', flush=True)
            continue
        def run(a=a):
            eligible_arm(s, a)
            return candidate_pairs(s, a, 'B')
        result = s.attempt(a['id'], run)
        if result and result['keep']: survivors.append(a)
    s.row('B.keep', {a['id']: s.arms.get(a['id']) for a in s.m['B']},
          'PASS' if survivors else 'NO-KEEP', REGISTERED['B.keep'])
    print('B ceilings overlap router body: do not add C1/C5 gains; D must measure any combination', flush=True)
    return survivors


def c_eligibility(s, a):
    eligible_arm(s, a)
    if a['id'] in ('NO_TOUCH', 'ISOLATE_POLLER'):
        # Existing switches, not a fusion prototype. Fresh exhaustive checks
        # and DSL/token gates run below, before any performance population.
        verify_wait_variant(s, a)
        if a.get('diagnostic_build'): verify_wait_variant(s, a, diagnostic=True)
        return a
    from examples.vllm_llm import import_certificate as cert
    from examples.vllm_llm.nvref_report import correctness
    proof = Path(a['prototype_certificate'])
    info = cert.validate(proof, timing=False)
    identity = cert.verified_build(Path(s.c['builds'][a['build']]), Path(s.c['model']))
    if info['build_sha256'] != cert.digest(identity): raise ValueError('prototype correctness belongs to another build')
    gate = (proof/'gate_chkx.log').read_text()
    cert.stamped(gate, identity)
    correctness(gate, a['id'])
    if 'CHECK-DSL OK' not in (proof/'gate_chkx.log').read_text(): raise ValueError('prototype CHECK-DSL missing')
    commit = a['prototype_commit']
    if not re.fullmatch('[a-f0-9]{40}', commit): raise ValueError('prototype full commit required')
    s.step('C_ancestry_'+a['id'], ['git', '-C', ROOT, 'merge-base', '--is-ancestor', commit, 'HEAD'],
           timeout=5, expected=1)
    if commit != s.c['build_commits'].get(a['build']): raise ValueError('prototype build/source commit mismatch')
    if a['kind'] == 'fusion':
        if a['expected_vgpr'] > 128 or a['expected_lds_bytes'] > 29752:
            raise ValueError('expected fusion resources exceed 128 VGPR / 29752 B LDS')
        log = (Path(s.c['builds'][a['build']])/'hipcc.log').read_text()
        blocks = re.split(r'Function Name:', log)
        native = [b for b in blocks if re.match(r'\s*etx_megakernel_d0\b', b)]
        if len(native) != 1: raise ValueError('whole-kernel native compiler resources missing')
        def value(label):
            match = re.search(re.escape(label)+r'(?: \[[^\]]+\])?:\s*(\d+)', native[0])
            if not match: raise ValueError('native resource field missing: '+label)
            return int(match[1])
        if value('VGPRs') > 128 or value('LDS Size') > 29752:
            raise ValueError('native fusion resources exceed budget')
        a['native_resources'] = dict(vgpr=value('VGPRs'), lds_bytes=value('LDS Size'))
    return a


def verify_wait_variant(s, a, diagnostic=False):
    """Only the registered switch (and diagnostic stamps) may differ from kept."""
    from examples.vllm_llm import import_certificate as cert
    build = a.get('diagnostic_build') if diagnostic else a['build']
    seal = verify_seal(s, build)
    b = Path(s.c['builds'][build])
    kept = Path(s.c['builds']['kept'])
    actual = cert.verified_build(b, Path(s.c['model']))
    base = cert.verified_build(kept, Path(s.c['model']))
    if not actual['check_enabled'] or not base['check_enabled']: raise ValueError('CHECK required')
    if base['hipcc_flags'].split() != BASE_FLAGS.split(): raise ValueError('kept compiler switches changed')
    if json.loads((b/'plan.json').read_text()) != json.loads((kept/'plan.json').read_text()):
        raise ValueError('wait arm requires identical full plan')
    if actual['files']['imports'] != base['files']['imports']:
        raise ValueError('wait arm changed imported math')
    expected = dict(base['overrides'], ETX_PREFETCH_ISOLATE_POLLER='1' if a['id'] == 'ISOLATE_POLLER' else '0')
    if base['overrides'].get('ETX_VL_TOUCH') != '1': raise ValueError('kept TOUCH must be ON')
    if base['overrides'].get('ETX_PREFETCH_ISOLATE_POLLER') != '0': raise ValueError('kept isolation must be OFF')
    # Output locations are provenance, not experiment switches.
    paths = {'ETX_VL_IMPORTS', 'ETX_VL_ADAPTERS', 'ETX_VL_PLAN', 'ETX_VL_CHAINS', 'ETX_VL_DSL_TILES'}
    clean = lambda d: {k: v for k, v in d.items() if k not in paths}
    if clean(actual['overrides']) != clean(expected): raise ValueError('unregistered wait-arm override')
    flags = base['hipcc_flags'].split()
    if a['id'] == 'NO_TOUCH': flags.append('-DETX_NO_TOUCH')
    if actual['hipcc_flags'].split() != flags: raise ValueError('unregistered wait-arm compiler flags')
    if seal['flags'].get('ETX_INBODY_BUILD', '0') != ('1' if diagnostic else '0'):
        raise ValueError('wait-arm stamp build mismatch')
    return actual


def no_touch_diagnostic(s, a):
    """Independent ON-stamp observation, never used as a wall sample/correction."""
    verify_wait_variant(s, a, diagnostic=True)
    arm = a['diagnostic_build']
    certify(s, arm)
    raw = s.out/'C_NO_TOUCH_inbody.bin'
    text = token_gate(host(s, arm, 'C_NO_TOUCH_stamp', ['--repeat', '1'],
                          {'ETX_INBODY_RAW': str(raw), 'ETX_HOUR_TRACE_STEP': '1024'}))
    if 'HOUR-SNAPSHOT run=0 step=1024' not in text: raise ValueError('ctx1024 snapshot missing')
    output = s.out/'C_NO_TOUCH_edges.json'
    s.step('C_NO_TOUCH_analysis', [sys.executable, '-u', '-m', 'examples.sessions.qwen300_hour',
           '--analyze', Path(s.c['builds'][arm])/'plan.json', raw, output], timeout=45, expected=15)
    return dict(path=str(output), basis='diagnostic stamps ON; no removable-time credit or subtraction')


def isolation_round_trip(s, a):
    """Run a preprepared measurement, never accept an offline/historical scalar.

    Probe stdout must contain exactly three POLLER-ROUND-TRIP JSON rows, bound
    to the candidate identity and this session. Missing probe/evidence leaves
    acceptance NOT-COMPLETE, even when all wall pairs pass.
    """
    from examples.vllm_llm import import_certificate as cert
    probe = s.c.get('isolation_probe')
    if not probe: raise ValueError('future isolated round-trip probe not staged; no KEEP')
    argv = probe['argv']
    if not argv or not Path(argv[0]).is_absolute() or cert.sha(Path(argv[0])) != probe['sha256']:
        raise ValueError('unsealed isolation probe executable')
    identity = cert.digest(cert.verified_build(Path(s.c['builds'][a['build']]), Path(s.c['model'])))
    nonce = str(s.out)
    text = s.step('C_ISOLATE_POLLER_round_trip', argv, dict(ETX_HOUR_SESSION=nonce,
                  ETX_HOUR_BUILD_SHA256=identity, ETX_PREFETCH_ISOLATE_POLLER='1'), timeout=120, expected=60)
    rows = [json.loads(l.split(' ', 1)[1]) for l in text.splitlines() if l.startswith('POLLER-ROUND-TRIP ')]
    if [r['sample'] for r in rows] != [0, 1, 2]: raise ValueError('NOT-COMPLETE round-trip population')
    if any(r['session'] != nonce or r['build_sha256'] != identity or r['isolate_poller'] != 1 for r in rows):
        raise ValueError('round-trip measurement belongs to another condition/session')
    xs = [r['round_trip_us'] for r in rows]
    if any(not math.isfinite(x) or x <= 0 for x in xs): raise ValueError('invalid measured round trip')
    return dict(samples=rows, median_us=st.median(xs), pass_lt_2us=st.median(xs) < 2.0)


def qualify_c(s):
    qualified = []
    for a in s.m['C']:
        if not a.get('enabled'):
            print(f'C ARM {a["id"]} SKIP: {a.get("reason", "disabled")}', flush=True)
            s.arms['C.eligibility.'+a['id']] = dict(status='SKIP', reason=a.get('reason', 'disabled'))
            continue
        result = s.attempt('C.eligibility.'+a['id'], lambda a=a: c_eligibility(s, a))
        if result:
            qualified.append(result)
            s.arms['C.eligibility.'+a['id']] = dict(status='QUALIFIED', native_resources=result.get('native_resources'))
    return qualified


def tranche_c(s):
    qualified = s.qualified_c if hasattr(s, 'qualified_c') else qualify_c(s)
    if not qualified:
        print(f'C SKIP: no already-correct committed full-model prototype qualifies; add {TRANCHE_SECONDS["C"]}s to D', flush=True)
        s.row('C.fusion', None, 'SKIP', REGISTERED['C.fusion'])
        s.row('C.body', None, 'SKIP', REGISTERED['C.body'])
        return [], TRANCHE_SECONDS['C']
    s.deadline = min(time.monotonic()+TRANCHE_SECONDS['C'], s.end)
    tranche_end = s.deadline
    budget = [dict(id=a['id'], allocated_s=a.get('budget_s', 0),
                   estimated_required_s=a.get('estimated_required_s')) for a in qualified]
    required = sum(a.get('estimated_required_s', 0) for a in qualified)
    s.row('C.budget', dict(available_s=TRANCHE_SECONDS['C'], arms=budget, estimated_required_s=required,
                         fits_estimate=required <= TRANCHE_SECONDS['C'], basis='estimate: cold host invocation expected=120s; not GPU measurement'),
          'OVER-BUDGET' if required > TRANCHE_SECONDS['C'] else 'ESTIMATE', 'C is 4020s; no time borrowed from D')
    survivors = []
    for a in qualified:
        s.deadline = min(time.monotonic()+a.get('budget_s', TRANCHE_SECONDS['C']), tranche_end)
        key = 'C.'+a['id']
        rule = REGISTERED.get(key, REGISTERED['C.body'])
        def run(a=a):
            diagnostic = no_touch_diagnostic(s, a) if a['id'] == 'NO_TOUCH' else None
            result = candidate_pairs(s, a, 'C', publish=False)
            result['diagnostic'] = diagnostic
            if a['id'] == 'ISOLATE_POLLER':
                try:
                    rt = isolation_round_trip(s, a)
                except (OSError, ValueError, KeyError, TimeoutError, RuntimeError) as exc:
                    result.update(keep=False, status='NOT-COMPLETE', round_trip=dict(status='NOT-COMPLETE', reason=str(exc)))
                else:
                    result['round_trip'] = rt
                    result['keep'] = result['keep'] and rt['pass_lt_2us']
                    result['status'] = 'KEEP' if result['keep'] else 'REJECT'
            s.row(key, result, result['status'], rule)
            return result
        result = s.attempt(a['id'], run)
        if result is None:
            s.row(key, None, 'NOT-COMPLETE', rule)
        if not result: continue
        if a['kind'] == 'fusion':
            net = result['median_ms']*1000/48
            accepted = result['keep'] and (net >= 1.042 or math.isclose(net, 1.042, abs_tol=1e-12))
            s.row('C.fusion.'+a['id'], dict(net_us_layer=net, pairs=result), 'PASS' if accepted else 'FAIL', REGISTERED['C.fusion'])
            result['keep'] = accepted
            result['status'] = 'KEEP' if accepted else 'REJECT'
        if result['keep']: survivors.append(a)
    for kind, key in (('fusion', 'C.fusion'), ('body', 'C.body')):
        s.row(key, {a['id']: dict(pairs=s.arms.get(a['id']), native_resources=a.get('native_resources'),
                                expected_vgpr=a.get('expected_vgpr'), expected_lds_bytes=a.get('expected_lds_bytes'))
                    for a in qualified if a['kind'] == kind},
              'PASS' if any(a['kind'] == kind for a in survivors) else 'NO-KEEP', REGISTERED[key])
    return survivors, 0


def system(s, name):
    """Existing S3 engine/host basis; local pinned image, no pull or downloads."""
    from examples.vllm_llm.vlbest_report import config, proof
    from examples.vllm_llm.nvref_report import vllm
    cfg = config('S3')
    output = s.out/(name+'.json')
    limit = int(min(235, s.deadline-time.monotonic()-3, s.end-time.monotonic()-3))
    if limit <= 0: raise TimeoutError('S3 NOT-COMPLETE remaining deadline')
    container_name = f'qwenhour-{os.getpid()}-{name.lower().replace("_", "-")}'
    cmd = ['sudo', 'docker', 'run', '--name', container_name, '--rm', '--pull=never', '--network=none',
           '--device=/dev/kfd', '--device=/dev/dri', '--group-add=video', '--ipc=host',
           '-e', 'HIP_VISIBLE_DEVICES='+os.environ.get('GPU', '0'),
           '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
           '-e', 'VLLM_ROCM_USE_AITER=0',
           '-v', f'{Path.home()}:{Path.home()}', '-w', str(ROOT), '--entrypoint', 'timeout',
           s.c['image'], '-k', '2', str(limit), 'python3', '-u', '-m', 'examples.sessions.qwen300_hour', '--vllm',
           '--model', s.c['hf'], '--golden', str(Path(s.c['model'])/'golden_c1024.txt'),
           '--tokens', '32', '--repeats', '1', '--batch', '1', '--label', 'S3', '--echo-config',
           '--gate-warmup',
           '--engine-kw', json.dumps(cfg['kw']), '--json', str(output)]
    try:
        text = s.step(name, cmd, timeout=limit+2, expected=min(180, limit))
    finally:
        # Killing the Docker CLI does not kill a daemon-managed container.
        # Internal timeout also bounds it if the outer runner kills this process.
        with (s.out/(name+'_cleanup.log')).open('x') as log:
            subprocess.run(['sudo', 'docker', 'rm', '-f', container_name], stdout=log,
                           stderr=subprocess.STDOUT, check=False, timeout=3)
    if proof(text, cfg): raise ValueError('S3 lever proof failed: '+str(proof(text, cfg)))
    if 'VLLM-WARMUP-GATE OK 32/32' not in text: raise ValueError('S3 correctness-before-timing gate missing')
    if re.findall(r'^VLLM-SAMPLE sample=(\d+) first_divergence=(\S+)', text, re.M) != [('0', '32/32')]:
        raise ValueError('S3 full token gate missing')
    vllm(text, name, '0', 'S3')
    data = json.loads(output.read_text())
    if len(data['per_token_ms']) != 1 or data['golden_prefix_match'] != 32 or len(data['tokens_out']) != 32:
        raise ValueError('S3 NOT-COMPLETE population')
    cost = data['per_token_ms'][0]
    if not math.isfinite(cost) or cost <= 0: raise ValueError('invalid S3 wall cost')
    return dict(wall_ms=cost, basis='(generate(32)-generate(1))/31 host wall', config=cfg)


def select_final(s, survivors):
    """Only prebuilt combinations; no summing overlapping singleton ceilings."""
    if not survivors: return 'kept'
    ids = {a['id'] for a in survivors}
    combined = [a for a in s.c.get('combinations', []) if set(a['members']) == ids]
    if len(ids) > 1 and combined:
        a = combined[0]
        eligible_arm(s, a)
        certify(s, a['build'])
        # D measures the combination afresh; singleton gains confer no KEEP.
        print('FINAL-COMBINATION prebuilt '+a['build']+' members='+','.join(sorted(ids)), flush=True)
        return a['build']
    if len(ids) > 1:
        print('FINAL-COMBINATION unavailable: no prebuilt compatible combination; use strongest measured singleton', flush=True)
    return max(survivors, key=lambda a: s.arms[a['id']]['median_ms'])['build']


def tranche_d(s, survivors, extra_seconds):
    from examples.vllm_llm.pairs import verdict as fusion_verdict, parse, pairs
    from examples.vllm_llm.nvref_report import verdict as system_verdict
    # D inherits all unused A/B/C time, including when C ran successfully.
    s.deadline = s.end
    if 'kept' not in s.certified:
        print('D NOT-COMPLETE: kept correctness gate failed', flush=True); return
    arm = s.attempt('D.selection', lambda: select_final(s, survivors))
    if arm is None: return
    # Fresh certificate from this session even for prequalified prototype builds.
    if s.attempt('D.certify', lambda: (certify(s, arm), True)[1]) is None: return
    if s.attempt('D.graph_gate', lambda: token_gate(host(s, arm, 'D_graph_gate', ['--repeat', '1', '--unfused']))) is None: return
    records = []; fusion_logs = []
    for rnd in range(5):
        row = {}
        # Two independent adjacent pinned pairs per round. F is remeasured
        # next to S3; F/G is not a nonadjacent triplet pretending to be a pair.
        for comparison in ('fusion', 'system'):
            order = ('fused', 'graph' if comparison == 'fusion' else 'S3')
            if rnd % 2: order = order[::-1]
            pair = {}
            for mode in order:
                name = f'D_{comparison}_r{rnd}_{mode}'
                def sample(mode=mode, name=name):
                    if mode == 'S3': return system(s, name)
                    text = host(s, arm, name, ['--repeat', '1']+(['--unfused'] if mode == 'graph' else []))
                    result = timing(text, mode)
                    if comparison == 'fusion': fusion_logs.extend((s.out/(name+'.log')).read_text().splitlines())
                    return result
                result = s.attempt(name, sample)
                if result is not None: pair[mode] = result
            row[comparison] = pair
        records.append(row)
    (s.out/'D_rounds.json').write_text(json.dumps(records, indent=2)+'\n')
    complete_fusion = all(set(r['fusion']) == {'fused', 'graph'} for r in records)
    complete_system = all(set(r['system']) == {'fused', 'S3'} for r in records)
    if complete_fusion:
        ps = pairs(parse(fusion_logs), 5)
        vf = fusion_verdict([(g.wall_ms-f.wall_ms)/g.wall_ms for f, g in ps])
        primary = fusion_verdict([(r['fusion']['graph']['wall_ms']-r['fusion']['fused']['wall_ms'])/r['fusion']['graph']['wall_ms'] for r in records])
        s.row('D.fusion', dict(result_wall_basis=vf, step_mean_wall_basis=primary),
              'ESTABLISHED' if vf['established'] and primary['established'] else 'NOT-ESTABLISHED', REGISTERED['D.fusion'])
    if complete_system:
        deltas = [r['system']['fused']['wall_ms']-r['system']['S3']['wall_ms'] for r in records]
        s.row('D.system', dict(deltas_ms=deltas, median_ms=st.median(deltas), spread_ms=max(deltas)-min(deltas), reference='S3 strongest previously tested legal configuration'),
              system_verdict(deltas), REGISTERED['D.system'])
    if complete_fusion and complete_system:
        values = [r['fusion']['fused']['wall_ms'] for r in records]
        s.row('D.target', dict(arm=arm, rounds_ms=values, median_ms=st.median(values),
                              system_pair_fused_ms=[r['system']['fused']['wall_ms'] for r in records]),
              'PASS' if st.median(values) <= 3.3333 else 'FAIL', REGISTERED['D.target'])
        s.arms['D'] = dict(status='COMPLETE', final=arm)
    else:
        s.arms['D'] = dict(status='NOT-COMPLETE', reason='missing final pairs; no shortened-run aggregate')


def environment_guard(s):
    from examples.vllm_llm.environment_guard import run
    return run(s)


def execute(s):
    preflight(s)  # CPU/file checks only; guard owns the first VM command.
    if s.attempt('GUARD', lambda: environment_guard(s)) is not True:
        if 'GUARD' not in s.rows:
            s.row('GUARD', None, 'NOT-COMPLETE', REGISTERED['GUARD'])
        print('END before timed tranches: environment guard did not pass', flush=True)
        return
    s.qualified_c = qualify_c(s)
    if not tranche_a(s):
        print('END after A: '+s.rows['A.stop']['value']['decision'], flush=True)
        for name in ('B.keep', 'B.stop', 'C.fusion', 'C.body', 'C.NO_TOUCH', 'C.ISOLATE_POLLER',
                     'D.target', 'D.fusion', 'D.system'):
            s.row(name, None, 'NOT-RUN', REGISTERED[name])
        return
    # X5 can change R2. Spend only its bounded isolated probes before S2,
    # never the C1/C5 wall pairs when the route has already closed.
    from examples.vllm_llm import x5
    s.deadline = min(time.monotonic()+120, s.end)
    if s.attempt('X5', lambda: x5.run(s)) is None:
        s.row('X5', None, 'NOT-COMPLETE', REGISTERED['X5'])
    if not second_stop(s):
        print('STOP after A: S2 closes target route', flush=True)
        for name in ('B.keep', 'C.fusion', 'C.body', 'C.NO_TOUCH', 'C.ISOLATE_POLLER', 'D.target', 'D.fusion', 'D.system'):
            s.row(name, None, 'NOT-RUN', REGISTERED[name])
        return
    b = tranche_b(s)
    c, extra = tranche_c(s)
    resets = [dict(id='DEV_RESET', build='reset', enabled=True, kind='reset', flags={'ETX_DEV_RESET':'1'})] if s.rows.get('RESET.keep', {}).get('status') == 'KEEP' else []
    tranche_d(s, b+c+resets, extra)


def vllm_entry(args):
    import torch
    if torch.cuda.device_count() != 1 or 'gfx942' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('one visible MI300X gfx942 required')
    print('HARDWARE '+str(torch.cuda.get_device_name(0))+' ROCm='+str(torch.version.hip), flush=True)
    from examples.llm import vllm_bench
    sys.argv = ['vllm_bench', *args]
    vllm_bench.main()


def dry_run():
    """Exercise CPU logic, shell syntax and runner registration; never spawn GPU commands."""
    for j in JOBS:
        subprocess.run(['bash', '-n', '-c', j['cmd']], check=True)
    assert MAX_SESSION_SECONDS == 8400
    assert sum(TRANCHE_SECONDS.values()) + ENVIRONMENT_GUARD_SECONDS + REPORT_SECONDS == MAX_SESSION_SECONDS
    c_arms = manifest()['C']
    assert all(a['enabled'] for a in manifest()['B']+c_arms)
    assert sum(a['budget_s'] for a in c_arms) == TRANCHE_SECONDS['C']
    assert all(a['estimated_required_s'] <= a['budget_s'] for a in c_arms)
    assert keep([.030, .031, .030])['keep']
    assert not keep([.030, -.001, .030])['keep']
    try:
        attention_price(dict(phases=[]))
    except ValueError:
        pass
    else:
        raise AssertionError('missing critical-body coverage must refuse')
    assert len({a['id'] for a in manifest()['B']+manifest()['C']}) == len(manifest()['B']+manifest()['C'])
    print(json.dumps(dict(budgets_s=dict(TRANCHE_SECONDS, guard=ENVIRONMENT_GUARD_SECONDS, D='remaining session time (>=1910s)', report=REPORT_SECONDS),
                          manifest=manifest(), thresholds=REGISTERED), indent=2))
    from etx.tools.gpu_jobs import main as runner
    return runner([str(Path(__file__)), '--dry-run'])


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == '--vllm': vllm_entry(args[1:]); return 0
    if args and args[0] == '--analyze':
        if len(args) != 4: raise ValueError('--analyze PLAN RAW OUTPUT required')
        from etx.tools import inbody
        print('CPU ANALYSIS begin: lossless edge/body records', flush=True)
        result = edge_prices(json.loads(Path(args[1]).read_text()), inbody.read_raw(args[2]))
        # Load analysis is independent: absent load stamps must not erase G.
        try:
            plan = json.loads(Path(args[1]).read_text())
            on = inbody.read_raw(args[2])
            loads = inbody.load_phases(plan, on)
            off_path = Path(args[2]).with_name('stamps_load_off.bin')
            if off_path.is_file():
                off_report = inbody.analyze(plan, inbody.read_raw(off_path))
                on_report = result['inbody']
                critical = set(off_report['critical_task_ids']) | set(on_report['critical_task_ids'])
                changes = [abs(on_report['tasks'][i]['body_us']-off_report['tasks'][i]['body_us'])
                           for i in critical if on_report['tasks'][i]['phase'] in inbody.LOAD_ROLES]
                if not changes: raise ValueError('missing load-phase perturbation population')
                loads['perturbation'] = dict(sampled_bodies=len(changes), max_abs_body_change_us=max(changes),
                    pass_le_0_5_us=max(changes) <= .5, basis='one matched snapshot ON/OFF, same stamp executable; original/OFF spread remains [U]')
                loads['validity'] = 'NOT-COMPLETE: original/OFF spread not established' if max(changes) <= .5 else 'INVALID: load stamps exceed 0.5 us/body'

        except ValueError as exc:
            loads = dict(status='NOT-COMPLETE', reason=str(exc))
        result['load_phases'] = loads
        output = Path(args[3])
        output.write_text(json.dumps(result, indent=2)+'\n')
        # The parent must not parse/rewrite the large lossless report after
        # its subprocess deadline. Its small decision view is emitted HERE,
        # under the same deadline as all raw analysis and serialization.
        summary = dict(critical_pools=result['critical_pools'], floor_us=result['floor_us'], gap_atoms=result['gap_atoms'],
                       basis=result['basis'], load_phases=result['load_phases'], inbody=dict(phases=result['inbody']['phases']))
        output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print('CPU ANALYSIS complete', flush=True)
        return 0
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--template', type=Path)
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--run', action='store_true')
    g.add_argument('--prepare', type=Path)
    g.add_argument('--prewarm', type=Path)
    a = ap.parse_args(args)
    if a.template: a.template.write_text(json.dumps(template(), indent=2)+'\n'); return 0
    if a.dry_run: return dry_run()
    if a.prepare: prepare(a.prepare); return 0
    if a.prewarm: prewarm(a.prewarm); return 0
    s = Session(dict(manifest=str(MANIFEST)), os.environ['ETX_SESS_LOGDIR'], float(os.environ.get('ETX_SESS_REMAINING_SECONDS', MAX_SESSION_SECONDS)))
    try:
        s.c = json.loads(Path(os.environ.get('ETX_QWEN300_CONFIG', ROOT/'qwen300-hour.json')).read_text())
        s.m = manifest(s.c.get('manifest', MANIFEST))
        execute(s)
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError, Refusal, subprocess.SubprocessError) as exc:
        s.arms['session'] = dict(status='NOT-COMPLETE', reason=str(exc))
        print('SESSION NOT-COMPLETE '+str(exc), flush=True)
    finally:
        s.finish()
    return 0 if s.arms.get('D', {}).get('status') == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
