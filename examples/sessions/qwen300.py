"""Qwen3-30B-A3B bf16 MoE phase pricing and the KEY32 decision.

One MI300X, batch-1, ctx1024; every configuration stays bf16.

CPU inspection: python3 -m examples.sessions.qwen300 --dry-run
Launch (after push): tools/session.sh start IP examples/sessions/qwen300.py 0 A
No provisioning, billing API, GPU access or image pull on import / dry-run.

Two serial jobs: bf16 MoE phase pricing, then the KEY32 decision.
Job 2 runs last even if Job 1 fails. Nothing is silently dropped.
Retries in the same log directory rerun every step. Before writing a log,
move its previous contents to log_history/<log stem>/attempt<N>/<filename>,
using the first unused per-log attempt directory. The canonical path holds the
latest full, unfiltered log; older logs remain readable under the same session
directory. This also applies to retained compiler logs and compiler_version.txt.
Per-step certificate and routing directories are created with exist_ok=True;
no other step's data is deleted or borrowed.
The job timeouts total 70 minutes.
The runner cap is 110 minutes; VM life is at most 120 minutes including setup.
At the historical $1.99/GPU-hour rate: estimated ceiling $3.98 (not a quote).
The operator must fetch results and delete the VM immediately on completion or
at the allocation deadline. This script neither provisions nor deletes a VM.

Correctness requires 32/32 greedy tokens equal to the HF bf16 reference.
No reduced precision, patched vLLM, donor tuning or checkpoint substitution.
KEY32 is scoped to the shipped unfolded Qwen plan. Section 8's folded
configuration is not qualified by this session; no other-model jobs are included.
The vLLM wall estimator is (generate(32)-generate(1))/31, five warm samples,
identical to examples/llm/vllm_bench.py. Trace timings are separate device data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from etx.session import proc
from examples.vllm_llm.qwen300_config import IMAGE, QWEN300_MAX_SESSION_SECONDS, validate_cap

MAX_SESSION_SECONDS = QWEN300_MAX_SESSION_SECONDS
validate_cap(MAX_SESSION_SECONDS, 'Qwen300')
HF = {'bf16': 'Qwen/Qwen3-30B-A3B'}
GOLDEN = Path('examples/llm/results/golden_c1024_qwen3-30b-a3b.txt')
FLAGS = '-DETX_DSL_WSUM -DETX_DSL_SCALAR -DETX_DSL_GU1 -DETX_DSL_RBATCH'
MODULE = 'examples.sessions.qwen300'
HEADER = '''QWEN300 pre-registration: one MI300X, batch=1, context=1024, greedy=32.
BUDGET estimated 1.5 hours / $2.99; runner HARD STOP 110 minutes;
VM-life HARD STOP 2 hours / $3.98 at estimated $1.99/GPU-hour, setup included.
Delete the VM immediately when finished; runner timeout does not delete a VM.
All configurations require bf16 and 32/32 greedy tokens equal to HF Qwen/Qwen3-30B-A3B.
Job 1: correctness 32/32 before bf16 host-wall timing; <=3.333333 ms meets 300 tok/s.
Then device AITER stage1+stage2 mean over 48 layers and 31 decode steps:
<=21 us/layer: importing may recover >=0.28 ms; 300 remains a multi-week target.
>24 us/layer: MoE swap CLOSED; bf16 must ship its own MoE tile.
(21,24] us/layer: unresolved. Missing stages/coverage: NOT-COMPLETE, never zero.
The recovery is an estimate from the bf16 phase comparison.
Job 2: KEY32 estimate 0.03--0.145 ms; five alternating pinned off/on pairs.
KEEP only if every wall saving >0, median >=0.03 ms and >2*(max-min).
Require fresh same-build import checks at positions 0 and 1040, 32/32 on every run.
Report fused and same-kernel HIP graph, device separately, and Job 1 bf16 vLLM.
No shorter-run numbers; missing/failed arms are explicitly NOT-COMPLETE.
'''
JOBS = [dict(name=f'job{i}_{name}', gpu=0, machine='A', quiet_s=180,
             timeout=seconds,
             cmd=f'stdbuf -oL -eL python3 -u -m {MODULE} --job {i}')
        for i, name, seconds in [(1, 'pricing', 1800), (2, 'key32', 2400)]]


def root():
    return Path(os.environ['ETX_SESS_LOGDIR']).resolve()


def model(precision):
    if precision != 'bf16':
        raise ValueError('Qwen300 requires bf16')
    return Path.home() / 'llm' / 'qwen3-30b-a3b'


def golden(precision):
    return root() / f'{precision}_golden.txt'


def condition(precision, basis):
    return f'Qwen3-30B-A3B precision={precision} GPU=MI300X GPU_ID={os.environ.get("GPU", "0")} batch=1 context=1024 tokens=32 basis={basis}'


def container(args, seconds, gpu=True):
    home = str(Path.home())
    cmd = ['sudo', 'docker', 'run', '--rm', '--init', '--ipc=host', '--shm-size=16g',
           '--security-opt', 'seccomp=unconfined', '-v', f'{home}:{home}', '-w', str(Path.cwd()),
           '-e', 'PYTHONUNBUFFERED=1', '-e', f'HOME={home}', '-e', f'ETX_SESS_LOGDIR={root()}']
    if gpu:
        cmd += ['--device=/dev/kfd', '--device=/dev/dri', '--group-add=video',
                '-e', f'HIP_VISIBLE_DEVICES={os.environ.get("GPU", "0")}',
                '-e', 'VLLM_ROCM_USE_AITER=1', '-e', 'VLLM_ENABLE_V1_MULTIPROCESSING=0']
    # Bound daemon-owned work independently of the docker client. Never exceed
    # the session deadline even when this substep started near the hard stop.
    seconds = min(seconds, max(1, int(DEADLINE - time.monotonic()) - 10))
    return cmd + ['--entrypoint', 'timeout', IMAGE, '-k', '5', str(seconds),
                  'stdbuf', '-oL', '-eL'] + args


DEADLINE = float('inf')


def prepare_log(path):
    """Archive this log alone before a retry; keep the canonical reader path."""
    if path.exists():
        history = root() / 'log_history' / path.stem
        attempt = 1
        while (history / f'attempt{attempt}').exists():
            attempt += 1
        directory = history / f'attempt{attempt}'
        directory.mkdir(parents=True, exist_ok=True)
        archived = directory / path.name
        path.rename(archived)
        print(f'ARCHIVE {path} -> {archived}', flush=True)
    return path


def step(name, cmd, seconds, env=None):
    """Archive stale logs, then run with unfiltered output and the same bounds."""
    limit = min(seconds, int(DEADLINE - time.monotonic()) - 5)
    if limit <= 0:
        raise RuntimeError(f'{name}: NOT-COMPLETE session deadline')
    path = prepare_log(root() / f'{name}.log')
    print(f'COMMAND {name} deadline={limit}s {shlex.join(cmd)}', flush=True)
    started = time.monotonic()
    with path.open('w') as f:
        f.write(f'# job {name} gpu={os.environ.get("GPU", "0")}\n# {shlex.join(cmd)}\n')
        f.flush()
        def stuck(elapsed):
            print(f'PROGRESS {name} elapsed={elapsed:.0f}s deadline={limit}s full_log={path}', flush=True)
            if time.time() - path.stat().st_mtime > 180:
                print(f'STUCK-STATE {name}', flush=True)
                for probe in (['ps', '-eo', 'pid,ppid,etime,stat,args'], ['rocm-smi']):
                    try:
                        subprocess.run(probe, timeout=8, check=False)
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        print(f'PROBE {exc}', flush=True)
        rc = proc.run(cmd, sink=f, deadline_s=limit, poll_s=15, on_poll=stuck, log=f,
                      env={k: v for k, v in dict(os.environ, **(env or {})).items() if v is not None})
    if rc == proc.DEADLINE_RC:
        raise RuntimeError(f'{name}: NOT-COMPLETE substep timeout')
    print(f'FINISH {name} rc={rc} full_log={path}', flush=True)
    if rc:
        print(path.read_text(errors='replace'), flush=True)
        raise RuntimeError(f'{name}: failed rc={rc}')
    return path.read_text(errors='replace')


def inside(mode, precision, seconds=480):
    return container(['python3', '-u', '-m', MODULE, '--inside', mode, '--precision', precision], seconds)


def download(precision):
    from huggingface_hub import model_info, snapshot_download
    rev = model_info(HF[precision]).sha
    dst = model(precision) / 'hf'
    print(f'DOWNLOAD {HF[precision]} revision={rev} path={dst}', flush=True)
    snapshot_download(HF[precision], revision=rev, local_dir=dst,
                      allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.txt', '*.model'])
    (root() / f'{precision}_revision.json').write_text(json.dumps({'repo': HF[precision], 'revision': rev}))
    print('DOWNLOAD COMPLETE', flush=True)


def vllm_run(precision, mode, *, input_binding=None):
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    prompt_file = Path(input_binding['prompt']) if input_binding else GOLDEN
    golden_file = Path(input_binding['golden']) if input_binding else golden(precision)
    checkpoint = input_binding['hf'] if input_binding else str(model(precision) / 'hf')
    if input_binding:
        print('INPUT-BINDING '+json.dumps(input_binding, sort_keys=True), flush=True)
    prompt = list(map(int, prompt_file.read_text().splitlines()[0].split()[1:]))
    assert len(prompt) == 1024
    kw = dict(model=checkpoint, dtype='bfloat16', trust_remote_code=True,
              max_model_len=1072, gpu_memory_utilization=.85, enforce_eager=False,
              attention_backend='ROCM_AITER_UNIFIED_ATTN', enable_prefix_caching=True)
    print('EXACT-LLM-ARGS ' + json.dumps(kw, sort_keys=True), flush=True)
    print(f'CONDITION {condition(precision, "device trace" if mode == "trace" else "host wall")} '
          f'image={IMAGE} vllm={vllm.__version__} ROCm={torch.version.hip} device={torch.cuda.get_device_name(0)}', flush=True)
    if torch.cuda.device_count() != 1 or 'gfx942' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('requires exactly one visible gfx942 MI300X; inspect recorded product identity')
    llm = LLM(**kw)
    p = [TokensPrompt(prompt_token_ids=prompt)]
    def gen(n):
        print(f'GENERATE begin tokens={n}', flush=True)
        start = time.perf_counter()
        output = llm.generate(p, SamplingParams(temperature=0., max_tokens=n, ignore_eos=True), use_tqdm=False)
        elapsed = (time.perf_counter() - start) * 1000
        tokens = list(output[0].outputs[0].token_ids)
        print(f'GENERATE complete tokens={len(tokens)}', flush=True)
        return elapsed, tokens
    def check(tokens):
        expected = list(map(int, golden_file.read_text().splitlines()[1].split()[1:]))
        if len(expected) != 32 or tokens != expected:
            raise RuntimeError(f'CORRECTNESS FAILED precision={precision}: expected={expected}, actual={tokens}')
        print(f'CORRECTNESS OK 32/32 precision={precision} HF={HF[precision]}', flush=True)
    _, tokens = gen(32)
    check(tokens)  # gate before even starting the timing population
    gen(1)
    if mode == 'trace':
        # Last two bursts are exactly generate(1), generate(32). Same engine,
        # prompt, precision and settings as bar, including prefix caching.
        time.sleep(.25); gen(1)
        time.sleep(.25); _, tokens = gen(32); check(tokens)
        time.sleep(.25)
        print('TRACE COMPLETE 32/32', flush=True)
        return
    full, one = [], []
    for n in range(5):
        dt, tokens = gen(32); check(tokens)
        d1, _ = gen(1)
        full.append(dt); one.append(d1)
        print(f'BENCH-PROGRESS sample={n+1}/5 correctness=32/32', flush=True)
    costs = [(a-b)/31 for a, b in zip(full, one)]
    if min(costs) <= 0:
        raise RuntimeError('nonpositive wall sample')
    data = dict(precision=precision, basis='host wall', batch=1, context=1024, tokens=32,
                image=IMAGE, rocm=torch.version.hip, gpu=torch.cuda.get_device_name(0),
                input_binding=input_binding, launch=kw, full_ms=full, one_ms=one, per_token_ms=costs,
                median_ms=statistics.median(costs), correct=32)
    (root() / f'{precision}_bar.json').write_text(json.dumps(data, indent=2))
    print(f'BAR {condition(precision, "host wall")} median_ms={data["median_ms"]:.6f} '
          f'tok_s={1000/data["median_ms"]:.3f} target300={data["median_ms"] <= 1000/300}', flush=True)


def price_trace(directory, precision):
    from examples.vllm_llm.vllm_profile import load, bursts, stats
    bs = bursts(load(str(directory)))
    if len(bs) < 2:
        raise RuntimeError('trace: missing isolated generate bursts')
    short, long = bs[-2:]
    # CK's Python ck_moe_stage1/2 wrappers launch kernel_moe_gemm (and
    # kernel_moe_gemm_2lds) templated symbols. Attribute ONLY that CK family
    # by execution order: two launches per layer, stage1 then stage2. Preserve
    # every raw name; reject a changed count or a changing symbol pair.
    def label_ck(burst):
        indices = [i for i, r in enumerate(burst) if 'kernel_moe_gemm' in r[2].lower()]
        if not indices:
            return burst
        if len(indices) % 96:
            raise RuntimeError('CK attribution: not exactly two stages per each of 48 layers')
        pairs = [(burst[indices[i]][2], burst[indices[i+1]][2]) for i in range(0, len(indices), 2)]
        if len(set(pairs)) != 1:
            raise RuntimeError('CK attribution: heterogeneous stage symbols; manual analysis required')
        labelled = list(burst)
        for n, i in enumerate(indices):
            start, end, name = burst[i]
            labelled[i] = (start, end, f'ck_moe_stage{n%2+1} ORDER-INFERRED ' + name)
        return labelled
    raw_short, raw_long = short, long
    short, long = label_ck(short), label_ck(long)
    (root()/f'{precision}_trace_symbols.json').write_text(json.dumps(
        {'short': sorted({r[2] for r in raw_short}), 'long': sorted({r[2] for r in raw_long}),
         'attribution': 'CK kernel_moe_gemm order within each 48-layer pass; explicit stage symbols otherwise'}, indent=2))
    # CK dispatch can choose another algorithm. Never silently price it as CK.
    def kind(name):
        n = name.lower()
        if 'ck_moe_stage1' in n: return 'stage1'
        if 'ck_moe_stage2' in n: return 'stage2'
        if 'reduce_segments' in n or ('reduce' in n and ('attn' in n or 'attention' in n)): return 'reduce'
        if 'attn' in n or 'attention' in n:
            if not any(x in n for x in ('prefill', 'flash', 'reshape')): return 'attention'
        return None
    rows = []
    for category in ('stage1', 'stage2', 'attention', 'reduce'):
        a = [r for r in short if kind(r[2]) == category]
        b = [r for r in long if kind(r[2]) == category]
        # Either category participates in prefill (48), or decode only (0).
        if len(a) not in (0, 48) or len(b) != len(a) + 31*48:
            raise RuntimeError(f'{precision} {category}: NOT-COMPLETE coverage short={len(a)} long={len(b)} expected difference=1488; no dispatch workaround')
        decode = b[len(a):]
        for layer in range(48):
            samples = [(e-s)/1000 for s, e, _ in decode[layer::48]]
            if len(samples) != 31 or min(samples) <= 0:
                raise RuntimeError('invalid layer samples')
            rows.append(dict(layer=layer, category=category, mean_us=statistics.mean(samples), samples_us=samples))
    result = []
    for layer in range(48):
        r = {x['category']: x['mean_us'] for x in rows if x['layer'] == layer}
        result.append(dict(layer=layer, **r))
        print(f'LAYER {condition(precision, "device kernels us")} layer={layer} {json.dumps(r)}', flush=True)
    total = statistics.mean(r['stage1'] + r['stage2'] for r in result)
    decision = 'IMPORT-CANDIDATE >=0.28ms estimated recovery; multi-week target' if total <= 21 else 'CLOSED: own bf16 MoE tile required' if total > 24 else 'UNRESOLVED'
    s1, b1, _ = stats(short); sn, bn, _ = stats(long)
    print(f'PRICING {condition(precision, "device")} moe_us_per_layer={total:.6f} {decision}', flush=True)
    print(f'CEILING {condition(precision, "device")} span_ms={(sn-s1)/31e6:.6f} busy_ms={(bn-b1)/31e6:.6f} gap_ms={(sn-s1-bn+b1)/31e6:.6f}', flush=True)
    (root() / f'{precision}_pricing.json').write_text(json.dumps(dict(layers=result, samples=rows), indent=2))
    return result


def job1():
    step('image', ['sudo', 'docker', 'pull', IMAGE], 720)
    step('record', ['bash', '-c', f'set -e; rocm-smi --showproductname; hipconfig --version; uname -a; sudo docker inspect {shlex.quote(IMAGE)}'], 30)
    model('bf16').mkdir(parents=True, exist_ok=True)  # host owns parent; container writes checkpoint below it
    step('bf16_download', inside('download', 'bf16', 600), 620)
    # Reuse the repository HF bf16 reference.
    if hashlib.sha256(GOLDEN.read_bytes()).hexdigest() != 'bab2f1fbe12f70179711ec40a215c74710845b73169b1cc6a3a6ca50e0a19484':
        raise RuntimeError('bf16 golden hash mismatch')
    golden('bf16').write_bytes(GOLDEN.read_bytes())
    step('bf16_bar', inside('bench', 'bf16'), 500)
    path = root() / 'trace_bf16'
    cmd = container(['rocprofv3', '--kernel-trace', '--output-format', 'csv', '-d', str(path), '--',
                     'python3', '-u', '-m', MODULE, '--inside', 'trace', '--precision', 'bf16'], 480)
    text = step('trace_bf16', cmd, 500)
    if 'TRACE COMPLETE 32/32' not in text:
        raise RuntimeError('trace correctness missing')
    price_trace(path, 'bf16')


def etx_capture(arm, extra='', env=None, name=None):
    from examples.vllm_llm import import_certificate as cert
    build = root() / f'b_{arm}'
    text = step(name or f'{arm}_run', ['stdbuf', '-oL', '-eL', '.venv/bin/python', '-m',
                'examples.vllm_llm.import_certificate', '--run-build', str(build), '--model', str(model('bf16')),
                '--'] + shlex.split(extra), 180,
                dict(ETX_SYNC='', ETX_VL_PROGRESS='64', ETX_UNFUSED_SYNC=None, ETX_UNFUSED_NOGRAPH=None,
                     ETX_VL_GRAPH=None, ETX_HOST_PROF=None, ETX_TRACE=None, ETX_TRACE_RAW=None,
                     ETX_CHECK_TOPK_DUMP=None) | (env or {}))
    cert.stamped(text, cert.verified_build(build, model('bf16')))
    return text


def correct_etx(text, dsl=False):
    from examples.vllm_llm.nvref_report import correctness
    correctness(text, 'KEY32')
    if dsl and 'CHECK-DSL OK' not in text:
        raise RuntimeError('CHECK-DSL missing')


def routing_gate(directory):
    expected = {f'L{layer}_{suffix}.bin' for layer in range(48) for suffix in ('ids', 'weights')}
    for pos in (0, 1040):
        off, on = (directory / f'routing_{arm}_p{pos}' for arm in ('off', 'on'))
        for path in (off, on):
            if {p.name for p in path.iterdir()} != expected:
                raise RuntimeError(f'ROUTING-AUDIT missing/extra layer outputs at {path}')
        for name in sorted(expected):
            a, b = (off/name).read_bytes(), (on/name).read_bytes()
            if not a or len(a) != 8*8*4 or a != b:
                raise RuntimeError(f'ROUTING-AUDIT mismatch position={pos} {name}')
        print(f'ROUTING-AUDIT OK position={pos} layers=48 replicas=8 ids=identical weights=bit-identical', flush=True)


def job2():
    from examples.vllm_llm import import_certificate as cert
    from examples.vllm_llm.pairs import parse, verdict
    from examples.vllm_llm.tile_ab import trace_bodies
    if not golden('bf16').exists():
        raise RuntimeError('KEY32 NOT-COMPLETE: Job 1 bf16 preparation failed')
    step('etx_env', ['bash', '-c', 'set -e; test -x .venv/bin/python || python3 -m venv --system-site-packages .venv; .venv/bin/python -c "import yaml" || .venv/bin/pip install pyyaml'], 120)
    step('pack', container(['python3', 'examples/vllm_llm/prep.py', '--src', str(model('bf16')/'hf'),
                           '--out', str(model('bf16')), '--maxlen', '1072'], 240), 260)
    (model('bf16') / 'golden_c1024.txt').write_bytes(golden('bf16').read_bytes())
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('ETX_VL_', 'ETX_DSL_', 'ETX_LLM_', 'DSL_ABLATE_'))
             and k not in ('ETX_DEV_RESET', 'ETX_TOPK_KEY32', 'ETX_HIPCC_FLAGS')}
    def build(arm):
        env = dict(clean, ETX_VL_CHECK='1', ETX_TOPK_KEY32=str(int(arm == 'on')),
                   ETX_HIPCC_FLAGS=FLAGS, ETX_OUT=str(root()/f'b_{arm}'),
                   VLLM=str(Path.home()/'vllm'), ETX_TORCH_HEADERS=str(Path.home()/'rocm-headers/pytorch'),
                   ETX_ROCM_HEADERS=str(Path.home()/'rocm-headers/inc'))
        # Clear inherited experimental toggles before selecting the baseline.
        print(f'BUILD-CONDITION arm={arm} KEY32={env["ETX_TOPK_KEY32"]} CHECK=1 flags={FLAGS} fold=shipped-default', flush=True)
        removed = {k: None for k in os.environ if k not in clean}
        try:
            step(f'build_{arm}', ['stdbuf', '-oL', '-eL', 'bash', 'examples/vllm_llm/build.sh', str(model('bf16'))],
                 600, dict(removed, **env))
        finally:
            # session.sh fetch excludes b_*/; retain complete nested compiler
            # logs at the top level even when compilation fails.
            for filename in ('compile.log', 'hipcc.log', 'compiler_version.txt'):
                source = root()/f'b_{arm}'/filename
                if source.exists():
                    prepare_log(root()/f'{arm}_{filename}').write_bytes(source.read_bytes())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(build, ('off', 'on')))
    if (root()/'b_off/vl_plan.txt').read_bytes() != (root()/'b_on/vl_plan.txt').read_bytes():
        raise RuntimeError('KEY32 changed the plan: invalid switch-only comparison')
    for arm in ('off', 'on'):
        certificate = root()/f'certificate_{arm}'
        certificate.mkdir(exist_ok=True)
        cert.export_build(root()/f'b_{arm}', model('bf16'), certificate)
        for pos in (0, 1040):
            audit = root()/f'routing_{arm}_p{pos}'
            audit.mkdir(exist_ok=True)
            text = etx_capture(arm, f'--check 48 --check-pos {pos}',
                               env={'ETX_CHECK_TOPK_DUMP':str(audit)}, name=f'{arm}_check_p{pos}')
            (certificate/f'check_p{pos}.txt').write_text(text)
        print('CERT ' + json.dumps(cert.validate(certificate, timing=False)), flush=True)
        for mode in ('fused', 'graph'):
            extra = '--repeat 1 --check-dsl' if mode == 'fused' else '--repeat 1 --unfused'
            correct_etx(etx_capture(arm, extra, name=f'{arm}_gate_{mode}'), mode == 'fused')
    routing_gate(root())
    # Both builds retain their full raw traces. Timing samples never enable tracing.
    bodies = {}
    for arm in ('off', 'on'):
        text = etx_capture(arm, '--repeat 1', env={'ETX_TRACE':'1', 'ETX_TRACE_RAW':str(root()/f'tr_{arm}.bin')}, name=f'tr_{arm}')
        correct_etx(text)
        bodies[arm] = trace_bodies(root()/f'tr_{arm}.log')
        if not {'topk','attn','reduce','prep','moe_up','moe_down'} <= bodies[arm].keys():
            raise RuntimeError(f'{arm}: required trace bodies missing')
        print(f'BODIES {condition("bf16", "device task body us")} arm={arm} '
              + json.dumps(bodies[arm]), flush=True)
    print(f'BODY-DELTA {condition("bf16", "device task body us")} on-minus-off ' +
          json.dumps({k: bodies['on'][k]-bodies['off'][k] for k in bodies['off'].keys() & bodies['on'].keys()}), flush=True)
    rounds = []
    for rnd in range(1, 6):
        row = {}
        for mode in ('fused', 'graph'):
            for arm in (('off', 'on') if rnd % 2 else ('on', 'off')):
                text = etx_capture(arm, '--unfused' if mode == 'graph' else '', name=f'r{rnd}_{arm}_{mode}')
                correct_etx(text)
                runs = parse(text.splitlines()).runs
                if len(runs) != 1 or runs[0].arm != mode:
                    raise RuntimeError('unexpected timing population')
                row[f'{arm}_{mode}'] = dict(wall=runs[0].wall_ms, device=runs[0].device_ms)
        rounds.append(row)
    savings = [r['off_fused']['wall']-r['on_fused']['wall'] for r in rounds]
    med, spread = statistics.median(savings), max(savings)-min(savings)
    keep = min(savings)>0 and med>=.03 and med>2*spread
    print(f'KEY32 {condition("bf16", "host wall")} scope=shipped-unfolded-plan savings_ms={savings} median={med} spread={spread} KEEP={keep}', flush=True)
    for arm in ('off', 'on'):
        for mode in ('fused', 'graph'):
            for basis in ('wall', 'device'):
                values = [r[f'{arm}_{mode}'][basis] for r in rounds]
                print(f'KEY32-ARM {condition("bf16", basis)} arm={arm} mode={mode} '
                      f'median_ms={statistics.median(values):.6f} rounds_ms={values}', flush=True)
        gains = [(r[f'{arm}_graph']['wall']-r[f'{arm}_fused']['wall'])/r[f'{arm}_graph']['wall'] for r in rounds]
        print(f'FUSION-DIAGNOSTIC (not adjacent fused/graph pairs) arm={arm} {condition("bf16", "host wall")} {json.dumps(verdict(gains))}', flush=True)
    reference = root()/'bf16_bar.json'
    if reference.exists():
        print('CONTEMPORANEOUS-SYSTEM-REFERENCE ' + reference.read_text(), flush=True)
    else:
        print('SYSTEM-REFERENCE NOT-COMPLETE: bf16 vLLM bar failed', flush=True)
    (root()/'key32_pairs.json').write_text(json.dumps(rounds, indent=2))


def dry_run():
    print(HEADER)
    print('ARITHMETIC measured bf16 wall 4.471 ms =>', 1000/4.471, 'tok/s; gap_ms=', 4.471-1000/300)
    print('ARITHMETIC estimated bf16 bytes=6.1904GB; 4.3TB/s utilization=', 6.1904/(4.471*4.3))
    for j in JOBS:
        subprocess.run(['bash', '-n', '-c', j['cmd']], check=True)
    from etx.tools.gpu_jobs import main as runner
    return runner([__file__, '--dry-run'])


def main(argv=None):
    global DEADLINE
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--job', type=int, choices=(1,2))
    mode.add_argument('--inside', choices=('download','bench','trace'))
    ap.add_argument('--precision', choices=('bf16',), default='bf16')
    a = ap.parse_args(argv)
    if a.dry_run:
        return dry_run()
    if a.inside:
        if a.inside == 'download': download(a.precision)
        else: vllm_run(a.precision, a.inside)
        return 0
    print(HEADER, flush=True)
    seconds = min(JOBS[a.job-1]['timeout']-30, int(os.environ['ETX_SESS_REMAINING_SECONDS'])-30)
    DEADLINE = time.monotonic() + seconds
    try:
        (job1, job2)[a.job-1]()
    except Exception as exc:
        print(f'JOB {a.job} NOT-COMPLETE: {type(exc).__name__}: {exc}', flush=True)
        expected = ({1: ['bf16_bar', 'trace_bf16'],
                     2: [f'r{rnd}_{arm}_{mode}' for rnd in range(1,6)
                         for arm in ('off','on') for mode in ('fused','graph')]})[a.job]
        for name in expected:
            if not (root()/f'{name}.log').exists():
                print(f'NOT-RUN {name}: prerequisite failed or deadline exhausted', flush=True)
        return 2
    print(f'JOB {a.job} COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
