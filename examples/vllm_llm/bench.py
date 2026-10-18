"""Standalone kept ETX benchmark: python3 -m examples.vllm_llm.bench [--batch N] [--recipe kept|plain] [model-dir].

--batch N (1, 2, 4 or 8) builds a plan that serves N concurrent requests by replication (ETX_VL_REQUESTS=N, no column
batching). Request r reads golden_c1024[_r<97 r>].txt and its prefill reference from the model directory. Every
decode step advances all N requests by one token, so tokens_per_s = N * 1000 / median_decode_wall_ms.
--recipe plain (N > 1 only) is the recipe batch_correctness.py qualified on hardware, plus the host-only timing report.
Each batch gets its own build and its own certificate; no gate or number is shared between batches.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from examples.vllm_llm import import_certificate as cert
from etx.session import proc
from examples.vllm_llm.kept_build import BASE_FLAGS, KEPT_ENV, ROOT, source_hashes
from examples.vllm_llm.nvref_report import correctness
from examples.vllm_llm.pairs import Refusal, parse

DEFAULT_MODEL = Path.home() / 'llm/qwen3-30b-a3b'


BATCHES = (1, 2, 4, 8)
RECIPES = ('kept', 'plain')
CONTEXT, GEN = 1024, 32
# Host-only: per-step HOUR-WALL lines and the 5 s step watchdog; no device code changes.
REPORT_FLAGS = '-DETX_INBODY_REPORT'
# Hardware-measured lower bounds (session 20261017T140801Z-988ffe, plain recipe) with margin: one
# position's byte check took about 1 / 3 / 11 min at B2 / B4 / B8, the 2-repeat runs about 1 min.
CHECK_TIMEOUT = {1: 180, 2: 600, 4: 900, 8: 1800}
RUN_TIMEOUT = {1: 180, 2: 600, 4: 600, 8: 900}


def recipe(batch, name='kept'):
    """Build overrides for one batch. B1 kept is the unchanged kept build."""
    if batch not in BATCHES:
        raise Refusal(f'batch must be one of {BATCHES}')
    if name not in RECIPES:
        raise Refusal(f'recipe must be one of {RECIPES}')
    if name == 'plain':
        if batch == 1:
            raise Refusal('the plain recipe is the multi-request correctness recipe; batch 1 uses kept')
        from examples.vllm_llm.batch_correctness import recipe as qualified
        return dict(qualified(batch)), REPORT_FLAGS
    env = dict(KEPT_ENV)
    if batch > 1:
        # Replication: every request its own launches; no column batching, no chaining.
        env.update(ETX_VL_REQUESTS=str(batch), ETX_VL_BATCH='0', ETX_VL_NB='1', ETX_VL_CHAIN='0')
    return env, BASE_FLAGS


def environment(build, batch=1, name='kept'):
    # Keep toolchain paths, but clear inherited experiments and diagnostics.
    tools = {'ETX_DEVLIBS', 'ETX_ROCM_HEADERS', 'ETX_TORCH_HEADERS'}
    env = {k: v for k, v in os.environ.items()
           if (not k.startswith(('ETX_', 'DSL_ABLATE_', 'BF16_')) or k in tools)}
    overrides, flags = recipe(batch, name)
    env.update(overrides, ETX_VL_CHECK='1', ETX_OUT=str(build), ETX_HIPCC_FLAGS=flags,
               ETX_VL_PROGRESS='1', PYTHONUNBUFFERED='1')
    return env


def command(argv, log, env, timeout):
    """Full logs, bounded subprocess groups and visible progress outside timers."""
    print(f'BENCH progress: {log.stem}; log={log}', file=sys.stderr, flush=True)
    with log.open('x') as out:
        out.write('# command ' + json.dumps(list(map(str, argv))) + '\n'); out.flush()
        rc = proc.run(argv, sink=out, deadline_s=timeout, poll_s=15, cwd=ROOT, env=env, log=out,
                      on_poll=lambda s: print(f'BENCH progress: {log.stem} elapsed={s:.0f}s bytes={log.stat().st_size}',
                                              file=sys.stderr, flush=True))
        if rc == proc.DEADLINE_RC:
            raise Refusal(f'{log.stem}: timeout; full log={log}')
        if rc:
            raise Refusal(f'{log.stem}: exit {rc}; full log={log}')
    return log.read_text()


def prerequisites(model, batch=1):
    if not Path('/dev/kfd').exists():
        raise Refusal('no AMD GPU available (/dev/kfd missing); run on an MI300X ROCm box')
    for tool in ('hipcc', 'rocm-smi', 'stdbuf', 'bash', 'git'):
        if shutil.which(tool) is None:
            raise Refusal(f'missing prerequisite: {tool}')
    if not (ROOT / '.venv/bin/python').is_file():
        raise Refusal('missing .venv/bin/python; run make setup in the ROCm environment')
    for name in ('hf/config.json', 'vl.bin', 'vl.manifest', 'golden_c1024.txt'):
        path = model / name
        if not path.is_file() or path.stat().st_size == 0:
            raise Refusal(f'missing input: {path}')
    config = json.loads((model / 'hf/config.json').read_text())
    shape = dict(model_type='qwen3_moe', num_hidden_layers=48, hidden_size=2048,
                 num_key_value_heads=4, num_experts=128, num_experts_per_tok=8)
    if any(config.get(k) != v for k, v in shape.items()):
        raise Refusal('requires staged Qwen3-30B-A3B model')
    if config.get('torch_dtype', config.get('dtype')) != 'bfloat16' or config.get('quantization_config'):
        raise Refusal('requires bf16, unquantized model')
    from examples.llm.vllm_bench import read_golden
    prompt, tokens = read_golden(model / 'golden_c1024.txt')
    if len(prompt) != 1024 or len(tokens) != 32:
        raise Refusal('golden requires 1024 prompt tokens and 32 independent HF greedy tokens')
    if batch > 1:
        prompts = []
        for name in cert.request_goldens(batch):
            path = model / name
            prefill = model / name.replace('golden', 'prefill', 1)
            for p in (path, prefill):
                if not p.is_file() or p.stat().st_size == 0:
                    raise Refusal(f'missing input: {p} (examples/llm/prep.py --context 1024 --rotate N --golden-only)')
            prompt, tokens = read_golden(path)
            if len(prompt) != CONTEXT or len(tokens) != GEN:
                raise Refusal(f'{name}: requires 1024 prompt tokens and 32 HF greedy tokens')
            prompts.append(tuple(prompt))
        if len(set(prompts)) != batch:
            raise Refusal('every request needs a distinct prompt')
    vllm = Path(os.environ.get('VLLM', str(Path.home() / 'vllm')))
    if not (vllm / 'csrc').is_dir():
        raise Refusal(f'missing vLLM kernel sources: {vllm}/csrc (set VLLM)')


def batch_correctness(text, batch):
    """Every request 32/32 free-running in the one run, and the host's RESULT agrees."""
    rows = re.findall(r'^run (\d+) request (\d+): (\d+)/(\d+) generated tokens match HF greedy(.*)$', text, re.M)
    if [r[:4] for r in rows] != [('0', str(r), '32', '32') for r in range(batch)] or any(r[4] for r in rows):
        raise Refusal(f'requires exactly one free-running 32/32 token verdict per request ({batch} requests)')
    results = re.findall(r'^RESULT .* requests=(\d+) matched=(\d+) gen=(\d+) .* matched_each=([\d,]+)$', text, re.M)
    if results != [(str(batch), '32', '32', ','.join(['32'] * batch))]:
        raise Refusal('missing or wrong RESULT for the batch')
    if 'no token check' in text or re.search(r'^CHECK(?:-DSL)? FAILED|FAILED|DIFFERS|^HANG |^ERROR:', text, re.M):
        raise Refusal('incomplete run or failed correctness check')


def coverage(text, batch):
    """Teacher-forced argmax at every one of the 1055 positions of every request: 0 wrong beyond HF ties."""
    rows = re.findall(r'^CHECK-COVERAGE OK request (\d+): (\d+) position\(s\), 0 wrong beyond ties, (\d+) tie-tolerated .*$', text, re.M)
    if [(r, n) for r, n, _ in rows] != [(str(r), str(CONTEXT + GEN - 1)) for r in range(batch)]:
        raise Refusal('incomplete per-request coverage')
    if re.search(r'^CHECK-COVERAGE FAILED|^HANG |^ERROR:', text, re.M):
        raise Refusal('failed coverage')
    return [int(t) for _, _, t in rows]


def decode_samples(text, batch=1):
    if batch == 1:
        correctness(text, 'bench')
        if re.findall(r'^run (\d+): (\d+)/(\d+) generated tokens match HF greedy$', text, re.M) != [('0', '32', '32')]:
            raise Refusal('requires exactly one free-running 32/32 token verdict')
        # --check-dsl needs a one-request plan (host.hip); batches rely on per-request bytes and coverage.
        if not re.search(r'^CHECK-DSL OK\b', text, re.M):
            raise Refusal('missing CHECK-DSL OK')
    else:
        batch_correctness(text, batch)
    runs = parse(text.splitlines()).runs
    if len(runs) != 1 or runs[0].arm != 'fused':
        raise Refusal('requires ordinary fused ETX timing')
    rows = [json.loads(line.split(' ', 1)[1]) for line in text.splitlines() if line.startswith('HOUR-WALL ')]
    if [(r['run'], r['step']) for r in rows] != [(0, t) for t in range(1024, 1055)]:
        raise Refusal('incomplete decode timing: requires 31 post-first-token steps')
    xs = [r['ms'] for r in rows]
    if any(not math.isfinite(x) or x <= 0 for x in xs):
        raise Refusal('invalid decode timing')
    return xs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', nargs='?', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--batch', type=int, choices=BATCHES, default=1, help='concurrent requests (replicated)')
    ap.add_argument('--recipe', choices=RECIPES, default='kept')
    ap.add_argument('--json', type=Path, default=None, help='also write the result record here')
    args = ap.parse_args(argv)
    batch = args.batch
    tokens = GEN * batch
    checked, planned, matched = 0, None, 0
    printed_identity = False
    try:
        model = args.model.expanduser().resolve()
        recipe(batch, args.recipe)
        prerequisites(model, batch)
        parent = ROOT / 'build/bench'; parent.mkdir(parents=True, exist_ok=True)
        build = Path(tempfile.mkdtemp(prefix=f'{args.recipe}-b{batch}-' if batch > 1 else 'kept-', dir=parent))
        env = environment(build, batch, args.recipe)
        hardware = command(['rocm-smi', '--showproductname'], build/'hardware.log', env, 15)
        if 'MI300X' not in hardware:
            raise Refusal('no MI300X available; see hardware.log')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        hashes = source_hashes()
        command(['bash', ROOT/'examples/vllm_llm/build.sh', model], build/'build.log', env, 900 if batch == 1 else 2100)
        identity = cert.verified_build(build, model)
        if cert.plan_requests(build) != batch:
            raise Refusal(f'plan serves {cert.plan_requests(build)} request(s), expected {batch}')
        overrides, flags = recipe(batch, args.recipe)
        if not identity['check_enabled'] or identity['hipcc_flags'] != flags \
                or any(identity['overrides'].get(k) != v for k, v in overrides.items()):
            raise Refusal('build does not use the selected recipe')
        conditions = dict(gpu='MI300X', batch=batch, context=CONTEXT, greedy_tokens=GEN)
        if batch > 1:
            conditions['recipe'] = args.recipe
        record = dict(source_commit=commit, source_hashes=hashes, kept_flags=KEPT_ENV if batch == 1 else overrides,
                      hipcc_flags=identity['hipcc_flags'],
                      hipcc_command=json.loads((build/'hipcc_command.json').read_text()),
                      build_identity=identity, build_sha256=cert.digest(identity),
                      conditions=conditions)
        (build/'bench_identity.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
        print('BUILD-IDENTITY '+json.dumps(record, sort_keys=True), flush=True)
        printed_identity = True
        archive = build/'certificate'
        cert.export_build(build, model, archive)
        expected = cert.expected_instances((build/'vl_plan.txt').read_text(),
                                          json.loads((build/'imp/imports.json').read_text()))
        planned = len(expected)*2
        def host(name, flags, timeout):
            return command([ROOT/'.venv/bin/python', '-u', '-m', 'examples.vllm_llm.import_certificate',
                            '--run-build', build, '--model', model, '--', *flags], build/(name+'.log'), env, timeout)
        per_request = {}
        for pos in (0, 1040):
            text = host(f'check_p{pos}', ['--check', '48', '--check-pos', str(pos)], CHECK_TIMEOUT[batch])
            (archive/f'check_p{pos}.txt').write_text(text)
            cert.check_output(text, expected, pos, identity)
            if batch > 1:
                from examples.vllm_llm.batch_correctness import check_identity
                per_request[str(pos)] = check_identity(text, batch, expected, pos)
            checked += len(expected)
        cert.validate(archive, timing=False)
        ties = None
        if batch > 1:
            text = host('coverage', ['--repeat', '1', '--check-coverage'], RUN_TIMEOUT[batch])
            cert.stamped(text, identity)
            ties = coverage(text, batch)
        text = host('decode', ['--repeat', '1'] + (['--check-dsl'] if batch == 1 else []), RUN_TIMEOUT[batch])
        cert.stamped(text, identity)
        if batch == 1:
            match = re.search(r'^run 0: (\d+)/32 generated tokens match HF greedy', text, re.M)
            matched = int(match[1]) if match else 0
        else:
            matched = sum(int(m) for m in re.findall(r'^run 0 request \d+: (\d+)/32 generated tokens match HF greedy', text, re.M))
        xs = decode_samples(text, batch)
        if cert.verified_build(build, model) != identity or source_hashes() != hashes:
            raise Refusal('source/build identity changed during benchmark')
        ms = statistics.median(xs); spread = max(xs)-min(xs); tps = batch*1000/ms
        if batch == 1:
            print(f'CORRECTNESS PASS tokens={matched}/32 imports={checked}/{planned} '
                  f'({len(expected)}/{len(expected)} at each position 0,1040; byte-identical host --check)', flush=True)
            print(f'TIMING median_decode_wall_ms={ms:.6f} tokens_per_s={tps:.2f} '
                  f'spread_ms={spread:.6f} spread=max-min steps={len(xs)}', flush=True)
        else:
            print(f'CORRECTNESS PASS batch={batch} recipe={args.recipe} tokens={matched}/{tokens} (32/32 each of {batch} requests) '
                  f'imports={checked}/{planned} ({len(expected) // batch}/request at each position 0,1040; byte-identical host --check) '
                  f'coverage={batch}x{CONTEXT + GEN - 1} positions, 0 wrong beyond ties, ties={",".join(map(str, ties))}', flush=True)
            print(f'TIMING batch={batch} recipe={args.recipe} median_decode_wall_ms={ms:.6f} tokens_per_s={tps:.2f} '
                  f'ms_per_token={ms / batch:.6f} spread_ms={spread:.6f} spread=max-min steps={len(xs)}', flush=True)
        result = dict(schema=1, status='PASS', batch=batch, recipe=args.recipe, build=str(build),
                      build_sha256=cert.digest(identity), source_commit=commit, conditions=conditions,
                      measured='host wall time of one decode step (all requests advance one token), steps 1024-1054',
                      median_decode_wall_ms=ms, tokens_per_s=tps, ms_per_token=ms / batch, spread_ms=spread,
                      samples_ms=xs, tokens_matched=matched, tokens_expected=tokens,
                      imports_checked=checked, imports_planned=planned, positions=[0, 1040],
                      byte_identity_per_request=per_request or None, coverage_ties=ties)
        (build/'bench_result.json').write_text(json.dumps(result, indent=2)+'\n')
        if args.json:
            args.json.write_text(json.dumps(result, indent=2)+'\n')
        print(f'BENCH PASS {ms:.6f} {tps:.2f}' + (f' batch={batch} recipe={args.recipe}' if batch > 1 else ''), flush=True)
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        reason = ' '.join(str(exc).split())
        if not printed_identity:
            print('BUILD-IDENTITY FAIL unavailable: '+reason, flush=True)
        print(f'CORRECTNESS FAIL tokens={matched}/{tokens} imports={checked}/{planned or "unknown"} reason={reason}', flush=True)
        if args.json:
            args.json.write_text(json.dumps(dict(schema=1, status='FAIL', batch=batch, recipe=args.recipe,
                                                 reason=reason, timing=None), indent=2)+'\n')
        print('BENCH FAIL '+reason, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
