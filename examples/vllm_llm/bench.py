"""Standalone kept ETX benchmark: python3 -m examples.vllm_llm.bench [model-dir]."""
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


def environment(build):
    # Keep toolchain paths, but clear inherited experiments and diagnostics.
    tools = {'ETX_DEVLIBS', 'ETX_ROCM_HEADERS', 'ETX_TORCH_HEADERS'}
    env = {k: v for k, v in os.environ.items()
           if (not k.startswith(('ETX_', 'DSL_ABLATE_', 'BF16_')) or k in tools)}
    env.update(KEPT_ENV, ETX_OUT=str(build), ETX_HIPCC_FLAGS=BASE_FLAGS,
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


def prerequisites(model):
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
    vllm = Path(os.environ.get('VLLM', str(Path.home() / 'vllm')))
    if not (vllm / 'csrc').is_dir():
        raise Refusal(f'missing vLLM kernel sources: {vllm}/csrc (set VLLM)')


def decode_samples(text):
    correctness(text, 'bench')
    if re.findall(r'^run (\d+): (\d+)/(\d+) generated tokens match HF greedy$', text, re.M) != [('0', '32', '32')]:
        raise Refusal('requires exactly one free-running 32/32 token verdict')
    if not re.search(r'^CHECK-DSL OK\b', text, re.M):
        raise Refusal('missing CHECK-DSL OK')
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', nargs='?', type=Path, default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    checked, planned, matched = 0, None, 0
    printed_identity = False
    try:
        model = args.model.expanduser().resolve()
        prerequisites(model)
        parent = ROOT / 'build/bench'; parent.mkdir(parents=True, exist_ok=True)
        build = Path(tempfile.mkdtemp(prefix='kept-', dir=parent))
        env = environment(build)
        hardware = command(['rocm-smi', '--showproductname'], build/'hardware.log', env, 15)
        if 'MI300X' not in hardware:
            raise Refusal('no MI300X available; see hardware.log')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        hashes = source_hashes()
        command(['bash', ROOT/'examples/vllm_llm/build.sh', model], build/'build.log', env, 900)
        identity = cert.verified_build(build, model)
        record = dict(source_commit=commit, source_hashes=hashes, kept_flags=KEPT_ENV,
                      hipcc_flags=identity['hipcc_flags'],
                      hipcc_command=json.loads((build/'hipcc_command.json').read_text()),
                      build_identity=identity, build_sha256=cert.digest(identity),
                      conditions=dict(gpu='MI300X', batch=1, context=1024, greedy_tokens=32))
        (build/'bench_identity.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
        print('BUILD-IDENTITY '+json.dumps(record, sort_keys=True), flush=True)
        printed_identity = True
        archive = build/'certificate'
        cert.export_build(build, model, archive)
        expected = cert.expected_instances((build/'vl_plan.txt').read_text(),
                                          json.loads((build/'imp/imports.json').read_text()))
        planned = len(expected)*2
        def host(name, flags):
            return command([ROOT/'.venv/bin/python', '-u', '-m', 'examples.vllm_llm.import_certificate',
                            '--run-build', build, '--model', model, '--', *flags], build/(name+'.log'), env, 180)
        for pos in (0, 1040):
            text = host(f'check_p{pos}', ['--check', '48', '--check-pos', str(pos)])
            (archive/f'check_p{pos}.txt').write_text(text)
            cert.check_output(text, expected, pos, identity)
            checked += len(expected)
        cert.validate(archive, timing=False)
        text = host('decode', ['--repeat', '1', '--check-dsl'])
        cert.stamped(text, identity)
        match = re.search(r'^run 0: (\d+)/32 generated tokens match HF greedy', text, re.M)
        matched = int(match[1]) if match else 0
        xs = decode_samples(text)
        if cert.verified_build(build, model) != identity or source_hashes() != hashes:
            raise Refusal('source/build identity changed during benchmark')
        ms = statistics.median(xs); spread = max(xs)-min(xs)
        print(f'CORRECTNESS PASS tokens={matched}/32 imports={checked}/{planned} '
              f'({len(expected)}/{len(expected)} at each position 0,1040; byte-identical host --check)', flush=True)
        print(f'TIMING median_decode_wall_ms={ms:.6f} tokens_per_s={1000/ms:.2f} '
              f'spread_ms={spread:.6f} spread=max-min steps={len(xs)}', flush=True)
        print(f'BENCH PASS {ms:.6f} {1000/ms:.2f}', flush=True)
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        reason = ' '.join(str(exc).split())
        if not printed_identity:
            print('BUILD-IDENTITY FAIL unavailable: '+reason, flush=True)
        print(f'CORRECTNESS FAIL tokens={matched}/32 imports={checked}/{planned or "unknown"} reason={reason}', flush=True)
        print('BENCH FAIL '+reason, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
