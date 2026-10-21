"""Qwen B2/B4/B8 functional qualification, without a performance verdict.

All GPU commands use pre-staged local assets. No download or provisioning.
See docs/glm/QWEN-BATCH-GLM53.md for reference preparation and session commands.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from etx.session import proc
from examples.llm.vllm_bench import read_golden
from examples.vllm_llm import import_certificate as cert

ROOT = Path(__file__).resolve().parents[2]
BATCHES = (2, 4, 8)
CONTEXT, GEN, REPEAT = 1024, 32, 2


def golden_names(batch):
    return ['golden_c1024.txt'] + [f'golden_c1024_r{97*r}.txt' for r in range(1, batch)]


def recipe(batch):
    if batch not in BATCHES:
        raise ValueError('batch must be 2, 4 or 8')
    # Replication is already implemented. Do not enable/tune column batching.
    return dict(ETX_VL_REQUESTS=str(batch), ETX_VL_BATCH='0', ETX_VL_NB='1',
                ETX_VL_CHAIN='0', ETX_VL_CHECK='1', ETX_DEV_RESET='0',
                ETX_VL_MAXLEN='1072')


def clean_env(batch):
    keep = {'ETX_DEVLIBS', 'ETX_LLVM', 'ETX_ROCM_HEADERS', 'ETX_TORCH_HEADERS'}
    env = {k: v for k, v in os.environ.items()
           if (not k.startswith(('ETX_', 'DSL_ABLATE_')) or k in keep)
           and k not in ('CFLAGS', 'HIPCC_COMPILE_FLAGS_APPEND')}
    return dict(env, **recipe(batch), ETX_VL_PROGRESS='64', HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')


def inputs(model, batch, *, hash_weights=True):
    """Validate the staged assets. hash_weights=False keeps the 60 GiB blob out of structure-only preflight."""
    config = model / 'hf/config.json'
    c = json.loads(config.read_text())
    reference = json.loads((ROOT / 'examples/llm/configs/qwen3-30b-a3b.json').read_text())
    for k in ('model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
              'num_key_value_heads', 'head_dim', 'num_experts', 'num_experts_per_tok',
              'moe_intermediate_size', 'vocab_size'):
        if c.get(k) != reference.get(k):
            raise ValueError(f'not the Qwen3-30B-A3B shape: {k}')
    revision = (model / 'hf/REVISION').read_text().strip()
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('hf/REVISION must pin a checkpoint commit')
    prompts = []
    paths = [config, model / 'hf/REVISION', model / 'vl.manifest']
    for name in golden_names(batch):
        path = model / name
        prompt, gen = read_golden(path)
        if len(prompt) != CONTEXT or len(gen) != GEN:
            raise ValueError(f'{name}: expected 1024 prompt / 32 greedy tokens')
        if any(t < 0 or t >= c['vocab_size'] for t in prompt + gen):
            raise ValueError('token outside vocabulary')
        prompts.append(tuple(prompt))
        prefill = model / name.replace('golden', 'prefill', 1)
        rows = prefill.read_text().splitlines()
        am = list(map(int, rows[0].split()[1:]))
        margins = list(map(float, rows[1].split()[1:]))
        if (len(am) < CONTEXT + GEN - 1 or len(am) != len(margins)
                or any(t < 0 or t >= c['vocab_size'] for t in am)
                or any(not math.isfinite(m) or m < 0 for m in margins)):
            raise ValueError('incomplete HF prefill reference')
        paths.extend([path, prefill])
    if len(set(prompts)) != batch:
        raise ValueError('every request needs a distinct prompt')
    if not (model / 'vl.bin').is_file():
        raise ValueError('missing pre-packed vl.bin')
    # Hash the packed weights too: a changed blob must not reuse a receipt.
    files = {str(p.relative_to(model)): cert.sha(p) for p in paths}
    if hash_weights:
        files[str((model / 'vl.bin').relative_to(model))] = cert.sha(model / 'vl.bin')
    else:
        files[str((model / 'vl.bin').relative_to(model))] = dict(bytes=(model / 'vl.bin').stat().st_size)
    return dict(model='Qwen/Qwen3-30B-A3B', revision=revision, batch=batch,
                context=CONTEXT, gen=GEN, files=files)


def execution_package(model=None):
    """Offline manifest, with optional complete binding to pre-staged assets."""
    sources = ['examples/sessions/qwen_batch_correctness.py',
               'examples/vllm_llm/batch_correctness.py', 'examples/vllm_llm/build.sh',
               'examples/vllm_llm/host.hip', 'examples/vllm_llm/model.py',
               'examples/vllm_llm/import_certificate.py', 'etx/codegen/kernel.py',
               'examples/vllm_llm/batch_local.sh', 'examples/vllm_llm/batch_preflight.sh',
               'examples/llm/configs/qwen3-30b-a3b.json']
    required = ['hf/config.json', 'hf/REVISION', 'vl.bin', 'vl.manifest']
    required += golden_names(8)
    required += [name.replace('golden', 'prefill', 1) for name in golden_names(8)]
    record = dict(schema=1, batches=list(BATCHES), context=CONTEXT, gen=GEN, repeat=REPEAT,
                  required_assets=required, recipes={str(b): recipe(b) for b in BATCHES},
                  sources={name: cert.sha(ROOT / name) for name in sources},
                  asset_status='UNBOUND', gpu_correctness='unmeasured', performance='unmeasured')
    if model is not None:
        record['inputs'] = {str(b): inputs(model, b) for b in BATCHES}
        record['asset_status'] = 'BOUND'
    return record


def host_args(model, build, batch):
    args = [str(build / 'run'), str(model), str(build / 'vl_plan.txt'),
            '--gen', str(GEN), '--repeat', str(REPEAT)]
    for r, name in enumerate(golden_names(batch)):
        args.extend(['--golden' + (str(r) if r else ''), name])
    return args


def check_tokens(text, batch, *, coverage=False):
    rows = re.findall(r'^run (\d+) request (\d+): (\d+)/(\d+) generated tokens match HF greedy(.*)$', text, re.M)
    expected = [(str(run), str(r), '32', '32') for run in range(REPEAT) for r in range(batch)]
    if [r[:4] for r in rows] != expected:
        raise ValueError('missing, duplicate, wrong or truncated per-request token results')
    if any(('teacher-forced' in r[4]) != coverage for r in rows):
        raise ValueError('wrong decode mode')
    results = re.findall(r'^RESULT .* requests=(\d+) matched=(\d+) gen=(\d+) .* matched_each=([\d,]+)$', text, re.M)
    if results != [(str(batch), '32', '32', ','.join(['32'] * batch))]:
        raise ValueError('missing or wrong RESULT')
    if coverage:
        rows = re.findall(r'^CHECK-COVERAGE OK request (\d+): (\d+) position\(s\), 0 wrong beyond ties, (\d+) tie-tolerated .*$', text, re.M)
        if [(r, n) for r, n, _ in rows] != [(str(r), '1055') for r in range(batch)]:
            raise ValueError('incomplete per-request coverage')
    if re.search(r'FAILED|DIFFERS|^HANG |^ERROR:|stopped after', text, re.M):
        raise ValueError('failed/truncated host output')


def check_identity(text, batch, expected, position):
    """Per-request byte-identity gate for replicated multi-request plans.

    cert.check_output already proves every planned instance ran and compared
    identical in plan order. This additionally proves each request owns the
    same non-empty planned set, so a dropped or duplicated request cannot hide
    behind a globally consistent instance list.
    """
    records = [json.loads(line[len('CHECK-INSTANCE '):]) for line in text.splitlines()
               if line.startswith('CHECK-INSTANCE ')]
    by_req = {r: [] for r in range(batch)}
    for record in records:
        req = record.get('req')
        if type(req) is not int or req not in by_req:
            raise ValueError('CHECK-INSTANCE lacks a valid request id')
        by_req[req].append(record)
    planned = {r: [e['id'] for e in expected if e['req'] == r] for r in range(batch)}
    for r in range(batch):
        if not planned[r]:
            raise ValueError(f'request {r}: plan has no imported byte comparison')
    if len({len(ids) for ids in planned.values()}) != 1:
        raise ValueError('requests have unequal planned imported instance counts')
    for r in range(batch):
        if [rec['id'] for rec in by_req[r]] != planned[r]:
            raise ValueError(f'request {r}: incomplete/duplicate/out-of-order byte comparisons at position {position}')
        if any(rec.get('result') != 'identical' or rec.get('first_byte') != -1
               or rec.get('position') != position or type(rec.get('bytes')) is not int or rec['bytes'] <= 0
               for rec in by_req[r]):
            raise ValueError(f'request {r}: imported bytes differ at position {position}')
    return dict(position=position, requests=batch, instances_per_request=len(planned[0]))


def execute(argv, path, env, seconds):
    """Full logs, deadline, regular progress and a silence probe; kill the entire tree."""
    last = [0.0]
    with path.open('x') as out:
        out.write('# command ' + json.dumps(list(map(str, argv))) + '\n'); out.flush()
        def poll(elapsed):
            if elapsed - last[0] >= 30:
                print(f'BATCH-PROGRESS elapsed_s={elapsed:.0f} bytes={path.stat().st_size} log={path}', flush=True)
                last[0] = elapsed
            if elapsed > 180 and time.time() - path.stat().st_mtime > 180:
                probe = subprocess.run(['ps', '-eo', 'pid,ppid,etime,pcpu,stat,comm'], capture_output=True, text=True, timeout=5)
                out.write(f'\nSILENCE-PROBE exit={probe.returncode}\n' + probe.stdout + probe.stderr); out.flush()
                # A long active compile may legitimately be quiet. Retain the
                # probe and the hard step deadline, rather than misclassifying
                # silence as a correctness failure.
                print(f'BATCH-SILENCE-PROBE elapsed_s={elapsed:.0f} log={path}', flush=True)
        rc = proc.run(argv, sink=out, env=env, cwd=ROOT, deadline_s=seconds, poll_s=5, on_poll=poll)
    if rc:
        raise ValueError(f'exit {rc}; full log: {path}')
    return path.read_text()


def qualify(model, build, out, batch):
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=1, status='NOT-COMPLETE', batch=batch, performance='unmeasured')
    try:
        before = inputs(model, batch)
        identity = cert.verified_build(build, model)
        overrides = identity['overrides']
        if not identity['check_enabled'] or any(overrides.get(k) != v for k, v in recipe(batch).items()):
            raise ValueError('build does not use the fixed correctness recipe')
        expected = cert.expected_instances((build / 'vl_plan.txt').read_text(), json.loads((build / 'imp/imports.json').read_text()))
        env = clean_env(batch)
        args = host_args(model, build, batch)
        identity_gate = {}
        for position in (0, 1040):
            raw = execute(args + ['--check', '48', '--check-pos', str(position)], out / f'check_p{position}.log', env, 900)
            tag = cert.digest(identity)
            cert.check_output(f'BUILD-IDENTITY phase=before sha256={tag}\n' + raw + f'\nBUILD-IDENTITY phase=after sha256={tag}\n', expected, position, identity)
            identity_gate[position] = check_identity(raw, batch, expected, position)
        for mode, extra in [('coverage', ['--check-coverage']), ('fused', []), ('unfused', ['--unfused'])]:
            raw = execute(args + extra, out / f'{mode}.log', env, 900)
            check_tokens(raw, batch, coverage=mode == 'coverage')
        if inputs(model, batch) != before or cert.verified_build(build, model) != identity:
            raise ValueError('assets changed during qualification')
        receipt.update(status='PASS', inputs=before, build_identity=identity,
                       imported_per_position=len(expected), positions=[0, 1040],
                       byte_identity_per_request={str(k): v for k, v in identity_gate.items()},
                       greedy_tokens_per_mode=batch * GEN * REPEAT, modes=['fused', 'unfused'],
                       coverage='HF margin >0.5 gate; exact 32/32 tail in every repetition')
    except Exception as exc:
        receipt['error'] = str(exc)
        raise
    finally:
        (out / 'correctness.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation', choices=['commands', 'build', 'qualify'])
    ap.add_argument('--batch', type=int, choices=BATCHES, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.model, a.build, a.out = (p.resolve() for p in (a.model, a.build, a.out))
    if a.operation == 'commands':
        print(json.dumps(dict(schema=1, recipe=recipe(a.batch), performance='unmeasured',
                              host=host_args(a.model, a.build, a.batch), positions=[0, 1040]), indent=2))
    elif a.operation == 'build':
        inputs(a.model, a.batch)
        a.out.mkdir(parents=True, exist_ok=True)
        execute(['bash', 'examples/vllm_llm/build.sh', str(a.model)], a.out / 'build.log',
                dict(clean_env(a.batch), ETX_OUT=str(a.build)), 1800)
    else:
        qualify(a.model, a.build, a.out, a.batch)
        print(f'BATCH-CORRECTNESS PASS batch={a.batch} performance=unmeasured', flush=True)


if __name__ == '__main__':
    main()
