#!/usr/bin/env python3
"""One fail-closed preflight for the two-GPU Qwen and GLM-5.3-Flash session.

Checks, without a GPU and without downloading anything:

  qwen-assets    the packed Qwen3-30B-A3B model for B2/B4/B8: shape, pinned
                 revision, 1024-token prompts, 32-token greedy outputs, prefill
                 references, distinct prompts and the packed vl.bin.
  glm-stage      the staged GLM-5.3-Flash checkpoint: pinned model/revision/config,
                 compiled dimensions, PP2 layer ownership, every tensor descriptor
                 and every blob/meta file size.
  glm-reference  the independent HF greedy record and its 32-step logits file,
                 bound to the same checkpoint index/shards as the stage.

Every failure prints `PREFLIGHT FAIL <item>: <reason>` and the command exits 2.
The default checks are structural and fast; --hash also re-hashes every staged
file (the Qwen vl.bin and the whole GLM stage), which takes minutes on the VM.

  python3 tools/two_gpu_preflight.py \
      --qwen-model ~/llm/qwen3-30b-a3b \
      --glm-stage /staged/glm53 \
      --glm-reference /staged/glm53-reference.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Fail(Exception):
    """A named preflight item is missing, malformed or inconsistent."""


def qwen_assets(path, hash_weights):
    from examples.vllm_llm import batch_correctness as bc

    package = bc.execution_package()
    missing = [name for name in package['required_assets'] if not (path / name).is_file()]
    if missing:
        raise Fail('missing ' + ', '.join(missing))
    if (path / 'vl.bin').stat().st_size == 0:
        raise Fail('vl.bin is empty')
    for batch in bc.BATCHES:
        try:
            bc.inputs(path, batch, hash_weights=hash_weights)
        except Exception as exc:
            raise Fail(f'batch {batch}: {exc}') from exc
    return dict(batches=list(bc.BATCHES), files=len(package['required_assets']),
                weights='hashed' if hash_weights else 'size only')


def glm_stage(path, full_hash, gate='production'):
    from examples.glm53 import stage
    from examples.glm53.prepare import gate_identity, metadata, sha

    evidence, model = gate_identity(gate)
    stage_json = path / 'stage.json'
    if not stage_json.is_file():
        raise Fail(f'missing {stage_json}')
    record = json.loads(stage_json.read_text())
    cfg, api = metadata(evidence)
    if record.get('schema') != 1:
        raise Fail('stage.json schema is not 1')
    if record.get('model') != model or record.get('revision') != api['sha']:
        raise Fail(f'staged model/revision is not the pinned {model.split("/")[-1]} checkpoint ({gate} gate)')
    if record.get('config_sha256') != sha(evidence / 'glm-config.json'):
        raise Fail(f'staged config is not the pinned {gate}-gate GLM config')
    if record.get('fixture'):
        raise Fail('synthetic fixtures cannot qualify the production model')
    if not isinstance(record.get('config'), dict) or not isinstance(record.get('dimensions'), dict) \
            or 'MAXLEN' not in record['dimensions']:
        raise Fail('stage.json has no compiled config/dimensions')
    if not isinstance(record.get('weights'), dict):
        raise Fail('stage.json has no weight table')
    if record['dimensions'].get('BF16') != 1:
        raise Fail('production activations must be BF16')
    if record.get('config') != cfg['text_config']:
        raise Fail('staged text config differs from the pinned config')
    if record['dimensions'] != stage.dimensions(record['config'], record['dimensions']['MAXLEN'], True):
        raise Fail('compiled dimensions are inconsistent with the staged config')
    layers = record['config']['num_hidden_layers']
    if set(record['weights']) != {*(f'l{i}' for i in range(layers)), 'embedding', 'norm', 'head'}:
        raise Fail('staged model is incomplete')
    ranks = record.get('ranks') or []
    owned = [i for rank in ranks for i in rank.get('layers', [])]
    if len(ranks) != 2 or sorted(owned) != list(range(layers)) or len(set(owned)) != layers:
        raise Fail('PP2 layer ownership is invalid')
    spec_file = path / 'glm53_spec.h'
    if not spec_file.is_file():
        raise Fail(f'missing {spec_file.name}')
    if spec_file.read_text() != stage.header(record['dimensions']):
        raise Fail('glm53_spec.h differs from the staged dimensions')
    for name, spec in record['weights'].items():
        for kind in ('blob', 'meta'):
            member = path / spec[kind]
            if not member.is_file():
                raise Fail(f'missing {spec[kind]}')
            if member.resolve().parent != path.resolve():
                raise Fail(f'unsafe staged path {spec[kind]}')
            if member.stat().st_size != spec[f'{kind}_bytes']:
                raise Fail(f'{spec[kind]} is {member.stat().st_size} bytes, '
                           f'stage.json records {spec[f"{kind}_bytes"]}')
        try:
            stage.validate_descriptors(path, name, spec, record['config'])
        except Exception as exc:
            raise Fail(f'{name}: {exc}') from exc
    if full_hash:
        stage.validate(path, gate=gate)
    summary = dict(layers=layers, ranks=[[r['layers'][0], r['layers'][-1]] for r in ranks],
                   payload_bytes=sum(w['blob_bytes'] for w in record['weights'].values()),
                   hashes='verified' if full_hash else 'not recomputed (--hash)')
    return summary, record


def glm_reference(path, staged, expected_batch, gate='production'):
    from examples.glm53 import reference
    from examples.glm53.prepare import gate_identity, metadata

    if not path.is_file():
        raise Fail(f'missing {path}')
    ref = json.loads(path.read_text())
    for key, value in reference.identity(gate).items():
        if ref.get(key) != value:
            raise Fail(f'{key} is {ref.get(key)!r}, expected {value!r}')
    if ref.get('backend') != 'hf':
        raise Fail(f"backend is {ref.get('backend')!r}, expected 'hf'")
    cfg, _ = metadata(gate_identity(gate)[0])
    text = cfg['text_config']
    batch = ref.get('batch')
    if type(batch) is not int or batch not in (1, 2, 4, 8):
        raise Fail('batch must be 1, 2, 4 or 8')
    requests = ref.get('requests')
    if not isinstance(requests, list) or len(requests) != batch:
        raise Fail('requests do not match the batch')
    prompts, lengths = [], set()
    for r, request in enumerate(requests):
        prompt, generated = request.get('prompt_ids'), request.get('generated_ids')
        if not isinstance(prompt, list) or not prompt:
            raise Fail(f'request {r}: missing prompt')
        if not isinstance(generated, list) or len(generated) != 32:
            raise Fail(f'request {r}: expected 32 generated tokens')
        if any(type(t) is not int or not 0 <= t < text['vocab_size'] for t in prompt + generated):
            raise Fail(f'request {r}: token outside the vocabulary')
        if len(prompt) + 32 > text['max_position_embeddings']:
            raise Fail(f'request {r}: prompt does not fit the context')
        prompts.append(tuple(prompt))
        lengths.add(len(prompt))
    if len(set(prompts)) != batch:
        raise Fail('requests do not use distinct prompts')
    if len(lengths) != 1:
        raise Fail('prompts have unequal lengths, which the compiled batch cannot run')
    provenance = ref.get('provenance') or {}
    if provenance.get('quantization') != 'FP8 block dequantize -> BF16':
        raise Fail('reference uses incompatible FP8 numerical semantics')
    if staged is not None:
        snapshot = staged.get('snapshot_inputs') or {}
        if provenance.get('index_sha256') != snapshot.get('index_sha256'):
            raise Fail('reference checkpoint index differs from the staged inputs')
        source = provenance.get('source_shards') or {}
        bad = [name for name, digest in snapshot.get('shards', {}).items() if source.get(name) != digest]
        if bad:
            raise Fail(f'reference source shard identity differs: {bad[0]}')
    if expected_batch is not None and batch != expected_batch:
        raise Fail(f'reference is batch {batch}, but ETX_GLM_BATCH expects {expected_batch}')
    logits = path.with_suffix('.logits')
    if not logits.is_file():
        raise Fail(f'missing {logits}')
    expected_bytes = 32 * batch * text['vocab_size'] * 4
    if logits.stat().st_size != expected_bytes:
        raise Fail(f'{logits.name} is {logits.stat().st_size} bytes, expected {expected_bytes}')
    return dict(batch=batch, prompt=lengths.pop(), steps=32, logits_bytes=expected_bytes,
                logits_mib=round(expected_bytes / 2**20, 1), revision=ref['revision'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--qwen-model', type=Path, help='packed Qwen3-30B-A3B directory')
    ap.add_argument('--glm-stage', type=Path, help='staged GLM-5.3-Flash directory')
    ap.add_argument('--glm-reference', type=Path, help='HF reference .json (its .logits must sit beside it)')
    ap.add_argument('--glm-batch', type=int, choices=[1, 2, 4, 8], help='batch the GLM session will run')
    ap.add_argument('--glm-gate', choices=['production', 'module'], default='production',
                    help='pinned GLM identity: the release, or the synthetic module-gate checkpoint')
    ap.add_argument('--hash', action='store_true', help='also re-hash every staged file (minutes)')
    ap.add_argument('--json', type=Path, help='write the result record here')
    args = ap.parse_args(argv)
    if not (args.qwen_model or args.glm_stage or args.glm_reference):
        ap.error('give --qwen-model, --glm-stage and/or --glm-reference')

    result, failed = {}, 0
    if args.qwen_model:
        try:
            result['qwen-assets'] = qwen_assets(args.qwen_model.resolve(), args.hash)
            print(f"PREFLIGHT OK qwen-assets: batches {','.join(map(str, result['qwen-assets']['batches']))}; "
                  f"{result['qwen-assets']['files']} files each; weights {result['qwen-assets']['weights']}")
        except Exception as exc:
            failed = 1
            print(f'PREFLIGHT FAIL qwen-assets: {exc}')
            result['qwen-assets'] = dict(error=str(exc))

    staged = None
    if args.glm_stage:
        try:
            summary, staged = glm_stage(args.glm_stage.resolve(), args.hash, args.glm_gate)
            print(f"PREFLIGHT OK glm-stage: {summary['layers']} layers, ranks "
                  f"{summary['ranks'][0][0]}-{summary['ranks'][0][1]}/{summary['ranks'][1][0]}-{summary['ranks'][1][1]}, "
                  f"{summary['payload_bytes'] / 2**30:.2f} GiB payload, hashes {summary['hashes']}")
            result['glm-stage'] = summary
        except Exception as exc:
            failed = 1
            print(f'PREFLIGHT FAIL glm-stage: {exc}')
            result['glm-stage'] = dict(error=str(exc))

    if args.glm_reference:
        try:
            if staged is None and args.glm_stage:
                raise Fail('not checked: glm-stage did not validate')
            summary = glm_reference(args.glm_reference.resolve(), staged, args.glm_batch, args.glm_gate)
            print(f"PREFLIGHT OK glm-reference: batch {summary['batch']}, prompt {summary['prompt']}, "
                  f"{summary['steps']} steps, {summary['logits_mib']} MiB logits"
                  + (f"; run the GLM session with ETX_GLM_BATCH={summary['batch']}"))
            result['glm-reference'] = summary
        except Exception as exc:
            failed = 1
            print(f'PREFLIGHT FAIL glm-reference: {exc}')
            result['glm-reference'] = dict(error=str(exc))

    record = dict(schema=1, status='FAIL' if failed else 'OK', hash=args.hash,
                  qwen=result.get('qwen-assets'), stage=result.get('glm-stage'),
                  reference=result.get('glm-reference'))
    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + '\n')
    if failed:
        return 2
    print('PREFLIGHT OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
