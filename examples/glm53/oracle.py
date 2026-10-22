"""Validate same-checkpoint HF versus actual ETX multi-GPU greedy records offline.

Token agreement alone does not certify all full-checkpoint operator numerics.
Record schema: model, revision, config_sha256, batch, mode='greedy',
speculative=false, backend='hf'/an explicit ETX backend, requests=[{prompt_ids, generated_ids}].
"""
import argparse
import json
from pathlib import Path
from examples.glm53.prepare import gate_identity, metadata, sha, check_slice


CANDIDATE_BACKENDS=('etx-pp2', 'etx-pp4', 'etx-pp8', 'etx-ffn-tp', 'etx-tp')


def build_backend(options):
    mode=options.get('parallel',0)
    if options.get('attention_tp',0):
        if mode != 1:raise ValueError('attention TP needs FFN TP')
        return 'etx-tp'
    return 'etx-ffn-tp' if mode==1 else f'etx-pp{options.get("devices",2)}'


def compare(reference, candidate, gate='production', candidate_backend='etx-pp2'):
    if candidate_backend not in CANDIDATE_BACKENDS:
        raise ValueError('unsupported ETX candidate backend')
    evidence, model = gate_identity(gate)
    cfg, api = metadata(evidence)
    identity = dict(schema=1, model=model, revision=api['sha'],
                    config_sha256=sha(evidence / 'glm-config.json'), mode='greedy', speculative=False)
    for record, backend in ((reference, 'hf'), (candidate, candidate_backend)):
        if any(record.get(k) != v for k, v in identity.items()) or record.get('backend') != backend:
            raise ValueError('wrong model/revision/config, decode mode or backend')
        if type(record.get('speculative')) is not bool or type(record.get('schema')) is not int:
            raise ValueError('invalid schema or speculative flag type')
        b = record.get('batch')
        if type(b) is not int or b not in (1, 2, 4, 8) or len(record['requests']) != b:
            raise ValueError('incomplete batch')
        prompts = []
        for request in record['requests']:
            p, g = request['prompt_ids'], request['generated_ids']
            if not p or len(g) != 32 or len(p) + 32 > cfg['text_config']['max_position_embeddings']:
                raise ValueError('incomplete prompt or 32-token output')
            if any(type(t) is not int or not 0 <= t < cfg['text_config']['vocab_size'] for t in p + g):
                raise ValueError('invalid token')
            prompts.append(tuple(p))
        if len(set(prompts)) != b:
            raise ValueError('duplicate request prompt')
    slices = {}
    for record, backend in ((reference, 'hf'), (candidate, candidate_backend)):
        sl = record.get('slice')
        if gate == 'slice':
            if not isinstance(sl, dict):
                raise ValueError(f'{backend} record carries no slice provenance')
            sl = check_slice(cfg['text_config'], tuple(sl.get('layers') or ()))
        elif sl is not None:
            raise ValueError(f'{backend} record is sliced but the gate is {gate}')
        slices[backend] = sl
    if slices['hf'] != slices[candidate_backend]:
        raise ValueError('reference and candidate slices differ')
    if reference['batch'] != candidate['batch']:
        raise ValueError('batch mismatch')
    matched = []
    for r, (ref, got) in enumerate(zip(reference['requests'], candidate['requests'])):
        if ref['prompt_ids'] != got['prompt_ids']:
            raise ValueError(f'request {r} prompt differs')
        matched.append(sum(a == b for a, b in zip(ref['generated_ids'], got['generated_ids'])))
    return dict(schema=1, status='TOKENS-EQUAL' if all(n == 32 for n in matched) else 'FAILED',
                matched_each=matched, required_each=32, model_certified=False,
                missing_gates=['per-operator numerical/state correctness', 'same-build native model/source identity',
                               'real multi-GPU full model execution'], performance='unmeasured')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gate', choices=['production', 'module', 'slice'], default='production')
    ap.add_argument('--candidate-backend', choices=CANDIDATE_BACKENDS, default='etx-pp2')
    a = ap.parse_args()
    try:
        result = compare(json.loads(a.reference.read_text()), json.loads(a.candidate.read_text()), a.gate, a.candidate_backend)
    except (ValueError, KeyError, TypeError) as exc:
        result = dict(schema=1, status='FAILED', error=str(exc), model_certified=False)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'TOKENS-EQUAL' else 2


if __name__ == '__main__':
    raise SystemExit(main())
