"""Validate same-checkpoint HF versus the actual ETX PP2 greedy token records offline.

Token agreement alone does not certify all full-checkpoint operator numerics.
Record schema: model, revision, config_sha256, batch, mode='greedy',
speculative=false, backend='hf'/'etx-pp2', requests=[{prompt_ids, generated_ids}].
"""
import argparse
import json
from pathlib import Path
from examples.glm53.prepare import MODEL, metadata, sha, EVIDENCE


def compare(reference, candidate):
    cfg, api = metadata()
    identity = dict(schema=1, model=MODEL, revision=api['sha'],
                    config_sha256=sha(EVIDENCE / 'glm-config.json'), mode='greedy', speculative=False)
    for record, backend in ((reference, 'hf'), (candidate, 'etx-pp2')):
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
                               'real two-GPU full model execution'], performance='unmeasured')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    try:
        result = compare(json.loads(a.reference.read_text()), json.loads(a.candidate.read_text()))
    except (ValueError, KeyError, TypeError) as exc:
        result = dict(schema=1, status='FAILED', error=str(exc), model_certified=False)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'TOKENS-EQUAL' else 2


if __name__ == '__main__':
    raise SystemExit(main())
