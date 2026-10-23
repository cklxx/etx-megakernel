"""Synthetic two-GPU qualification records (Qwen batch receipts + GLM verdicts) for verifier tests; not evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _inventory():
    return {e['batch']: e['imported_instances']
            for e in json.loads((ROOT / 'docs/evidence/qwen-batch-glm53/qwen-plan-inventory.json').read_text())}


def qwen_files(batch):
    from examples.vllm_llm import batch_correctness as bc
    names = bc.golden_names(batch)
    names += [n.replace('golden', 'prefill', 1) for n in names]
    names += ['hf/config.json', 'hf/REVISION', 'vl.manifest', 'vl.bin']
    return {name: f'{i:064x}' for i, name in enumerate(names)}


def qwen_receipt(batch):
    from examples.vllm_llm import batch_correctness as bc
    inventory = _inventory()
    return dict(status='PASS', performance='unmeasured', batch=batch, positions=[0, 1040],
                inputs=dict(model='Qwen/Qwen3-30B-A3B', revision='a' * 40, batch=batch,
                            context=1024, gen=32, files=qwen_files(batch)),
                build_identity=dict(check_enabled=True, overrides=bc.recipe(batch)),
                imported_per_position=inventory[batch],
                byte_identity_per_request={
                    str(p): dict(position=p, requests=batch,
                                 instances_per_request=inventory[batch] // batch) for p in (0, 1040)},
                greedy_tokens_per_mode=batch * 32 * 2, modes=['fused', 'unfused'])


def glm_receipt(batch=2):
    from examples.glm53.prepare import MODEL
    verdict = dict(status='TOKENS-EQUAL', matched_each=[32] * batch,
                   checkpoint_execution='real two-GPU ETX model', logit_gate='REPORTED',
                   logit_max_abs=0.25, logit_max_abs_per_step=[0.01] * 32,
                   logit_argmax_equal=True, model_certified=False)
    return dict(status='TOKENS-EQUAL', performance='unmeasured', model=MODEL, revision='a' * 40,
                batch=batch, generated_each=32, repetitions=2, matched_each=[32] * batch,
                model_certified=False,
                build_hashes={name: 'b' * 64 for name in
                              ('run', 'build-spec.json', 'plan.json', 'plan_data.h', 'glm53_spec.h',
                               'glm53_contract.h', 'load.tsv', 'megakernel_d0.hip', 'megakernel_d1.hip')},
                reference_hashes={'ref.json': 'c' * 64, 'ref.logits': 'd' * 64},
                verdicts=[dict(verdict, repetition=0), dict(verdict, repetition=1)],
                logit_gate='REPORTED')


def qualification_record(tmp_path, qwen=True, glm=True):
    """A complete synthetic record directory as the two-GPU session leaves it (under tmp_path/record)."""
    root = Path(tmp_path) / 'record'
    for batch in (2, 4, 8):
        directory = root / f'qualify_b{batch}'
        directory.mkdir(parents=True)
        (directory / 'correctness.json').write_text(json.dumps(qwen_receipt(batch)))
    if glm:
        directory = root / 'qualification'
        directory.mkdir(parents=True)
        (directory / 'qualification.json').write_text(json.dumps(glm_receipt()))
    return root


def qwen_staged_model(tmp_path, batch=8):
    """A packed-model directory shaped like a staged Qwen3-30B-A3B (config, revision, goldens); placeholder weights."""
    from examples.vllm_llm import batch_correctness as bc
    model = Path(tmp_path) / 'model'
    (model / 'hf').mkdir(parents=True)
    (model / 'hf/config.json').write_text((ROOT / 'examples/llm/configs/qwen3-30b-a3b.json').read_text())
    (model / 'hf/REVISION').write_text('a' * 40)
    (model / 'vl.manifest').write_text('test fixture, not GPU inputs')
    (model / 'vl.bin').write_bytes(b'fixture')
    for r, name in enumerate(bc.golden_names(batch)):
        (model / name).write_text('prompt: ' + ' '.join([str(r)] * 1024) + '\ngen: ' + ' '.join(['1'] * 32) + '\n')
        (model / name.replace('golden', 'prefill', 1)).write_text('argmax: ' + ' '.join(['1'] * 1055) + '\nmargin: ' + ' '.join(['1.0'] * 1055))
    return model
