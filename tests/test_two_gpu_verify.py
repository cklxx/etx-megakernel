import json
from pathlib import Path

import pytest

from examples.glm53.prepare import MODEL
from examples.vllm_llm import batch_correctness as bc
from tools import two_gpu_verify as vf

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = {e['batch']: e['imported_instances']
             for e in json.loads((ROOT / 'docs/evidence/qwen-batch-glm53/qwen-plan-inventory.json').read_text())}


def qwen_receipt(batch):
    return dict(status='PASS', performance='unmeasured', batch=batch, positions=[0, 1040],
                inputs=dict(model='Qwen/Qwen3-30B-A3B', revision='a' * 40, batch=batch,
                            context=1024, gen=32, files={f'f{i}': f'{i:064x}' for i in range(20)}),
                build_identity=dict(check_enabled=True, overrides=bc.recipe(batch)),
                imported_per_position=INVENTORY[batch],
                byte_identity_per_request={
                    str(p): dict(position=p, requests=batch,
                                 instances_per_request=INVENTORY[batch] // batch) for p in (0, 1040)},
                greedy_tokens_per_mode=batch * 32 * 2, modes=['fused', 'unfused'])


def glm_receipt(batch=2):
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


def record(tmp_path, qwen=True, glm=True):
    root = tmp_path / 'record'
    for batch in (2, 4, 8):
        directory = root / f'qualify_b{batch}'
        directory.mkdir(parents=True)
        (directory / 'correctness.json').write_text(json.dumps(qwen_receipt(batch)))
    if glm:
        directory = root / 'qualification'
        directory.mkdir(parents=True)
        (directory / 'qualification.json').write_text(json.dumps(glm_receipt()))
    return root


def run(path, *args):
    return vf.main([str(path), *args])


def test_complete_receipts_verify(tmp_path, capsys):
    root = record(tmp_path)
    assert run(root) == 0
    out = capsys.readouterr().out
    for batch in (2, 4, 8):
        assert f'B{batch} {INVENTORY[batch]} instances (97/request)' in out
    assert 'VERIFY OK glm: batch 2, 2 repetitions' in out
    assert out.rstrip().endswith('VERIFY OK')


def test_missing_receipt_is_named(tmp_path, capsys):
    root = record(tmp_path)
    (root / 'qualify_b4/correctness.json').unlink()
    assert run(root) == 2
    assert 'VERIFY FAIL qwen: missing' in capsys.readouterr().out


@pytest.mark.parametrize('change, match', [
    (dict(status='NOT-COMPLETE', error='exit 124'), 'status'),
    (dict(performance='measured'), 'must not report performance'),
    (dict(imported_per_position=1), 'imported_per_position'),
    (dict(positions=[0]), 'positions'),
    (dict(modes=['fused']), 'modes'),
    (dict(greedy_tokens_per_mode=64), 'greedy_tokens_per_mode'),
])
def test_qwen_receipts_that_do_not_prove_the_claim_fail(tmp_path, change, match):
    root = record(tmp_path)
    path = root / 'qualify_b8/correctness.json'
    receipt = json.loads(path.read_text())
    receipt.update(change)
    path.write_text(json.dumps(receipt))
    assert run(root) == 2
    with pytest.raises(vf.Fail, match=match):
        vf.qwen(root)


def test_qwen_per_request_identity_must_cover_every_request(tmp_path):
    root = record(tmp_path)
    path = root / 'qualify_b2/correctness.json'
    receipt = json.loads(path.read_text())
    receipt['byte_identity_per_request']['1040']['instances_per_request'] = 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(vf.Fail, match='position 1040 does not cover all 2 requests'):
        vf.qwen(root)


def test_qwen_recipe_must_be_the_fixed_one(tmp_path):
    root = record(tmp_path)
    path = root / 'qualify_b4/correctness.json'
    receipt = json.loads(path.read_text())
    receipt['build_identity']['overrides']['ETX_VL_BATCH'] = '1'
    path.write_text(json.dumps(receipt))
    with pytest.raises(vf.Fail, match='ETX_VL_BATCH'):
        vf.qwen(root)


@pytest.mark.parametrize('change, match', [
    (dict(status='FAILED'), 'status'),
    (dict(model_certified=True), 'model_certified'),
    (dict(matched_each=[32, 31]), 'matched_each'),
    (dict(repetitions=1), 'two independent repetitions'),
    (dict(logit_gate='SKIPPED'), 'full-logit gate'),
    (dict(build_hashes={}), 'build file hashes'),
    (dict(build_hashes={'run': 'b' * 64}), 'build file hashes'),
    (dict(reference_hashes={'ref.json': 'c' * 64}), 'reference/logits hashes'),
])
def test_glm_receipts_that_do_not_prove_the_claim_fail(tmp_path, change, match):
    root = record(tmp_path)
    path = root / 'qualification/qualification.json'
    receipt = json.loads(path.read_text())
    receipt.update(change)
    path.write_text(json.dumps(receipt))
    with pytest.raises(vf.Fail, match=match):
        vf.glm(root)


@pytest.mark.parametrize('change, match', [
    (dict(status='FAILED'), 'not 32/32'),
    (dict(matched_each=[32, 31]), 'not 32/32'),
    (dict(logit_gate='SKIPPED'), 'no full-logit comparison'),
    (dict(logit_max_abs=float('nan')), 'no finite full-logit error'),
    (dict(logit_max_abs_per_step=[0.1] * 31), 'per-step logit errors'),
    (dict(logit_max_abs_per_step=[float('nan')] * 32), 'finite per-step logit errors'),
    (dict(logit_argmax_equal=False), 'argmax differs'),
    (dict(checkpoint_execution='simulated'), 'did not execute the real two-GPU model'),
])
def test_glm_verdicts_are_checked_per_repetition(tmp_path, change, match):
    root = record(tmp_path)
    path = root / 'qualification/qualification.json'
    receipt = json.loads(path.read_text())
    receipt['verdicts'][1].update(change)
    path.write_text(json.dumps(receipt))
    with pytest.raises(vf.Fail, match=match):
        vf.glm(root)


def test_qwen_only_and_glm_only_do_not_require_the_other(tmp_path):
    root = record(tmp_path)
    (root / 'qualification/qualification.json').unlink()
    assert run(root, '--qwen-only') == 0
    assert run(root, '--glm-only') == 2


def test_require_seal_refuses_an_unsealed_record(tmp_path, capsys):
    root = record(tmp_path)
    assert run(root, '--qwen-only', '--require-seal') == 2
    assert 'VERIFY FAIL seal' in capsys.readouterr().out
