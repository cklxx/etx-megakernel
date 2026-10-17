from copy import deepcopy
import json
import subprocess
import sys

import pytest

from examples.glm53 import boundary, oracle, prepare
from examples.llm.model import load_dims
from etx.ir import Scope
from etx.passes import compile_graph, verify_plan
from etx.tools.gpu_jobs import load_session


def test_exact_identity_architecture_and_complete_ownership():
    p = prepare.prepare()
    assert p['model'] == 'zai-org/GLM-5.3-Flash'
    assert p['revision'] == '46a050bec13002ad29ee795c16e61cf4fa3c4a9c'
    assert p['architecture'] == dict(wrapper='Glm5NextForConditionalGeneration', hidden=4096, layers=45,
                                     kda=34, sparse_mla=11, experts=288, topk=8, hc_mult=4)
    assert p['ranks'][0]['layers'] == list(range(24))
    assert p['ranks'][1]['layers'] == list(range(24, 45))
    assert [r['moe_layers'] for r in p['ranks']] == [21, 21]
    assert [r['kda_layers'] for r in p['ranks']] == [18, 16]
    assert p['status'] == 'BLOCKED' and p['performance'] == 'unmeasured'


def test_memory_is_an_estimate_and_bf16_cannot_fit_two_192gib_cards():
    p = prepare.prepare(batch=8)
    b = p['budget']
    assert b['checkpoint_payload_bytes'] == 328250014584
    assert b['boundary_bytes_per_decode_step'] == 8 * 4 * 4096 * 2
    assert b['bf16_fits_payload_floor'] is False
    assert b['native_payload_reserve_bytes'] > 0
    assert b['unknown'] and b['memory_fit'].startswith('UNESTABLISHED')


@pytest.mark.parametrize('kwargs', [dict(batch=0), dict(batch=3), dict(context=0), dict(context=1048577), dict(gpu_gib=0)])
def test_reject_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        prepare.prepare(**kwargs)


def test_source_evidence_mutation_is_rejected(tmp_path):
    import shutil
    for p in prepare.EVIDENCE.glob('glm-*'):
        shutil.copyfile(p, tmp_path / p.name)
    shutil.copyfile(prepare.EVIDENCE / 'sources.json', tmp_path / 'sources.json')
    (tmp_path / 'glm-config.json').write_text('{}')
    with pytest.raises(ValueError, match='changed source evidence'):
        prepare.metadata(tmp_path)


def test_existing_qwen_loader_refuses_nested_and_flat_glm(tmp_path):
    config = json.loads((prepare.EVIDENCE / 'glm-config.json').read_text())
    for c in (config, config['text_config']):
        path = tmp_path / 'config.json'
        path.write_text(json.dumps(c))
        with pytest.raises(ValueError, match='GLM-5.3-Flash needs'):
            load_dims(str(path))


def test_executable_blocker_is_offline_and_exits_nonzero(tmp_path):
    out = tmp_path / 'blocker.json'
    run = subprocess.run([sys.executable, '-m', 'examples.glm53.prepare', '--require-model', '--out', str(out)],
                         cwd=prepare.ROOT, capture_output=True, text=True)
    assert run.returncode == 2 and json.loads(out.read_text())['status'] == 'BLOCKED'


@pytest.mark.parametrize('batch', [1, 2, 4, 8])
def test_boundary_scope_memory_and_full_bit_oracle(batch):
    b = boundary.bindings(B=batch)
    p = compile_graph(boundary.build(), 'gfx942', b, {})
    assert not verify_plan(p) and p.n_devices == 2
    assert p.events['ready'].scope == p.events['feedback'].scope == Scope.SYSTEM
    assert p.events['ready'].memory == p.events['feedback'].memory == 'fine_grained'
    assert p.tensor_placement['mailbox'] == p.tensor_placement['token'] == 'fine_grained'
    for step in (0, 1, 7):
        values, feedback = boundary.oracle(batch, step)
        assert len(values) == batch * 16384 and len(feedback) == batch
        assert len(set(values[:16384])) == 16384
        assert values != boundary.oracle(batch, step + 1)[0]
        assert feedback == [values[r*16384] + values[(r+1)*16384-1] for r in range(batch)]


def token_record(backend='hf', batch=2):
    p = prepare.prepare(batch)
    return dict(schema=1, backend=backend, model=p['model'], revision=p['revision'],
                config_sha256=p['config_sha256'], batch=batch, mode='greedy', speculative=False,
                requests=[dict(prompt_ids=[1, 2+r], generated_ids=[r]*32) for r in range(batch)])


def test_future_model_token_oracle_is_exact_but_not_a_model_certificate():
    r, c = token_record(), token_record('etx-pp2')
    verdict = oracle.compare(r, c)
    assert verdict['status'] == 'TOKENS-EQUAL' and verdict['matched_each'] == [32, 32]
    assert not verdict['model_certified']
    c['requests'][1]['generated_ids'][-1] = 77
    assert oracle.compare(r, c)['status'] == 'FAILED'


@pytest.mark.parametrize('mutation', [
    lambda c: c.update(revision='0'*40),
    lambda c: c.update(mode='teacher'),
    lambda c: c.update(speculative=True),
    lambda c: c['requests'][0].update(generated_ids=[0]*31),
    lambda c: c['requests'][1].update(prompt_ids=[1, 2]),
    lambda c: c['requests'].pop(),
    lambda c: c['requests'][0].update(prompt_ids=[-1]),
])
def test_oracle_rejects_wrong_identity_and_incomplete_requests(mutation):
    candidate = deepcopy(token_record('etx-pp2'))
    mutation(candidate)
    with pytest.raises(ValueError):
        oracle.compare(token_record(), candidate)


def test_transport_session_preserves_two_visible_devices():
    jobs, limit = load_session(prepare.ROOT / 'examples/sessions/glm53_boundary.py')
    assert limit == 300 and len(jobs) == 2
    job = jobs[1]
    assert job.lane == 'cpu' and job.env['HIP_VISIBLE_DEVICES'] == '0,1'
    assert job.reads == ['boundary/run']
