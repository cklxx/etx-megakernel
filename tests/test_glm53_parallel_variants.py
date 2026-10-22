"""Authored-only regression cases for the options emitter; never GPU qualification."""
import hashlib
import json
import sys

import pytest

from examples.glm53 import parallel_variants
from examples.glm53.options import Options


MODES = ('pp', 'ffn-tp', 'ep', 'full-tp', 'head-tp-ep')


def test_fixed_matrix_has_independent_unbalanced_and_balanced_arms():
    entries = parallel_variants.variants()
    by_name = {entry['name']:entry for entry in entries}
    expected = {f'{mode}{world}{suffix}' for world in (2,4,8) for mode in MODES
                for suffix in ('','-up-balanced')}
    assert len(entries) == len(by_name) == 30 and set(by_name) == expected
    for world in (2,4,8):
        for mode, parallel, attention in [('pp',0,0), ('ffn-tp',1,0), ('ep',2,0),
                                          ('full-tp',1,1), ('head-tp-ep',2,1)]:
            base = by_name[f'{mode}{world}']
            balanced = by_name[f'{mode}{world}-up-balanced']
            assert base['degree'] == balanced['degree'] == world
            assert base['mode'] == balanced['mode'] == mode
            assert base['options'] == Options(devices=world, parallel=parallel, attention_tp=attention).to_dict()
            assert balanced['options'] == dict(base['options'], up_tail_balance=1)
            assert base['effective_up_pairs'] == 36 and balanced['effective_up_pairs'] == 32
            assert balanced['options_sha256'] != base['options_sha256']
            assert base['options']['up_pairs'] == balanced['options']['up_pairs'] == 36
    assert [e['name'] for e in entries if e['measured_default_configuration']] == ['pp2']
    assert Options() == Options(devices=2, parallel=0, attention_tp=0, up_tail_balance=0)


def test_generation_is_only_options_and_manifest_with_canonical_hashes(tmp_path):
    out = tmp_path/'new-options'
    manifest = parallel_variants.generate(out)
    assert json.loads((out/'manifest.json').read_text()) == manifest
    assert manifest['purpose'] == 'options-generation-only'
    assert manifest['status'] == 'uncompiled-untested' and manifest['launches_performed'] == 0
    assert manifest['measured_default_options'] == Options().to_dict()
    assert {p.name for p in out.iterdir()} == {'manifest.json'} | {e['options_file'] for e in manifest['variants']}
    assert len({e['options_sha256'] for e in manifest['variants']}) == 30
    for entry in manifest['variants']:
        value = json.loads((out/entry['options_file']).read_text())
        canonical = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(canonical).hexdigest() == entry['options_sha256']
        assert value == entry['options']
        assert entry['status'] == 'uncompiled-untested'
        assert 'batch' not in entry['options'] and 'batch' not in entry
        assert entry['checks_still_required'] == list(parallel_variants.REQUIRED_CHECKS)
        assert '32/32' in ' '.join(entry['checks_still_required'])
        if entry['mode'] == 'ep':
            assert 'EP preservation' in entry['numerical_contract']
        if entry['mode'] in ('ffn-tp','full-tp','head-tp-ep'):
            assert 'TP numerical qualification' in entry['numerical_contract']


def test_batch_axis_reuses_options_but_has_unique_planning_case_identity():
    entries = parallel_variants.variants()
    cases = parallel_variants.cases(entries)
    by_name = {entry['name']:entry for entry in entries}
    assert len(cases) == len({c['name'] for c in cases}) == 120
    assert len({c['case_sha256'] for c in cases}) == 120
    assert len({c['options_file'] for c in cases}) == 30
    assert len({c['options_sha256'] for c in cases}) == 30
    assert {c['name'] for c in cases} == {f'{e["name"]}-b{b}' for e in entries for b in (1,2,4,8)}
    assert [c['name'] for c in cases if c['measured_default_case']] == ['pp2-b1']
    for case in cases:
        entry = by_name[case['variant']]
        batch = case['batch']
        assert case['bindings'] == {'B':batch}
        assert case['options_file'] == entry['options_file']
        assert case['options_sha256'] == entry['options_sha256']
        assert case['degree'] == entry['degree'] and case['mode'] == entry['mode']
        value = dict(options_sha256=entry['options_sha256'], batch=batch)
        expected = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert case['case_sha256'] == expected
        assert case['status'] == 'uncompiled-untested'
        assert case['checks_still_required'] == list(parallel_variants.REQUIRED_CHECKS)
        assert 'batch' not in entry['options']


@pytest.mark.parametrize('batch', [0,3,16,True,False,1.0,'2',None])
def test_planning_identity_rejects_unsupported_or_noninteger_batch(batch):
    with pytest.raises(ValueError, match='batch'):
        parallel_variants.case_digest(Options().digest(), batch)


def test_generated_manifest_preserves_batch_acceptance_contract(tmp_path):
    manifest = parallel_variants.generate(tmp_path/'batch-options')
    assert manifest['schema'] == 2 and manifest['batches'] == [1,2,4,8]
    assert len(manifest['variants']) == 30 and len(manifest['cases']) == 120
    assert 'NOT a build certificate' in manifest['case_hash_basis']
    assert manifest['correctness'] == dict(logits_shape=[32,'B','VOCAB'], preserve_all_logits=True,
                                           hf_greedy_tokens_required_each=32, request_isolation_required=True)
    assert 'per-WG LDS unchanged' in manifest['batch_capacity']
    assert manifest['launches_performed'] == 0


@pytest.mark.parametrize('existing', ['directory', 'file', 'symlink'])
def test_existing_output_is_never_reused_or_overwritten(tmp_path, existing):
    out = tmp_path/'existing'
    if existing == 'directory':
        out.mkdir()
        sentinel = out/'keep'
    elif existing == 'symlink':
        target = tmp_path/'target'
        target.mkdir()
        out.symlink_to(target, target_is_directory=True)
        sentinel = target/'keep'
    else:
        sentinel = out
    sentinel.write_text('untouched')
    with pytest.raises(FileExistsError):
        parallel_variants.generate(out)
    assert sentinel.read_text() == 'untouched'
    if existing != 'file':assert sorted(p.name for p in out.iterdir()) == ['keep']


def test_cli_requires_an_explicit_new_output_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['parallel_variants'])
    with pytest.raises(SystemExit):parallel_variants.main()
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(sys, 'argv', ['parallel_variants', '--out', str(tmp_path)])
    with pytest.raises(SystemExit):parallel_variants.main()
    assert 'output already exists' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
