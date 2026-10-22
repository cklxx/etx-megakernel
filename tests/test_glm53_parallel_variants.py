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
        assert entry['batch'] == 1 and entry['bindings'] == {'B':1}
        assert entry['checks_still_required'] == list(parallel_variants.REQUIRED_CHECKS)
        assert '32/32' in ' '.join(entry['checks_still_required'])
        if entry['mode'] == 'ep':
            assert 'EP preservation' in entry['numerical_contract']
        if entry['mode'] in ('ffn-tp','full-tp','head-tp-ep'):
            assert 'TP numerical qualification' in entry['numerical_contract']


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
