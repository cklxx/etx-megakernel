"""Offline MTP preflight; the required 32 x 5 acceptance remains blocked.

Do not skip the acceptance gate when the synthetic MTP weights or independent
HF oracle are missing. A green preflight alone is not token equivalence.
"""
import pytest

from examples.glm53 import module_gate
from examples.glm53.mtp_graph import MTPUnavailable, prerequisites, require_prerequisites


def test_preflight_is_unqualified_and_describes_required_comparison():
    report = prerequisites()
    assert report['gate'] == 'module8'
    assert report['qualified'] is False
    assert report['required_comparison'] == dict(positions=32, k=5, reference='HF MTP head')


def test_module8_inventory_has_no_draft_decoder():
    text = module_gate.module_config('module8')['text_config']
    names = [name for i in range(text['num_hidden_layers'])
             for name, _, _, _ in module_gate.native_tensors(text, i)]
    assert names
    assert not any(name.startswith('model.language_model.layers.4.') for name in names)
    assert any('no MTP layer 4 tensors' in b for b in prerequisites()['blockers'])


def test_missing_prerequisites_refuse_qualification():
    with pytest.raises(MTPUnavailable, match='no MTP layer'):
        require_prerequisites()


def test_module8_mtp_32_positions_k5_equal_hf():
    report = prerequisites()
    assert not report['blockers'], 'MTP acceptance NOT COMPLETE: ' + '; '.join(report['blockers'])
    # Removing the blockers is necessary, but cannot by itself certify tokens.
    pytest.fail('MTP acceptance NOT COMPLETE: implement the ETX draft graph and compare '
                'all 160 draft IDs against the independently loaded HF MTP head')
