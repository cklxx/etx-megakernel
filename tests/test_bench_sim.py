"""Protocol conformance; lifecycle tests live under tests/e2e/test_api_experiment.py."""
import pytest
from etx.testing import protocol_conformance, provenance_conformance, operator_provider_conformance


@pytest.mark.parametrize('mode', ['static', 'dynamic', 'hybrid'])
def test_protocol_completion(mode):
    assert protocol_conformance(mode=mode)['errors'] == []


@pytest.mark.parametrize('fault,expected', [
    ('missing', 'coverage'), ('duplicate', 'coverage'), ('nan', 'nonfinite'),
    ('sync', 'dependency'), ('deadlock', 'deadlock'), ('budget', 'disagrees'),
    ('inflated', 'makespan disagrees'),
])
def test_injected_faults_fail(fault, expected):
    assert any(expected in error for error in protocol_conformance(fault=fault)['errors'])


def test_dynamic_waits_for_last_finishing_producer():
    assert protocol_conformance(fault='last-producer')['errors'] == []


def test_provenance_binds_non_python_sources(tmp_path):
    result = provenance_conformance(tmp_path)
    assert result['non_python_sources'] == 7
    assert result['commit_unchanged']


@pytest.mark.parametrize('small,expected,times', [
    (False, '0700f260fcd5862e5dec66945d0d6dc9c26f6aa23be4c7484f80f3778edc9b09',
     (47.69066472105026, 112.89506472105025, 61.5362370148727)),
    (True, '31fd16299cf5024dc045f4e4aa263adeea997d698aae55a46c6f8c88b09b24a1',
     (41.80908771966494, 102.82923802728456, 55.589887719664915)),
])
def test_package_operator_provider_preserves_legacy_bytes_and_reload(small, expected, times):
    # Captured from the original examples before extraction, 2026-10-23.
    # Hash covers constants, bindings, runtime, graph, all three plan contracts
    # and every generated legacy source/header/plan byte at size 2.
    result = operator_provider_conformance(small=small)
    assert result['signatures'] == {
        'splitk': 'cb56ef588e69a81850a8f773f21c497770bf6a01bc4d866dd929b44eb40e2af6',
        'moe': expected,
    }
    assert result['immutable'] is True
    assert result['reload_dimensions'] == [1024, 256, 1024]
    assert result['frozen_small_refused'] is small
    row = result['study']['rows'][0]
    assert tuple(row['makespan_us'][mode] for mode in ('static', 'dynamic', 'hybrid')) == times
    assert row['max_mean'] == 4.0
    isolated = operator_provider_conformance(small=small, kind='isolated').to_dict()
    assert isolated == {key: value for key, value in result.to_dict().items() if key != 'reload_dimensions'}


@pytest.mark.parametrize('layout,scope', [
    ('installed', ['etx']),
    ('copied', ['etx', 'examples/moe_layer.py', 'examples/splitk_sum.py', 'pyproject.toml']),
])
def test_protocol_provenance_explicit_layout_scope_and_mutation_refusal(tmp_path, layout, scope):
    result = provenance_conformance(tmp_path, layout=layout)
    assert result['scope'] == scope
    assert result['commit'] is None and result['dirty'] is None
    assert result['mutation_refused'] is True
