"""Caller contract tests: declarations are not builds or hardware evidence."""
from dataclasses import FrozenInstanceError
import json

import pytest
from etx.external import Configuration, KernelSpec, WorkloadSpec, inspect, kernel, model


def test_model_roundtrip_and_identity():
    first = model('fixture', config={'batch': 2, 'shape': [8, 16]})
    second = model('fixture', config={'shape': [8, 16], 'batch': 2})
    assert first == second
    assert first.identity == second.identity
    assert WorkloadSpec.from_dict(json.loads(json.dumps(first.to_dict()))) == first
    assert model('fixture', config={'batch': 4, 'shape': [8, 16]}).identity != first.identity
    result = inspect(first)
    assert result.status == 'DECLARED'
    assert result.identity == first.identity
    assert result.declaration == first


@pytest.mark.parametrize('kind', ['graph', 'imported', 'native'])
def test_operator_kinds_use_the_same_workload_record(kind):
    declaration = kernel(KernelSpec('operator', kind, {'abi': {'args': ['x', 'y']}}))
    assert isinstance(declaration, WorkloadSpec)
    assert declaration.kind == kind
    assert inspect(declaration).status == 'DECLARED'


def test_configuration_is_detached_in_both_directions():
    config = {'layers': [{'width': 8}]}
    declaration = model('fixture', config=config)
    identity = declaration.identity
    config['layers'][0]['width'] = 16
    declaration.config['layers'][0]['width'] = 32
    declaration.to_dict()['config']['layers'][0]['width'] = 64
    assert declaration.config['layers'] == [{'width': 8}]
    assert declaration.identity == identity
    with pytest.raises(FrozenInstanceError):
        declaration.name = 'changed'
    with pytest.raises(FrozenInstanceError):
        declaration.config = Configuration({})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf'), object(), {1: 'x'}, {1, 2}])
def test_rejects_non_json_configuration(value):
    with pytest.raises(ValueError):
        model('fixture', config={'value': value})


@pytest.mark.parametrize('name', ['', ' leading', 'trailing ', None, 4])
def test_rejects_invalid_name(name):
    with pytest.raises(ValueError):
        model(name, config={})


@pytest.mark.parametrize('version', [0, 2, True, 1.0, '1'])
def test_rejects_unknown_or_coerced_schema(version):
    with pytest.raises(ValueError, match='schema'):
        WorkloadSpec.from_dict(dict(schema_version=version, name='fixture', kind='model', config={}))


def test_rejects_unknown_fields_kinds_and_objects():
    record = model('fixture', config={}).to_dict()
    with pytest.raises(ValueError):
        WorkloadSpec.from_dict(dict(record, unknown=True))
    with pytest.raises(ValueError):
        WorkloadSpec.from_dict(dict(record, kind='arbitrary'))
    with pytest.raises(ValueError):
        Configuration([])
    with pytest.raises(TypeError):
        inspect(object())
    with pytest.raises(TypeError):
        kernel(object())
