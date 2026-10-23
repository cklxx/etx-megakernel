"""Registered operators reuse public planning, simulation and source artifacts."""
from dataclasses import replace
import hashlib
import json
import shutil

import pytest

from etx.external import (KernelSpec, PlanRequest, kernel, plan, simulate, emit,
                          save_sources, load_sources, check_device_sources)


def request(name, *, small=False, size=2):
    config = {'provider': name}
    if name == 'moe_layer':
        config['small'] = small
    return PlanRequest(kernel(KernelSpec(name, 'graph', config)),
                       bindings={'n' if name == 'splitk_sum' else 'B': size})


@pytest.mark.parametrize('name,small', [('splitk_sum', False), ('moe_layer', False), ('moe_layer', True)])
@pytest.mark.parametrize('size', [1, 2, 4, 8])
def test_registered_operator_uses_shared_lifecycle(tmp_path, name, small, size):
    spec = request(name, small=small, size=size)
    planned = plan(spec)
    assert plan(spec).identity == planned.identity
    assert planned.report['graph'] == name
    assert planned.report['bindings']['n' if name == 'splitk_sum' else 'B'] == size
    result = simulate(planned, seed=17)
    assert result.backend == 'simulation' and not result.deadlock
    source = emit(planned)
    assert source.status == 'SOURCES' and not source.external_sources
    tile = 'examples/tiles/' + ('splitk.hip' if name == 'splitk_sum' else 'moe.hip')
    expected = ('beedd82376c66cd0f869d53018baf8d020f4553b9fcd155e7fc0604ea237ee09'
                if name == 'splitk_sum' else
                'e4010dfc8809f64f746e0767fa65c36f4d9a768554381ee3ac14195b3a25d4d4')
    assert hashlib.sha256(source.files[tile].encode()).hexdigest() == expected
    defines = {} if name == 'splitk_sum' else (
        dict(D=256, H=2, S=128, FF=128) if small else dict(D=1024, H=8, S=512, FF=512))
    assert json.loads(source.files['etx_native_inputs.json']) == dict(
        schema_version=1, kind='native-tile-inputs',
        units=[dict(source=tile, language='hip', defines=defines)])
    assert 'include/etx/abi.h' in source.files
    assert 'include/etx/tile_util.h' in source.files
    save_sources(source, tmp_path / 'sources')
    shutil.move(tmp_path / 'sources', tmp_path / 'relocated')
    assert load_sources(tmp_path / 'relocated') == source


@pytest.mark.parametrize('damage', ['missing-recipe', 'missing-tile', 'wrong-dimensions', 'unknown-unit'])
def test_native_recipe_rejected_before_launch(tmp_path, damage):
    source = emit(plan(request('moe_layer', small=True)))
    files = source.files.to_dict()
    if damage == 'missing-recipe':
        del files['etx_native_inputs.json']
    elif damage == 'missing-tile':
        del files['examples/tiles/moe.hip']
    else:
        recipe = json.loads(files['etx_native_inputs.json'])
        if damage == 'wrong-dimensions':
            recipe['units'][0]['defines']['D'] = 1024
        else:
            recipe['units'][0]['source'] = 'unknown.hip'
        files['etx_native_inputs.json'] = json.dumps(recipe, sort_keys=True, indent=2) + '\n'
    with pytest.raises(ValueError, match='native inputs'):
        check_device_sources(replace(source, files=files), output=tmp_path / 'rejected',
                             docker='/not/docker', image='not-an-image')
    assert not (tmp_path / 'rejected').exists()


@pytest.mark.parametrize('payload', ['examples/tiles/moe.hip', 'etx_native_inputs.json'])
@pytest.mark.parametrize('damage', ['mutate', 'remove'])
def test_native_input_corruption_rejected(tmp_path, payload, damage):
    source = emit(plan(request('moe_layer', small=True)))
    save_sources(source, tmp_path / 'source')
    path = tmp_path / 'source' / payload
    if damage == 'mutate':
        path.write_text(path.read_text() + '\ncorruption\n')
        with pytest.raises(ValueError, match='checksum'):
            load_sources(tmp_path / 'source')
    else:
        path.unlink()
        with pytest.raises(FileNotFoundError):
            load_sources(tmp_path / 'source')


@pytest.mark.parametrize('bindings', [{}, {'M': 256, 'K': 64, 'N': 128}])
@pytest.mark.parametrize('target', ['gfx90a', 'gfx942'])
def test_gemm_registered_provider_reuses_multi_rank_lifecycle(tmp_path, bindings, target):
    spec = PlanRequest(kernel(KernelSpec('gemm', 'graph', {'provider': 'gemm_rs'})),
                       bindings=bindings, target=target)
    planned = plan(spec)
    assert planned.report['devices'] == 2
    assert not simulate(planned, seed=17).deadlock
    source = emit(planned)
    assert not source.external_sources and len(source.rank_sources) == 2
    assert hashlib.sha256(source.files['examples/tiles/gemm_rs.hip'].encode()).hexdigest() == (
        'f142df93503bbb7f7a814272d63bbe1b192135c9dd8f26b3a3947c03e49f00bb')
    assert hashlib.sha256(source.files['examples/hosts/gemm_rs.hip'].encode()).hexdigest() == (
        '2882e0fc44cca5b7ae51b24746ff0c4e01108374e2b09899a5f13b35425405d0')
    save_sources(source, tmp_path / 'source')
    assert load_sources(tmp_path / 'source') == source


@pytest.mark.parametrize('bindings', [{'M': 128}, {'K': 33}, {'N': 127}, {'M': 0}, {'world': 8}])
def test_gemm_provider_rejects_unrepresentable_native_shapes(bindings):
    with pytest.raises(ValueError):
        plan(PlanRequest(kernel(KernelSpec('gemm', 'graph', {'provider': 'gemm_rs'})), bindings=bindings))


def test_operator_dimensions_do_not_read_ambient_small_mode(monkeypatch):
    default = request('moe_layer')
    small = request('moe_layer', small=True)
    first = plan(default)
    second = plan(small)
    assert first.identity != second.identity
    monkeypatch.setenv('ETX_MOE_SMALL', '1')
    assert plan(default) == first
    assert plan(small) == second
    assert plan(request('moe_layer', size=4)).identity != first.identity


@pytest.mark.parametrize('config,bindings,runtime', [
    ({'provider': 'missing'}, {}, {}),
    ({'provider': 'splitk_sum', 'small': False}, {}, {}),
    ({'provider': 'moe_layer', 'small': 1}, {}, {}),
    ({'provider': 'moe_layer', 'tensors': []}, {}, {}),
    ({'provider': 'moe_layer'}, {'n_gg': 1}, {}),
    ({'provider': 'moe_layer'}, {'B': 0}, {}),
    ({'provider': 'splitk_sum'}, {'B': 2}, {}),
    ({'provider': 'splitk_sum'}, {'n': 0}, {}),
    ({'provider': 'moe_layer'}, {}, {'topk': [[0, 1]]}),
])
def test_operator_provider_rejects_ambiguous_or_derived_inputs(config, bindings, runtime):
    with pytest.raises(ValueError):
        plan(PlanRequest(kernel(KernelSpec('operator', 'graph', config)),
                         bindings=bindings, runtime=runtime))
