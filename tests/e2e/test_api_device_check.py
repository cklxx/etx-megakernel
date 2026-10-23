"""Opt-in offline Docker checks: real gfx942 objects, never device execution."""
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
from etx.external import PlanRequest, plan, emit, check_device_sources, KernelSpec, kernel
from etx.testing import fixture


@pytest.mark.skipif(not os.environ.get('ETX_TEST_ROCM_DOCKER'), reason='offline Docker compiler check not requested')
@pytest.mark.parametrize('broken_rank', [None, 7])
def test_all_rank_device_objects_and_rank_failure(tmp_path, broken_rank):
    artifact = emit(plan(PlanRequest(fixture(devices=8, batch=1))))
    if broken_rank is not None:
        files = artifact.files.to_dict()
        files[artifact.rank_sources[broken_rank]] += '\n#error deliberate_rank_7_failure\n'
        artifact = replace(artifact, files=files)
    root = Path(os.environ.get('ETX_API_BUILD_EVIDENCE', str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    directory = root / ('valid' if broken_rank is None else 'rank7-failure')
    result = check_device_sources(artifact, output=directory, docker='/usr/bin/docker',
                                  image=os.environ['ETX_TEST_ROCM_DOCKER'], deadline_s=90)
    assert result.source_identity == artifact.identity
    assert result.toolchain['gpu_access'] is False
    assert result.toolchain['network'] == 'none'
    assert len(result.ranks) == 8
    for rank, invocation in enumerate(result.ranks):
        if invocation.status != 'complete':
            print((directory / 'checks' / f'rank-{rank}' / 'output.log').read_text())
    if broken_rank is None:
        assert result.status == 'DEVICE_CHECKED'
        assert len(result.objects) == 8
        assert all(rank.status == 'complete' for rank in result.ranks)
    else:
        assert result.status == 'FAILED'
        assert result.ranks[7].status == 'failed'
        assert all(rank.status == 'complete' for rank in result.ranks[:7])
        assert len(result.objects) == 7
    assert result.status != 'EXECUTABLE'


@pytest.mark.skipif(not os.environ.get('ETX_TEST_ROCM_DOCKER'), reason='offline Docker compiler check not requested')
@pytest.mark.parametrize('name,small,inline,broken', [
    ('splitk_sum', False, False, False),
    ('moe_layer', False, False, False),
    ('moe_layer', True, False, False),
    ('moe_layer', True, True, False),
    ('moe_layer', True, False, True),
])
def test_registered_native_tiles_compile_or_fail_as_a_unit(tmp_path, name, small, inline, broken):
    config = dict(provider=name)
    if name == 'moe_layer':
        config['small'] = small
    artifact = emit(plan(PlanRequest(kernel(KernelSpec(name, 'graph', config)),
                                     bindings={'n' if name == 'splitk_sum' else 'B': 2},
                                     options={'inline_tiles': inline})))
    tile = 'examples/tiles/' + ('splitk.hip' if name == 'splitk_sum' else 'moe.hip')
    if broken:
        files = artifact.files.to_dict()
        files[tile] += '\n#error deliberate_native_tile_failure\n'
        artifact = replace(artifact, files=files)
    root = Path(os.environ.get('ETX_API_BUILD_EVIDENCE', str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f'{name}-small{small}-inline{inline}-broken{broken}'
    result = check_device_sources(artifact, output=directory, docker='/usr/bin/docker',
                                  image=os.environ['ETX_TEST_ROCM_DOCKER'], deadline_s=90)
    assert result.source_identity == artifact.identity
    assert result.toolchain['gpu_access'] is False and result.toolchain['network'] == 'none'
    assert len(result.ranks) == len(result.tiles) == 1
    for kind, invocations in [('rank', result.ranks), ('tile', result.tiles)]:
        for index, invocation in enumerate(invocations):
            if invocation.status != 'complete':
                print((directory / 'checks' / f'{kind}-{index}' / 'output.log').read_text())
    assert result.ranks[0].status == 'complete'
    if broken:
        assert result.status == 'FAILED' and result.tiles[0].status == 'failed'
        assert set(result.objects) == {'megakernel_d0.o'}
    else:
        assert result.status == 'DEVICE_CHECKED' and result.tiles[0].status == 'complete'
        assert set(result.objects) == {'megakernel_d0.o', 'native_tile_0.o'}
    defines = result.toolchain['native_units'][0]['defines']
    if name == 'moe_layer':
        assert defines['D'] == (256 if small else 1024)
        assert defines['S'] == (128 if small else 512)
        invocation = json.loads((directory / 'checks' / 'tile-0' / 'request.json').read_text())
        assert f'-DD={defines["D"]}' in invocation['argv']
    assert result.status != 'EXECUTABLE'
