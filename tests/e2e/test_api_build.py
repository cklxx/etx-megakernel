"""Public linked artifacts: offline compiler evidence, never device execution."""
from dataclasses import replace
import os
from pathlib import Path
import shutil

import pytest
from etx.external import (KernelSpec, PlanRequest, kernel, plan, emit, build, load_build,
                          BuildArtifact, DeviceCheck, ProcessResult)
from etx.testing import fixture


def sources(name='splitk_sum', *, small=False, inline=False, seed=12345):
    config = dict(provider=name)
    bindings = {'n': 2} if name == 'splitk_sum' else (
        {'M': 256, 'K': 64, 'N': 128} if name == 'gemm_rs' else {'B': 2, 'seed': seed})
    if name == 'moe_layer':
        config['small'] = small
    return emit(plan(PlanRequest(kernel(KernelSpec(name, 'graph', config)), bindings=bindings,
                                 options={'inline_tiles': inline})))


@pytest.mark.parametrize('kind', ['fixture', 'missing-host', 'seed', 'multi-rank-inline'])
def test_unsupported_host_refuses_before_launch(tmp_path, kind):
    artifact = sources(seed=7, name='moe_layer') if kind == 'seed' else sources()
    if kind == 'multi-rank-inline':
        artifact = sources('gemm_rs', inline=True)
    elif kind == 'fixture':
        artifact = emit(plan(PlanRequest(fixture())))
    elif kind == 'missing-host':
        files = artifact.files.to_dict()
        del files['examples/hosts/splitk.hip']
        artifact = replace(artifact, files=files)
    with pytest.raises(ValueError):
        build(artifact, output=tmp_path / 'not-created', docker='/not/docker', image='unused')
    assert not (tmp_path / 'not-created').exists()


def test_link_status_requires_real_success_and_exact_payloads():
    actual = ProcessResult('0' * 64, 'host-process', 'complete', 0, '0' * 64)
    checked = DeviceCheck('0' * 64, {}, (actual,), {'megakernel_d0.o': '0' * 64}, 'DEVICE_CHECKED')
    failed = replace(actual, status='failed', returncode=1)
    for link, files, status in [
        (None, {}, 'FAILED'),
        (replace(actual, backend='scripted'), {'run': '0' * 64}, 'EXECUTABLE'),
        (failed, {'run': '0' * 64}, 'EXECUTABLE'),
        (actual, {}, 'EXECUTABLE'),
        (actual, {'../run': '0' * 64}, 'EXECUTABLE'),
    ]:
        with pytest.raises(ValueError):
            BuildArtifact(checked, link, files, status)
    assert BuildArtifact(checked, failed, {}, 'FAILED').status == 'FAILED'


@pytest.mark.skipif(not os.environ.get('ETX_TEST_ROCM_DOCKER'), reason='offline Docker linker check not requested')
@pytest.mark.parametrize('name,small,inline,damage', [
    ('splitk_sum', False, False, None),
    ('moe_layer', False, False, None),
    ('moe_layer', True, False, None),
    ('moe_layer', True, True, None),
    ('gemm_rs', False, False, None),
    ('splitk_sum', False, False, 'host'),
    ('splitk_sum', False, False, 'tile'),
])
def test_linked_operator_artifacts_and_failure_gates(tmp_path, name, small, inline, damage):
    artifact = sources(name, small=small, inline=inline)
    if damage:
        files = artifact.files.to_dict()
        path = 'examples/' + ('hosts/' if damage == 'host' else 'tiles/') + 'splitk.hip'
        files[path] += '\n#error deliberate_link_input_failure\n'
        artifact = replace(artifact, files=files)
    root = Path(os.environ.get('ETX_API_BUILD_EVIDENCE', str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f'linked-{name}-{small}-{inline}-{damage}'
    result = build(artifact, output=directory, docker='/usr/bin/docker',
                   image=os.environ['ETX_TEST_ROCM_DOCKER'], deadline_s=90)
    if result.link is not None and result.link.status != 'complete':
        print((directory / 'checks/link/output.log').read_text())
    assert result.device_check.source_identity == artifact.identity
    assert result.device_check.toolchain['gpu_access'] is False
    if damage:
        assert result.status == 'FAILED' and not result.files
        if damage == 'tile':
            assert result.link is None
            assert not (directory / 'checks/link').exists()
        else:
            assert result.device_check.status == 'DEVICE_CHECKED'
            assert result.link.status == 'failed'
    else:
        assert result.status == 'EXECUTABLE' and set(result.files) == {'run'}
        assert result.link.status == 'complete'
    assert load_build(directory) == result
    relocated = tmp_path / 'relocated'
    shutil.copytree(directory, relocated)
    assert load_build(relocated) == result
    if not damage:
        host_name = {'splitk_sum': 'splitk.hip', 'moe_layer': 'moe.hip', 'gemm_rs': 'gemm_rs.hip'}[name]
        for path in ('run', 'native_tile_0.o', 'examples/hosts/' + host_name,
                     'checks/link/output.log', 'checks/version/output.log', 'checks/link-cleanup/output.log'):
            target = relocated / path
            original = target.read_bytes()
            target.write_bytes(original + b'corruption')
            with pytest.raises(ValueError):
                load_build(relocated)
            target.write_bytes(original)
        assert load_build(relocated) == result
