"""Source artifacts use the same planner/emitter and do not claim native builds."""
from dataclasses import replace
import json
import shutil

import pytest
from etx.external import PlanRequest, plan, emit, save_sources, load_sources, check_device_sources, DeviceCheck
from etx.external import ProcessResult
from etx.testing import fixture


def test_device_check_rejects_substituted_evidence_and_missing_objects():
    scripted = ProcessResult('0' * 64, 'scripted', 'complete', 0, '0' * 64)
    with pytest.raises(ValueError, match='actual host compiler'):
        DeviceCheck('0' * 64, {}, (scripted,), {'megakernel_d0.o': '0' * 64}, 'DEVICE_CHECKED')
    actual = replace(scripted, backend='host-process')
    with pytest.raises(ValueError, match='objects for every successful rank'):
        DeviceCheck('0' * 64, {}, (actual,), {}, 'DEVICE_CHECKED')
    failed = replace(actual, status='failed', returncode=1)
    with pytest.raises(ValueError, match='status disagrees'):
        DeviceCheck('0' * 64, {}, (failed,), {}, 'DEVICE_CHECKED')


def test_native_tile_receipts_cannot_admit_partial_compilation():
    actual = ProcessResult('0' * 64, 'host-process', 'complete', 0, '0' * 64)
    failed = replace(actual, status='failed', returncode=1)
    rank_object = {'megakernel_d0.o': '0' * 64}
    with pytest.raises(ValueError, match='status disagrees'):
        DeviceCheck('0' * 64, {}, (actual,), rank_object, 'DEVICE_CHECKED', (failed,))
    with pytest.raises(ValueError, match='objects for every'):
        DeviceCheck('0' * 64, {}, (actual,), rank_object, 'DEVICE_CHECKED', (actual,))
    with pytest.raises(ValueError, match='actual host compiler'):
        DeviceCheck('0' * 64, {}, (actual,), rank_object, 'FAILED',
                    (replace(actual, backend='scripted'),))
    with pytest.raises(ValueError, match='objects for every'):
        DeviceCheck('0' * 64, {}, (actual,),
                    dict(rank_object, native_tile_0='0' * 64), 'FAILED', (failed,))
    receipt = DeviceCheck('0' * 64, {}, (actual,), rank_object, 'FAILED', (failed,))
    assert receipt.to_dict()['tiles'][0]['status'] == 'failed'


def test_device_check_rejects_unresolved_source_dependencies_before_launch(tmp_path):
    artifact = emit(plan(PlanRequest(fixture())))
    with pytest.raises(ValueError, match='self-contained'):
        check_device_sources(replace(artifact, external_sources=('unresolved.hip',)),
                             output=tmp_path / 'rejected', docker='/not/docker', image='not-an-image')
    assert not (tmp_path / 'rejected').exists()


@pytest.mark.parametrize('devices', [1, 2, 8])
def test_all_rank_source_artifacts_roundtrip_and_relocate(tmp_path, devices):
    planned = plan(PlanRequest(fixture(devices=devices, batch=2)))
    artifact = emit(planned)
    assert artifact.status == 'SOURCES'
    assert artifact.rank_sources == tuple(f'megakernel_d{i}.hip' for i in range(devices))
    assert artifact.rank_headers == tuple(f'plan_data_d{i}.h' for i in range(devices))
    assert artifact.files['plan_data.h'] == artifact.files['plan_data_d0.h']
    assert 'include/etx/abi.h' in artifact.files
    assert artifact.identity == emit(planned).identity
    save_sources(artifact, tmp_path / 'original')
    shutil.copytree(tmp_path / 'original', tmp_path / 'relocated')
    assert load_sources(tmp_path / 'relocated') == artifact
    with pytest.raises(FileExistsError):
        save_sources(artifact, tmp_path / 'original')
    with pytest.raises(ValueError, match='compilation or execution'):
        replace(artifact, status='EXECUTABLE')


@pytest.mark.parametrize('damage', ['missing-rank', 'payload', 'manifest', 'traversal', 'symlink'])
def test_source_artifact_rejects_corruption(tmp_path, damage):
    artifact = emit(plan(PlanRequest(fixture(devices=2))))
    directory = tmp_path / 'artifact'
    save_sources(artifact, directory)
    if damage == 'missing-rank':
        (directory / artifact.rank_headers[-1]).unlink()
    elif damage == 'payload':
        (directory / artifact.rank_sources[-1]).write_text('changed')
    elif damage == 'manifest':
        data = json.loads((directory / 'manifest.json').read_text())
        data['identity'] = '0' * 64
        (directory / 'manifest.json').write_text(json.dumps(data))
    elif damage == 'traversal':
        data = json.loads((directory / 'manifest.json').read_text())
        data['payloads']['../outside'] = '0' * 64
        (directory / 'manifest.json').write_text(json.dumps(data))
    else:
        payload = directory / artifact.rank_sources[0]
        external = tmp_path / 'external.hip'
        external.write_text(payload.read_text())
        payload.unlink()
        payload.symlink_to(external)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_sources(directory)


def test_source_rank_inventory_and_replanning_are_checked():
    planned = plan(PlanRequest(fixture(devices=2)))
    artifact = emit(planned)
    with pytest.raises(ValueError, match='every rank'):
        replace(artifact, rank_headers=artifact.rank_headers[:-1])
    report = planned.report.to_dict()
    report['devices'] = 42
    with pytest.raises(ValueError, match='identity drift'):
        emit(replace(planned, report=report))


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a//b', './x', 'a\\b', 'manifest.json'])
def test_source_paths_are_relative_and_normalized(name):
    artifact = emit(plan(PlanRequest(fixture())))
    with pytest.raises(ValueError):
        replace(artifact, files={**artifact.files.to_dict(), name: 'payload'})
