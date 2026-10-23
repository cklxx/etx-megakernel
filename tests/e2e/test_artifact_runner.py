"""Artifact runner uses real offline builds with explicitly scripted execution."""
from dataclasses import replace
import json
import os
import sys
from pathlib import Path
import shutil

import pytest
from etx.external import (ArtifactRunRequest, ArtifactRun, KernelSpec, PlanRequest,
                          build, emit, kernel, plan, load_build, run_artifact, read_artifact_run,
                          ProcessRequest, execute)
from etx.testing import scripted_executor, recorded_executor


def request_for(directory, identity, cwd):
    return ArtifactRunRequest(str(directory), identity, str(cwd), 10,
                              arguments=('--check',), environment={'ETX_EXPLICIT': '1'}, silence_s=2)


def test_run_requires_explicit_native_activation_before_reading_artifact(tmp_path):
    request = request_for(tmp_path / 'absent', '0' * 64, tmp_path)
    with pytest.raises(ValueError, match='explicit'):
        run_artifact(request, output=tmp_path / 'not-created')
    with pytest.raises(ValueError, match='substitute'):
        run_artifact(request, output=tmp_path / 'not-created', execute_native=True,
                     executor=scripted_executor(request.process_request()))
    assert not (tmp_path / 'not-created').exists()
    request_file = tmp_path / 'run-request.json'
    request_file.write_text(json.dumps(request.to_dict()))
    command = ProcessRequest((sys.executable, '-m', 'etx', 'run', str(request_file),
                              '--out', str(tmp_path / 'not-created')),
                             str(Path(__file__).resolve().parents[2]), 20)
    assert execute(command, output=tmp_path / 'cli-refusal').status == 'failed'
    assert 'explicit execute_native=True' in (tmp_path / 'cli-refusal/output.log').read_text()
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('fields', [
    {'directory': 'relative'}, {'build_identity': 'bad'}, {'cwd': 'relative'},
    {'arguments': '--check'}, {'arguments': ('bad\0arg',)}, {'deadline_s': 0},
    {'silence_s': float('nan')}, {'environment': {'invalid': 1}}, {'schema_version': True},
])
def test_invalid_run_requests(tmp_path, fields):
    request = request_for(tmp_path, '0' * 64, tmp_path)
    with pytest.raises((TypeError, ValueError)):
        replace(request, **fields)


@pytest.fixture(scope='module')
def linked_operator(tmp_path_factory):
    retained = os.environ.get('ETX_API_RUNNER_BUILD')
    if retained:
        directory = Path(retained).resolve()
        receipt = load_build(directory)
        assert receipt.status == 'EXECUTABLE'
        return directory, receipt
    image = os.environ.get('ETX_TEST_ROCM_DOCKER')
    if not image:
        pytest.skip('offline Docker runner artifact check not requested')
    root = Path(os.environ.get('ETX_API_BUILD_EVIDENCE', str(tmp_path_factory.mktemp('artifact-runner'))))
    root.mkdir(parents=True, exist_ok=True)
    directory = root / 'runner-splitk'
    source = emit(plan(PlanRequest(kernel(KernelSpec('splitk', 'graph', {'provider': 'splitk_sum'})),
                                  bindings={'n': 2})))
    receipt = build(source, output=directory, docker='/usr/bin/docker', image=image, deadline_s=90)
    assert receipt.status == 'EXECUTABLE'
    return directory, receipt


@pytest.mark.parametrize('returncode', [0, 7, 124])
def test_artifact_run_relocation_replay_and_evidence(tmp_path, linked_operator, returncode):
    directory, receipt = linked_operator
    relocated = tmp_path / 'relocated'
    shutil.copytree(directory, relocated)
    assert load_build(relocated) == receipt
    request = request_for(relocated, receipt.identity, tmp_path)
    process = request.process_request()
    output = tmp_path / 'scripted'
    result = run_artifact(request, output=output,
                          executor=scripted_executor(process, output=b'explicit substitute\x00\xff\n',
                                                     returncode=returncode))
    assert isinstance(result, ArtifactRun)
    assert result.process.backend == 'scripted'
    assert result.process.returncode == returncode
    assert result.process.status == {0: 'complete', 7: 'failed', 124: 'deadline'}[returncode]
    assert read_artifact_run(output, request=request) == result
    evidence = tmp_path / 'relocated-evidence'
    shutil.copytree(output, evidence)
    assert read_artifact_run(evidence, request=request) == result
    request_file = tmp_path / 'run-request.json'
    request_file.write_text(json.dumps(request.to_dict()))
    cli = execute(ProcessRequest((sys.executable, '-m', 'etx', 'inspect-run', str(evidence),
                                  '--request', str(request_file)),
                                 str(Path(__file__).resolve().parents[2]), 20), output=tmp_path / 'inspect-cli')
    assert cli.returncode == (0 if returncode == 0 else 1)
    assert json.loads((tmp_path / 'inspect-cli/output.log').read_text()) == dict(
        result.to_dict(), identity=result.identity)
    replay = run_artifact(request, output=tmp_path / 'replay', executor=recorded_executor(output))
    assert replay.process.backend == 'recorded'
    assert replay.process.output_sha256 == result.process.output_sha256
    with pytest.raises(ValueError, match='identity'):
        read_artifact_run(output, request=replace(request, build_identity='0' * 64))
    with pytest.raises(ValueError, match='process identity'):
        replace(result, request=replace(request, arguments=()))
    (evidence / 'output.log').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        read_artifact_run(evidence, request=request)


@pytest.mark.parametrize('damage', ['identity', 'binary', 'source', 'symlink'])
def test_artifact_rejected_before_executor(tmp_path, linked_operator, damage):
    directory, receipt = linked_operator
    relocated = tmp_path / 'relocated'
    shutil.copytree(directory, relocated)
    request = request_for(relocated, receipt.identity, tmp_path)
    if damage == 'identity':
        request = replace(request, build_identity='0' * 64)
    elif damage == 'binary':
        with (relocated / 'run').open('ab') as binary:
            binary.write(b'corruption')
    elif damage == 'source':
        (relocated / 'examples/hosts/splitk.hip').write_text('corruption')
    else:
        (relocated / 'run').unlink()
        (relocated / 'run').symlink_to(directory / 'run')
    executor = scripted_executor(request.process_request())
    with pytest.raises(ValueError):
        run_artifact(request, output=tmp_path / 'not-created', executor=executor)
    assert not executor.used
    assert not (tmp_path / 'not-created').exists()
