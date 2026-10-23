"""Validate the linked artifact before delegating to the shared process supervisor."""
import json
import platform
from pathlib import Path

from etx.api.runs import ArtifactRunRequest, ArtifactRun
from etx.api.execution import ProcessResult
from etx.artifacts.builds import load_build
from .execution import execute, read_process_result, _write


def run_artifact(request, *, output, executor=None, execute_native=False):
    if not isinstance(request, ArtifactRunRequest):
        raise TypeError('run_artifact requires ArtifactRunRequest')
    if type(execute_native) is not bool:
        raise ValueError('execute_native must be boolean')
    if executor is None and not execute_native:
        raise ValueError('native artifact execution requires explicit execute_native=True')
    if executor is not None and execute_native:
        raise ValueError('substitute execution cannot request native execution')
    artifact = load_build(request.directory)
    if artifact.identity != request.build_identity:
        raise ValueError('artifact build identity mismatch')
    if artifact.status != 'EXECUTABLE':
        raise ValueError('artifact is not executable')
    if executor is None and (platform.system() != 'Linux' or platform.machine() != 'x86_64'):
        raise ValueError('linked artifact requires a Linux x86-64 host')
    directory = Path(output).absolute()
    result = execute(request.process_request(), output=directory, executor=executor)
    record = ArtifactRun(request, result)
    _write(directory / 'artifact_run.json', dict(record.to_dict(), identity=record.identity))
    return record


def read_artifact_run(directory, *, request):
    if not isinstance(request, ArtifactRunRequest):
        raise TypeError('read_artifact_run requires ArtifactRunRequest')
    directory = Path(directory)
    for name in ('artifact_run.json', 'request.json', 'result.json', 'output.log'):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('artifact run requires regular evidence files')
    data = json.loads((directory / 'artifact_run.json').read_text())
    record = ArtifactRun(ArtifactRunRequest(**data['request']), ProcessResult(**data['process']),
                         schema_version=data['schema_version'])
    if record.request != request or record.identity != data['identity']:
        raise ValueError('artifact run identity mismatch')
    process = read_process_result(directory, request=request.process_request())
    if process != record.process:
        raise ValueError('artifact run process evidence mismatch')
    return record
