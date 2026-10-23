"""Process adapter: durable records and the existing isolated session supervisor."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
import subprocess
import sys

from etx.api.execution import ProcessRequest, ProcessResult


def _write(path, record):
    with path.open('x') as output:
        json.dump(record, output, sort_keys=True, indent=2)
        output.write('\n')


def _digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def execute(request, *, output, executor=None):
    if not isinstance(request, ProcessRequest):
        raise TypeError('execute requires ProcessRequest')
    directory = Path(output).absolute()
    if not Path(request.cwd).is_dir():
        raise ValueError('process cwd is not a directory')
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / 'request.json', request.to_dict())
    if executor is not None:
        result = executor.execute(request, directory)
        if not isinstance(result, ProcessResult) or result.backend not in ('scripted', 'recorded'):
            raise ValueError('substitute executor must return explicitly non-device evidence')
    else:
        # The worker invokes session.proc.run. This transport does not implement
        # a second deadline or tree sweep, and never reaps unrelated caller children.
        worker = Path(__file__).with_name('process_worker.py').resolve()
        with (directory / 'supervisor.log').open('xb') as log:
            completed = subprocess.run([sys.executable, '-I', str(worker), str(directory)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0 or not (directory / 'worker.json').is_file():
            raise RuntimeError(f'process supervisor incomplete; evidence retained in {directory}')
        fields = json.loads((directory / 'worker.json').read_text())
        result = ProcessResult(request.identity, 'host-process', **fields)
    if result.request_identity != request.identity or result.output_sha256 != _digest(directory / 'output.log'):
        raise ValueError('executor evidence identity mismatch')
    _write(directory / 'result.json', result.to_dict())
    return result


def run_logged(argv, log, *, seconds):
    """Legacy argv/log contract, with inherited environment kept out of receipts."""
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError('argv list required; shell commands forbidden')
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('session deadline/budget exhausted')
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print('STAGE START', json.dumps(argv), flush=True)
    with log.open('xb'):
        pass
    from etx.contracts.worker import stream_worker
    with tempfile.TemporaryDirectory(prefix='etx-log-worker-') as temporary:
        directory = Path(temporary)
        _write(directory / 'request.json', dict(argv=argv, log=str(log.absolute()), seconds=seconds))
        worker = Path(__file__).with_name('log_worker.py').resolve()
        code = stream_worker([sys.executable, '-I', str(worker), str(directory), str(os.getpid())])
        response = directory / 'response.json'
        if code != 0 or not response.is_file():
            raise RuntimeError(f'stage supervisor incomplete; log retained at {log}')
        fields = json.loads(response.read_text())
        if 'error' in fields:
            error = fields['error']
            exceptions = {cls.__name__: cls for cls in (ValueError, TypeError, FileNotFoundError,
                          PermissionError, OSError, RuntimeError, TimeoutError, KeyboardInterrupt)}
            raise exceptions.get(error['type'], RuntimeError)(error['message'])
        print('STAGE EXIT', fields['returncode'], str(log), flush=True)
        return fields['returncode']


def read_process_result(directory, *, request):
    """Revalidate stored request and complete output before returning evidence."""
    if not isinstance(request, ProcessRequest):
        raise TypeError('read_process_result requires ProcessRequest')
    directory = Path(directory)
    stored = ProcessRequest(**json.loads((directory / 'request.json').read_text()))
    result = ProcessResult(**json.loads((directory / 'result.json').read_text()))
    if stored != request or result.request_identity != request.identity:
        raise ValueError('process request identity mismatch')
    if result.output_sha256 != _digest(directory / 'output.log'):
        raise ValueError('process output checksum mismatch')
    return result
