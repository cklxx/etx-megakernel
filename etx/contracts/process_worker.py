"""Isolated transport for session.proc; never supervise a caller's unrelated children.

Invoked with python -I and absolute paths. The only loaded implementation is the
packaged supervisor next to this adapter, not code from cwd or the evidence tree.
"""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import sys
import time


def main(directory):
    directory = Path(directory)
    request = json.loads((directory / 'request.json').read_text())
    spec = importlib.util.spec_from_file_location(
        'etx_process_supervisor', Path(__file__).resolve().parents[1] / 'session' / 'proc.py')
    proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proc)
    events = []
    status, returncode = 'error', None
    diagnostics = io.StringIO()
    last_output, last_size, last_warning = time.monotonic(), 0, 0.0

    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)

    with (directory / 'output.log').open('xb') as output:
        def progress(elapsed):
            nonlocal last_output, last_size, last_warning
            size = (directory / 'output.log').stat().st_size
            now = time.monotonic()
            if size != last_size:
                last_output, last_size = now, size
            silent = now - last_output
            if silent >= request['silence_s'] and now - last_warning >= request['silence_s']:
                event = f'SILENCE elapsed={elapsed:.3f}s silent={silent:.3f}s bytes={size}'
                events.append(event)
                print(event, flush=True)
                last_warning = now

        try:
            print('START bounded host process', flush=True)
            returncode = proc.run(request['argv'], sink=output, cwd=request['cwd'],
                                  env=request['environment'], deadline_s=request['deadline_s'],
                                  poll_s=min(1.0, request['silence_s']), on_poll=progress,
                                  log=diagnostics)
            status = 'complete' if returncode == 0 else 'deadline' if returncode == proc.DEADLINE_RC else 'failed'
            if diagnostics.getvalue():
                events.append(diagnostics.getvalue())
                status, returncode = 'error', None
        except InterruptedError as error:
            status, returncode = 'interrupted', None
            events.append(str(error))
        except Exception as error:
            events.append(f'{type(error).__name__}: {error}')
        print(f'END status={status} returncode={returncode}', flush=True)
    with (directory / 'output.log').open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    result = dict(status=status, returncode=returncode, events=events, output_sha256=digest)
    with (directory / 'worker.json').open('x') as output:
        json.dump(result, output, sort_keys=True)


if __name__ == '__main__':
    main(sys.argv[1])
