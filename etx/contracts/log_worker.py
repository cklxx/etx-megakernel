"""Compatibility worker for inherited-environment, byte-preserving stage logs."""
import io
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from etx.session import proc


def main(directory, parent):
    directory = Path(directory)
    request = json.loads((directory / 'request.json').read_text())
    stopping = None

    def interrupted(signum, frame):
        nonlocal stopping
        # Never raise asynchronously: a first signal may arrive while proc.run
        # is already sweeping descendants or establishing its cleanup scope.
        if stopping is None:
            stopping = signum

    def check_stop():
        if stopping is not None:
            raise KeyboardInterrupt(f'stage interrupted by signal {stopping}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        if sys.platform.startswith('linux'):
            import ctypes
            ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)
        if os.getppid() != parent:
            raise KeyboardInterrupt('stage caller exited before worker setup')
        log = Path(request['log'])
        diagnostics = io.StringIO()
        progress_at = 0.0

        def progress(elapsed):
            nonlocal progress_at
            check_stop()
            if elapsed >= request['seconds']:
                print(f'NOT-COMPLETE deadline supervisor={os.getpid()} log={log}', flush=True)
                raise TimeoutError('stage deadline; no performance verdict')
            if elapsed - progress_at >= 10:
                print(f'PROGRESS supervisor={os.getpid()} elapsed_s={elapsed:.1f} log_bytes={log.stat().st_size} log={log}', flush=True)
                progress_at = elapsed

        # The parent reserves the log exclusively before launching this worker.
        with log.open('ab') as output:
            check_stop()
            code = proc.run(request['argv'], sink=output, poll_s=.01,
                            on_poll=progress, log=diagnostics)
        # A signal during normal-exit cleanup must still report interruption,
        # but only after proc.run has finished removing its descendants.
        check_stop()
        if diagnostics.getvalue():
            raise RuntimeError(diagnostics.getvalue())
        response = dict(returncode=code)
    except BaseException as error:
        response = dict(error=dict(type=type(error).__name__, message=str(error)))
    with (directory / 'response.json').open('x') as output:
        json.dump(response, output)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
