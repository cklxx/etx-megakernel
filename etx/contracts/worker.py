"""Shared isolated-worker transport; the worker owns deadlines and tree cleanup."""
import signal
import subprocess
import sys


def stream_worker(argv):
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True,
                               text=True, errors='replace', bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        return process.wait()
    except BaseException:
        process.send_signal(signal.SIGTERM)
        # Do not kill a supervisor out from under its separately grouped children.
        while process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print('Waiting for child cleanup; supervisor remains alive', file=sys.stderr, flush=True)
        raise
    finally:
        process.stdout.close()
