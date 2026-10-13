"""etx/session/proc.py: one step's output, deadline and callback, and no process of the step alive afterwards."""
import io
import os
import sys
import time

import pytest

from etx.session import proc


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_streams_and_returns_the_exit_code():
    out = []
    assert proc.run([sys.executable, "-c", "print('hello'); raise SystemExit(3)"], sink=out.append) == 3
    assert "".join(out) == "hello\n"


def test_deadline_kills_a_child_that_holds_the_pipe(tmp_path):
    # review 08, R14: a build-shaped shell whose compiler child keeps stdout open past the deadline
    marker = tmp_path / "late"
    start = time.monotonic()
    rc = proc.run(["bash", "-c", f"(sleep 1; touch {marker}) & wait"], sink=lambda s: None, deadline_s=0.2)
    assert rc == proc.DEADLINE_RC and time.monotonic() - start < 0.9
    time.sleep(1.2)
    assert not marker.exists()


def test_a_background_child_does_not_outlive_its_step(tmp_path):
    pidfile = tmp_path / "pid"
    log = io.StringIO()
    with (tmp_path / "out.log").open("w") as sink:
        assert proc.run(["bash", "-c", f"sleep 30 & echo $! > {pidfile}"], sink=sink, log=log) == 0
    assert not _alive(int(pidfile.read_text()))


def test_a_raising_callback_stops_the_step(tmp_path):
    def watchdog(elapsed):
        raise TimeoutError("stuck")
    with pytest.raises(TimeoutError, match="stuck"), (tmp_path / "out.log").open("w") as sink:
        proc.run(["sleep", "30"], sink=sink, poll_s=0.05, on_poll=watchdog)
