"""The one process supervisor of a session (docs/SESSION-ARCH-2026-10-13.md, component A).

Invariant: when a step (a runner job, or an invocation inside one) ends -- normally, at its deadline, or stopped --
no process it started is alive. Process groups cannot hold that (GNU `timeout` leaves them, compilers fork during a
kill: R01, R14, R15), so the supervisor is a Linux child subreaper and `sweep` kills its whole remaining tree,
re-scanning until nothing is left (a process with SIGKILL pending cannot fork). stdlib only: the runner starts each
job as `python -I etx/session/proc.py -- argv...` (the shim), from any job directory.
"""
import os
import selectors
import signal
import subprocess
import sys
import time

DEADLINE_RC = 124            # GNU timeout's code: one meaning of "deadline" for every caller
SWEEP_S = 30.0               # a process stuck in the kernel (a hung GPU call) may take a while to die


def subreaper() -> None:
    """Linux: orphans of this process's children re-parent here, not to init. A no-op elsewhere."""
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)          # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        pass


def _parents() -> dict[int, int]:
    """pid -> ppid of every process, from /proc (Linux; elsewhere empty: the step's process group is all we reach)."""
    out = {}
    for d in filter(str.isdigit, os.listdir("/proc") if os.path.isdir("/proc/self/task") else ()):
        try:
            out[int(d)] = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            pass
    return out


def tree(roots, parents) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        children.setdefault(ppid, []).append(pid)
    out, stack = set(), [p for p in roots if p in parents]
    while stack:
        pid = stack.pop()
        if pid not in out:
            out.add(pid)
            stack.extend(children.get(pid, []))
    return out


def sweep(keep=frozenset(), group=None) -> list[int]:
    """SIGKILL every descendant of this process outside the subtrees at `keep` (and the process group `group`, all
    that reaches orphans without a subreaper), reap ours, repeat until none is left. Returns survivors after SWEEP_S."""
    me, end = os.getpid(), time.monotonic() + SWEEP_S
    while True:
        if group is not None:
            try:
                os.killpg(group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        parents = _parents()
        victims = tree([p for p, pp in parents.items() if pp == me and p not in keep], parents) - tree(keep, parents)
        if not victims or time.monotonic() > end:
            return sorted(victims)
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
                if parents[pid] == me:
                    os.waitpid(pid, os.WNOHANG)
            except (ProcessLookupError, PermissionError, ChildProcessError):
                pass
        time.sleep(0.005)


def run(argv, *, sink, deadline_s=None, poll_s=1.0, on_poll=None, env=None, cwd=None, log=None) -> int:
    """Run one step in this process; leave nothing of it alive. sink: an open file, or a callable taking the output
    text (streamed through a pipe). on_poll(elapsed_s) runs every poll_s; if it raises, the step is killed and the
    exception propagates. Returns the exit code, or DEADLINE_RC when deadline_s passed first."""
    subreaper()
    keep = {p for p, pp in _parents().items() if pp == os.getpid()}        # children this process already had
    pipe = callable(sink)
    p = subprocess.Popen(list(map(str, argv)), stdout=subprocess.PIPE if pipe else sink, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, env=env, cwd=cwd)
    selector, started, rc = selectors.DefaultSelector(), time.monotonic(), None
    if pipe:
        selector.register(p.stdout, selectors.EVENT_READ)
    try:
        while rc is None:
            left = poll_s if deadline_s is None else max(0.0, min(poll_s, deadline_s - (time.monotonic() - started)))
            if selector.get_map():
                for key, _ in selector.select(timeout=left):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        sink(chunk.decode(errors="replace"))
                    else:
                        selector.unregister(key.fileobj)
            else:
                try:
                    p.wait(timeout=left)
                except subprocess.TimeoutExpired:
                    pass
            elapsed = time.monotonic() - started
            if p.poll() is not None and not selector.get_map():
                rc = p.returncode
            elif deadline_s is not None and elapsed >= deadline_s:
                rc = DEADLINE_RC
            elif on_poll is not None:
                on_poll(elapsed)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        survivors = sweep(keep, group=p.pid)
        selector.close()
        if pipe:
            p.stdout.close()
        for stream in {log, sys.stderr} - {None} if survivors else ():
            print(f"SURVIVORS after sweep (stuck in the kernel?): {survivors}", file=stream, flush=True)
    return rc


def shim(argv) -> int:
    """The runner's per-job process: run argv; on its exit or on SIGTERM / SIGINT / SIGHUP, kill all the job started."""
    subreaper()
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: stop.append(signum))
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
    while child.poll() is None and not stop:
        time.sleep(0.02)
    if child.poll() is None:
        child.kill()
        child.wait()
    survivors = sweep(group=child.pid)
    if survivors:
        print(f"SURVIVORS after sweep (stuck in the kernel?): {survivors}", flush=True)
    return 128 + stop[0] if stop else child.returncode


if __name__ == "__main__":
    if sys.argv[1:2] != ["--"] or len(sys.argv) < 3:
        raise SystemExit("usage: python -I etx/session/proc.py -- argv...")
    raise SystemExit(shim(sys.argv[2:]))
