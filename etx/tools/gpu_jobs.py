"""Run a GPU session's jobs on every GPU at once (and on several machines), with the discipline CLAUDE.md asks for.

A session is a Python file that defines JOBS, a list of dicts:

  name      unique; the log is <logdir>/<name>.log (full output, never filtered)
  cmd       bash command (run with `bash -c`, from `cwd`); $GPU holds the GPU id the job got
  lane      "gpu" (default): one job per GPU at a time, STRICTLY in list order per GPU (a job never overtakes an
            earlier one that is still pending; a skip advances the queue); "cpu" (builds, downloads: --cpu-slots in
            parallel); "final": one at a time, after every other job is terminal (reports)
  gpu       pin to one GPU id (A/B pairs that must share a device); default: any free GPU. A pin to a GPU the
            machine does not have folds onto the first GPU (so a two-GPU session also runs, serially, on one)
  writes    files the job produces: relative to the record, or absolute / ~ / $VAR paths (the job's env first). A job
            that exits 0 without one of them FAILS ("declared output missing"); a leading "?" marks an output the job
            may decide not to write (a GO token): its readers are skipped when it is absent
  reads     what the job needs: a file (written by the job that declares it, or present before the session) or a job
            name (that job's success). The job waits until each producer finished, and is skipped if one did not
            succeed or a file without producer is absent. A leading "?" waits for the producer, whatever its result.
  timeout   seconds (default 900); on expiry the job's whole process tree is killed and the job reported "timeout"
  env       extra environment
  machine   which machine runs it (default "A"): one runner per machine, `--machine B` on the second VM
  summary   regexes; matching log lines are printed when the job ends (the log keeps everything)
  quiet_s   warn when the job has printed nothing for this long (default 180)

A session may define MAX_SESSION_SECONDS for an aggregate hard stop from runner
start, including setup. At expiry all running process groups are killed and
pending jobs skipped; the runner exits nonzero. VM deletion remains the operator's
responsibility; start earlier to respect an allocation-time budget.
ETX_SESS_MAX_SECONDS overrides MAX_SESSION_SECONDS for one run and must be finite and positive.

Every GPU job gets HIP_VISIBLE_DEVICES=$GPU, so a single-GPU program runs on the lane's GPU; a docker command passes
`-e HIP_VISIBLE_DEVICES=$GPU` itself. Order the list by priority (microbenchmarks and correctness first, captures last):
GPU jobs run strictly in list order per GPU, so the list order IS the priority.

  python3 -m etx.tools.gpu_jobs examples/sessions/next.py --gpus 0,1 --logdir ~/vlnext [--machine A] [--dry-run]

Output: one line per start / end with the elapsed session time, a progress block every --progress-s seconds (running
jobs, their last log line, and a warning for any job silent past quiet_s), and at the end a status table.
<logdir>/status.json is atomically refreshed on every transition (including skipped jobs). Exit code 0 only if every job that ran finished OK.
The session's record is a NEW directory <logdir>/<session id> (etx/session/record.py), printed at start; every job
receives it as ETX_SESS_LOGDIR. After the last job the runner seals it (MANIFEST + SEAL); a session file may name
LOCAL_ONLY globs (what a fetch leaves on the box).
Bare Python commands use the runner's interpreter directory first on PATH;
commands naming an explicit interpreter path keep that choice.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from etx.session import proc as sproc
from etx.session import record


@dataclass
class Job:
    name: str
    cmd: str
    lane: str = "gpu"
    gpu: int | None = None
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    timeout: float = 900.0
    env: dict[str, str] = field(default_factory=dict)
    machine: str = "A"
    summary: list[str] = field(default_factory=list)
    quiet_s: float = 180.0
    cwd: str = "~/etx"
    # runtime
    status: str = "pending"          # pending | running | ok | fail | timeout | skipped
    rc: int | None = None
    start: float = 0.0
    end: float = 0.0
    on_gpu: int | None = None
    proc: subprocess.Popen | None = None
    log: Path | None = None
    note: str = ""

    def deps(self, optional=False) -> list[str]:
        """The jobs this one needs to succeed, or (optional) waits for whatever their result: its bare-word reads."""
        return [r.lstrip("?") for r in self.reads if r.startswith("?") == optional and not set("/.") & set(r)]


def session_limit(limit: float | None) -> float | None:
    if "ETX_SESS_MAX_SECONDS" in os.environ:
        try:
            limit = float(os.environ["ETX_SESS_MAX_SECONDS"])
        except ValueError:
            raise SystemExit("MAX_SESSION_SECONDS must be finite and positive") from None
    if limit is not None and (not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0):
        raise SystemExit("MAX_SESSION_SECONDS must be finite and positive")
    return limit


def load_session(path: str | Path) -> tuple[list[Job], float | None]:
    jobs, limit, _ = _load(path)
    return jobs, limit


def _load(path):
    spec = importlib.util.spec_from_file_location("etx_session", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    jobs = []
    for d in mod.JOBS:
        d = {k: [v] if k in ("reads", "writes", "summary") and isinstance(v, str) else v for k, v in d.items()}
        try:
            jobs.append(Job(**d))
        except TypeError as exc:
            raise SystemExit(f"job {d.get('name')}: {exc}") from None
    names = [j.name for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"duplicate job names: {sorted(dup)}")
    writers = [w.lstrip("?") for j in jobs for w in j.writes]
    for j in jobs:
        if j.lane not in ("gpu", "cpu", "final"):
            raise SystemExit(f"job {j.name}: lane must be gpu, cpu or final")
        for w in (w.lstrip("?") for w in j.writes):
            if writers.count(w) > 1 or w in names:
                raise SystemExit(f"job {j.name}: {w!r} has more than one writer or is a job name")
        by, writer = {x.name: x for x in jobs}, {w.lstrip("?"): x for x in jobs for w in x.writes}
        for r in (r.lstrip("?") for r in j.reads):
            if not set("/.") & set(r) and r not in names:         # a bare word names a job: catch typos at load
                raise SystemExit(f"job {j.name}: unknown dependency {r!r}")
            p = by.get(r, writer.get(r))
            if p is not None and p.lane == "final" and j.lane != "final":
                raise SystemExit(f"job {j.name}: reads the final job {p.name!r}, which runs after it")
            if p is not None and p.lane == j.lane == "gpu" and names.index(p.name) > names.index(j.name) and (
                    p.gpu is None or j.gpu is None or p.gpu == j.gpu):
                raise SystemExit(f"job {j.name}: reads {p.name!r}, a later job on its GPU lane (it would never run)")
    limit = session_limit(getattr(mod, "MAX_SESSION_SECONDS", None))
    return jobs, limit, tuple(getattr(mod, "LOCAL_ONLY", record.LOCAL_ONLY))


def load_jobs(path: str | Path) -> list[Job]:
    return load_session(path)[0]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class Runner:
    def __init__(self, jobs: list[Job], gpus: list[int], cpu_slots: int, logdir: Path, progress_s: float = 60.0,
                 out=None, poll_s: float = 1.0, max_seconds: float | None = None, local_only=record.LOCAL_ONLY):
        self.jobs = jobs
        self.gpus = gpus
        self.cpu_slots = cpu_slots
        self.logdir = logdir
        self.progress_s = progress_s
        self.out = out or sys.stdout
        self.poll_s = poll_s
        self.t0 = time.time()
        self.deadline = time.monotonic() + max_seconds if max_seconds is not None else None
        self.by_name = {j.name: j for j in jobs}
        self.writer = {w.lstrip("?"): j for j in jobs for w in j.writes}
        for j in jobs:
            if j.lane == "gpu" and j.gpu is not None and gpus and j.gpu not in gpus:
                # a session written for two GPUs still runs on a one-GPU machine: the pin folds onto the first
                j.note, j.gpu = f"pinned GPU {j.gpu} absent, run on GPU {gpus[0]}", gpus[0]
        self.local_only = local_only

    # -------------------------------------------------------------- helpers
    def say(self, msg: str) -> None:
        el = time.time() - self.t0
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())} +{int(el // 60):3d}m{int(el % 60):02d}s] {msg}", file=self.out, flush=True)

    def path(self, j: Job, name: str) -> Path:
        env = {**os.environ, **j.env, "ETX_SESS_LOGDIR": str(self.logdir)}
        p = os.path.expanduser(re.sub(r"\$\{?(\w+)\}?", lambda m: env.get(m[1], m[0]), name))
        return Path(p) if os.path.isabs(p) else self.logdir / p

    def _state(self, j: Job) -> str:
        """ready | wait | skip, from what the job reads (module docstring)"""
        for r in j.reads:
            key = r.lstrip("?")
            producer = self.by_name.get(key, self.writer.get(key))
            if producer is not None and producer.status in ("pending", "running"):
                return "wait"
            if r.startswith("?"):
                continue
            if producer is not None and producer.status != "ok":
                j.note = f"{key}: {producer.name} {producer.status}"
                return "skip"
            if key not in self.by_name and not self.path(j, key).exists():
                j.note = f"absent: {key}"
                return "skip"
        return "ready"

    def _start(self, j: Job, gpu: int | None) -> None:
        j.log = self.logdir / f"{j.name}.log"
        env = dict(os.environ)
        env.update({k: str(v) for k, v in j.env.items()})
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", os.defpath)
        if gpu is not None:
            env["HIP_VISIBLE_DEVICES"] = str(gpu)
            env["GPU"] = str(gpu)
        env["ETX_SESS_LOGDIR"] = str(self.logdir.resolve())
        env["PYTHONUNBUFFERED"] = "1"
        if self.deadline is not None:
            env["ETX_SESS_REMAINING_SECONDS"] = str(max(0, int(self.deadline - time.monotonic())))
        cwd = os.path.expanduser(j.cwd)
        fh = open(j.log, "w")
        fh.write(f"# job {j.name} gpu={gpu} cwd={cwd}\n# {j.cmd}\n")
        fh.flush()
        # the shim (etx/session/proc.py) owns the job's process tree: nothing the job started outlives it
        j.proc = subprocess.Popen([sys.executable, "-I", sproc.__file__, "--", "bash", "-c", j.cmd],
                                  cwd=cwd if os.path.isdir(cwd) else None, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        j.status, j.start, j.on_gpu = "running", time.time(), gpu
        self._write_status()
        self.say(f"start {j.name}" + (f" on GPU {gpu}" if gpu is not None else " (cpu)"))

    def _finish(self, j: Job, status: str) -> None:
        missing = [w for w in j.writes if w[0] != "?" and not self.path(j, w).exists()] if status == "ok" else []
        if missing:
            status, j.note = "fail", f"declared output missing: {missing}"
        j.status, j.end = status, time.time()
        j.rc = j.proc.returncode if j.proc else None
        self._write_status()
        tail = []
        if j.log and j.log.exists():
            text = j.log.read_text(errors="replace")
            for rx in j.summary:
                tail += [ln for ln in text.splitlines() if re.search(rx, ln)]
            if status != "ok" and not tail:
                tail = text.splitlines()[-5:]
        self.say(f"{status:7} {j.name} rc={j.rc} {j.end - j.start:.0f}s" + (f" [{j.note}]" if j.note else ""))
        for ln in tail[:40]:
            print(f"    {ln[:240]}", file=self.out)
        self.out.flush()

    def _kill(self, *jobs: Job) -> None:
        """Stop jobs: each shim kills its job's whole tree; a shim that does not finish is killed, and whatever it
        left re-parents to the runner (a subreaper) and is swept here."""
        for j in jobs:
            j.proc.send_signal(signal.SIGTERM)
        for j in jobs:
            try:
                j.proc.wait(sproc.SWEEP_S + 5)
            except subprocess.TimeoutExpired:
                j.proc.kill()
                j.proc.wait()
        survivors = sproc.sweep(keep={x.proc.pid for x in self.jobs if x.proc is not None})
        if survivors:
            self.say(f"SURVIVORS after sweep (stuck in the kernel?): {survivors}")

    def _progress(self) -> None:
        run = [j for j in self.jobs if j.status == "running"]
        waiting = sum(j.status == "pending" for j in self.jobs)
        self.say(f"progress: {len(run)} running, {waiting} waiting, "
                 f"{sum(j.status in ('ok', 'fail', 'timeout', 'skipped') for j in self.jobs)} done")
        now = time.time()
        for j in run:
            last, mtime = "", j.start
            if j.log and j.log.exists():
                mtime = j.log.stat().st_mtime
                lines = [ln for ln in j.log.read_text(errors="replace").splitlines() if ln.strip()]
                last = lines[-1][:160] if lines else ""
            silent = now - mtime
            flag = f"  SILENT {silent:.0f}s (probe it)" if silent > j.quiet_s else ""
            where = f"GPU {j.on_gpu}" if j.on_gpu is not None else "cpu"
            print(f"    {j.name} [{where}] {now - j.start:.0f}s / {j.timeout:.0f}s{flag}: {last}", file=self.out)
        self.out.flush()

    # -------------------------------------------------------------- main loop
    def run(self) -> int:
        """Run every job, then seal the record. The record directory is this session's alone: never one that holds
        anything (a record is write-once)."""
        sproc.subreaper()
        if self.logdir.exists() and any(self.logdir.iterdir()):
            raise SystemExit(f"refusing {self.logdir}: not empty; a session record is a new directory")
        self.logdir.mkdir(parents=True, exist_ok=True)
        out = self.out
        try:
            with (self.logdir / "runner.log").open("x") as log:
                self.out = _Tee(out, log)
                self.say(f"session record: {self.logdir}")
                return self._run()
        finally:
            self.out = out
            plan = json.dumps([[j.name, j.cmd, j.lane, j.gpu, j.reads, j.writes, j.timeout, j.env]
                               for j in self.jobs], default=str)
            s = record.seal(self.logdir, jobs={j.name: j.status for j in self.jobs},
                            plan=hashlib.sha256(plan.encode()).hexdigest(), local_only=self.local_only)
            print(f"SEALED {self.logdir} complete={s['complete']}", file=out, flush=True)

    def _run(self) -> int:
        self._write_status()
        last_progress = time.time()
        try:
            while True:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.say("HARD-STOP: session budget exhausted; killing running jobs")
                    self._kill(*[j for j in self.jobs if j.status == "running"])
                    for j in self.jobs:
                        if j.status == "running":
                            j.proc.wait()
                            j.note = "session hard stop"
                            self._finish(j, "timeout")
                        elif j.status == "pending":
                            j.status, j.note = "skipped", "session hard stop"
                    self._write_status()
                    self.report()
                    return 1
                now = time.time()
                for j in self.jobs:                                   # reap
                    if j.status == "running":
                        if j.proc.poll() is not None:
                            self._finish(j, "ok" if j.proc.returncode == 0 else "fail")
                        elif now - j.start > j.timeout:
                            self._kill(j)
                            j.note = f"timeout {j.timeout:.0f}s"
                            self._finish(j, "timeout")
                for j in self.jobs:                                   # skips propagate
                    if j.status == "pending" and self._state(j) == "skip":
                        j.status = "skipped"
                        self._write_status()
                        self.say(f"skip    {j.name} [{j.note}]")
                busy_gpus = {j.on_gpu for j in self.jobs if j.status == "running" and j.lane == "gpu"}
                cpu_busy = sum(j.status == "running" and j.lane == "cpu" for j in self.jobs)
                others_done = all(j.status not in ("pending", "running") for j in self.jobs if j.lane != "final")
                final_free = not any(j.status == "running" and j.lane == "final" for j in self.jobs)
                held = set()                                          # GPUs an earlier pending job holds in order
                for j in self.jobs:                                   # start, in list order
                    if j.status != "pending":
                        continue
                    ready = self._state(j) == "ready"
                    if j.lane == "cpu":
                        if ready and cpu_busy < self.cpu_slots:
                            self._start(j, None); cpu_busy += 1
                    elif j.lane == "final":
                        if ready and others_done and final_free:
                            self._start(j, None)
                        final_free = False                            # finals run one at a time, in order
                    else:
                        lanes = set(self.gpus) if j.gpu is None else {j.gpu}
                        free = [g for g in self.gpus if g in lanes - busy_gpus - held]
                        if ready and free:
                            self._start(j, free[0]); busy_gpus.add(free[0])
                        else:
                            held |= lanes
                if all(j.status not in ("pending", "running") for j in self.jobs):
                    break
                if time.time() - last_progress > self.progress_s:
                    self._progress(); last_progress = time.time()
                remaining = self.deadline - time.monotonic() if self.deadline is not None else self.poll_s
                time.sleep(max(0, min(self.poll_s, remaining)))
        except KeyboardInterrupt:
            for j in self.jobs:
                if j.status == "running":
                    self._kill(j); j.note = "interrupted"; self._finish(j, "fail")
            raise
        return self.report()

    def _write_status(self) -> None:
        # One scheduler thread owns all state transitions. Unique sibling temp
        # files + replace keep concurrent job readers on complete snapshots.
        payload = json.dumps(
            [{"name": j.name, "status": j.status, "rc": j.rc, "gpu": j.on_gpu,
              "seconds": (j.end - j.start) if j.end else None, "note": j.note}
             for j in self.jobs], indent=1)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.logdir, prefix=".status-", delete=False) as fh:
                tmp = Path(fh.name)
                fh.write(payload)
            os.replace(tmp, self.logdir / "status.json")
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def report(self) -> int:
        self.say("session done")
        print(f"  {'job':32} {'status':8} {'rc':>4} {'gpu':>4} {'seconds':>8}", file=self.out)
        for j in self.jobs:
            sec = f"{j.end - j.start:.0f}" if j.end else ""
            gpu = "" if j.on_gpu is None else str(j.on_gpu)
            print(f"  {j.name:32} {j.status:8} {'' if j.rc is None else j.rc:>4} {gpu:>4} {sec:>8}  {j.note}", file=self.out)
        self._write_status()
        bad = [j for j in self.jobs if j.status in ("fail", "timeout")]
        return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m etx.tools.gpu_jobs", description=__doc__.split("\n\n")[0])
    ap.add_argument("session", help="Python file defining JOBS")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids this machine uses (default 0)")
    ap.add_argument("--cpu-slots", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--logdir", default="~/vlnext")
    ap.add_argument("--machine", default="A", help="run only the jobs tagged with this machine")
    ap.add_argument("--only", help="regex over job names: run only these (dependencies outside it count as done)")
    ap.add_argument("--progress-s", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true", help="print the plan (order, lanes, dependencies) and exit")
    a = ap.parse_args(argv)
    jobs, max_seconds, local_only = _load(a.session)
    jobs = [j for j in jobs if j.machine == a.machine]
    if a.only:
        keep = {j.name for j in jobs if re.search(a.only, j.name)}
        dropped = {w for j in jobs if j.name not in keep for w in j.writes} | {j.name for j in jobs} - keep
        for j in jobs:
            j.reads = [r for r in j.reads if r.lstrip("?") not in dropped]
        jobs = [j for j in jobs if j.name in keep]
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    if a.dry_run:
        if max_seconds is not None:
            print(f"Session hard stop: {max_seconds:g}s from runner start (includes setup)")
        for i, j in enumerate(jobs):
            pin = f"GPU {j.gpu}" if j.gpu is not None else ("any GPU" if j.lane == "gpu" else "cpu")
            pin = "final" if j.lane == "final" else pin
            print(f"{i:3d} {j.name:32} {pin:8} timeout {j.timeout:5.0f}s  reads: {', '.join(j.reads) or '-'}"
                  + (f"  writes: {', '.join(j.writes)}" if j.writes else ""))
        return 0
    logdir = record.create(Path(os.path.expanduser(a.logdir)))
    return Runner(jobs, gpus, a.cpu_slots, logdir, a.progress_s, max_seconds=max_seconds, local_only=local_only).run()


if __name__ == "__main__":
    raise SystemExit(main())
