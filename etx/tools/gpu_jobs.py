"""Run a GPU session's jobs on every GPU at once (and on several machines), with the discipline CLAUDE.md asks for.

A session is a Python file that defines JOBS, a list of dicts:

  name      unique; the log is <logdir>/<name>.log (full output, never filtered)
  cmd       bash command (run with `bash -c`, from `cwd`); $GPU holds the GPU id the job got
  lane      "gpu" (default: one job per GPU at a time) or "cpu" (builds, downloads: --cpu-slots in parallel)
  gpu       pin to one GPU id (A/B pairs that must share a device); default: any free GPU. A pin to a GPU the
            machine does not have folds onto the first GPU (so a two-GPU session also runs, serially, on one)
  after     names that must have finished OK first (a failed or skipped one skips this job)
  after_any names that must have finished, whatever the result
  gate      name of an earlier job and a regex: run only if that job's log matches, e.g. ("sync_gate", "UC_GO=1")
  timeout   seconds (default 900); on expiry the job's process group is killed and the job is reported "timeout"
  env       extra environment
  machine   which machine runs it (default "A"): one runner per machine, `--machine B` on the second VM
  summary   regexes; matching log lines are printed when the job ends (the log keeps everything)
  quiet_s   warn when the job has printed nothing for this long (default 180)

Every GPU job gets HIP_VISIBLE_DEVICES=$GPU, so a single-GPU program runs on the lane's GPU; a docker command passes
`-e HIP_VISIBLE_DEVICES=$GPU` itself. Order the list by priority (microbenchmarks and correctness first, captures last):
a free lane takes the first ready job in list order.

  python3 -m etx.tools.gpu_jobs examples/sessions/next.py --gpus 0,1 --logdir ~/vlnext [--machine A] [--dry-run]

Output: one line per start / end with the elapsed session time, a progress block every --progress-s seconds (running
jobs, their last log line, and a warning for any job silent past quiet_s), and at the end a status table plus
<logdir>/status.json. Exit code 0 only if every job that ran finished OK.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    name: str
    cmd: str
    lane: str = "gpu"
    gpu: int | None = None
    after: list[str] = field(default_factory=list)
    after_any: list[str] = field(default_factory=list)
    gate: tuple[str, str] | None = None
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


def load_jobs(path: str | Path) -> list[Job]:
    spec = importlib.util.spec_from_file_location("etx_session", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    jobs = []
    for d in mod.JOBS:
        d = dict(d)
        for k in ("after", "after_any", "summary"):
            if isinstance(d.get(k), str):
                d[k] = [d[k]]
        if d.get("gate") is not None:
            d["gate"] = tuple(d["gate"])
        jobs.append(Job(**d))
    names = [j.name for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"duplicate job names: {sorted(dup)}")
    known = set(names)
    for j in jobs:
        for dep in j.after + j.after_any + ([j.gate[0]] if j.gate else []):
            if dep not in known:
                raise SystemExit(f"job {j.name}: unknown dependency {dep!r}")
        if j.lane not in ("gpu", "cpu"):
            raise SystemExit(f"job {j.name}: lane must be gpu or cpu")
    return jobs


class Runner:
    def __init__(self, jobs: list[Job], gpus: list[int], cpu_slots: int, logdir: Path, progress_s: float = 60.0,
                 out=None, poll_s: float = 1.0):
        self.jobs = jobs
        self.gpus = gpus
        self.cpu_slots = cpu_slots
        self.logdir = logdir
        self.progress_s = progress_s
        self.out = out or sys.stdout
        self.poll_s = poll_s
        self.t0 = time.time()
        self.by_name = {j.name: j for j in jobs}

    # -------------------------------------------------------------- helpers
    def say(self, msg: str) -> None:
        el = time.time() - self.t0
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())} +{int(el // 60):3d}m{int(el % 60):02d}s] {msg}", file=self.out, flush=True)

    def _deps_state(self, j: Job) -> str:
        """ready | wait | skip"""
        for d in j.after:
            s = self.by_name[d].status
            if s in ("fail", "timeout", "skipped"):
                return "skip"
            if s != "ok":
                return "wait"
        for d in j.after_any:
            if self.by_name[d].status in ("pending", "running"):
                return "wait"
        if j.gate:
            g = self.by_name[j.gate[0]]
            if g.status in ("pending", "running"):
                return "wait"
            text = g.log.read_text(errors="replace") if g.log and g.log.exists() else ""
            if not re.search(j.gate[1], text, re.M):
                j.note = f"gate {j.gate[0]} !~ {j.gate[1]!r}"
                return "skip"
        return "ready"

    def _start(self, j: Job, gpu: int | None) -> None:
        j.log = self.logdir / f"{j.name}.log"
        env = dict(os.environ)
        env.update({k: str(v) for k, v in j.env.items()})
        if gpu is not None:
            env["HIP_VISIBLE_DEVICES"] = str(gpu)
            env["GPU"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        cwd = os.path.expanduser(j.cwd)
        fh = open(j.log, "w")
        fh.write(f"# job {j.name} gpu={gpu} cwd={cwd}\n# {j.cmd}\n")
        fh.flush()
        j.proc = subprocess.Popen(["bash", "-c", j.cmd], cwd=cwd if os.path.isdir(cwd) else None, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        j.status, j.start, j.on_gpu = "running", time.time(), gpu
        self.say(f"start {j.name}" + (f" on GPU {gpu}" if gpu is not None else " (cpu)"))

    def _finish(self, j: Job, status: str) -> None:
        j.status, j.end = status, time.time()
        j.rc = j.proc.returncode if j.proc else None
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

    def _kill(self, j: Job) -> None:
        try:
            os.killpg(j.proc.pid, signal.SIGTERM)
            try:
                j.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(j.proc.pid, signal.SIGKILL)
                j.proc.wait(10)
        except ProcessLookupError:
            pass

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
        self.logdir.mkdir(parents=True, exist_ok=True)
        last_progress = time.time()
        try:
            while True:
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
                    if j.status == "pending" and self._deps_state(j) == "skip":
                        j.status = "skipped"
                        self.say(f"skip    {j.name}" + (f" [{j.note}]" if j.note else " [a dependency failed]"))
                busy_gpus = {j.on_gpu for j in self.jobs if j.status == "running" and j.lane == "gpu"}
                cpu_busy = sum(j.status == "running" and j.lane == "cpu" for j in self.jobs)
                for j in self.jobs:                                   # start, in list order
                    if j.status != "pending" or self._deps_state(j) != "ready":
                        continue
                    if j.lane == "cpu":
                        if cpu_busy < self.cpu_slots:
                            self._start(j, None); cpu_busy += 1
                        continue
                    if j.gpu is not None and j.gpu not in self.gpus:
                        # a session written for two GPUs still runs on a one-GPU machine: the pin folds onto the first
                        j.note = f"pinned GPU {j.gpu} absent, run on GPU {self.gpus[0]}"
                        j.gpu = self.gpus[0]
                    free = [g for g in self.gpus if g not in busy_gpus and (j.gpu is None or g == j.gpu)]
                    if free:
                        self._start(j, free[0]); busy_gpus.add(free[0])
                if all(j.status not in ("pending", "running") for j in self.jobs):
                    break
                if time.time() - last_progress > self.progress_s:
                    self._progress(); last_progress = time.time()
                time.sleep(self.poll_s)
        except KeyboardInterrupt:
            for j in self.jobs:
                if j.status == "running":
                    self._kill(j); j.note = "interrupted"; self._finish(j, "fail")
            raise
        return self.report()

    def report(self) -> int:
        self.say("session done")
        print(f"  {'job':32} {'status':8} {'rc':>4} {'gpu':>4} {'seconds':>8}", file=self.out)
        for j in self.jobs:
            sec = f"{j.end - j.start:.0f}" if j.end else ""
            gpu = "" if j.on_gpu is None else str(j.on_gpu)
            print(f"  {j.name:32} {j.status:8} {'' if j.rc is None else j.rc:>4} {gpu:>4} {sec:>8}  {j.note}", file=self.out)
        (self.logdir / "status.json").write_text(json.dumps(
            [{"name": j.name, "status": j.status, "rc": j.rc, "gpu": j.on_gpu, "seconds": (j.end - j.start) if j.end else None,
              "note": j.note} for j in self.jobs], indent=1))
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
    jobs = [j for j in load_jobs(a.session) if j.machine == a.machine]
    if a.only:
        keep = {j.name for j in jobs if re.search(a.only, j.name)}
        for j in jobs:
            j.after = [d for d in j.after if d in keep]
            j.after_any = [d for d in j.after_any if d in keep]
            if j.gate and j.gate[0] not in keep:
                j.gate = None
        jobs = [j for j in jobs if j.name in keep]
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    if a.dry_run:
        for i, j in enumerate(jobs):
            pin = f"GPU {j.gpu}" if j.gpu is not None else ("any GPU" if j.lane == "gpu" else "cpu")
            deps = ", ".join(j.after + [f"~{d}" for d in j.after_any] + ([f"gate {j.gate[0]}~{j.gate[1]}"] if j.gate else []))
            print(f"{i:3d} {j.name:32} {pin:8} timeout {j.timeout:5.0f}s  after: {deps or '-'}")
        return 0
    return Runner(jobs, gpus, a.cpu_slots, Path(os.path.expanduser(a.logdir)), a.progress_s).run()


if __name__ == "__main__":
    raise SystemExit(main())
