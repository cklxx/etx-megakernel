"""Run session jobs with ordered GPU lanes, CPU workers, deadlines and sealed records.

Python JOBS files remain supported executable input. The implementation lives in
etx.experiments.jobs; new callers use immutable declarations through etx.external.
"""
import argparse
import os

from etx.external import JobSessionRequest, run_jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python3 -m etx.tools.gpu_jobs', description=__doc__.split('\n\n')[0])
    ap.add_argument('session', help='Python file defining JOBS')
    ap.add_argument('--gpus', default='0', help='comma-separated GPU ids this machine uses (default 0)')
    ap.add_argument('--cpu-slots', type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument('--logdir', default='~/vlnext')
    ap.add_argument('--machine', default='A', help='run only the jobs tagged with this machine')
    ap.add_argument('--only', help='regex over job names: run only these (dependencies outside it count as done)')
    ap.add_argument('--progress-s', type=float, default=60.0)
    ap.add_argument('--dry-run', action='store_true', help='print the plan (order, lanes, dependencies) and exit')
    a = ap.parse_args(argv)
    return run_jobs(JobSessionRequest(
        session=a.session, gpus=tuple(int(x) for x in a.gpus.split(',') if x != ''),
        cpu_slots=a.cpu_slots, logdir=a.logdir, machine=a.machine, only=a.only,
        progress_s=a.progress_s, dry_run=a.dry_run)).returncode


if __name__ == '__main__':
    raise SystemExit(main())
