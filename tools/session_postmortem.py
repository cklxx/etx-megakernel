"""Summarize every captured session, refusing incomplete evidence."""
import json
from pathlib import Path
import re
import sys


def report(dest):
    missing = False

    def uncaptured(path):
        nonlocal missing
        print(f'UNCAPTURED {path}')
        missing = True

    directories = {p.parent for name in ('status.json', '*.log') for p in dest.rglob(name)}
    if not directories:
        directories = {dest}
    for directory in sorted(directories):
        status = directory / 'status.json'
        runner = directory / 'runner.log'
        print(f'SESSION {directory}')
        try:
            jobs = json.loads(status.read_text())
            if not isinstance(jobs, list) or not jobs or any(
                not isinstance(j, dict) or not {'name', 'status'} <= j.keys() for j in jobs
            ):
                raise ValueError('invalid status')
        except (OSError, ValueError):
            uncaptured(status)
            jobs = []
        for job in jobs:
            if job['status'] == 'ok':
                continue
            print(f"JOB {job['name']} status={job['status']} rc={job.get('rc')} note={job.get('note', '')}")
            path = directory / (job['name'] + '.log')
            try:
                lines = path.read_text().splitlines()
            except OSError:
                uncaptured(path)
                continue
            if len(lines) < 2 or not lines[1].startswith('# '):
                uncaptured(f'{path}:2')
            else:
                print(f'COMMAND {path}:2: {lines[1]}')
            print(f'TAIL {path}')
            print('\n'.join(lines[-5:]))
        try:
            text = runner.read_text()
        except OSError:
            uncaptured(runner)
            continue
        stamps = re.findall(r'^\[\d{2}:\d{2}:\d{2} \+\s*(\d+)m(\d+)s\]', text, re.M)
        if len(stamps) < 2:
            uncaptured(f'{runner}:timestamps')
        else:
            elapsed = [int(m) * 60 + int(s) for m, s in stamps]
            print(f'PAID MINUTES {runner}: {(elapsed[-1] - elapsed[0]) / 60:.2f} (captured runner span; not provider billing)')
            if 'session done' not in text:
                uncaptured(f'{runner}:session done')
    return int(missing)


if __name__ == '__main__':
    sys.exit(report(Path(sys.argv[1])))
