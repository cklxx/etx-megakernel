"""Plan, run and independently check the experimental gfx942 handoff probe.

No GPU activity on import, plan, or analyze. Synthetic pressure is not model evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from etx.external import NativeProbeRequest, run_native_probe

# Implementation: etx/experiments/fresh_handoff.py behind etx.external.run_native_probe.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('plan')
    execute = commands.add_parser('run')
    execute.add_argument('--binary', type=Path, required=True)
    execute.add_argument('--out', type=Path, required=True)
    execute.add_argument('--budget-seconds', type=float, default=7200)
    execute.add_argument('--case-seconds', type=float, default=420)
    inspect = commands.add_parser('analyze')
    inspect.add_argument('directory', type=Path)
    args = parser.parse_args()
    arguments = (dict(binary=str(args.binary), out=str(args.out), budget_seconds=args.budget_seconds,
                      case_seconds=args.case_seconds) if args.command == 'run' else
                 dict(directory=str(args.directory)) if args.command == 'analyze' else {})
    root = str(Path(__file__).resolve().parents[2])
    raise SystemExit(run_native_probe(NativeProbeRequest('fresh-handoff', args.command, root, arguments)))


if __name__ == '__main__':
    main()
