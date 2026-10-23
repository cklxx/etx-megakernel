"""Run an existing example: python -m examples.cli <dotted-name> [arguments]."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from examples.commands import REGISTRY, Command


def command_help(command: Command) -> str:
    """Describe the source without importing optional dependencies or running work."""
    lines = [f'usage: python -m examples.cli {command.name} [arguments]', '',
             f'Module: {command.module}', 'Entrypoint: original __main__ block', '']
    if command.description:
        lines.extend([command.description, ''])
    lines.append('Argument declarations (from source; expressions are not evaluated):')
    lines.extend(f'  {argument}' for argument in command.arguments)
    if not command.arguments:
        lines.append('  No argparse declarations; see the original entrypoint below.')
    lines.extend(['', 'Original entrypoint:', command.entrypoint or '  (no main guard)', '',
                  'Execution forwards arguments unchanged. Original runtime dependencies apply.'])
    return '\n'.join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=tuple(REGISTRY), help='stable dotted example name')
    # Parse only the command; all following tokens belong to the original script.
    if not args or args[0] in {'-h', '--help'}:
        parser.parse_args(args)
        return
    parsed = parser.parse_args(args[:1])
    command = REGISTRY[parsed.command]
    remaining = args[1:]
    before_separator = remaining[:remaining.index('--')] if '--' in remaining else remaining
    if any(arg in {'-h', '--help'} for arg in before_separator):
        print(command_help(command))
        return
    command.callable(remaining)


if __name__ == '__main__':
    main()
