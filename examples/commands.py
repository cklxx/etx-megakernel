"""Source-only registry of the existing example scripts.

Discovery never imports an example: some scripts allocate devices or perform work at
import time. Each callable executes the original module's __main__ block lazily.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import runpy
import sys
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent


def _run(module: str, path: Path, argv: Sequence[str]) -> None:
    """Use the original script entrypoint, including its exit status and exceptions."""
    old_argv, old_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(path), *argv]
        # Several examples use sibling imports when run as scripts.
        sys.path.insert(0, str(path.parent))
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


@dataclass(frozen=True)
class Command:
    name: str
    module: str
    path: Path
    arguments: tuple[str, ...]
    description: str
    entrypoint: str
    callable: Callable[[Sequence[str]], None]

    @property
    def family(self) -> str:
        return self.name.split('.')[0] if '.' in self.name else 'standalone'


def discover(root: Path = ROOT) -> tuple[Command, ...]:
    """Include every main guard or ArgumentParser constructor, in path order.

    Argument declarations are source expressions, not evaluated values: this keeps
    dynamic defaults, choices, helper-generated arguments and GPU imports safe.
    """
    result = []
    for path in sorted(root.rglob('*.py')):
        if path.parent == root and path.name in {'commands.py', 'cli.py'}:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        guards = [n for n in tree.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Compare)
                  and any(isinstance(v, ast.Constant) and v.value == '__main__'
                          for v in [n.test.left, *n.test.comparators])]
        constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and ((isinstance(n.func, ast.Attribute) and n.func.attr == 'ArgumentParser')
                             or (isinstance(n.func, ast.Name) and n.func.id == 'ArgumentParser'))]
        if not guards and not constructors:
            continue
        declarations = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)
                               and isinstance(n.func, ast.Attribute)
                               and n.func.attr in {'add_argument', 'add_subparsers', 'add_parser'}),
                              key=lambda n: (n.lineno, n.col_offset))
        arguments = tuple(ast.unparse(n) for n in declarations)
        name = '.'.join(path.relative_to(root).with_suffix('').parts)
        module = f'examples.{name}'
        entrypoint = '\n'.join(ast.unparse(n) for guard in guards for n in guard.body)
        result.append(Command(name, module, path, arguments, ast.get_docstring(tree) or '',
                              entrypoint, partial(_run, module, path)))
    return tuple(result)


COMMANDS = discover()
REGISTRY = {command.name: command for command in COMMANDS}
