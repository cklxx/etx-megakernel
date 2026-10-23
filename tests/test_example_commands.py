"""CPU-only checks: importing the dispatch adapters never imports GPU scripts."""
import ast
from contextlib import redirect_stdout
import importlib
import importlib.util
from io import StringIO
from pathlib import Path
import subprocess
import sys

import pytest

from examples import cli, commands


@pytest.mark.parametrize('command', commands.COMMANDS, ids=lambda c: c.name)
def test_entry_imports_and_is_listed(command):
    # The callable is deliberately a lazy adapter. Importing the target itself
    # would run arbitrary module-level work (including GPU allocation).
    adapter = importlib.import_module(command.callable.func.__module__)
    assert getattr(adapter, command.callable.func.__name__) is command.callable.func
    assert importlib.util.find_spec(command.module).origin == str(command.path)
    assert command.name in TOP_HELP


with redirect_stdout(StringIO()) as _help:
    with pytest.raises(SystemExit) as _exit:
        cli.main(['--help'])
TOP_HELP = _help.getvalue()
assert _exit.value.code == 0


@pytest.mark.parametrize('command', commands.COMMANDS, ids=lambda c: c.name)
def test_help_without_import_or_execution(command, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('help must not execute an example')
    monkeypatch.setattr(commands.runpy, 'run_module', forbidden)
    cli.main([command.name, '--help'])
    text = capsys.readouterr().out
    assert command.name in text
    for argument in command.arguments:
        assert argument in text


@pytest.mark.parametrize('command', commands.COMMANDS, ids=lambda c: c.name)
def test_arguments_pass_through(command, monkeypatch):
    original_argv, original_path = sys.argv, sys.path[:]
    remaining = ['input with spaces', '--unknown=literal', '--repeat', '2',
                 '--repeat', '3', '--', '-h', '']
    seen = []

    def run_module(module, *, run_name, alter_sys):
        seen.append((module, run_name, alter_sys, sys.argv[:], sys.path[0]))
        raise SystemExit(7)

    monkeypatch.setattr(commands.runpy, 'run_module', run_module)
    with pytest.raises(SystemExit) as error:
        cli.main([command.name, *remaining])
    assert error.value.code == 7
    assert seen == [(command.module, '__main__', True,
                     [str(command.path), *remaining], str(command.path.parent))]
    assert sys.argv is original_argv
    assert sys.path == original_path


def test_discovery_covers_sources_and_is_deterministic():
    expected = set()
    for path in commands.ROOT.rglob('*.py'):
        if path.name in {'cli.py', 'commands.py'} and path.parent == commands.ROOT:
            continue
        tree = ast.parse(path.read_text())
        has_main = any(isinstance(n, ast.If) and '__main__' in ast.unparse(n.test)
                       for n in tree.body)
        has_parser = any(isinstance(n, ast.Call) and
                         ((isinstance(n.func, ast.Attribute) and n.func.attr == 'ArgumentParser')
                          or (isinstance(n.func, ast.Name) and n.func.id == 'ArgumentParser'))
                         for n in ast.walk(tree))
        if has_main or has_parser:
            expected.add(path)
    assert {c.path for c in commands.COMMANDS} == expected
    assert len(commands.REGISTRY) == len(expected)
    assert [c.name for c in commands.discover()] == [c.name for c in commands.COMMANDS]


def test_actual_module_execution_matches_original():
    # pairs is a pure CPU parser; no files means a deterministic refusal.
    args = ['--pairs', '5']
    original = subprocess.run([sys.executable, '-m', 'examples.vllm_llm.pairs', *args],
                              capture_output=True, text=True, timeout=5)
    unified = subprocess.run([sys.executable, '-m', 'examples.cli', 'vllm_llm.pairs', *args],
                             capture_output=True, text=True, timeout=5)
    assert (unified.returncode, unified.stdout, unified.stderr) == (
        original.returncode, original.stdout, original.stderr)


def test_unknown_command(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(['does.not.exist'])
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err


def test_documented_inventory_matches_registry():
    from collections import Counter
    text = (commands.ROOT / 'COMMANDS.md').read_text()
    assert f'Total commands: **{len(commands.COMMANDS)}**.' in text
    rows = [line for line in text.splitlines() if line.startswith('| `')]
    assert len(rows) == len(commands.COMMANDS)
    assert {row.split('`')[1] for row in rows} == set(commands.REGISTRY)
    for family, count in Counter(c.family for c in commands.COMMANDS).items():
        assert f'| {family} | {count} |' in text
        assert f'## {family} ({count})' in text
