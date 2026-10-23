"""Static import policy for the external-test tree (not a security sandbox)."""
import ast
from pathlib import Path
import sys


_FACADES = ('etx.external', 'etx.testing')
_OPAQUE = {'importlib', 'runpy', 'subprocess', 'ctypes', 'multiprocessing'}
_PATCH = {'patch', 'setattr', 'delattr', 'getfixturevalue'}
_BUILTINS = {'__import__', 'eval', 'exec', 'compile', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars'}
_PROCESS = {'system', 'popen', 'fork', 'forkpty', 'posix_spawn', 'posix_spawnp',
            'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe',
            'spawnl', 'spawnle', 'spawnlp', 'spawnlpe', 'spawnv', 'spawnve', 'spawnvp', 'spawnvpe'}


def _qualified(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, aliases)
        return f'{base}.{node.attr}' if base else None
    return None


def _allowed(module):
    root = module.split('.')[0]
    return (module == 'pytest' or
            (root in sys.stdlib_module_names and root not in _OPAQUE) or
            module in _FACADES or module == 'tests.e2e' or
            module.startswith('tests.e2e.'))


def check_api_layers(root: Path) -> tuple[str, ...]:
    """Check imports in the public library and caller facade below etx/.

    Only adapter modules in etx.contracts may bridge the library to internals.
    Testing probes are separate and intentionally allowed to inspect internals.
    """
    root = Path(root)
    violations = []
    for layer, permitted in [('external', ('etx.api', 'etx.external')),
                             ('api', ('etx.api', 'etx.contracts'))]:
        directory = root / layer
        if not directory.is_dir():
            raise ValueError(f'missing API layer: {directory}')
        for path in sorted(directory.rglob('*.py')):
            package = ['etx', *path.relative_to(root).parts[:-1]]
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    prefix = package[:len(package) - node.level + 1] if node.level else []
                    module = '.'.join([*prefix, *([node.module] if node.module else [])])
                    modules = [module]
                for module in modules:
                    if (module.split('.')[0] not in sys.stdlib_module_names and
                            not any(module == name or module.startswith(name + '.') for name in permitted)):
                        violations.append(f'{path}:{node.lineno}: {layer} layer cannot import {module}')
    return tuple(violations)


def check_migrated_entry_points(root: Path) -> tuple[str, ...]:
    """Import guard for migrated command adapters, not all legacy tools.

    Compiler CLI's old private loader shim remains an explicit migration debt;
    it is not silently included as a compliant adapter.
    """
    root = Path(root)
    violations = []
    for name in ('etx/tools/bench_sim.py', 'etx/tools/cpu_check.py', 'bench/sim_experiments.py',
                 'examples/experiment_program/stage.py'):
        path = root / name
        if not path.is_file():
            violations.append(f'{path}: missing migrated command adapter')
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = ['<relative>'] if node.level else [node.module or '']
            else:
                continue
            for module in modules:
                if module != 'etx.external' and (module.split('.')[0] not in sys.stdlib_module_names
                                                 or module.split('.')[0] in _OPAQUE):
                    violations.append(f'{path}:{node.lineno}: command adapter cannot import {module}')
    return tuple(violations)


def check_external_sources(root: Path) -> tuple[str, ...]:
    """Scan every Python helper under root; return file:line policy violations.

    External tests use public executors rather than subprocess Python or dynamic
    loading. Helpers outside this tree are forbidden; relative imports are also
    refused until a resolver can prove they stay within the checked tree.
    """
    from etx import external, testing

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'external-test root is not a directory: {root}')
    exports = {'etx.external': external.__all__, 'etx.testing': testing.__all__}
    violations = []
    for path in sorted(root.rglob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f'{node.module}.{alias.name}'
        assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
        for _ in range(len(assignments) + 1):
            changed = False
            for node in assignments:
                value = _qualified(node.value, aliases)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and value and aliases.get(target.id) != value:
                        aliases[target.id] = value
                        changed = True
            if not changed:
                break
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = ['<relative>'] if node.level else [node.module or '']
                if node.module in exports:
                    for alias in node.names:
                        if alias.name not in exports[node.module]:
                            violations.append(f'{path}:{node.lineno}: unsupported facade export {alias.name}')
            for module in modules:
                if not _allowed(module):
                    violations.append(f'{path}:{node.lineno}: forbidden import {module}')
                elif module.startswith('tests.e2e.'):
                    helper = root.joinpath(*module.split('.')[2:])
                    if not helper.with_suffix('.py').is_file() and not (helper / '__init__.py').is_file():
                        violations.append(f'{path}:{node.lineno}: unchecked E2E helper {module}')
            if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
                violations.append(f'{path}:{node.lineno}: private/reflection attribute {node.attr}')
            qualified = _qualified(node, aliases)
            if qualified:
                if qualified.startswith('etx.') and not any(qualified == facade or qualified.startswith(facade + '.') for facade in _FACADES):
                    violations.append(f'{path}:{node.lineno}: nonpublic package access {qualified}')
                if qualified in ('sys.modules', 'sys.path', 'sys.meta_path'):
                    violations.append(f'{path}:{node.lineno}: import state access {qualified}')
                for facade, names in exports.items():
                    if qualified.startswith(facade + '.'):
                        member = qualified[len(facade) + 1:].split('.')[0]
                        if member not in names:
                            violations.append(f'{path}:{node.lineno}: unsupported facade export {member}')
                if qualified.startswith('tests.e2e.'):
                    parts = qualified.split('.')[2:]
                    if not any((root.joinpath(*parts[:n]).with_suffix('.py')).is_file() or
                               (root.joinpath(*parts[:n]) / '__init__.py').is_file()
                               for n in range(1, len(parts) + 1)):
                        violations.append(f'{path}:{node.lineno}: unchecked E2E helper {qualified}')
            if isinstance(node, ast.Call):
                name = _qualified(node.func, aliases) or ''
                leaf = name.rsplit('.', 1)[-1]
                if leaf in _BUILTINS:
                    violations.append(f'{path}:{node.lineno}: opaque execution/reflection {name}')
                if leaf in _PROCESS or name.startswith('asyncio.create_subprocess_'):
                    violations.append(f'{path}:{node.lineno}: use a public executor')
                if leaf == 'getfixturevalue':
                    violations.append(f'{path}:{node.lineno}: declare fixture dependencies explicitly')
                elif leaf in _PATCH or name.startswith('unittest.mock.patch'):
                    violations.append(f'{path}:{node.lineno}: use supported fixtures/executors, not private patches')
    return tuple(violations)
