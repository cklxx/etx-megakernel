"""Keep the test suite independent of a developer's home directory."""
import ast
import io
from pathlib import Path
import re
import tokenize

import pytest


ROOT = Path(__file__).resolve().parents[1]
HOME_PREFIXES = tuple('/' + name + '/' for name in ('Users', 'home'))
HOME_CALL = re.compile(r'\bPath\s*\.\s*home\s*\(')


def home_dependencies(source):
    tree = ast.parse(source)
    path_names = {'Path'}
    pathlib_names = {'pathlib'}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'pathlib':
            path_names.update(a.asname or a.name for a in node.names if a.name == 'Path')
        elif isinstance(node, ast.Import):
            pathlib_names.update(a.asname or a.name for a in node.names if a.name == 'pathlib')

    violations = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(HOME_PREFIXES):
                violations.add(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'home':
            owner = node.func.value
            if ((isinstance(owner, ast.Name) and owner.id in path_names)
                    or (isinstance(owner, ast.Attribute) and owner.attr == 'Path'
                        and isinstance(owner.value, ast.Name) and owner.value.id in pathlib_names)):
                violations.add(node.lineno)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and 'EXTERNAL_TESTS' not in token.string:
            if any(prefix in token.string for prefix in HOME_PREFIXES) or HOME_CALL.search(token.string):
                violations.add(token.start[0])
    return sorted(violations)


def test_test_modules_have_no_home_dependencies():
    violations = [f'{path.relative_to(ROOT)}:{line}'
                  for path in sorted((ROOT / 'tests').rglob('*.py'))
                  for line in home_dependencies(path.read_text())]
    assert not violations, 'Home-directory dependencies in tests:\n' + '\n'.join(violations)


@pytest.mark.parametrize('source', [
    'ROOT = Path(' + repr('/' + 'Users/example/repo') + ')',
    'ROOT = ' + repr('/' + 'home/example/repo'),
    'ROOT = Path.' + 'home()',
    'import pathlib as pl\nROOT = pl.Path.' + 'home()',
    'from pathlib import Path as P\nROOT = P.home()',
    '# machine-specific ' + '/' + 'Users/example/repo',
    'ROOT = ' + repr('/' + 'home/example/repo') + ' # EXTERNAL_TESTS',
])
def test_home_dependency_guard_rejects_machine_paths(source):
    assert home_dependencies(source)


def test_home_dependency_guard_allows_portable_paths_and_external_comments():
    source = ('ROOT = Path(__file__).resolve().parents[1]\n'
              'DATA = Path(os.environ["ETX_DATA_ROOT"])\n'
              '# EXTERNAL_TESTS: archived at ' + '/' + 'home/example/data\n')
    assert home_dependencies(source) == []
