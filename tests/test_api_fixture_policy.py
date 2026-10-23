"""Conformance tests for the public pytest policy plugin."""
import pytest

pytest_plugins = ['pytester']


@pytest.mark.parametrize('body,autouse', [
    ('def test_example(private): pass', False),
    ('def test_example(public): pass', False),
    ('def test_example(): pass', True),
    ('def test_example(request): request.getfixturevalue("private")', False),
])
def test_rejects_inherited_private_fixtures(pytester, body, autouse):
    pytester.makeconftest(f'''
from pathlib import Path
import pytest
from etx.testing import external_test_policy

def pytest_configure(config):
    config.pluginmanager.register(external_test_policy(Path(__file__).parent / 'e2e'))

@pytest.fixture(autouse={autouse!r})
def private():
    raise AssertionError('private fixture must not execute')
''')
    root = pytester.path / 'e2e'
    root.mkdir()
    (root / 'test_example.py').write_text('import pytest\n' + body + '\n')
    (root / 'conftest.py').write_text(
        'import pytest\n@pytest.fixture\ndef public(private): return private\n')
    result = pytester.runpytest('-q')
    assert result.ret != 0
    output = result.stdout.str() + result.stderr.str()
    expected = ('declare fixture dependencies explicitly' if 'getfixturevalue' in body else
                'external test uses nonpublic fixture private')
    assert expected in output
    assert 'AssertionError: private fixture must not execute' not in output


@pytest.mark.parametrize('source', [
    'from etx.api import model',
    'from etx.testing.boundary import check_external_sources',
    'request.getfixturevalue("private")',
])
def test_focused_e2e_run_checks_even_unused_helpers(pytester, source):
    pytester.makeconftest('''
from pathlib import Path
from etx.testing import external_test_policy

def pytest_configure(config):
    config.pluginmanager.register(external_test_policy(Path(__file__).parent / 'e2e'))
''')
    root = pytester.path / 'e2e'
    root.mkdir()
    (root / 'test_example.py').write_text('def test_example(): pass\n')
    (root / 'unused.py').write_text(source + '\n')
    result = pytester.runpytest('-q', str(root / 'test_example.py'))
    assert result.ret != 0
    assert 'unused.py:1:' in result.stderr.str()


def test_accepts_external_and_builtin_fixtures_without_affecting_internal_tests(pytester):
    pytester.makeconftest('''
from pathlib import Path
import pytest
from etx.testing import external_test_policy

def pytest_configure(config):
    config.pluginmanager.register(external_test_policy(Path(__file__).parent / 'e2e'))

@pytest.fixture
def private(): return 7
''')
    pytester.makepyfile('def test_internal(private): assert private == 7')
    root = pytester.path / 'e2e'
    root.mkdir()
    (root / 'conftest.py').write_text(
        'import pytest\n@pytest.fixture\ndef public(tmp_path): return tmp_path\n')
    (root / 'test_example.py').write_text(
        'def test_example(public): assert public.is_dir()\n')
    pytester.runpytest('-q').assert_outcomes(passed=2)
