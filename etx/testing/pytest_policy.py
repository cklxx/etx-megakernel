"""Pytest integration for the caller boundary; pytest internals stay here."""
import inspect
from pathlib import Path

import pytest


def external_test_policy(root: Path):
    """Return a pytest plugin enforcing fixtures for tests below root.

    Register once with config.pluginmanager.register() in the root conftest.
    Collection checks fixture closures; setup also catches getfixturevalue().
    This is an architectural guard, not a sandbox for hostile Python.
    """
    return _ExternalTestPolicy(Path(root).resolve())


class _ExternalTestPolicy:
    def __init__(self, root):
        self.root = root

    def _external(self, node):
        path = getattr(node, 'path', None)
        return path is not None and Path(path).resolve().is_relative_to(self.root)

    def _check(self, fixturedef, node):
        func = inspect.unwrap(fixturedef.func)
        module = getattr(func, '__module__', '')
        if module.startswith('_pytest.'):
            return
        path = Path(inspect.getfile(func)).resolve()
        if path.is_relative_to(self.root) or path.is_relative_to(Path(__file__).parent):
            return
        raise pytest.UsageError(
            f'{node.nodeid}: external test uses nonpublic fixture '
            f'{fixturedef.argname} from {path}; move it behind etx.testing')

    def pytest_collection_modifyitems(self, items):
        from .boundary import check_external_sources

        external = [item for item in items if self._external(item)]
        if external:
            violations = check_external_sources(self.root)
            if violations:
                raise pytest.UsageError('\n'.join(violations))
        for item in external:
            for definitions in item._fixtureinfo.name2fixturedefs.values():
                if definitions:
                    self._check(definitions[-1], item)

    def pytest_fixture_setup(self, fixturedef, request):
        if self._external(request._pyfuncitem):
            self._check(fixturedef, request._pyfuncitem)
