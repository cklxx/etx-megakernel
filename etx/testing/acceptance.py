"""Bounded CPU acceptance probes; no GPU execution or persisted ambient environment."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from unittest.mock import patch

from etx.api.types import Configuration
from etx.experiments import acceptance


def acceptance_results(path):
    """Classify a JUnit report using the production external-skip policy."""
    return Configuration(acceptance.test_results(Path(path)))


def verify_acceptance_archive(directory, expected_index):
    """Exercise the archive integrity gate with a caller-owned pinned fixture."""
    return Configuration(acceptance.verify_archive(Path(directory), expected_index))


@contextmanager
def acceptance_dependency_failure():
    """Inject only the dependency gate failure; the external lifecycle still runs."""
    with patch.object(acceptance, 'dependencies',
                      side_effect=ValueError('archive member checksum mismatch')):
        yield


def acceptance_probe(kind, *, output, env=None, suite=None, selection=()):
    """Run fixed acceptance child seams with real CPU children and complete logs.

    Environment overrides are test inputs, held only in memory. Pytest uses the
    production sanitization policy; script probes preserve inherited variables.
    The returned record reports child exit status, never an acceptance verdict.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = None if env is None else {**os.environ, **env}
    if kind == 'command-failure':
        command = [sys.executable, '-c', 'print("diagnostic"); raise SystemExit(7)']
    elif kind == 'formal':
        command = ['bash', acceptance.ROOT / 'formal/check.sh', 'old', output / 'formal']
    elif kind == 'matrix':
        command = ['bash', acceptance.ROOT / 'examples/vllm_llm/local_matrix.sh', output / 'matrix']
    elif kind == 'pytest':
        if suite is None:
            raise ValueError('pytest probe requires a fixture suite')
        if env is not None:
            raise ValueError('pytest probe uses the ambient acceptance environment')
        environment = acceptance.pytest_environment()
        command = [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=', '-p',
                   'etx.tools.pytest_acceptance',
                   f'--acceptance-inventory={output / "inventory.json"}', str(suite), *selection]
    else:
        raise ValueError(f'unknown acceptance probe: {kind}')
    return Configuration(acceptance.command_step(kind, command, output, env=environment))
