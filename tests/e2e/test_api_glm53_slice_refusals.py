"""GLM slice-gate refusals of the stage and reference commands, moved from tests/test_glm53_slice.py."""
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def command(module, *argv):
    return run_command([sys.executable, '-m', module, *map(str, argv)], cwd=ROOT)


@pytest.mark.parametrize('argv,message', [
    (('--gate', 'slice'), 'slice gate requires --layers'),
    (('--gate', 'production', '--layers', '22-25'), 'only valid with --gate slice'),
    (('--gate', 'slice', '--layers', '22-23'), 'straddle the PP2 boundary'),
])
def test_staging_refuses_before_touching_the_checkpoint(tmp_path, argv, message):
    run = command('examples.glm53.stage', '--src', tmp_path / 'missing', '--out', tmp_path / 'out', *argv)
    assert run.returncode != 0 and message in run.stdout, run.stdout[-1500:]
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('argv,message', [
    (('--gate', 'slice'), 'slice gate requires --layers'),
    (('--gate', 'slice', '--layers', '24-25'), 'straddle the PP2 boundary'),
])
def test_reference_refuses_before_touching_the_checkpoint(tmp_path, argv, message):
    pytest.importorskip('torch'); pytest.importorskip('transformers')
    run = command('examples.glm53.reference', '--src', tmp_path / 'missing', '--prompts', tmp_path / 'p.json',
                  '--out', tmp_path / 'out', '--offload', tmp_path / 'off', '--max-memory', '{}', *argv)
    assert run.returncode != 0 and message in run.stdout, run.stdout[-1500:]
