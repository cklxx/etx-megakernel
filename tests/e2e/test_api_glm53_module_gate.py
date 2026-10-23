"""Staging the GLM module-gate checkpoint and the two-GPU preflight over it, moved from tests/test_glm53_module_gate.py."""
from pathlib import Path
import sys

import pytest
from etx.testing import glm_module_checkpoint, run_command

ROOT = Path(__file__).resolve().parents[2]


def command(module, *argv):
    return run_command([sys.executable, '-m', module, *map(str, argv)], cwd=ROOT, timeout=600)


@pytest.fixture(scope='module')
def staged(tmp_path_factory):
    pytest.importorskip('torch')
    d = tmp_path_factory.mktemp('module-gate')
    glm_module_checkpoint(d / 'checkpoint')
    run = command('examples.glm53.stage', '--src', d / 'checkpoint', '--out', d / 'stage', '--gate', 'module')
    assert run.returncode == 0, run.stdout[-1500:]
    return d


def test_the_release_stager_refuses_the_module_checkpoint(staged):
    run = command('examples.glm53.stage', '--src', staged / 'checkpoint', '--out', staged / 'as-release')
    assert run.returncode != 0 and 'production gate' in run.stdout


def test_preflight_keeps_the_gates_apart(staged):
    assert command('tools.two_gpu_preflight', '--glm-stage', staged / 'stage', '--glm-gate', 'module', '--hash').returncode == 0
    run = command('tools.two_gpu_preflight', '--glm-stage', staged / 'stage')
    assert run.returncode == 2 and 'pinned GLM-5.3-Flash checkpoint' in run.stdout


# examples.glm53.prepare.MODULE_MODEL and examples.glm53.module_gate.BATCH, pinned in tests/test_e2e_pins.py.
MODULE_MODEL = 'etx-synthetic/GLM-5.3-Flash-module-gate'
MODULE_BATCH = 2


def test_prepare_loads_the_checkpoint_exactly_and_the_cpu_body_passes(tmp_path):
    for module in ('torch', 'transformers', 'accelerate'):
        pytest.importorskip(module)
    import json
    d = tmp_path / 'out'
    run = command('examples.glm53.module_gate', 'prepare', '--out', d)
    assert run.returncode == 0, run.stdout[-2000:]
    verdict = json.loads(run.stdout[run.stdout.index('\n{') + 1:] if '\n{' in run.stdout else run.stdout)
    ref = json.loads((d / 'reference.json').read_text())
    assert ref['model'] == MODULE_MODEL and ref['batch'] == MODULE_BATCH
    assert ref['provenance']['quantization'] == 'FP8 block dequantize -> BF16'
    assert verdict['status'] == 'TOKENS-EQUAL' and verdict['matched_each'] == [32] * MODULE_BATCH
    assert verdict['module_rule']['ok'] and verdict['logit_argmax_equal']
    assert all(len(set(r['generated_ids'])) > 16 for r in ref['requests'])   # not a degenerate repeat
    assert not any(p.name.startswith('cpu') for p in (d / 'stage').iterdir())  # the pushed stage stays clean
