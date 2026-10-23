"""Command-level cases moved from tests/test_two_gpu_preflight.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.testing import glm_cpu_stage, glm_reference_record, qwen_staged_model as staged_model, run_command

ROOT = Path(__file__).resolve().parents[2]
_LAST = []


def pf_main(argv=None):
    """`pf.main(argv)` as the command `python -m tools.two_gpu_preflight`."""
    result = run_command([sys.executable, "-m", "tools.two_gpu_preflight", *map(str, argv or [])], cwd=ROOT)
    _LAST.append(result.stdout)
    return result.returncode


def last_output():
    return _LAST[-1] if _LAST else ""


@pytest.fixture(scope='module')
def glm_fixture(tmp_path_factory):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    return glm_cpu_stage(tmp_path_factory.mktemp('glm-preflight') / 'stage')


reference = glm_reference_record


def staged():
    return dict(snapshot_inputs=dict(index_sha256='a' * 64, shards={'model-00001.safetensors': 'b' * 64}))


def test_missing_glm_stage_is_named_and_fails_closed(tmp_path, capsys):
    assert pf_main(['--glm-stage', str(tmp_path / 'absent')]) == 2
    out = last_output()
    assert 'PREFLIGHT FAIL glm-stage: missing' in out and 'stage.json' in out
    assert 'PREFLIGHT OK' not in out


def test_synthetic_glm_stage_cannot_pass_production_preflight(glm_fixture, capsys):
    assert pf_main(['--glm-stage', str(glm_fixture)]) == 2
    out = last_output()
    assert 'PREFLIGHT FAIL glm-stage: synthetic fixtures' in out


def test_tampered_glm_identity_is_named(glm_fixture, tmp_path, capsys):
    import shutil
    copied = tmp_path / 'stage'
    shutil.copytree(glm_fixture, copied)
    stage = json.loads((copied / 'stage.json').read_text())
    stage['revision'] = '0' * 40
    (copied / 'stage.json').write_text(json.dumps(stage))
    assert pf_main(['--glm-stage', str(copied)]) == 2
    assert 'pinned GLM-5.3-Flash checkpoint' in last_output()


def test_qwen_assets_pass_and_name_a_missing_reference(tmp_path, capsys):
    model = staged_model(tmp_path, 8)
    assert pf_main(['--qwen-model', str(model)]) == 0
    out = last_output()
    assert 'PREFLIGHT OK qwen-assets: batches 2,4,8; 20 files each' in out
    assert out.rstrip().endswith('PREFLIGHT OK')

    # Any one staged reference; the preflight must name it.
    missing = sorted(p.name for p in model.glob('golden*'))[-1]
    (model / missing).unlink()
    assert pf_main(['--qwen-model', str(model)]) == 2
    out = last_output()
    assert 'PREFLIGHT FAIL qwen-assets: missing' in out and missing in out


def test_qwen_hash_mode_rehashes_the_packed_weights(tmp_path, capsys):
    model = staged_model(tmp_path, 8)
    assert pf_main(['--qwen-model', str(model), '--hash']) == 0
    assert 'weights hashed' in last_output()
