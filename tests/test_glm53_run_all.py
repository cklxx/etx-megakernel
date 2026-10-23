"""examples/glm53/run_all.py: one command, every input checked before any work.

Command cases are in tests/e2e/test_api_glm53_run_all.py; these are the in-process floor/certificate probes.

CPU only. The dry run prints the exact step commands and the refusal list and
executes nothing; a real run that is missing an input exits 2 without creating
the output root. No test needs a GPU, a model or torch.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Preflight only checks existence; these tests never execute the supplied compiler.
HIPCC_STUB = sys.executable
EVIDENCE = ROOT / 'docs/evidence/qwen-batch-glm53'
sys.path.insert(0, str(ROOT))

from etx.models.glm53.run_all import (MONO_GFX942, WEIGHT_COPY_FACTOR, Failure,
                                       annotate, reading_budget)


def run(*flags, env=None):
    environment = {k: v for k, v in os.environ.items() if not k.startswith('GLM_')}
    environment['PYTHONPATH'] = str(ROOT)
    environment.update(env or {})
    return subprocess.run([sys.executable, '-m', 'examples.glm53.run_all', *map(str, flags)],
                          cwd=ROOT, env=environment, capture_output=True, text=True)


def pinned(tmp_path):
    """A model root carrying the pinned config/REVISION/index, without the weights."""
    api = json.loads((EVIDENCE / 'glm-model-api.json').read_text())
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_bytes((EVIDENCE / 'glm-config.json').read_bytes())
    (model / 'REVISION').write_text(api['sha'] + '\n')
    (model / 'model-00001.safetensors').write_bytes(b'')
    (model / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {'model.embed_tokens.weight': 'model-00001.safetensors'}}))
    return model


def fake_python(tmp_path, missing=(), gpus=2):
    """An interpreter that answers the dependency/GPU probe without importing anything."""
    script = tmp_path / 'python'
    payload = json.dumps(dict(missing=list(missing), gpus=gpus))
    assert "'" not in payload
    script.write_text(f"#!/bin/sh\nprintf '%s\\n' '{payload}'\n")
    script.chmod(0o755)
    return script


def staged_model():
    """The stage.json fields the budget reads, at hand-checkable size."""
    return dict(
        dimensions=dict(MAXLEN=1072, STATE_KDA=100, STATE_MLA=10720),
        config=dict(layer_types=['linear_attention', 'deepseek_sparse_attention']),
        weights=dict(
            embedding=dict(blob_bytes=10 * 8 * 4, meta_bytes=32,
                           tensors={'model.embed_tokens.weight': {'shape': [10, 8]}}),
            l0=dict(blob_bytes=1000, meta_bytes=64, tensors={}),
            norm=dict(blob_bytes=16, meta_bytes=32, tensors={}),
            head=dict(blob_bytes=800, meta_bytes=32, tensors={})))


def test_the_floor_uses_bytes_read_and_keeps_the_copy_factor_separate():
    budget = reading_budget(staged_model(), batch=4, tbps=4.3)
    # 1944 bytes of layer/norm/head weights and descriptors, shared per step,
    # plus the embedding descriptors and one 32-byte row per request:
    # (1944 + 32 + 4 * 32) / 4 = 526 per token.
    assert budget['weights_read_per_token_bytes'] == 526
    # 4 bytes per state element: KDA 100 + MLA 10720 scaled 1024/1072.
    assert budget['state_read_per_token_bytes'] == 41360
    assert budget['kernel_bytes_per_token'] == 41886
    assert budget['floor_ms_per_token'] == pytest.approx(41886 / 4.3e12 * 1000)
    assert budget['weight_copy']['factor'] == WEIGHT_COPY_FACTOR == 1.72
    assert budget['weight_copy']['separate'] is True
    assert 'never folded' in budget['weight_copy']['note']
    assert budget['weight_copy']['weight_copy_bytes_per_token'] == round(1.72 * 526)
    assert budget['mono_gfx942'] == MONO_GFX942


def test_the_weight_read_is_amortised_over_the_batch_and_state_is_not():
    one = reading_budget(staged_model(), batch=1, tbps=4.3)
    four = reading_budget(staged_model(), batch=4, tbps=4.3)
    assert one['weights_read_per_token_bytes'] == 1944 + 32 + 32
    assert four['weights_read_per_token_bytes'] == 526
    assert one['state_read_per_token_bytes'] == four['state_read_per_token_bytes']
    assert one['floor_ms_per_token'] > four['floor_ms_per_token']


def test_the_certificate_carries_gpus_the_fraction_and_the_mono_note(tmp_path):
    path = tmp_path / 'bench_result.json'
    path.write_text(json.dumps(dict(schema=1, status='PASS', batch=4, median_decode_wall_ms=1.0,
                                    tokens_per_s=4000.0, ms_per_token=0.25)))
    annotate(path, staged_model(), batch=4, gpus=2, tbps=4.3)
    saved = json.loads(path.read_text())
    floor = saved['bandwidth_floor']
    assert saved['gpus'] == 2
    assert saved['mono_gfx942_availability'] == MONO_GFX942
    assert floor['gpus'] == 2
    assert floor['measured_ms_per_token'] == 0.25
    assert floor['fraction_of_floor'] == pytest.approx(floor['floor_ms_per_token'] / 0.25)
    assert floor['percent_of_floor'] == pytest.approx(100 * floor['fraction_of_floor'])
    assert floor['measured_over_floor'] == pytest.approx(0.25 / floor['floor_ms_per_token'])


def test_annotate_refuses_a_failed_or_mismatched_certificate(tmp_path):
    path = tmp_path / 'bench_result.json'
    path.write_text(json.dumps(dict(status='FAIL', batch=4)))
    with pytest.raises(Failure):
        annotate(path, staged_model(), batch=4, gpus=2, tbps=4.3)
    path.write_text(json.dumps(dict(status='PASS', batch=2, ms_per_token=0.25)))
    with pytest.raises(Failure):
        annotate(path, staged_model(), batch=4, gpus=2, tbps=4.3)


def test_annotate_refuses_a_stage_without_the_budget_fields(tmp_path):
    path = tmp_path / 'bench_result.json'
    path.write_text(json.dumps(dict(status='PASS', batch=4, ms_per_token=0.25)))
    with pytest.raises(Failure):
        annotate(path, dict(config=dict(layer_types=[])), batch=4, gpus=2, tbps=4.3)
