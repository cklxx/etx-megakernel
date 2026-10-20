"""examples/glm53/run_all.py: one command, every input checked before any work.

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
EVIDENCE = ROOT / 'docs/evidence/qwen-batch-glm53'
sys.path.insert(0, str(ROOT))

from examples.glm53.run_all import (MONO_GFX942, WEIGHT_COPY_FACTOR, Failure,
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


def test_missing_model_is_refused_and_the_download_command_is_named(tmp_path):
    out = tmp_path / 'out'
    model = tmp_path / 'model'
    r = run('--dry-run', '--model-root', model, '--out', out, '--hipcc', '/bin/true')
    assert r.returncode == 3, r.stdout + r.stderr
    assert 'REFUSED before any work' in r.stdout
    assert f'hf download zai-org/GLM-5.3-Flash --revision ' in r.stdout
    assert f'--local-dir {model}' in r.stdout
    assert 'examples.glm53.stage' in r.stdout          # the plan is still printed
    assert not out.exists()


def test_the_plan_lists_every_step_in_order(tmp_path):
    out = tmp_path / 'out'
    r = run('--dry-run', '--model-root', tmp_path / 'model', '--out', out, '--hipcc', '/bin/true')
    names = [line.split()[2].rstrip(':') for line in r.stdout.splitlines()
             if line.startswith('PLAN ') and line.split()[1].isdigit()]
    assert names == ['prompts', 'stage', 'prepare', 'reference', 'build', 'qualify',
                     'oracle-run0', 'oracle-run1', 'bench']
    assert '--layers 22-25' in r.stdout
    assert '--gate slice' in r.stdout
    assert 'ETX_GLM_BATCH=4' in r.stdout and 'ETX_GLM_GATE=slice' in r.stdout
    assert f'--prompts {out}/prompts.json' in r.stdout
    assert f'--reference {out}/reference.json' in r.stdout
    assert f'--candidate {out}/qualification/run0.json' in r.stdout
    assert f'--candidate {out}/qualification/run1.json' in r.stdout
    assert 'CERTIFICATE ' + str(out / 'certificate' / 'bench_result.json') in r.stdout


def test_an_existing_output_root_is_refused(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    r = run('--dry-run', '--model-root', tmp_path / 'model', '--out', out, '--hipcc', '/bin/true')
    assert r.returncode == 3
    assert 'already exists' in r.stdout
    assert f'--out {out}-<label>' in r.stdout


def test_a_window_that_does_not_straddle_the_pp2_split_is_refused(tmp_path):
    r = run('--dry-run', '--model-root', tmp_path / 'model', '--out', tmp_path / 'out',
            '--layers', '0-3', '--hipcc', '/bin/true')
    assert r.returncode == 3
    assert 'straddle' in r.stdout
    assert '--layers 22-25' in r.stdout


def test_a_gpu_count_other_than_two_is_refused(tmp_path):
    r = run('--dry-run', '--model-root', tmp_path / 'model', '--out', tmp_path / 'out',
            '--gpus', '1', '--hipcc', '/bin/true')
    assert r.returncode == 3
    assert 'requires exactly two GPUs' in r.stdout
    assert '--gpus 2 (provision with tools/hotaisle_grab.sh 2x)' in r.stdout


def test_environment_variables_fill_every_parameter(tmp_path):
    model, out = tmp_path / 'model', tmp_path / 'out'
    r = run('--dry-run', '--hipcc', '/bin/true', env=dict(
        GLM_MODEL_ROOT=model, GLM_OUT_ROOT=out, GLM_LAYERS='23-26', GLM_BATCH='2',
        GLM_GPUS='2', GLM_MAX_MEMORY='{"0":"1GiB"}'))
    assert f'PLAN model={model} out={out} layers=23-26 batch=2 gpus=2' in r.stdout
    assert '--layers 23-26' in r.stdout
    assert '--batch 2' in r.stdout
    assert '1GiB' in r.stdout


def test_a_real_run_refuses_without_creating_the_output_root(tmp_path):
    out = tmp_path / 'out'
    r = run('--model-root', tmp_path / 'model', '--out', out, '--hipcc', '/bin/true')
    assert r.returncode == 2
    assert 'no work was done; the output root was not created' in r.stdout
    assert not out.exists()


def test_missing_modules_name_one_install_command(tmp_path):
    python = fake_python(tmp_path, missing=('torch', 'accelerate'), gpus=2)
    out = tmp_path / 'out'
    r = run('--dry-run', '--model-root', tmp_path / 'model', '--out', out,
            '--python', python, '--hipcc', '/bin/true')
    assert r.returncode == 3
    assert f'{python} cannot import torch, accelerate' in r.stdout
    assert f"{python} -m pip install -e '.[reference]' accelerate" in r.stdout
    assert not out.exists()


def test_a_ready_environment_plans_without_running(tmp_path):
    model = pinned(tmp_path)
    out = tmp_path / 'out'
    r = run('--dry-run', '--model-root', model, '--out', out,
            '--python', fake_python(tmp_path, gpus=2), '--hipcc', '/bin/true')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'DRY-RUN READY' in r.stdout
    assert 'REFUSED' not in r.stdout
    assert not out.exists()


def test_a_wrong_revision_or_missing_shard_is_refused(tmp_path):
    model = pinned(tmp_path)
    python, out = fake_python(tmp_path, gpus=2), tmp_path / 'out'
    (model / 'REVISION').write_text('0' * 40 + '\n')
    r = run('--dry-run', '--model-root', model, '--out', out, '--python', python, '--hipcc', '/bin/true')
    assert r.returncode == 3 and 'different revision' in r.stdout
    api = json.loads((EVIDENCE / 'glm-model-api.json').read_text())
    (model / 'REVISION').write_text(api['sha'] + '\n')
    (model / 'model-00001.safetensors').unlink()
    r = run('--dry-run', '--model-root', model, '--out', out, '--python', python, '--hipcc', '/bin/true')
    assert r.returncode == 3 and '1 checkpoint shard(s) are missing' in r.stdout


def test_an_unusable_max_memory_names_the_default(tmp_path):
    r = run('--dry-run', '--model-root', tmp_path / 'model', '--out', tmp_path / 'out',
            '--max-memory', 'not-json', '--hipcc', '/bin/true')
    assert r.returncode == 3
    assert '--max-memory is not usable' in r.stdout
    assert '{"0":"80GiB","1":"80GiB","cpu":"32GiB"}' in r.stdout


def test_an_unsupported_batch_is_a_parser_error(tmp_path):
    r = run('--dry-run', '--model-root', tmp_path / 'model', '--out', tmp_path / 'out', '--batch', '3')
    assert r.returncode == 2
    assert 'invalid choice' in r.stderr


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


def test_a_nonpositive_bandwidth_is_refused(tmp_path):
    r = run('--dry-run', '--model-root', tmp_path / 'model', '--out', tmp_path / 'out',
            '--bandwidth-tbps', '0', '--hipcc', '/bin/true')
    assert r.returncode == 3
    assert 'not a positive rate' in r.stdout
    assert '--bandwidth-tbps 4.3' in r.stdout


def test_annotate_refuses_a_stage_without_the_budget_fields(tmp_path):
    path = tmp_path / 'bench_result.json'
    path.write_text(json.dumps(dict(status='PASS', batch=4, ms_per_token=0.25)))
    with pytest.raises(Failure):
        annotate(path, dict(config=dict(layer_types=[])), batch=4, gpus=2, tbps=4.3)
