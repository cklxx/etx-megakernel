"""The GLM-5.3 CPU fixture check (same HIP bodies, 32 exact greedy tokens), moved from tests/test_glm53_model.py."""
import json
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('batch,bf16,fp8', [(1, False, False), (2, False, False), (4, True, False), (8, True, True)])
def test_same_hip_body_full_hybrid_model_32_exact_tokens_and_states(tmp_path, batch, bf16, fp8):
    pytest.importorskip('torch'); pytest.importorskip('transformers')
    argv = ['--out', str(tmp_path/'fixture'), '--batch', str(batch), '--steps', '32']
    argv += ['--bf16'] if bf16 else []
    argv += ['--fp8'] if fp8 else []
    run = run_command([sys.executable, '-m', 'examples.glm53.cpu_check', *argv], cwd=ROOT, timeout=900)
    assert run.returncode == 0, run.stdout[-2000:]
    result = json.loads(run.stdout[run.stdout.index('{'):])
    assert result['matched'] == batch*32 and result['status'] == 'PASS'
    assert result['fixture'] and not result['full_checkpoint']
