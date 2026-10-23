"""examples.vllm_llm.bench on a host without an AMD GPU, moved from tests/test_etx_bench.py, through external APIs only."""
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(Path('/dev/kfd').exists(), reason='this host has an AMD GPU; the CPU-box refusal does not apply')
def test_cpu_box_is_a_clear_failure():
    run = run_command([sys.executable, '-m', 'examples.vllm_llm.bench'], cwd=ROOT)
    assert run.returncode == 1
    assert 'BENCH FAIL no AMD GPU available' in run.stdout
    assert 'Traceback' not in run.stdout and 'TIMING ' not in run.stdout
