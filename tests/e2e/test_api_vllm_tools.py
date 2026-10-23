"""vLLM analysis/check commands behind the qwen-vllm harness answer through their historical module paths."""
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('module', ['dense_bytes', 'router_study_guards', 'verify_gaps_cpu', 'polling_isa_cpu', 'watchdog_cpu'])
def test_command_help(module):
    run = run_command([sys.executable, '-m', f'examples.vllm_llm.{module}', '--help'], cwd=ROOT)
    assert run.returncode == 0 and 'usage:' in run.stdout, run.stdout[-1000:]


def test_dense_bytes_requires_both_plans():
    run = run_command([sys.executable, '-m', 'examples.vllm_llm.dense_bytes'], cwd=ROOT)
    assert run.returncode == 2 and '--replicated' in run.stdout
