"""Attention-early-merge harness commands answer --help through the external API (no build, no GPU)."""
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('module', ('build', 'offline', 'build_model', 'session', 'simulate'))
def test_harness_commands_answer_help(module):
    result = run_command([sys.executable, '-m', 'examples.vllm_llm.attention_early_merge.' + module, '--help'],
                         cwd=ROOT, check=True)
    assert 'usage:' in result.stdout
