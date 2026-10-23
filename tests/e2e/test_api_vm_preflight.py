"""The tools/vm_preflight.sh entry, moved from tests/test_vm_preflight.py, through external APIs only."""
import os
from pathlib import Path

from etx.testing import run_command

REPO = Path(__file__).resolve().parents[2]


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\n'+body+'\n')
    path.chmod(0o755)


def test_cli_exit_and_shell_entry(tmp_path):
    executable(tmp_path/'bin/docker', 'exit 1')
    executable(tmp_path/'bin/rocm-smi', 'exit 1')
    env = dict(os.environ, HOME=str(tmp_path), VLLM=str(tmp_path/'vllm'),
               ETX_ROCM_HEADERS=str(tmp_path/'headers'),
               PATH=str(tmp_path/'bin')+os.pathsep+os.environ['PATH'])
    result = run_command(['bash', str(REPO/'tools/vm_preflight.sh'), str(tmp_path/'model'),
                          str(tmp_path/'arms'), '--gates', str(tmp_path/'gates')], env=env, cwd=tmp_path)
    assert result.returncode == 1
    assert result.stdout.count('\nFAIL ') + result.stdout.startswith('FAIL ') >= 6
    assert 'PREFLIGHT FAIL ' in result.stdout
