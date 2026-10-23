"""examples.vllm_llm.g6_session refusals, moved from tests/test_qwen_g6.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
# examples.vllm_llm.g6_session.BUILD_INPUTS, in its check order.
BUILD_INPUTS = ('interpreter', 'vllm_source', 'packed_model', 'rocm_headers', 'gpu')


def g6(*argv, env):
    return run_command([sys.executable, '-m', 'examples.vllm_llm.g6_session', *argv], cwd=ROOT,
                       env=dict(os.environ, **env))


def test_pairs_refuses_an_uncertified_build(tmp_path):
    (tmp_path / "certificate_kept").mkdir()
    run = g6('pairs', 'kept_l1', env=dict(ETX_SESS_LOGDIR=str(tmp_path)))
    assert run.returncode != 0 and 'kept_l1 has no certificate' in run.stdout


def test_preflight_refuses_a_box_without_the_build_inputs_before_anything_runs(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(dict(model=str(tmp_path / "absent"), hf=str(tmp_path / "absent/hf"), golden_sha256="0" * 64)))
    run = g6('preflight', env=dict(ETX_QWEN_G6_CONFIG=str(cfg), ETX_SESS_LOGDIR=str(tmp_path), VLLM=str(tmp_path / "no-vllm")))
    out = run.stdout
    assert run.returncode != 0 and 'box inputs missing' in out
    assert "FAIL packed_model" in out and "missing=vl.bin,vl.manifest,hf/config.json,golden_c1024.txt" in out
    assert "FAIL vllm_source" in out and "git clone --depth 1 --branch v0.27.1" in out
    assert not (tmp_path / "inputs.json").exists()
    assert [l.split()[1].rstrip(":") for l in out.splitlines() if l.startswith(("PASS", "FAIL"))] == list(BUILD_INPUTS)
