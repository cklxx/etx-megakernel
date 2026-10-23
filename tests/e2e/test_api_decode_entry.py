"""The GQA decode entry (examples/vllm_llm/decode.py) dry runs through the external API; CPU only, no step runs."""
import json
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
KNOWN = ('qwen3-8b', 'qwen3-30b-a3b', 'mixtral-8x7b')


def decode(*argv):
    return run_command([sys.executable, "-m", "examples.vllm_llm.decode", *map(str, argv)], cwd=ROOT, timeout=120)


def test_mla_config_is_refused(tmp_path):
    c = json.loads((ROOT / "examples/llm/configs/qwen3-8b.json").read_text())
    c.update(kv_lora_rank=512, q_lora_rank=1536)
    (tmp_path / "config.json").write_text(json.dumps(c))
    r = decode(tmp_path, "--dry-run")
    assert r.returncode == 2 and "examples/dsv2lite" in r.stdout


def test_config_dir_names_the_model(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text((ROOT / "examples/llm/configs/mixtral-8x7b.json").read_text())
    r = decode(hf, "--dry-run")
    assert r.returncode == 0 and f"model {tmp_path.name}:" in r.stdout, r.stdout


@pytest.mark.parametrize("name", KNOWN)
def test_dry_run_cli(name, tmp_path):
    r = decode(name, "--dry-run", "--hf", tmp_path / "hf", "--work", tmp_path / name)
    assert r.returncode == 0, r.stdout
    assert f"model {name}:" in r.stdout and "[prep]" in r.stdout and "[build]" in r.stdout and "[run]" in r.stdout
    has_golden = (ROOT / f"examples/llm/results/golden_c1024_{name}.txt").is_file()
    assert ("no golden file" in r.stdout) == (not has_golden)
