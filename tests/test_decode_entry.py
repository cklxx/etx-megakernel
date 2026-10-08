"""Item 8 of docs/PLAN-generic-decoder-2026-10-08.md section 7: one entry point for the three GQA models
(examples/vllm_llm/decode.py). CPU only: the tests check the selection and the commands, and run no step."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.llm.model import load_dims
from examples.vllm_llm import decode
from etx.select import plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    import os
    for k in [k for k in os.environ if k.startswith("ETX_")]:
        monkeypatch.delenv(k)


@pytest.mark.parametrize("name", decode.KNOWN)
def test_entry_selection_is_the_planner_selection(name):
    n, cfg = decode.resolve(name)
    assert n == name and cfg == ROOT / f"examples/llm/configs/{name}.json"
    d = decode.check_gqa(cfg)
    assert d.NH % d.NKV == 0 and d.G > 1                     # GQA
    assert decode.select(cfg).manifest() == plan(str(cfg), "gfx942").manifest()


@pytest.mark.parametrize("name", decode.KNOWN)
def test_model_py_builds_what_the_entry_prints(name, monkeypatch):
    """Every plan instance of model.py has the export that the printed selection names for its phase."""
    _, cfg = decode.resolve(name)
    monkeypatch.setenv("ETX_LLM_CONFIG", str(cfg))
    from examples.vllm_llm import model as M
    p = decode.select(cfg)
    seen = set()
    for i in M.instances(load_dims(str(cfg))):
        key = f"{i['role']}@0" if i["layer"] == 0 and f"{i['role']}@0" in p.phases else i["role"]
        if key in p.phases:
            assert i["export"] == p.phases[key].export, (name, key, i["export"], p.phases[key].export)
            seen.add(key)
    assert seen == set(p.phases), (name, set(p.phases) - seen)


@pytest.mark.parametrize("name", decode.KNOWN)
def test_steps_are_the_same_for_every_model(name, tmp_path):
    hf, work = tmp_path / "hf", tmp_path / name
    st = decode.steps(name, hf, work)
    assert [k for k, _ in st] == ["prep", "build", "run"]
    prep, build, run = (c for _, c in st)
    assert prep[1].endswith("examples/vllm_llm/prep.py") and prep[2:] == ["--src", str(hf), "--out", str(work),
                                                                         "--maxlen", str(decode.MAXLEN)]
    assert build[1:] == [str(ROOT / "examples/vllm_llm/build.sh"), str(work)]
    out = ROOT / "build/vl" / work.name                      # build.sh: OUT=$ROOT/build/vl/$(basename DIR)
    assert run == [str(out / "run"), str(work), str(out / "vl_plan.txt"), "--repeat", "2"]


def test_maxlen_matches_vm_run():
    assert f"--maxlen {decode.MAXLEN}" in (ROOT / "examples/vllm_llm/vm_run.sh").read_text()


def test_entry_has_no_branch_on_a_model_name():
    """No comparison, condition or lookup key in decode.py names a model. Messages may name one."""
    tree = ast.parse((ROOT / "examples/vllm_llm/decode.py").read_text())
    tests = [n for n in ast.walk(tree) if isinstance(n, (ast.Compare, ast.Subscript, ast.Dict))]
    tests += [n.test for n in ast.walk(tree) if isinstance(n, (ast.If, ast.IfExp, ast.While))]
    hits = [c.value for t in tests for c in ast.walk(t) if isinstance(c, ast.Constant) and isinstance(c.value, str)
            and any(m in c.value.lower() for m in ("qwen", "mixtral", "deepseek", "phi"))]
    assert not hits, hits


def test_mla_config_is_refused(tmp_path):
    c = json.loads((ROOT / "examples/llm/configs/qwen3-8b.json").read_text())
    c.update(kv_lora_rank=512, q_lora_rank=1536)
    (tmp_path / "config.json").write_text(json.dumps(c))
    with pytest.raises(decode.NotGQA, match="examples/dsv2lite"):
        decode.check_gqa(tmp_path / "config.json")
    r = subprocess.run([sys.executable, "-m", "examples.vllm_llm.decode", str(tmp_path), "--dry-run"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "examples/dsv2lite" in r.stderr


def test_config_path_and_hf_dir(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text((ROOT / "examples/llm/configs/mixtral-8x7b.json").read_text())
    assert decode.resolve("mixtral-8x7b", str(hf)) == ("mixtral-8x7b", hf / "config.json")
    assert decode.resolve(str(hf / "config.json")) == (tmp_path.name, hf / "config.json")
    assert decode.resolve(str(hf)) == (tmp_path.name, hf / "config.json")        # <dir>/hf names the model <dir>
    other = tmp_path / "m2"
    other.mkdir()
    (other / "config.json").write_text((hf / "config.json").read_text())
    assert decode.resolve(str(other)) == ("m2", other / "config.json")
    r = subprocess.run([sys.executable, "-m", "examples.vllm_llm.decode", str(hf), "--dry-run"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and f"model {tmp_path.name}:" in r.stdout, r.stderr
    with pytest.raises(FileNotFoundError):
        decode.resolve("no-such-model")


@pytest.mark.parametrize("name", decode.KNOWN)
def test_dry_run_cli(name, tmp_path):
    r = subprocess.run([sys.executable, "-m", "examples.vllm_llm.decode", name, "--dry-run",
                        "--hf", str(tmp_path / "hf"), "--work", str(tmp_path / name)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert f"model {name}:" in r.stdout and "[prep]" in r.stdout and "[build]" in r.stdout and "[run]" in r.stdout
    has_golden = (ROOT / f"examples/llm/results/golden_c1024_{name}.txt").is_file()
    assert ("no golden file" in r.stdout) == (not has_golden)
