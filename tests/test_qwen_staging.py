"""examples/sessions/qwen_g6.py staging jobs, executed for real in a sandbox HOME (no network: only the branches that
decide locally -- the vLLM tag check, the refusal of an unpinned download, the golden + config step on a packed model)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from etx.tools.gpu_jobs import load_session
from examples.vllm_llm.vm_preflight import GOLDEN_SHA256, VLLM_TAG

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def box(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))                     # CONFIG (~/qwen-g6.json) resolves here at load
    jobs, _ = load_session(ROOT / "examples/sessions/qwen_g6.py")
    env = dict(os.environ, HOME=str(home), ETX_SESS_LOGDIR=str(tmp_path / "rec"))
    run = lambda name: subprocess.run(["bash", "-c", {j.name: j for j in jobs}[name].cmd], cwd=ROOT, env=env,
                                      capture_output=True, text=True)
    return home, run


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=F", "-c", "user.email=f@x", "-c", "commit.gpgsign=false",
                    "-c", "tag.gpgsign=false", *args], cwd=cwd, check=True, capture_output=True)


def test_stage_vllm_accepts_the_tag_and_refuses_a_checkout_past_it(box):
    home, run = box
    v = home / "vllm"
    (v / "csrc/rocm").mkdir(parents=True)
    (v / "csrc/rocm/attention.cu").write_text("//\n")
    git("init", "-q", ".", cwd=v); git("add", ".", cwd=v); git("commit", "-qm", "x", cwd=v); git("tag", VLLM_TAG, cwd=v)
    r = run("stage_vllm")
    assert r.returncode == 0 and f"= {VLLM_TAG}" in r.stdout
    git("commit", "-q", "--allow-empty", "-m", "past the tag", cwd=v)
    r = run("stage_vllm")
    assert r.returncode == 1 and f"is not at {VLLM_TAG}" in r.stdout


def test_stage_dl_refuses_an_unpinned_earlier_download_before_fetching(box):
    home, run = box
    hf = home / "llm/qwen3-30b-a3b/hf"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text("{}")                     # an old download, no REVISION
    r = run("stage_dl")
    assert r.returncode == 1 and "refusing to reuse it" in r.stdout
    (hf / "REVISION").write_text("0" * 40 + "\n")             # a different revision: refused too
    assert run("stage_dl").returncode == 1


def test_stage_pack_on_a_packed_model_writes_the_pinned_golden_and_the_config(box):
    home, run = box
    m = home / "llm/qwen3-30b-a3b"
    m.mkdir(parents=True)
    (m / "vl.manifest").write_text("packed")                  # already packed: prep.py is not rerun
    r = run("stage_pack")
    assert r.returncode == 0, r.stdout + r.stderr
    assert hashlib.sha256((m / "golden_c1024.txt").read_bytes()).hexdigest() == GOLDEN_SHA256
    cfg = json.loads((home / "qwen-g6.json").read_text())
    assert cfg == dict(model=str(m), hf=str(m / "hf"), golden_sha256=GOLDEN_SHA256)
