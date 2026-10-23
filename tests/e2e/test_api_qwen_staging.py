"""Existing Qwen staging branches through external APIs, sandbox HOME, no downloads."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from etx.external import ProcessRequest, execute, load_job_session

ROOT = Path(__file__).resolve().parents[2]
# Pinned runtime and golden expected by the original staging regression.
VLLM_TAG = 'v0.27.1'
GOLDEN_SHA256 = 'bab2f1fbe12f70179711ec40a215c74710845b73169b1cc6a3a6ca50e0a19484'


@pytest.fixture
def box(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv('ETX_QWEN_G6_CONFIG', raising=False)
    session = load_job_session(str(ROOT / 'examples/sessions/qwen_g6.py'))
    jobs = {job.name: job for job in session.jobs}
    env = {'HOME': str(home), 'PATH': '/usr/bin:/bin', 'ETX_SESS_LOGDIR': str(tmp_path / 'rec')}
    counter = 0

    def run(name):
        nonlocal counter
        assert name in ('stage_vllm', 'stage_dl', 'stage_pack')
        if name == 'stage_vllm':
            assert (home / 'vllm/.git').is_dir()  # Never enter the clone branch.
        elif name == 'stage_dl':
            assert (home / 'llm/qwen3-30b-a3b/hf/config.json').is_file()
            revision = home / 'llm/qwen3-30b-a3b/hf/REVISION'
            assert not revision.exists() or revision.read_text().strip() == '0' * 40
        else:
            assert (home / 'llm/qwen3-30b-a3b/vl.manifest').read_text() == 'packed'
        output = tmp_path / f'stage-{counter}'
        counter += 1
        result = execute(ProcessRequest(('/bin/bash', '-c', jobs[name].cmd), str(ROOT), 30,
                                         environment=env), output=output)
        return result, (output / 'output.log').read_text()

    return home, run


def git(tmp_path, *args, cwd):
    command = shutil.which('git')
    assert command
    output = tmp_path / f'git-{len(list(tmp_path.glob("git-*")))}'
    result = execute(ProcessRequest((command, '-c', 'user.name=F', '-c', 'user.email=f@x',
                                     '-c', 'commit.gpgsign=false', '-c', 'tag.gpgsign=false', *args),
                                    str(cwd), 20, environment={'PATH': '/usr/bin:/bin',
                                                            'HOME': str(tmp_path)}), output=output)
    assert result.status == 'complete', (output / 'output.log').read_text()


def test_stage_vllm_accepts_the_tag_and_refuses_a_checkout_past_it(box, tmp_path):
    home, run = box
    v = home / 'vllm'
    (v / 'csrc/rocm').mkdir(parents=True)
    (v / 'csrc/rocm/attention.cu').write_text('//\n')
    git(tmp_path, 'init', '-q', '.', cwd=v)
    git(tmp_path, 'add', '.', cwd=v)
    git(tmp_path, 'commit', '-qm', 'x', cwd=v)
    git(tmp_path, 'tag', VLLM_TAG, cwd=v)
    result, text = run('stage_vllm')
    assert result.returncode == 0 and f'= {VLLM_TAG}' in text
    git(tmp_path, 'commit', '-q', '--allow-empty', '-m', 'past the tag', cwd=v)
    result, text = run('stage_vllm')
    assert result.returncode == 1 and f'is not at {VLLM_TAG}' in text


def test_stage_dl_refuses_an_unpinned_earlier_download_before_fetching(box):
    home, run = box
    hf = home / 'llm/qwen3-30b-a3b/hf'
    hf.mkdir(parents=True)
    (hf / 'config.json').write_text('{}')
    result, text = run('stage_dl')
    assert result.returncode == 1 and 'refusing to reuse it' in text
    (hf / 'REVISION').write_text('0' * 40 + '\n')
    assert run('stage_dl')[0].returncode == 1


def test_stage_pack_on_a_packed_model_writes_the_pinned_golden_and_the_config(box):
    home, run = box
    model = home / 'llm/qwen3-30b-a3b'
    model.mkdir(parents=True)
    (model / 'vl.manifest').write_text('packed')
    result, text = run('stage_pack')
    assert result.returncode == 0, text
    assert hashlib.sha256((model / 'golden_c1024.txt').read_bytes()).hexdigest() == GOLDEN_SHA256
    cfg = json.loads((home / 'qwen-g6.json').read_text())
    assert cfg == dict(model=str(model), hf=str(model / 'hf'), golden_sha256=GOLDEN_SHA256)
