"""CPU-only box fixtures; no Docker daemon, downloads or GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import venv

import pytest

from examples.vllm_llm import vm_preflight as p

REPO = Path(__file__).resolve().parents[1]
NAMES = ('interpreter', 'vllm_source', 'packed_model', 'arms', 'rocm_headers', 'runtime_image', 'gpu', 'gates')


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\n'+body+'\n')
    path.chmod(0o755)


@pytest.fixture
def box(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    root.mkdir()
    monkeypatch.setattr(p, 'ROOT', root)
    # A genuine isolated venv with fixture modules, no pip/bootstrap/network.
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root/'.venv')
    site = Path(subprocess.check_output([root/'.venv/bin/python', '-c',
                                       'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    for name in ('yaml', 'numpy'):
        (site/f'{name}.py').write_text('')
    source = tmp_path/'vllm'
    (source/'csrc/rocm').mkdir(parents=True)
    (source/'csrc/rocm/attention.cu').write_text('// fixture\n')
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.test', '-c', 'commit.gpgsign=false',
                    'commit', '-qm', 'fixture'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'tag.gpgsign=false', 'tag', p.VLLM_TAG], check=True)   # a release clone is at its tag
    monkeypatch.setenv('VLLM', str(source))
    model = tmp_path/'llm/qwen3-30b-a3b'
    (model/'hf').mkdir(parents=True)
    for name in ('vl.bin', 'vl.manifest', 'hf/config.json'):
        (model/name).write_text('fixture')
    golden = root/'examples/llm/results/golden_c1024_qwen3-30b-a3b.txt'
    golden.parent.mkdir(parents=True)
    shutil.copyfile(REPO/golden.relative_to(root), golden)
    shutil.copyfile(golden, model/'golden_c1024.txt')
    arms = tmp_path/'arms'; arms.mkdir()
    gates = tmp_path/'cpu-gates.json'
    gates.write_text(json.dumps(dict.fromkeys(p.GATES, True)))
    headers = tmp_path/'rocm-headers/inc'
    (headers/'hip').mkdir(parents=True)
    (headers/'hip/hip_runtime.h').write_text('// fixture')
    monkeypatch.setenv('ETX_ROCM_HEADERS', str(headers))
    kfd = tmp_path/'dev/kfd'; kfd.parent.mkdir(); kfd.touch()
    monkeypatch.setattr(p, 'KFD', kfd)
    bin_dir = tmp_path/'bin'
    executable(bin_dir/'rocm-smi', 'echo "Card Series: AMD Instinct MI300X"')
    executable(bin_dir/'docker', 'test "$1 $2 $3" = "image inspect '+p.IMAGE+'" || exit 1\necho sha256:fixture')
    monkeypatch.setenv('PATH', str(bin_dir)+os.pathsep+os.environ['PATH'])
    return dict(root=root, model=model, arms=arms, gates=gates, source=source,
                headers=headers, kfd=kfd, bin=bin_dir, site=site)


def run(box, *extra):
    return p.main([str(box['model']), str(box['arms']), '--gates', str(box['gates']), *extra])


def verdicts(capsys):
    return capsys.readouterr().out.splitlines()


def test_all_good(box, capsys):
    assert run(box) == 0
    lines = verdicts(capsys)
    assert [line.split(':')[0] for line in lines[:-1]] == ['PASS '+n for n in NAMES]
    assert lines[-1] == 'PREFLIGHT PASS'
    assert 'commit=' in lines[1] and lines[1].endswith('= v0.27.1')
    assert p.IMAGE in lines[5]
    for name in ('vl.bin', 'vl.manifest', 'hf/config.json', 'golden_c1024.txt'):
        assert f'{name}={(box["model"]/name).stat().st_size} bytes' in lines[2]


@pytest.mark.parametrize('item', NAMES)
def test_each_item_fails_alone(box, capsys, item):
    if item == 'interpreter':
        shutil.rmtree(box['root']/'.venv')
    elif item == 'vllm_source':
        shutil.rmtree(box['source'])
    elif item == 'packed_model':
        (box['model']/'vl.bin').unlink()
    elif item == 'arms':
        box['arms'].rmdir()
    elif item == 'rocm_headers':
        shutil.rmtree(box['headers'])
    elif item == 'runtime_image':
        executable(box['bin']/'docker', 'echo "image absent" >&2; exit 1')
    elif item == 'gpu':
        box['kfd'].unlink()
    elif item == 'gates':
        box['gates'].unlink()
    before = snapshot(box['root'].parent)
    assert run(box) == 1
    assert snapshot(box['root'].parent) == before
    lines = verdicts(capsys)
    failures = [line for line in lines if line.startswith('FAIL ')]
    assert len(failures) == 1 and failures[0].startswith(f'FAIL {item}:')
    assert '; supply: ' in failures[0]
    assert sum(line.startswith('PASS ') for line in lines) == 7
    assert lines[-1] == 'PREFLIGHT FAIL 1 item(s)'
    if item == 'interpreter':
        assert 'python3 -m venv --without-pip .venv' in failures[0]
        assert 'https://bootstrap.pypa.io/get-pip.py' in failures[0]
        assert '.venv/bin/python -m pip install -e ".[dev]"' in failures[0]
    if item == 'vllm_source':
        assert f'git clone --depth 1 --branch {p.VLLM_TAG} https://github.com/vllm-project/vllm.git {box["source"]}' in failures[0]
    if item == 'gates':
        assert all(f'{name}=None' in failures[0] for name in p.GATES)


@pytest.mark.parametrize('missing', ['yaml', 'numpy'])
def test_dependency_import_failure(box, capsys, missing):
    (box['site']/f'{missing}.py').unlink()
    assert run(box) == 1
    assert f"No module named '{missing}'" in '\n'.join(verdicts(capsys))


def test_golden_hash_mismatch(box, capsys):
    data = b'wrong golden\n'
    (box['model']/'golden_c1024.txt').write_bytes(data)
    assert run(box) == 1
    text = '\n'.join(verdicts(capsys))
    assert 'FAIL packed_model:' in text
    assert f'sha256={hashlib.sha256(data).hexdigest()} expected={p.GOLDEN_SHA256}' in text


@pytest.mark.parametrize('name', p.GATES)
@pytest.mark.parametrize('value', [False, 1, 'True', None])
def test_gates_require_literal_true(box, capsys, name, value):
    values = dict.fromkeys(p.GATES, True); values[name] = value
    box['gates'].write_text(json.dumps(values))
    assert run(box) == 1
    line = next(line for line in verdicts(capsys) if line.startswith('FAIL gates:'))
    assert all(f'{k}={values[k]!r}' in line for k in p.GATES)


@pytest.mark.parametrize('broken', ['not json', '[]'])
def test_malformed_gates(box, capsys, broken):
    box['gates'].write_text(broken)
    assert run(box) == 1
    line = next(line for line in verdicts(capsys) if line.startswith('FAIL gates:'))
    assert all(f'{k}=None' in line for k in p.GATES)


@pytest.mark.parametrize('broken', ['checkout', 'attention', 'headers', 'writable', 'product'])
def test_partial_inputs_refused(box, capsys, broken):
    item = dict(checkout='vllm_source', attention='vllm_source', headers='rocm_headers',
                writable='arms', product='gpu')[broken]
    if broken == 'checkout':
        shutil.rmtree(box['source']/'.git')
    elif broken == 'attention':
        (box['source']/'csrc/rocm/attention.cu').unlink()
    elif broken == 'headers':
        (box['headers']/'hip/hip_runtime.h').unlink()
    elif broken == 'writable':
        box['arms'].chmod(0o555)
    elif broken == 'product':
        executable(box['bin']/'rocm-smi', 'echo "Card Series: AMD Instinct MI308X"')
    assert run(box) == 1
    assert any(line.startswith('FAIL '+item+':') for line in verdicts(capsys))


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_fix_required_before_writes(box, capsys):
    box['arms'].rmdir()
    before = snapshot(box['root'].parent)
    assert run(box) == 1
    assert snapshot(box['root'].parent) == before
    assert not box['arms'].exists()
    assert run(box, '--fix') == 0
    assert box['arms'].is_dir()
    assert 'FIX arms:' in '\n'.join(verdicts(capsys))


def test_fix_restores_registered_golden(box, capsys):
    (box['model']/'golden_c1024.txt').write_text('wrong')
    assert run(box, '--fix') == 0
    assert hashlib.sha256((box['model']/'golden_c1024.txt').read_bytes()).hexdigest() == p.GOLDEN_SHA256


def test_fix_never_provisions_gpu(box, capsys):
    box['kfd'].unlink()
    executable(box['root']/'tools/hotaisle_grab.sh', 'touch GPU_PROVISIONED')
    before = snapshot(box['root'].parent)
    assert run(box, '--fix') == 1
    assert snapshot(box['root'].parent) == before
    assert 'FIX gpu:' not in '\n'.join(verdicts(capsys))


def test_probe_timeout_is_a_named_failure(box, monkeypatch, capsys):
    original = p.command
    def timeout(argv):
        if argv[0] == 'docker':
            raise subprocess.TimeoutExpired(argv, 10)
        return original(argv)
    monkeypatch.setattr(p, 'command', timeout)
    assert run(box) == 1
    assert any(line.startswith('FAIL runtime_image:') for line in verdicts(capsys))


def test_prepare_refuses_before_build_or_writes(box, capsys):
    from examples.sessions import qwen300_hour as h
    config = box['root']/'prepared.json'
    config.write_text(json.dumps(dict(model=str(box['model']), cpu_gates=str(box['gates']),
                                     builds=dict(kept=str(box['arms']/'kept')))))
    shutil.rmtree(box['source'])
    before = snapshot(box['root'].parent)
    with pytest.raises(ValueError, match='VM preflight refused staging'):
        h.prepare(config)
    assert not (box['arms']/'kept').exists()
    assert snapshot(box['root'].parent) == before
    assert 'FAIL vllm_source:' in '\n'.join(verdicts(capsys))


# The tools/vm_preflight.sh entry is checked in tests/e2e/test_api_vm_preflight.py.


def test_fix_failure_keeps_gate_values(box, monkeypatch, capsys):
    monkeypatch.delenv('ETX_STAGE_SOURCE', raising=False)
    box['gates'].unlink()
    assert run(box, '--fix') == 1
    line = next(line for line in verdicts(capsys) if line.startswith('FAIL gates:'))
    assert all(f'{name}=None' in line for name in p.GATES)
    assert 'fix failed:' in line


def test_stage_command_quotes_paths(tmp_path, monkeypatch):
    # Shell metacharacters in a model directory must remain literal path bytes.
    source = tmp_path/'source'; source.mkdir()
    name = 'model $(touch SHOULD_NOT_EXIST)'
    (source/name).mkdir(); (source/name/'vl.bin').write_bytes(b'fixture')
    destination = tmp_path/'dest space'
    executable(tmp_path/'bin/rsync', 'cp -R "$2". "$3"')
    monkeypatch.setenv('PATH', str(tmp_path/'bin')+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('ETX_STAGE_SOURCE', str(source))
    subprocess.run(['bash', '-c', p.stage(name, destination, directory=True)], cwd=tmp_path, check=True)
    assert (destination/'vl.bin').read_bytes() == b'fixture'
    assert not (tmp_path/'SHOULD_NOT_EXIST').exists()


def test_a_checkout_off_the_runtime_tag_is_refused(box, capsys):
    """2026-10-14: a v0.27.0-5 checkout passed while the detail printed v0.27.1; the tag is now verified."""
    git = ['git', '-C', str(box['source']), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
           '-c', 'commit.gpgsign=false']
    subprocess.run([*git, 'commit', '-q', '--allow-empty', '-m', 'past the tag'], check=True)
    assert run(box) == 1
    line = next(l for l in verdicts(capsys) if l.startswith('FAIL vllm_source'))
    assert f'not {p.VLLM_TAG}' in line and f'git clone --depth 1 --branch {p.VLLM_TAG}' in line


def test_only_runs_the_named_items(box, capsys):
    assert p.check(box['model'], box['arms'], box['gates'], only=('interpreter', 'packed_model')) == 0
    lines = verdicts(capsys)
    assert [l.split(':')[0] for l in lines[:-1]] == ['PASS interpreter', 'PASS packed_model']
    with pytest.raises(ValueError, match='unknown preflight items'):
        p.check(box['model'], box['arms'], box['gates'], only=('nope',))
