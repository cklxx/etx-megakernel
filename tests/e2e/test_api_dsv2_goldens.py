"""examples.dsv2lite.batch_goldens certify/verify/export commands, moved from tests/test_dsv2_batch.py."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
from etx.testing import run_command, write_npz

ROOT = Path(__file__).resolve().parents[2]
# examples.dsv2lite.batch_goldens.{FILES, ROTATE}, pinned in tests/test_e2e_pins.py.
FILES = ('golden.npz', 'golden_tokens.txt', 'golden_hidden.bin', 'fleet_cache.bin')
ROTATE = 97


def goldens(*argv):
    return run_command([sys.executable, '-m', 'examples.dsv2lite.batch_goldens', *map(str, argv)], cwd=ROOT)


def request_dir(data, r):
    return data if r == 0 else data / f'r{r}'


def rotate(ids, r):
    k = (ROTATE * r) % len(ids)
    return ids[k:] + ids[:k]


@pytest.mark.parametrize('damage', ['mixed', 'missing', 'prompt'])
def test_golden_certificate_binds_prompt_and_execution_bytes(tmp_path, damage):
    for r in range(2):
        d = request_dir(tmp_path, r)
        d.mkdir(exist_ok=True)
        prompt = rotate(list(range(128)), r)
        write_npz(d / 'golden.npz', prompt_ids=prompt, token_ids=[7] * 33)
        (d / 'golden_tokens.txt').write_text(' '.join(['7'] * 33))
        for name in ('fleet_cache.bin', 'golden_hidden.bin'):
            (d / name).write_bytes(f'{name}: request {r}'.encode())
        cert = dict(prompt_ids=prompt, sha256={f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in FILES})
        (d / 'golden.binding.json').write_text(json.dumps(cert))
    if damage == 'mixed':
        for name in ('fleet_cache.bin', 'golden_hidden.bin'):
            (tmp_path / 'r1' / name).write_bytes((tmp_path / name).read_bytes())
    elif damage == 'missing':
        (tmp_path / 'r1/golden.binding.json').unlink()
    else:
        cert['prompt_ids'] = list(range(128))
        (tmp_path / 'r1/golden.binding.json').write_text(json.dumps(cert))
    run = goldens(tmp_path, '--requests', '2')
    assert run.returncode != 0 and 'request 1:' in run.stdout and 'binding' in run.stdout, run.stdout[-1500:]


def test_fresh_golden_binding_verifies_and_exports(tmp_path):
    write_npz(tmp_path / 'golden.npz', prompt_ids=list(range(128)), token_ids=[7] * 33)
    (tmp_path / 'golden_tokens.txt').write_text(' '.join(['7'] * 33))
    for name in ('fleet_cache.bin', 'golden_hidden.bin'):
        (tmp_path / name).write_bytes(b'fresh reference')
    for argv in (('--certify',), ('--verify-binding',), ('--requests', '1')):
        run = goldens(tmp_path, *argv)
        assert run.returncode == 0, (argv, run.stdout[-1500:])
    assert (tmp_path / 'vllm/golden_r0.txt').read_text().startswith('prompt 0 1 2')
