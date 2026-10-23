"""Expert-merge prepare/inputs commands, moved from tests/test_expert_merge.py, through external APIs only."""
import json
import os
from pathlib import Path
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]


def merge(module, *argv, env=None):
    return run_command([sys.executable, '-m', f'examples.vllm_llm.expert_merge.{module}', *map(str, argv)], cwd=ROOT, env=env)


def test_generated_graph_and_identity(tmp_path):
    # The caller's model switches must not label a different executable.
    env = dict(os.environ, ETX_VL_NB='2', ETX_VL_DSL_EXT='1')
    for arm in ('control', 'candidate'):
        out = tmp_path/arm
        argv = ['generate', out, '--arm', arm, '--diagnostic'] + (['--head'] if arm == 'candidate' else [])
        run = merge('prepare', *argv, env=env)
        assert run.returncode == 0, run.stdout[-1500:]
        identity = json.loads((out/'identity.json').read_text())
        verified = merge('prepare', 'verify', out, env=env)
        assert verified.returncode == 0 and json.loads(verified.stdout.strip().splitlines()[-1]) == identity
        p = json.loads((out/'plan.json').read_text())
        assert len(p['descs']) == 192+256+256
        assert sum(len(e['counts']) for e in p['events']) == 9
        assert sorted(t['type'] for t in p['descs']).count(0) == 192
        assert [(e['scope'], e['counts']) for e in p['events']] == [('DOMAIN', [24]*8), ('DEVICE', [256])]
        for t in p['descs']:
            if t['type'] in (0, 1): assert t['domain'] == t['coord'][0]
        assert 'ETX_IMPORT_LDS 21120' in (out/'tiles.hip').read_text()
        assert identity.get('full_model_default') == 'OFF'


def test_input_pack_hash(tmp_path):
    out = tmp_path/'pack'
    assert merge('inputs', 'synthetic', out).returncode == 0
    run = merge('inputs', 'validate', out)
    assert run.returncode == 0 and json.loads(run.stdout.strip().splitlines()[-1])['kind'] == 'synthetic'
    (out/'x.bin').write_bytes(b'\0'*4096)
    run = merge('inputs', 'validate', out)
    assert run.returncode != 0 and 'hash mismatch' in run.stdout
