"""qwen300_hour commands, moved from tests/test_qwen300_hour.py, through external APIs only."""
import ast
import json
import os
from pathlib import Path
import struct
import sys

import pytest
from etx.testing import run_command

ROOT = Path(__file__).resolve().parents[2]
# etx.tools.inbody.MAGIC, pinned in tests/test_e2e_pins.py.
MAGIC = '0x3142444F424E4954'
# len(examples.sessions.qwen300_hour.REGISTERED), pinned in tests/test_e2e_pins.py.
REGISTERED_COUNT = 20


def hour(*argv, env=None):
    return run_command([sys.executable, '-m', 'examples.sessions.qwen300_hour', *map(str, argv)], cwd=ROOT,
                       env=env, timeout=15)


def trace(previous=True, ambiguous=False):
    p = dict(types={'attn_0': dict(id=0), 'prep_0': dict(id=1), 'reduce_0': dict(id=2)},
             descs=[dict(id=i, type=i, coord=[0], domain=0, mode='static') for i in range(3)],
             domains=1, workers_per_domain=2, events=[dict(name='done', offset=0, shape=[1])],
             ev_inst=[['done', [0], [0]]], task_in=[[], [], [0]])
    r = dict(version=1, tick_ns=10, waves_per_task=1,
             legacy=[[100, 110, 200, 0], [150, 160, 250, 1], [260, 290, 400, 1 if previous else 2]],
             poll=[[0], [0], [280]], events=[[215, 220, 0]],
             waves=[[180, 190], [220, 230], [350, 380]])
    if ambiguous:
        r['events'] = [[251, 255, 0]]
    return p, r


def test_analysis_cli_emits_small_decision_view_under_its_deadline(tmp_path):
    plan, raw = trace()
    p = tmp_path/'plan.json'; p.write_text(json.dumps(plan))
    words = [int(MAGIC, 16), 1, 3, 1, 1, 10]
    for k in ('legacy', 'poll', 'events', 'waves'): words += [v for row in raw[k] for v in row]
    r = tmp_path/'raw.bin'; r.write_bytes(struct.pack('<'+'Q'*len(words), *words))
    out = tmp_path/'edges.json'
    run = hour('--analyze', p, r, out)
    assert run.returncode == 0, run.stdout
    full = json.loads(out.read_text()); small = json.loads(out.with_suffix('.summary.json').read_text())
    assert full['inbody']['raw'] == raw
    assert small['critical_pools'] == full['critical_pools']
    assert 'raw' not in small['inbody']
    assert small['inbody']['phases'] == full['inbody']['phases']
    assert small['stamp_perturbation'] == full['stamp_perturbation']
    assert small['stamp_perturbation']['status'] == 'NOT-COMPLETE'
    assert small['gap_atoms']['G_ms'] is None


@pytest.mark.parametrize('seconds,accepted', [
    ('8400', True), ('6601', True), ('3600', True), ('8401', False),
    ('0', False), ('-1', False), ('nan', False), ('inf', False), ('invalid', False),
])
def test_cap_environment_validation(seconds, accepted):
    result = hour('--dry-run', env=dict(os.environ, ETX_SESS_MAX_SECONDS=seconds))
    assert (result.returncode == 0) is accepted
    if accepted: assert f'Session hard stop: {seconds}s' in result.stdout
    else: assert 'cap must be positive and <=8400 seconds' in result.stdout


def test_generated_template_carries_all_session_and_guard_keys(tmp_path):
    keys=set()
    for module in ('examples/sessions/qwen300_hour.py', 'examples/vllm_llm/environment_guard.py'):
        tree=ast.parse((ROOT/module).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant):
                base=node.value
                if ((module.endswith('qwen300_hour.py') and isinstance(base,ast.Name) and base.id=='c') or
                    (isinstance(base,ast.Attribute) and base.attr=='c')):
                    keys.add(node.slice.value)
    path=tmp_path/'template.json'
    assert hour('--template', path).returncode==0
    config=json.loads(path.read_text())
    assert keys <= config.keys()
    assert config['environment_guard']['trees']=={}
    assert config['weight_sha256']=={}


def test_missing_staging_still_prints_all_registered_rows(tmp_path):
    run = hour('--run', env=dict(os.environ, ETX_QWEN300_CONFIG=str(tmp_path/'missing.json'), ETX_SESS_LOGDIR=str(tmp_path/'out')))
    assert run.returncode == 2, run.stdout
    report = json.loads((tmp_path/'out/report.json').read_text())
    assert len(report['thresholds']) >= REGISTERED_COUNT
    assert all(r['value'] is None for r in report['thresholds'].values())
    assert 'REGISTERED THRESHOLDS' in run.stdout
