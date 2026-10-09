"""Portable session acceptance: only synthetic paths and stub processes."""
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from etx.tools.gpu_jobs import load_session
from examples.sessions import vlbest as s
from examples.vllm_llm import vlbest_report as r

SESSION = Path(__file__).resolve().parents[1] / 'examples/sessions/vlbest.py'


def test_budget_and_graph(capsys):
    jobs, limit = load_session(SESSION)
    graph = {j.name: j for j in jobs}
    assert s.VM_LIFE_SECONDS == 9000
    assert s.SETUP_ALLOWANCE_SECONDS == 600
    assert limit == 8400 == s.VM_LIFE_SECONDS - s.SETUP_ALLOWANCE_SECONDS
    assert s.BUDGET == 'BUDGET vm-life=150m runner-cap=140m'
    assert graph['budget'].cmd == "printf '%s\\n' 'BUDGET vm-life=150m runner-cap=140m'"
    assert s.main(['--dry-run']) == 0
    out = capsys.readouterr().out
    assert s.BUDGET in out
    rows = [line for line in out.splitlines() if '  after: ' in line]
    assert len(rows) == len(jobs)
    assert all(j.name in rows[i] for i, j in enumerate(jobs))
    seen = set()
    for j in jobs:
        assert set(j.after + j.after_any) <= seen, j.name
        seen.add(j.name)
        assert subprocess.run(['bash', '-n', '-c', j.cmd], capture_output=True).returncode == 0, j.name
    assert set(graph['report'].after_any) == set(graph) - {'report'}


def test_correctness_and_screening_dependencies():
    jobs, _ = load_session(SESSION)
    graph = {j.name: j for j in jobs}
    assert 'ETX_VL_CHECK=1' in graph['build_chkx'].cmd
    assert '--check-pos 0' in graph['check_timed'].cmd
    assert '--check-pos 1040' in graph['check_timed'].cmd
    assert '--checks-only' in graph['check_certificate'].cmd
    for j in jobs:
        if j.name.startswith(('a_r', 'b_r', 'S0_r', 'F_r', 'etx_r', 'etxu_r')):
            assert j.gpu == 0
            assert 'check_certificate' in j.after
    for rnd in (1, 2):
        names = [j.name for j in jobs if j.name.startswith(f'a_r{rnd}_')]
        assert names[0] == f'a_r{rnd}_S0_open'
        assert names[-1] == f'a_r{rnd}_S0_close'
        assert set(names[1:-1]) == {f'a_r{rnd}_S{i}' for i in range(1, 10)}
        assert 'moe_tune' in graph[f'a_r{rnd}_S9'].after
        assert 'moe_configs' in graph[f'a_r{rnd}_S1'].after
    timed = [j for j in jobs if j.name.startswith(('a_r', 'b_r', 'S0_r', 'F_r', 'etx_r', 'etxu_r'))]
    for i, j in enumerate(timed):
        # Every previous arm remains a completion barrier, including skipped arms.
        assert {p.name for p in timed[:i]} <= set(j.after_any)
    for rnd in range(1, 6):
        names = [j.name.split('_')[0] for j in jobs if j.name in {f'{a}_r{rnd}' for a in ('S0','F','etx','etxu')}]
        assert names == (['S0','F','etx','etxu'] if rnd % 2 else ['etxu','etx','F','S0'])
    assert '--entrypoint timeout' in graph['moe_tune'].cmd
    assert '-k 5 1200' in graph['moe_tune'].cmd
    assert '--batch-size 1 --tune' in graph['moe_tune'].cmd


@pytest.mark.parametrize('arm', [f'S{i}' for i in range(10)])
def test_arm_commands(arm):
    cfg = r.config(arm)
    cmd = s.arm_command(cfg, 'synthetic')
    assert s.IMAGE in cmd
    assert '--echo-config' in cmd and '--repeats 5' in cmd
    assert '--label ' + arm in cmd
    assert '--entrypoint timeout' in cmd and '-k 5 360' in cmd
    assert subprocess.run(['bash', '-n', '-c', cmd], capture_output=True).returncode == 0


def test_prepare_donor_synthetic(tmp_path):
    root, target = tmp_path / 'package', tmp_path / 'configs'
    root.mkdir()
    donor = root / 'E=128,N=768,device_name=AMD_Instinct_MI308X.json'
    donor.write_text('{"1":{"BLOCK_SIZE_M":16}}')
    s.prepare_moe(root, target)
    assert (target / 'mi308' / 'E=128,N=768,device_name=AMD_Instinct_MI300X.json').read_bytes() == donor.read_bytes()
    assert (target / 'tuned').is_dir()


@pytest.mark.parametrize('count', [0, 2])
def test_donor_ambiguity_refuses(tmp_path, count):
    for n in range(count):
        (tmp_path / f'E=128,N=768,device_name=AMD_Instinct_MI308{n}.json').write_text('{}')
    with pytest.raises(ValueError, match='exactly one'):
        s.prepare_moe(tmp_path, tmp_path / 'out')


@pytest.mark.parametrize('mode,arm,selected', [('combination','C','S4'), ('combination','S0_open','S0'), ('final','F','S5')])
def test_selected_arm_uses_actual_logdir(tmp_path, monkeypatch, mode, arm, selected):
    cfg = r.config('S4' if mode == 'combination' else 'S5')
    path = tmp_path / ('screen.json' if mode == 'combination' else 'final.json')
    path.write_text(json.dumps({mode: cfg}))
    calls = []
    def run(cmd, **kw):
        calls.append((cmd, kw))
        return types.SimpleNamespace(returncode=7)
    monkeypatch.setattr(s.subprocess, 'run', run)
    assert s.main(['--run-selected', mode, '--arm', arm, '--name', 'fixture', '--directory', str(tmp_path)]) == 7
    cmd, kw = calls[0]
    assert '--label ' + selected in cmd[-1]
    assert kw['env']['ETX_SESS_LOGDIR'] == str(tmp_path)


def test_no_combination_skips_without_process(tmp_path, monkeypatch, capsys):
    (tmp_path / 'screen.json').write_text('{"combination":null}')
    def fail(*a, **kw):
        pytest.fail('no subprocess permitted')
    monkeypatch.setattr(s.subprocess, 'run', fail)
    assert s.main(['--run-selected','combination','--arm','C','--name','fixture','--directory',str(tmp_path)]) == 0
    assert 'NOT-APPLIED C' in capsys.readouterr().out
