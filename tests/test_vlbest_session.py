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
    assert graph['moe_tune'].timeout == 2400
    assert '-k 5 2340' in graph['moe_tune'].cmd
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


@pytest.mark.parametrize('available', [True, False])
def test_ray_probe(available, monkeypatch, capsys):
    import importlib
    def load(name):
        assert name == 'ray'
        if not available:
            raise ModuleNotFoundError('ray')
    monkeypatch.setattr(importlib, 'import_module', load)
    assert s.main(['--probe-ray']) == 0
    assert capsys.readouterr().out == f"MOE-TUNE ray={'yes' if available else 'no'}\n"


@pytest.mark.parametrize('stage,threshold,reserve,label', [
    ('moe', 6240, 3660, 'MOE-TUNE'), ('b', 1680, 1140, 'STAGE-B'),
])
def test_budget_guard(stage, threshold, reserve, label, monkeypatch, capsys):
    for remaining in (0, threshold - 1, threshold):
        monkeypatch.setenv('ETX_SESS_REMAINING_SECONDS', str(remaining))
        assert s.main(['--budget-guard', stage]) == 0
        assert capsys.readouterr().out == (
            f'{label} SKIPPED budget remaining={remaining} reserve={reserve}\n'
            if remaining < threshold else f'{label} GO\n')
    monkeypatch.delenv('ETX_SESS_REMAINING_SECONDS')
    assert s.main(['--budget-guard', stage]) == 0
    assert 'SKIPPED budget remaining=0' in capsys.readouterr().out


def decision_runner(tmp_path, ray=True, budget=True, complete=True, stage_b=True):
    from etx.tools.gpu_jobs import Runner
    jobs, _ = load_session(SESSION)
    runner = Runner(jobs, [0], 1, tmp_path)
    for j in jobs:
        j.status = 'ok'
    for name, line in [
        ('docker_pull', f"MOE-TUNE ray={'yes' if ray else 'no'}"),
        ('moe_budget', 'MOE-TUNE GO' if budget else 'MOE-TUNE SKIPPED budget remaining=6000 reserve=3660'),
        ('moe_tune', 'MOE-TUNE COMPLETE' if complete else ''),
        ('b_budget', 'STAGE-B GO' if stage_b else 'STAGE-B SKIPPED budget remaining=1600 reserve=1140'),
    ]:
        j = runner.by_name[name]
        j.log = tmp_path / f'{name}.log'
        j.log.write_text(line + '\n')
    return runner


def propagate(runner, names):
    states = {}
    for name in names:
        j = runner.by_name[name]
        states[name] = runner._deps_state(j)
        if states[name] == 'skip':
            j.status = 'skipped'
    return states


@pytest.mark.parametrize('ray', [True, False])
def test_ray_omits_tune_and_s9(tmp_path, ray):
    runner = decision_runner(tmp_path, ray=ray)
    assert runner.jobs[0].name == 'docker_pull'
    assert '--probe-ray' in runner.jobs[0].cmd
    names = ['moe_tune', 'a_r1_S9', 'a_r2_S9']
    states = propagate(runner, names)
    assert set(states.values()) == ({'ready'} if ray else {'skip'})
    assert runner._deps_state(runner.by_name['a_r1_S1']) == 'ready'
    excluded = r.screen(tmp_path)['excluded']
    if ray:
        assert 'S9' not in excluded
    else:
        assert excluded['S9'] == 'NOT-APPLIED S9: ray missing in image'


def test_budget_omits_configs_s1_s9(tmp_path):
    runner = decision_runner(tmp_path, budget=False)
    names = ['moe_configs', 'moe_tune', 'a_r1_S1', 'a_r1_S9', 'a_r2_S1', 'a_r2_S9']
    assert set(propagate(runner, names).values()) == {'skip'}
    excluded = r.screen(tmp_path)['excluded']
    for arm in ('S1', 'S9'):
        assert excluded[arm].startswith(f'NOT-APPLIED {arm}: MOE-TUNE SKIPPED budget')
    assert 'S1' not in r.screen(tmp_path)['tested']


def test_incomplete_tune_never_screens_s9(tmp_path):
    runner = decision_runner(tmp_path, complete=False)
    for rnd in (1, 2):
        assert runner._deps_state(runner.by_name[f'a_r{rnd}_S9']) == 'skip'


@pytest.mark.parametrize('stage_b', [True, False])
def test_stage_b_reserves_c(tmp_path, stage_b):
    runner = decision_runner(tmp_path, stage_b=stage_b)
    names = [j.name for j in runner.jobs if j.name.startswith('b_r')]
    assert len(names) == 6
    assert set(propagate(runner, names).values()) == ({'ready'} if stage_b else {'skip'})
    for j in runner.jobs:
        if j.name.startswith(('S0_r', 'F_r', 'etx_r', 'etxu_r')):
            assert runner._deps_state(j) == 'ready'
    if not stage_b:
        assert 'NOT-APPLIED C: STAGE-B SKIPPED budget' in r.screen(tmp_path, 'b')['excluded']['C']


def test_omission_overrides_arm_logs(tmp_path, monkeypatch):
    # Even an otherwise valid arm must never be counted after explicit omission.
    decision_runner(tmp_path, ray=False)
    monkeypatch.setattr(r.Session, 'read', lambda *a, **kw: 'synthetic valid log')
    monkeypatch.setattr(r, 'observation', lambda *a, **kw: 5.0)
    result = r.screen(tmp_path, candidates=[r.config('S9')])
    assert result['tested'] == ['S0']
    assert result['excluded']['S9'] == 'NOT-APPLIED S9: ray missing in image'


def test_failed_tune_never_marks_complete():
    import os
    tune = next(j for j in s.JOBS if j['name'] == 'moe_tune')
    # Exercise the actual shell, replacing only Docker execution.
    cmd = 'sudo() { return 7; }; ' + tune['cmd']
    proc = subprocess.run(['bash', '-c', cmd], env=dict(os.environ, GPU='0'),
                          capture_output=True, text=True)
    assert proc.returncode == 7
    assert 'MOE-TUNE COMPLETE' not in proc.stdout


@pytest.mark.parametrize('ray,budget', [(True, True), (False, True), (True, False)])
def test_runner_omits_processes(tmp_path, ray, budget):
    from etx.tools.gpu_jobs import Runner
    jobs, _ = load_session(SESSION)
    for j in jobs:
        j.cmd = 'true'
        if j.name == 'docker_pull':
            j.cmd = f"echo 'MOE-TUNE ray={'yes' if ray else 'no'}'"
        elif j.name == 'moe_budget':
            j.cmd = "echo 'MOE-TUNE GO'" if budget else "echo 'MOE-TUNE SKIPPED budget remaining=6000 reserve=3660'"
        elif j.name == 'moe_tune':
            j.cmd = "echo 'MOE-TUNE COMPLETE'"
        elif j.name == 'b_budget':
            j.cmd = "echo 'STAGE-B GO'"
    runner = Runner(jobs, [0], 4, tmp_path, poll_s=.001)
    assert runner.run() == 0
    for name in ('moe_tune', 'a_r1_S9', 'a_r2_S9'):
        assert runner.by_name[name].status == ('ok' if ray and budget else 'skipped')
        assert (tmp_path / f'{name}.log').exists() == (ray and budget)
    for name in ('a_r1_S1', 'a_r2_S1'):
        assert runner.by_name[name].status == ('ok' if budget else 'skipped')
    assert runner.by_name['etxu_r5'].status == 'ok'
