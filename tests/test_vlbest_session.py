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


@pytest.mark.parametrize('override,cap', [(None, 8400), ('6000', 6000)])
def test_effective_budget_print(monkeypatch, override, cap):
    monkeypatch.delenv('ETX_SESS_MAX_SECONDS', raising=False)
    if override is not None:
        monkeypatch.setenv('ETX_SESS_MAX_SECONDS', override)
    budget = f'BUDGET vm-life=150m runner-cap={cap / 60:g}m'
    if override is not None:
        budget += ' override=ETX_SESS_MAX_SECONDS'
    jobs, limit = load_session(SESSION)
    assert limit == cap
    assert next(j.cmd for j in jobs if j.name == 'budget') == f"printf '%s\\n' '{budget}'"
    proc = subprocess.run([sys.executable, '-m', 'examples.sessions.vlbest', '--dry-run'],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[:2] == [
        budget, f'Session hard stop: {cap}s from runner start (includes setup)']


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
    assert set(graph['report'].after_any) == {j.name for j in jobs[:next(i for i,j in enumerate(jobs) if j.name == 'report')]}


def test_correctness_and_screening_dependencies():
    jobs, _ = load_session(SESSION)
    graph = {j.name: j for j in jobs}
    assert 'ETX_VL_CHECK=1' in graph['build_chkx'].cmd
    assert 'check_timed' not in graph and 'check_certificate' not in graph
    assert '--verify-build' in graph['build_identity'].cmd
    assert graph['gate_chkx'].after == ['build_identity', 'record', 'budget']
    for j in jobs:
        if j.name.startswith(('a_r', 'b_r', 'S0_r', 'F_r', 'etx_r', 'etxu_r')):
            assert j.gpu == 0
            assert 'build_identity' in j.after and 'gate_unf' in j.after
    for rnd in (1, 2):
        names = [j.name for j in jobs if j.name.startswith(f'a_r{rnd}_')]
        assert names[0] == f'a_r{rnd}_S0_open'
        assert names[-1] == f'a_r{rnd}_S0_close'
        assert set(names[1:-1]) == {f'a_r{rnd}_{arm}' for arm in ('S1', 'S3', 'S4f', 'S5f', 'S6f')}
        assert 'moe_configs' in graph[f'a_r{rnd}_S1'].after
    timed = [j for j in jobs if j.name.startswith(('a_r', 'b_r', 'S0_r', 'F_r', 'etx_r', 'etxu_r'))]
    for i, j in enumerate(timed):
        # Every previous arm remains a completion barrier, including skipped arms.
        assert {p.name for p in timed[:i]} <= set(j.after_any)
    for rnd in range(1, 6):
        names = [j.name.split('_')[0] for j in jobs if j.name in {f'{a}_r{rnd}' for a in ('S0','F','etx','etxu')}]
        assert names == (['S0','F','etx','etxu'] if rnd % 2 else ['etxu','etx','F','S0'])
    assert 'moe_tune' not in graph and 'moe_budget' not in graph
    assert graph['moe_configs'].after == ['docker_pull', 'dl']
    assert graph['moe_configs'].gate is None
    assert graph['a_r1_S1'].gate is None
    assert next(j.name for j in jobs if j.name.startswith('a_r1_S') and 'open' not in j.name) == 'a_r1_S1'


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
    assert not (target / 'tuned').exists()


@pytest.mark.parametrize('count', [0, 2])
def test_donor_ambiguity_refuses(tmp_path, count):
    for n in range(count):
        (tmp_path / f'E=128,N=768,device_name=AMD_Instinct_MI308{n}.json').write_text('{}')
    with pytest.raises(ValueError, match='exactly one'):
        s.prepare_moe(tmp_path, tmp_path / 'out')


@pytest.mark.parametrize('mode,arm,selected', [('combination','C2','S4'), ('combination','S0_open','S0'), ('final','F','S5')])
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
    assert 'NOT-APPLIED C2' in capsys.readouterr().out


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




def test_archived_certificate_identity():
    result = s.cert.validate(s.CERTIFICATE, timing=False)
    assert result['build_sha256'] == s.CERTIFIED_SHA256
    assert result['checked'] == result['planned'] == 97
    assert result['positions'] == [0, 1040]
    assert not result['eligible']


@pytest.mark.parametrize('mutation', [None, 'digest', 'check_disabled', 'actual_bytes', 'prior_digest'])
def test_build_identity_gate(tmp_path, monkeypatch, mutation):
    from examples.vllm_llm.pairs import Refusal
    identity = json.loads((s.CERTIFICATE / 'build_identity.json').read_text())
    if mutation == 'digest':
        identity['files']['executable'] = 'a' * 64
    if mutation == 'check_disabled':
        identity['check_enabled'] = False
        monkeypatch.setattr(s.cert, 'digest', lambda identity: s.CERTIFIED_SHA256)
    def verified(*args):
        if mutation == 'actual_bytes':
            raise Refusal('build/model fingerprint changed')
        return identity
    monkeypatch.setattr(s.cert, 'verified_build', verified)
    copied = []
    def export(build, model, dest):
        copied.append(True)
        for name in ('build_identity.json', 'vl_plan.txt', 'imports.json'):
            (dest / name).write_bytes((s.CERTIFICATE / name).read_bytes())
    monkeypatch.setattr(s.cert, 'export_build', export)
    if mutation == 'prior_digest':
        real = s.cert.validate
        monkeypatch.setattr(s.cert, 'validate', lambda *a, **kw: dict(real(*a, **kw), build_sha256='b'*64))
    if mutation:
        with pytest.raises(Refusal):
            s.verify_build(tmp_path, tmp_path / 'build', tmp_path / 'model')
        assert not copied and not (tmp_path / 'check_p0.txt').exists()
    else:
        assert s.verify_build(tmp_path, tmp_path / 'build', tmp_path / 'model') == 0
        assert s.cert.validate(tmp_path, timing=False)['checked'] == 97
        assert not list(tmp_path.glob('*_r*.log'))  # no historical timing transferred
        with pytest.raises(OSError):
            s.cert.validate(tmp_path)  # fresh correctness and timing still required


def test_failed_identity_stops_all_gpu_jobs(tmp_path):
    from etx.tools.gpu_jobs import Runner
    jobs, _ = load_session(SESSION)
    for j in jobs:
        j.cmd = 'exit 2' if j.name == 'build_identity' else 'true'
    runner = Runner(jobs, [0], 4, tmp_path, poll_s=.001)
    assert runner.run() != 0
    assert all(j.status == 'skipped' for j in runner.jobs if j.gpu is not None and j.name != 'prep')


def test_triplets_are_tail_and_keep_nvchk_commands():
    jobs, _ = load_session(SESSION)
    original = {j['name']: j for j in s.nvchk.JOBS}
    tail = [j for j in jobs if j.name.startswith(('poll_r', 'block_r', 'graph_r'))]
    assert [j.name for j in tail] == [f'{arm}_r{rnd}' for rnd in (1,2,3)
        for arm in (('graph','block','poll') if rnd == 2 else ('poll','block','graph'))]
    for j in tail:
        assert 'report' in j.after_any
        assert j.cmd == original[j.name]['cmd']
    assert jobs[-1].cmd.endswith('examples.sessions.nvchk --report-sync ' + s.LOGS)


@pytest.mark.parametrize('remaining,accepted', [(3179, []), (3180, ['S3']),
    (3660, ['S3','S5f']), (4140, ['S3','S5f','S4f']),
    (4680, ['S3','S5f','S4f','C2']), (4950, ['S3','S5f','S4f','C2','triplets'])])
def test_drop_order(remaining, accepted, monkeypatch, capsys):
    monkeypatch.setenv('ETX_SESS_REMAINING_SECONDS', str(remaining))
    assert s.budget_guard('screen') == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[1] for line in lines if line.endswith(' GO')] == accepted


@pytest.mark.parametrize('stage,threshold,label,arm', [('b',1680,'STAGE-B','C2'), ('triplets',390,'TRIPLETS','triplets')])
def test_optional_budget_guard(tmp_path, stage, threshold, label, arm, monkeypatch, capsys):
    monkeypatch.setenv('ETX_SESS_LOGDIR', str(tmp_path))
    for admission in (False, True):
        (tmp_path / 'screen_budget.log').write_text(f'SCREEN {arm} ' + ('GO' if admission else 'SKIPPED budget'))
        for remaining in (threshold-1, threshold):
            monkeypatch.setenv('ETX_SESS_REMAINING_SECONDS', str(remaining))
            s.budget_guard(stage)
            out = capsys.readouterr().out
            assert (out == f'{label} GO\n') is (admission and remaining == threshold)


@pytest.mark.parametrize('failure', ['Disabling fuse_qk_norm_rope_kvcache',
    'ValueError: use_inductor_graph_partition requires torch >= 2.9.0.dev', 'unrelated error'])
def test_fallback_preserves_full_logs(tmp_path, monkeypatch, capsys, failure):
    from tests.test_vlbest_report import log
    calls = []
    def command(cfg, name, timeout=360):
        calls.append(cfg.copy() | {'kw': dict(cfg['kw'])})
        import shlex
        text = failure if len(calls) == 1 else log('S6f', cfg=cfg)
        return 'printf %s ' + shlex.quote(text)
    monkeypatch.setattr(s, 'arm_command', command)
    assert s.run_arm(r.config('S6f'), 'a_r1_S6f', tmp_path) == 0
    output = capsys.readouterr().out
    assert (tmp_path / 'a_r1_S6f.primary.log').read_text() == failure
    if failure == 'unrelated error':
        assert len(calls) == 1
        assert r.proof(output, r.config('S6f')) if r.CONFIG_PREFIX in output else True
    else:
        assert len(calls) == 2
        assert calls[1]['kw']['compilation_config'] == {'splitting_ops': []}
        assert failure not in output
        assert r.observation(output, r.config('S6f')) == 5
        actual = json.loads((tmp_path / 'a_r1_S6f.config.json').read_text())
        assert actual['kw']['compilation_config'] == {'splitting_ops': []}
        assert s.run_arm(r.config('S6f'), 'a_r2_S6f', tmp_path) == 0
        assert len(calls) == 3 and calls[2]['kw']['compilation_config'] == {'splitting_ops': []}


def test_new_guard_source_mutations(tmp_path, monkeypatch, capsys):
    """Each guard's test must fail when that guard is removed."""
    import inspect
    cases = [
        (s.verify_build, "identity.get('check_enabled') is not True or cert.digest(identity) != CERTIFIED_SHA256", 'False',
         lambda: test_build_identity_gate(tmp_path, monkeypatch, 'digest')),
        (s.verify_build, "prior['build_sha256'] != CERTIFIED_SHA256", 'False',
         lambda: test_build_identity_gate(tmp_path, monkeypatch, 'prior_digest')),
        (s.budget_guard, 'remaining >= reserve', 'True',
         lambda: test_drop_order(3179, [], monkeypatch, capsys)),
        (s.budget_guard, 'not admitted or remaining < cost + reserve', 'False',
         lambda: test_optional_budget_guard(tmp_path, 'b', 1680, 'STAGE-B', 'C2', monkeypatch, capsys)),
        (s.partition_failed, "'Disabling fuse_qk_norm_rope_kvcache' in text", 'False',
         lambda: test_fallback_preserves_full_logs(tmp_path, monkeypatch, capsys, 'Disabling fuse_qk_norm_rope_kvcache')),
        (s.partition_failed, "'ValueError' in text", 'False',
         lambda: test_fallback_preserves_full_logs(tmp_path, monkeypatch, capsys, 'ValueError: use_inductor_graph_partition requires torch >= 2.9.0.dev')),
    ]
    for func, old, new, oracle in cases:
        source = inspect.getsource(func)
        assert old in source
        namespace = dict(s.__dict__)
        exec(source.replace(old, new), namespace)
        # Isolate mocks and artifacts between mutation oracles.
        for path in tmp_path.iterdir():
            if path.is_file():
                path.unlink()
        capsys.readouterr()
        with monkeypatch.context() as patch:
            patch.setattr(s, func.__name__, namespace[func.__name__])
            # Oracle mocks must be undone before the next mutant too.
            with pytest.raises((AssertionError, pytest.fail.Exception)):
                if func.__name__ == 'verify_build':
                    test_build_identity_gate(tmp_path, patch, 'digest' if "identity.get" in old else 'prior_digest')
                elif func.__name__ == 'partition_failed':
                    test_fallback_preserves_full_logs(tmp_path, patch, capsys,
                        'Disabling fuse_qk_norm_rope_kvcache' if 'Disabling' in old else
                        'ValueError: use_inductor_graph_partition requires torch >= 2.9.0.dev')
                else:
                    oracle()


def test_fallback_deadline(tmp_path, monkeypatch):
    import time
    ticks = iter([0, 400, 400, 400])
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    class Process:
        killed = False
        calls = 0
        def wait(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired('fixture', 30)
            return 0
        def kill(self):
            self.killed = True
    proc = Process()
    monkeypatch.setattr(s.subprocess, 'Popen', lambda *a, **kw: proc)
    assert s.run_arm(r.config('S4f'), 'deadline', tmp_path) == 124
    assert proc.killed
    assert not (tmp_path / 'deadline.config.json').exists()


def test_fallback_deadline_mutation(tmp_path, monkeypatch):
    import inspect
    source = inspect.getsource(s.run_arm)
    namespace = dict(s.__dict__)
    exec(source.replace('time.monotonic() >= deadline', 'False'), namespace)
    monkeypatch.setattr(s, 'run_arm', namespace['run_arm'])
    with pytest.raises(AssertionError):
        test_fallback_deadline(tmp_path, monkeypatch)
