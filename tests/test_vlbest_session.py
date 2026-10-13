"""Portable session acceptance: only synthetic paths and stub processes."""
import json
import os
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
    rows = [line for line in out.splitlines() if '  reads: ' in line]
    assert len(rows) == len(jobs)
    assert all(j.name in rows[i] for i, j in enumerate(jobs))
    seen = set()
    for j in jobs:
        assert set(j.deps() + j.deps(True)) <= seen, j.name
        seen.add(j.name)
        assert subprocess.run(['bash', '-n', '-c', j.cmd], capture_output=True).returncode == 0, j.name
    assert set(graph['report'].deps(True)) == {j.name for j in jobs[:next(i for i,j in enumerate(jobs) if j.name == 'report')]}


def test_correctness_and_screening_dependencies():
    jobs, _ = load_session(SESSION)
    graph = {j.name: j for j in jobs}
    assert 'ETX_VL_CHECK=1' in graph['build_chkx'].cmd
    assert 'check_timed' not in graph and 'check_certificate' not in graph
    assert '--verify-build' in graph['build_identity'].cmd
    assert graph['gate_chkx'].deps() == ['build_identity', 'record', 'budget']
    for j in jobs:
        if j.name.startswith(('a_r', 'b_r', 'S0_r', 'F_r', 'etx_r', 'etxu_r')):
            assert j.gpu == 0
            assert 'build_identity' in j.deps() and 'gate_unf' in j.deps()
    for rnd in (1, 2):
        names = [j.name for j in jobs if j.name.startswith(f'a_r{rnd}_')]
        assert names[0] == f'a_r{rnd}_S0_open'
        assert names[-1] == f'a_r{rnd}_S0_close'
        assert set(names[1:-1]) == {f'a_r{rnd}_{arm}' for arm in ('S1', 'S3', 'S4f', 'S5f', 'S6f')}
        assert 'moe_configs' in graph[f'a_r{rnd}_S1'].deps()
    timed = [j for j in jobs if j.name.startswith(('a_r', 'b_r', 'S0_r', 'F_r', 'etx_r', 'etxu_r'))]
    # Every previous arm remains a completion barrier, including skipped arms: one GPU lane, strictly in list order.
    assert all(j.lane == 'gpu' and j.gpu == 0 for j in timed)
    for rnd in range(1, 6):
        names = [j.name.split('_')[0] for j in jobs if j.name in {f'{a}_r{rnd}' for a in ('S0','F','etx','etxu')}]
        assert names == (['S0','F','etx','etxu'] if rnd % 2 else ['etxu','etx','F','S0'])
    assert 'moe_tune' not in graph and 'moe_budget' not in graph
    assert graph['moe_configs'].deps() == ['docker_pull', 'dl']
    assert not [r for r in graph['moe_configs'].reads + graph['a_r1_S1'].reads if r.startswith('go_')]
    assert 'go_screen_S3.ok' in graph['a_r1_S3'].reads
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
        assert 'report' in j.deps(True) and 'go_triplets.ok' in j.reads
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
    from tests.test_vlbest_report import log, PLAN
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
        assert r.observation(output, r.config('S6f'), expected=r.expected_replacements(PLAN)) == 5
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


def test_g0_local_source_baseline(monkeypatch, tmp_path):
    from examples.vllm_llm import vlbest3_g0 as g0
    source_root = os.environ.get('ETX_VLLM_SOURCE_ROOT')
    if not source_root:
        pytest.skip('set ETX_VLLM_SOURCE_ROOT to inject a vLLM source checkout')
    root = Path(source_root)
    monkeypatch.setenv('HOME', str(tmp_path))
    if not (root / g0.TUNER).exists():
        pytest.skip('local source baseline unavailable')
    sources = {key: (root / path).read_text() for key, path in dict(
        envs='vllm/envs.py', moe='vllm/model_executor/layers/fused_moe/fused_moe.py',
        unquantized='vllm/model_executor/layers/fused_moe/oracle/unquantized.py',
        tuner=g0.TUNER).items()}
    filename = g0.source_function(sources['moe'], 'get_config_file_name',
        {'get_device_name_as_file_name': lambda: 'AMD_Instinct_MI300X'})(128, 768, None, None)
    values = dict(torch='2.12.0', layername=True, tuner_exists=True,
                  folder=str(tmp_path / 'moecfg/tuned'), filename=filename)
    assert g0.validate(values, sources) == []


def test_vlbest3_graph():
    from examples.sessions import vlbest3 as v3
    jobs, limit = load_session(SESSION.with_name('vlbest3.py'))
    graph = {j.name: j for j in jobs}
    assert limit == 8400
    assert r.config('M0') == r.config('S3')
    assert r.config('M1')['env'] == r.config('S9')['env']
    assert r.config('M1')['kw'] == r.config('S3')['kw']
    assert len([j for j in jobs if j.name.startswith('a_r')]) == 8
    assert len([j for j in jobs if j.name.startswith(('F_r', 'S0_r', 'etx_r', 'etxu_r'))]) == 20
    assert graph['moe_tune'].timeout == 2700
    assert graph['g0'].lane == 'cpu'
    assert '--network=none' in graph['g0'].cmd and '--device' not in graph['g0'].cmd
    assert all(j.timeout == 390 for j in jobs if j.name.startswith('a_r'))
    seen = set()
    for job in jobs:
        assert set(job.deps() + job.deps(True)) <= seen
        seen.add(job.name)
        assert subprocess.run(['bash', '-n', '-c', job.cmd], capture_output=True).returncode == 0
    assert 'ps -eo pid,args' in __import__('inspect').getsource(v3.cleanup_tuner)


@pytest.mark.parametrize('failure', ['g0', 'moe_tune', 'moe_gate', 'a_r1_M1', 'F_r1', 'etx_r1'])
def test_vlbest3_red_stop(tmp_path, failure):
    from etx.tools.gpu_jobs import Runner
    jobs, _ = load_session(SESSION.with_name('vlbest3.py'))
    for job in jobs:
        job.cmd = 'exit 2' if job.name == failure else 'true'
    runner = Runner(jobs, [0], 4, tmp_path, poll_s=.001)
    assert runner.run() != 0
    failed_index = next(i for i, j in enumerate(jobs) if j.name == failure)
    assert all(j.status == 'skipped' for j in jobs[failed_index+1:] if j.lane == 'gpu')


@pytest.mark.parametrize('data', [None, {}, {'2': {'BLOCK_SIZE_M': 16}}, {'1': {}}, {'1': {'BLOCK_SIZE_M': 16}}])
def test_vlbest3_table_gate(tmp_path, monkeypatch, capsys, data):
    import hashlib
    from examples.sessions import vlbest3 as v3
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(v3, 'cleanup_tuner', lambda d: None)
    target = v3.target_table()
    target.parent.mkdir(parents=True)
    if data is not None:
        target.write_text(json.dumps(data))
        target.with_suffix('.progress.json').write_text(json.dumps({'configs_done':25}))
        (tmp_path / 'moe_tuned.json').write_text(json.dumps({'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))
    if data and data.get('1'):
        assert v3.tune_gate(tmp_path) == 0
    else:
        with pytest.raises(r.Refusal, match='NOT-COMPLETE'):
            v3.tune_gate(tmp_path)
    output = capsys.readouterr().out
    assert str(target.resolve()) in output and 'key_1=' in output
    expected_count = 25 if data is not None else 'unknown'
    assert f'configs_tuned={expected_count}' in output


def test_vlbest3_tuner_cpu_adapter(tmp_path):
    from examples.vllm_llm import vlbest3_tuner as t
    source = '''
import ray
from ray.experimental.tqdm_ray import tqdm
@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed):
        self.device_id = int(ray.get_gpu_ids()[0])
    def tune(self, *args):
        calls.append(args)
        search_space = args[9][:3]
        best_config = None
        for idx, config in enumerate(tqdm(search_space)):
            if idx == 1:
                continue
            best_config = {'BLOCK_SIZE_M': 16}
        return best_config
def main(args):
    ray.init()
if __name__ == '__main__':
    main(None)
'''
    calls, saves = [], []
    ns = {'calls': calls}
    exec(compile(t.adapted_source(source), '<synthetic>', 'exec'), ns)
    dtype = object()
    ranges = dict(BLOCK_SIZE_M=[16,32,64,128,256], BLOCK_SIZE_N=[16,32,64,128,256],
                  BLOCK_SIZE_K=[16,32,64,128,256], GROUP_SIZE_M=[1,4,8,16,32],
                  num_warps=[1,2,4,8], num_stages=[2], waves_per_eu=[0,1,2,4],
                  matrix_instr_nonkdim=[16,32], kpack=[1,2])
    ns.update(torch=types.SimpleNamespace(bfloat16=dtype), get_config=lambda **kw: None,
              get_model_params=lambda cfg:(128,8,768,2048),
              current_platform=types.SimpleNamespace(is_rocm=lambda:True),
              get_rocm_tuning_space=lambda flag:ranges.copy(),
              get_config_file_name=lambda *a:r.MOE_FILENAME, sort_config=lambda c:c,
              _get_config_dtype_str=lambda *a, **kw:None, save_configs=lambda *a:saves.append(a))
    args = types.SimpleNamespace(model='synthetic', tp_size=1, batch_size=1, save_dir=str(tmp_path))
    assert t.run(args, ns) == 0
    assert calls[0][:6] == (1,128,1536,2048,8,dtype)
    assert len(calls[0][9]) == 8000
    assert all(p['BLOCK_SIZE_M'] == 16 for p in calls[0][9])
    assert json.loads((tmp_path / r.MOE_FILENAME).read_text()) == {'1':{'BLOCK_SIZE_M':16}}
    assert json.loads((tmp_path / r.MOE_FILENAME).with_suffix('.progress.json').read_text())['configs_done'] == 3
    assert ns['resolve_dtype'](None) is dtype
    with pytest.raises(ValueError):
        t.adapted_source(source.replace('import ray', 'import ray\nray.init()'))


def test_g0_records_failure(monkeypatch, capsys):
    from examples.vllm_llm import vlbest3_g0 as g0
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(__version__='2.10.0'))
    monkeypatch.setitem(sys.modules, 'vllm', None)
    assert g0.probe() == 1
    output = capsys.readouterr().out
    for name in ('torch.__version__', 'vllm.utils.torch_utils._USE_LAYERNAME', 'tuner', 'config'):
        assert f'G0 {name}=' in output
    assert 'PREFLIGHT FAIL g0:' in output


@pytest.mark.parametrize('outcome', ['timeout', 'deadline', 'missing', 'success'])
def test_vlbest3_tune_failures(tmp_path, monkeypatch, capsys, outcome):
    from examples.sessions import vlbest3 as v3
    monkeypatch.setenv('HOME', str(tmp_path))
    cleanup = []
    monkeypatch.setattr(v3, 'cleanup_tuner', lambda d: cleanup.append(d))
    target = v3.target_table()
    target.parent.mkdir(parents=True)
    target.write_text('{"1":{"stale":1}}')
    receipt = tmp_path / 'moe_tuned.json'
    receipt.write_text('{}')
    waits = []
    class Process:
        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                assert not target.exists() and not receipt.exists()
                raise subprocess.TimeoutExpired('tune', timeout)
            if outcome == 'success':
                target.write_text('{"1":{"BLOCK_SIZE_M":16}}')
            return 124 if outcome == 'timeout' else 0
        def kill(self):
            cleanup.append('killed-client')
    ticks = iter([0, 2701 if outcome == 'deadline' else 60])
    monkeypatch.setattr(v3.time, 'monotonic', lambda:next(ticks))
    monkeypatch.setattr(v3.subprocess, 'Popen', lambda *a, **kw:Process())
    if outcome == 'success':
        assert v3.tune(tmp_path) == 0 and receipt.exists()
    else:
        with pytest.raises((r.Refusal, FileNotFoundError)):
            v3.tune(tmp_path)
        assert not receipt.exists()
    assert cleanup and waits[0] == 60
    assert 'TUNER-PROGRESS' in capsys.readouterr().out
    if outcome == 'deadline':
        assert 'killed-client' in cleanup


@pytest.mark.parametrize('change', ['version', 'layername', 'file', 'folder', 'filename'])
def test_g0_value_rejections(change, monkeypatch, tmp_path):
    from examples.vllm_llm import vlbest3_g0 as g0
    monkeypatch.setenv('HOME', str(tmp_path))
    values = dict(torch='2.12.0', layername=True, tuner_exists=True,
                  folder=str(tmp_path / 'moecfg/tuned'), filename=r.MOE_FILENAME)
    key, value = dict(version=('torch', '2.10'), layername=('layername', False),
                      file=('tuner_exists', False), folder=('folder', '/wrong'),
                      filename=('filename', 'wrong.json'))[change]
    values[key] = value
    errors = g0.validate(values, dict(envs='', moe='', tuner=''))
    expected = dict(version='torch <', layername='_USE_LAYERNAME', file='missing benchmarks',
                    folder='value mismatch', filename='filename mismatch')[change]
    assert any(expected in error for error in errors)


def test_vlbest3_checkpoint_budget_and_interruption(tmp_path, monkeypatch, capsys):
    from examples.vllm_llm import vlbest3_tuner as t
    now = [0.0]
    monkeypatch.setattr(t.time, 'monotonic', lambda: now[0])
    target = tmp_path / r.MOE_FILENAME
    checkpoint = t.Checkpoints(target, lambda c:c, 0, budget=100)
    items = checkpoint.progress(list(range(704)))
    for i in range(25):
        assert next(items) == i
        now[0] += 2
        checkpoint.checkpoint({'winner': i})
    assert checkpoint.limit == 47  # 50 seconds left, 2 s/config + 10% margin
    assert json.loads(target.read_text()) == {'1': {'winner':24}}
    output = capsys.readouterr().out
    assert 'configs_done=25' in output and 'estimated_total_seconds=1408.000' in output
    assert 'TUNER prune old=704 new=47' in output
    # A kill before atomic replacement preserves the last complete winner.
    def killed(src, dst):
        raise KeyboardInterrupt
    monkeypatch.setattr(Path, 'replace', killed)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.checkpoint({'winner':25})
    assert json.loads(target.read_text()) == {'1': {'winner':24}}
    assert json.loads(target.with_suffix('.progress.json').read_text())['configs_done'] == 25


def test_vlbest3_adapts_real_tuning_loop():
    from examples.vllm_llm import vlbest3_tuner as t
    source = Path('examples/vllm_llm/vendor/benchmark_moe.py').read_text()
    tree = t.adapted_source(source)
    compile(tree, '<pinned tuner>', 'exec')
    import ast
    assert ast.unparse(tree).count('_checkpoint(best_config)') == 1
    with pytest.raises(ValueError, match='tuning loop contract'):
        t.adapted_source(source.replace('enumerate(tqdm(search_space))', 'enumerate(search_space)'))
