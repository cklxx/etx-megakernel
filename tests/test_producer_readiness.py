"""Readiness regressions. All model and compute fixtures here are synthetic."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from examples.vllm_llm import producer_session as ps

ROOT = Path(__file__).resolve().parents[1]


def inputs(tmp_path):
    model = tmp_path/'model'
    hf = model/'hf'
    hf.mkdir(parents=True)
    for name in ('vl.bin', 'vl.manifest'):
        (model/name).write_bytes(b'synthetic packed input')
    (model/'golden_c1024.txt').write_text('prompt 1 2\nexpected 3\n')
    for name in ('config.json', 'REVISION', 'model.safetensors'):
        (hf/name).write_text('synthetic checkpoint')
    return dict(model=str(model), hf=str(hf))


def test_checkpoint_mismatch_rejected_before_compute(tmp_path, monkeypatch):
    c = inputs(tmp_path)
    c['hf'] = str(tmp_path/'different-hf')
    Path(c['hf']).mkdir()
    for p in (Path(c['model'])/'hf').iterdir():
        (Path(c['hf'])/p.name).write_bytes(p.read_bytes())
    monkeypatch.setattr(ps.g6, 'config', lambda: c)
    calls = []
    monkeypatch.setattr(ps, 'run', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='HF path'):
        ps.input_identity(c)
    assert calls == []


def test_trace_taken_before_producer_done_is_legal():
    from examples.vllm_llm.producer_trace import summarize
    types, rows = {}, []
    for layer in range(48):
        for name, taken, ready, done in [(f'o_proj_{layer}', 0, 10, 20),
                                         (f'moe_up_{layer}', 15, 29, 35)]:
            ty = len(types)
            types[name] = dict(id=ty, grid=[1])
            rows.append([ty, 0, 0, 0, 0, taken, ready, done, 0])
    out = summarize({'types': types}, np.array(rows, dtype='<i8').tobytes(), True)
    layer = out['layers'][0]
    assert layer['o_start_to_all_expert_tasks_ready_us'] == .19
    assert layer['o_start_to_routing_body_done_us'] == .1
    assert layer['routing_body_done_to_all_expert_ready_us'] == .09
    assert 'publication' in out['substage_costs']
    assert not any('published' in k for k in layer)


def test_stage_attach_external_venv(tmp_path):
    # Real interpreter/dependency probe; no preceding checkout .venv is used.
    from examples.vllm_llm.producer_stage import attach
    archive = tmp_path/'fresh'
    archive.mkdir()
    attach(archive, Path(sys.prefix))
    probe = subprocess.run([str(archive/'.venv/bin/python'), '-I', '-c',
                            'import numpy,yaml; print("DEPENDENCIES OK")'],
                           capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stderr
    assert 'DEPENDENCIES OK' in probe.stdout


def test_binding_rejects_reference_path_and_digest_changes(tmp_path):
    c = inputs(tmp_path)
    hashes = ps.input_identity(c)
    binding = ps.input_binding(c, hashes)
    experiment = dict(input_binding=binding, input_sha256=hashes)
    ref = dict(input_binding=dict(binding), launch=dict(model=binding['hf']))
    ps.validate_reference(ref, experiment)
    for key in ('hf', 'golden', 'prompt', 'input_sha256_digest', 'golden_sha256', 'prompt_sha256'):
        bad = dict(ref, input_binding=dict(binding, **{key: 'different'}))
        with pytest.raises(ValueError, match='identity'):
            ps.validate_reference(bad, experiment)
    with pytest.raises(ValueError, match='identity'):
        ps.validate_reference(dict(ref, launch=dict(model='/other/hf')), experiment)
    alias = tmp_path/'alias'
    alias.symlink_to(c['hf'], target_is_directory=True)
    assert ps.input_binding(dict(c, hf=str(alias)), hashes) == binding


def test_actual_vllm_call_uses_bound_checkpoint_prompt_and_golden(tmp_path, monkeypatch, capsys):
    """Run the real estimator with labelled CPU torch/vLLM substitutes."""
    import types
    from examples.sessions import qwen300 as q
    c = inputs(tmp_path)
    golden = Path(c['model'])/'golden_c1024.txt'
    golden.write_text('prompt '+' '.join(['42']*1024)+'\nexpected '+' '.join(['7']*32)+'\n')
    binding = ps.input_binding(c, ps.input_identity(c))
    seen = []
    class LLM:
        def __init__(self, **kw):
            seen.append(kw)
        def generate(self, prompts, params, **kwargs):
            assert prompts == [dict(prompt_token_ids=[42]*1024)]
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[7]*params.max_tokens)])]
    torch = types.SimpleNamespace(version=types.SimpleNamespace(hip='CPU-SUBSTITUTE'),
        cuda=types.SimpleNamespace(get_device_name=lambda i: 'CPU-SUBSTITUTE', device_count=lambda: 1,
                                  get_device_properties=lambda i: types.SimpleNamespace(gcnArchName='gfx942')))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'vllm', types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: types.SimpleNamespace(**kw), __version__='CPU-SUBSTITUTE'))
    monkeypatch.setitem(sys.modules, 'vllm.inputs', types.SimpleNamespace(TokensPrompt=lambda **kw: kw))
    monkeypatch.setattr(q, 'root', lambda: tmp_path)
    monkeypatch.setattr(q, 'GOLDEN', tmp_path/'DO-NOT-READ')
    monkeypatch.setattr(q, 'golden', lambda p: tmp_path/'DO-NOT-READ-EITHER')
    # A deterministic clock only supplies a positive estimator to this CPU fixture.
    times = iter([x for n in range(12) for x in (n*10, n*10+(2 if n%2 == 0 else 1))])
    monkeypatch.setattr(q.time, 'perf_counter', lambda: next(times))
    q.vllm_run('bf16', 'bench', input_binding=binding)
    ref = json.loads((tmp_path/'bf16_bar.json').read_text())
    assert seen[0]['model'] == binding['hf']
    ps.validate_reference(ref, dict(input_binding=binding,input_sha256=ps.input_identity(c)))
    assert 'INPUT-BINDING ' in capsys.readouterr().out


def test_preflight_rejects_mismatch_before_any_probe(tmp_path, monkeypatch):
    c = inputs(tmp_path)
    c['hf'] = str(tmp_path)
    monkeypatch.setattr(ps.g6, 'config', lambda: c)
    def forbidden(*a, **kw):
        pytest.fail('a mismatch reached compute/preflight probes')
    monkeypatch.setattr(ps.g6, 'preflight', forbidden)
    monkeypatch.setattr(ps, 'run', forbidden)
    with pytest.raises(ValueError, match='HF path'):
        ps.preflight()


def test_stage_invalid_venv_leaves_archive_unmodified(tmp_path):
    from examples.vllm_llm.producer_stage import attach
    archive = tmp_path/'fresh'
    archive.mkdir()
    bad = tmp_path/'bad'
    bad.mkdir()
    with pytest.raises(OSError):
        attach(archive, bad)
    assert not (archive/'.venv').exists()


def test_start_staging_failure_and_total_budget(monkeypatch):
    from tools import producer_start as start
    from etx.session import proc
    calls = []
    now = [1000.]
    monkeypatch.setattr(start.time, 'time', lambda: now[0])
    def run(argv, **kw):
        calls.append((argv, kw))
        now[0] += 20
        return 0
    monkeypatch.setattr(proc, 'run', run)
    start.launch('test-only', '/env', '/config', 2000.)
    assert len(calls) == 3
    assert calls[0][1]['deadline_s'] == 300
    assert calls[1][1]['deadline_s'] == 280
    assert int(calls[2][1]['env']['ETX_SESS_MAX_SECONDS']) == 945
    calls.clear()
    monkeypatch.setattr(proc, 'run', lambda argv, **kw: calls.append(argv) or 124)
    with pytest.raises(RuntimeError, match='exit=124'):
        start.launch('test-only', '/env', '/config', 2000.)
    assert len(calls) == 1  # Timeout cannot reach attach or start.


def test_fresh_archive_interpreter_preflight_and_failure_report(tmp_path):
    """Real git archive, real Python/deps, actual job graph and report. No GPU.

    Only the hardware/source preflight and compute are CPU substitutes. The
    missing interpreter failure is exercised before attachment, not mocked.
    """
    import os
    import shutil
    import tarfile
    import venv
    from examples.vllm_llm.producer_stage import attach
    archive = tmp_path/'archive'
    archive.mkdir()
    bundle = tmp_path/'source.tar'
    with bundle.open('wb') as out:
        subprocess.run(['git', 'archive', 'HEAD'], cwd=ROOT, stdout=out, check=True, timeout=30)
    with tarfile.open(bundle) as tar:
        tar.extractall(archive, filter='data')
    bundle.unlink()
    assert not (archive/'.venv').exists()
    # This is the real pre-fix command failure in a new archive.
    missing = subprocess.run(['bash', '-c', '.venv/bin/python -m examples.vllm_llm.producer_session preflight'],
                             cwd=archive, capture_output=True, text=True, timeout=10)
    assert missing.returncode == 127
    print('ARCHIVE before attach exit=127:', missing.stderr.strip())
    # A new environment independent of both the archive and the previous checkout.
    envdir = tmp_path/'independent-env'
    venv.EnvBuilder(with_pip=False).create(envdir)
    py = envdir/'bin/python'
    purelib = Path(subprocess.check_output([str(py), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    import yaml
    for module in (np, yaml):
        source = Path(module.__file__).parent
        shutil.copytree(source, purelib/source.name)
    attach(archive, envdir)
    # Actual VM interpreter probe at the resolved archive root.
    code = 'from examples.vllm_llm.vm_preflight import check; raise SystemExit(check("/unused", "/unused", only=("interpreter",)))'
    probe = subprocess.run([str(archive/'.venv/bin/python'), '-c', code], cwd=archive,
                           capture_output=True, text=True, timeout=30)
    print(probe.stdout)
    assert probe.returncode == 0, probe.stderr
    # Actual producer preflight and report must resolve even without GPU inputs.
    c = inputs(tmp_path)
    config = tmp_path/'config.json'
    config.write_text(json.dumps(c))
    records = tmp_path/'records'
    fixture = tmp_path/'cpu_session.py'
    fixture.write_text(f'''from examples.sessions.qwen_producer import JOBS
MAX_SESSION_SECONDS=30
SESSION_LEASE={str(tmp_path/'lease')!r}
for job in JOBS:
    job['cwd']={str(archive)!r}
    job['timeout']=10
    if job['name'] not in ('preflight','report'):
        job['cmd']='echo CPU-COMPUTE-SUBSTITUTE-MUST-NOT-RUN; exit 99'
''')
    env = dict(os.environ, ETX_QWEN_G6_CONFIG=str(config), ETX_SESS_MAX_SECONDS='30')
    run = subprocess.run([str(py), '-u', '-m', 'etx.tools.gpu_jobs', str(fixture), '--gpus', '0',
                          '--logdir', str(records), '--progress-s', '1'], cwd=archive, env=env,
                         capture_output=True, text=True, timeout=40)
    print('ARCHIVE runner exit=', run.returncode, '\n', run.stdout, run.stderr)
    assert run.returncode != 0
    record, = records.iterdir()
    statuses = {j['name']: j['status'] for j in json.loads((record/'status.json').read_text())}
    assert statuses['preflight'] == 'fail'
    assert all(statuses[n] == 'skipped' for n in ('build_baseline','build_producer','qualify','measure','diagnose'))
    preflight = (record/'preflight.log').read_text()
    print(preflight)
    assert 'FAIL phase=preflight' in preflight
    report = json.loads((record/'report.json').read_text())
    print('ARCHIVE actual report:', json.dumps(report))
    assert report['status'] == 'NOT-COMPLETE' and report['performance_claim'] is False
