import json
from pathlib import Path

import pytest

from examples.vllm_llm import batch_correctness as bc
from etx.experiments.jobs import load_session
from test_vllm_moe import _graph
from etx.passes import compile_graph, verify_plan


@pytest.mark.parametrize('batch', bc.BATCHES)
def test_replicated_plan_has_every_independent_request(tmp_path, monkeypatch, batch):
    monkeypatch.setenv('ETX_VL_REQUESTS', str(batch))
    monkeypatch.setenv('ETX_VL_CHAIN', '0')
    vm, graph = _graph(tmp_path, monkeypatch, xcd=False, layers=2)
    assert vm.REQUESTS == batch
    plan = compile_graph(graph, 'gfx942', {}, {})
    assert not verify_plan(plan)
    assert len(graph.grids) == len({gr.name for gr in graph.grids})
    for r in range(batch):
        prefix = f'r{r}_'
        grids = [gr for gr in graph.grids if gr.name.startswith(prefix)]
        assert grids[0].name == prefix + 'prologue'
        assert grids[-1].name == prefix + 'argmax'
        assert all(e.startswith(prefix) for gr in grids for e in list(gr.in_edges) + list(gr.out_edges))
    rows = [ln.split() for ln in (tmp_path / 'vl_plan.txt').read_text().splitlines()[2:]]
    assert {int(row[-2]) for row in rows} == set(range(batch))
    assert {row[-1] for row in rows} == {'1'}
    assert len(graph.grids) == len(rows) + 2 * batch


def output(batch, coverage=False):
    rows = []
    for run in range(2):
        for r in range(batch):
            rows.append(f'run {run} request {r}: 32/32 generated tokens match HF greedy' + (' (teacher-forced argmax)' if coverage else ''))
            if coverage and run == 0:
                rows.append(f'CHECK-COVERAGE OK request {r}: 1055 position(s), 0 wrong beyond ties, 0 tie-tolerated (HF margin <= 0.5)')
    rows.append(f'RESULT model=/model requests={batch} matched=32 gen=32 median_ms=0 wall_ms=0 tokens_per_s=0 matched_each=' + ','.join(['32'] * batch))
    return '\n'.join(rows)


@pytest.mark.parametrize('batch', bc.BATCHES)
@pytest.mark.parametrize('coverage', [True, False])
def test_all_request_gates(batch, coverage):
    bc.check_tokens(output(batch, coverage), batch, coverage=coverage)


@pytest.mark.parametrize('mutation', [
    lambda s: s.replace('run 0 request 1:', 'run 0 request 0:'),
    lambda s: s.replace('run 0 request 0: 32/32', 'run 0 request 0: 31/32'),
    lambda s: s.replace('run 0 request 0:', 'missing:'),
    lambda s: s + '\nCHECK-TOKENS FAILED',
    lambda s: s.replace('matched_each=32,32', 'matched_each=32,31'),
    lambda s: s.replace('HF greedy', 'HF greedy (teacher-forced argmax)'),
])
def test_reject_partial_or_wrong_results(mutation):
    with pytest.raises(ValueError):
        bc.check_tokens(mutation(output(2)), 2)


def test_coverage_is_per_request():
    with pytest.raises(ValueError):
        bc.check_tokens(output(4, True).replace('OK request 3:', 'OK request 2:'), 4, coverage=True)


def test_environment_is_fixed(monkeypatch):
    monkeypatch.setenv('ETX_VL_BATCH', '1')
    monkeypatch.setenv('ETX_VL_ABLATE', 'broken')
    monkeypatch.setenv('ETX_HIPCC_FLAGS', '-DBROKEN')
    monkeypatch.setenv('ETX_DEVLIBS', '/devlibs')
    e = bc.clean_env(8)
    assert e['ETX_VL_BATCH'] == '0' and e['ETX_VL_REQUESTS'] == '8'
    assert e['ETX_DEVLIBS'] == '/devlibs'
    assert 'ETX_VL_ABLATE' not in e and 'ETX_HIPCC_FLAGS' not in e


def test_session_is_gpu_free_to_load_and_orders_qualifications():
    jobs, _ = load_session(bc.ROOT / 'examples/sessions/qwen_batch_correctness.py')
    assert len(jobs) == 9
    # a staged ~/etx has no .venv: every build waits for the setup job that creates it
    assert [j.name for j in jobs[:2]] == ['apt', 'etx_env'] and jobs[1].reads == ['apt']
    assert all(j.reads == ['etx_env'] for j in jobs if j.name.startswith('build_b'))
    assert [j.gpu for j in jobs if j.lane == 'gpu'] == [0, 0, 0]
    assert 'qualify_b4/correctness.json' in next(j for j in jobs if j.name == 'qualify_b8').reads
    verify = next(j for j in jobs if j.name == 'verify')
    assert verify.lane == 'final' and 'tools/two_gpu_verify.py' in verify.cmd
    # "?" makes it run after a failed qualification too; it writes the verdict file.
    assert verify.reads == [f'?qualify_b{b}/correctness.json' for b in (2, 4, 8)]
    assert verify.writes == ['qwen-verification.json']


def test_failure_writes_incomplete_receipt(tmp_path):
    out = tmp_path / 'evidence'
    with pytest.raises(FileNotFoundError):
        bc.qualify(tmp_path / 'model', tmp_path / 'build', out, 2)
    assert json.loads((out / 'correctness.json').read_text())['status'] == 'NOT-COMPLETE'


def test_subprocess_execution_and_deadline(tmp_path):
    import sys
    text = bc.execute([sys.executable, '-c', 'print("hello")'], tmp_path / 'ok.log', bc.clean_env(2), 5)
    assert 'hello' in text
    with pytest.raises(ValueError, match='exit 124'):
        bc.execute([sys.executable, '-c', 'import time; time.sleep(5)'], tmp_path / 'timeout.log', bc.clean_env(2), 0.1)


def test_silent_active_compile_is_probed_and_keeps_its_hard_deadline(tmp_path, monkeypatch):
    now = bc.time.time()
    monkeypatch.setattr(bc.time, 'time', lambda: now + 1000)
    def supervised(argv, **kw):
        assert kw['deadline_s'] == 1200
        kw['on_poll'](181)
        kw['sink'].write('compile completed\n')
        return 0
    monkeypatch.setattr(bc.proc, 'run', supervised)
    text = bc.execute(['synthetic-long-compile'], tmp_path / 'compile.log', bc.clean_env(2), 1200)
    assert 'SILENCE-PROBE' in text and 'compile completed' in text


def test_native_script_supervises_both_compiler_stages(tmp_path):
    import os
    import subprocess
    import sys
    toolchain = tmp_path / 'synthetic-compiler'
    toolchain.mkdir()
    # Only transport/orchestration is tested here; these are NOT GPU compilers.
    for name in ('clang++', 'llc'):
        p = toolchain / name
        p.write_text(f'#!{sys.executable}\nimport pathlib, sys\n'
                     'pathlib.Path(sys.argv[sys.argv.index("-o")+1]).write_text("synthetic compiler fixture\\n")\n')
        p.chmod(0o755)
    out = tmp_path / 'batch'
    d = out / 'b2'
    d.mkdir(parents=True)
    (d / 'megakernel_d0.hip').write_text('#include "/code/fixture.h"\n')
    (d / 'imports.bc').write_bytes(b'synthetic')
    run = subprocess.run(['bash', 'examples/vllm_llm/batch_native.sh', str(out), '2'], cwd=bc.ROOT,
                         env=dict(os.environ, ETX_LLVM=str(toolchain), ETX_QWEN_NATIVE_SECONDS='10'),
                         capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
    receipt = json.loads((d / 'native-check.json').read_text())
    assert receipt['status'] == 'COMPILED' and receipt['gpu_correctness'] == 'unmeasured'
    assert all((d / name).exists() for name in ('native-ir.log', 'native-isa.log'))


from etx.testing import qwen_staged_model as staged_model  # noqa: E402  (shared with E2E preflight tests)


def test_asset_fingerprint_includes_weights_and_every_reference(tmp_path):
    model = staged_model(tmp_path)
    before = bc.inputs(model, 8)
    assert len(before['files']) == 20
    (model / 'vl.bin').write_bytes(b'changed')
    assert bc.inputs(model, 8) != before
    name = bc.golden_names(8)[-1]
    (model / name).write_text((model / bc.golden_names(8)[0]).read_text())
    with pytest.raises(ValueError, match='distinct prompt'):
        bc.inputs(model, 8)


def instance_text(rows, position=0):
    return '\n'.join('CHECK-INSTANCE ' + json.dumps(row) for row in rows)


def instances(requirements, position=0, **override):
    rows = [dict(id=i, export='attn_reduce', layer=0, replicas=1, req=req,
                 grid=[1, 1, 1], block=[128, 1, 1], position=position, bytes=4,
                 result='identical', first_byte=-1)
            for i, req in enumerate(requirements)]
    for row, changes in override.items():
        rows[int(row)].update(changes)
    return rows


def planned(requirements):
    return [dict(id=i, export='attn_reduce', layer=0, replicas=1, req=req,
                 grid=[1, 1, 1], block=[128, 1, 1])
            for i, req in enumerate(requirements)]


@pytest.mark.parametrize('batch', bc.BATCHES)
def test_byte_identity_gate_is_per_request(batch):
    requirements = [r for r in range(batch) for _ in range(3)]
    result = bc.check_identity(instance_text(instances(requirements, 1040), 1040), batch, planned(requirements), 1040)
    assert result == dict(position=1040, requests=batch, instances_per_request=3)


@pytest.mark.parametrize('mutate, match', [
    (lambda rows: rows[:2], 'request 0: incomplete'),
    (lambda rows: rows + [dict(rows[1], id=3)], 'request 1: incomplete'),
    (lambda rows: [dict(rows[0], req=None)] + rows[1:], 'valid request id'),
    (lambda rows: [dict(rows[0], result='differs')] + rows[1:], 'request 0: imported bytes differ'),
    (lambda rows: [dict(rows[0], position=0)] + rows[1:], 'request 0: imported bytes differ'),
])
def test_byte_identity_gate_refuses_partial_or_wrong_request(mutate, match):
    requirements = [0, 1, 0, 1, 0, 1]
    rows = instances(requirements, position=1040)
    with pytest.raises(ValueError, match=match):
        bc.check_identity(instance_text(mutate(rows), 1040), 2, planned(requirements), 1040)


def test_byte_identity_gate_requires_a_plan_entry_for_every_request():
    with pytest.raises(ValueError, match='request 1: plan has no imported byte comparison'):
        bc.check_identity(instance_text(instances([0, 0])), 2, planned([0, 0]), 0)
    with pytest.raises(ValueError, match='unequal planned'):
        bc.check_identity(instance_text(instances([0, 1, 0])), 2, planned([0, 1, 0]), 0)


def test_qualify_runs_all_five_processes_and_records_identity(tmp_path, monkeypatch):
    import sys
    model = staged_model(tmp_path, 2)
    build = tmp_path / 'build'
    build.mkdir()
    # A real subprocess fixture exercises log capture and the certificate parser.
    # This is deliberately labelled synthetic; it proves no GPU property.
    (build / 'vl_plan.txt').write_text('# synthetic\n'
                                       '0 attn_reduce 0 reduce 1 1 1 0 1 0 0 0 1 0 -1 0 1\n'
                                       '1 attn_reduce 0 reduce 1 1 1 0 1 0 0 0 1 0 -1 1 1\n')
    (build / 'imp').mkdir()
    (build / 'imp/imports.json').write_text(json.dumps({'attn_reduce': {'block': [128, 1, 1]}}))
    script = build / 'fixture.py'
    script.write_text('import json, sys\n'
                      'if "--check" in sys.argv:\n'
                      ' p = int(sys.argv[sys.argv.index("--check-pos")+1])\n'
                      ' for i in range(2):\n'
                      '  print("CHECK-INSTANCE " + json.dumps(dict(id=i, export="attn_reduce", layer=0, replicas=1, req=i, grid=[1,1,1], block=[128,1,1], position=p, bytes=4, result="identical", first_byte=-1)))\n'
                      ' print(f"CHECK-IMPORTED checked=2 planned=2 replicas=2 position={p}")\n'
                      ' print("CHECK OK: 0 launch(es) differ")\n'
                      'else:\n'
                      ' print(' + repr(output(2, True)) + ' if "--check-coverage" in sys.argv else ' + repr(output(2)) + ')\n')
    identity = dict(check_enabled=True, overrides=bc.recipe(2))
    monkeypatch.setattr(bc.cert, 'verified_build', lambda *_: identity)
    monkeypatch.setattr(bc, 'host_args', lambda *_: [sys.executable, str(script)])
    out = tmp_path / 'qualification'
    bc.qualify(model, build, out, 2)
    receipt = json.loads((out / 'correctness.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['performance'] == 'unmeasured'
    assert receipt['imported_per_position'] == 2 and receipt['greedy_tokens_per_mode'] == 128
    assert receipt['byte_identity_per_request'] == {
        '0': dict(position=0, requests=2, instances_per_request=1),
        '1040': dict(position=1040, requests=2, instances_per_request=1)}
    assert len(list(out.glob('*.log'))) == 5


def test_execution_package_has_full_asset_closure_and_sources():
    package = bc.execution_package()
    assert package['batches'] == [2, 4, 8]
    assert len(package['required_assets']) == 20
    assert len(set(package['required_assets'])) == 20
    assert 'prefill_c1024_r679.txt' in package['required_assets']
    assert all(len(v) == 64 for v in package['sources'].values())
    assert package['asset_status'] == 'UNBOUND'
    assert package['gpu_correctness'] == 'unmeasured'


def test_producer_asset_identity_passes_the_session_verifier_for_every_batch(tmp_path):
    from tests.test_two_gpu_verify import qwen_receipt
    from tools import two_gpu_verify as vf
    model = staged_model(tmp_path)
    record = tmp_path / 'record'
    for batch in bc.BATCHES:
        receipt = dict(qwen_receipt(batch), inputs=bc.inputs(model, batch))
        assert len(receipt['inputs']['files']) == 4 + 2 * batch
        (record / f'qualify_b{batch}').mkdir(parents=True)
        (record / f'qualify_b{batch}/correctness.json').write_text(json.dumps(receipt))
    assert set(vf.qwen(record)) == {'2', '4', '8'}
    # B2 bound to the B8 reference set is not what B2 read: refused.
    receipt = dict(qwen_receipt(2), inputs=dict(bc.inputs(model, 2), files=bc.inputs(model, 8)['files']))
    (record / 'qualify_b2/correctness.json').write_text(json.dumps(receipt))
    with pytest.raises(vf.Fail, match='incomplete staged-asset identity'):
        vf.qwen(record)
