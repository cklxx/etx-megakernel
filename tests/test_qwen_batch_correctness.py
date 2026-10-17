import json
from pathlib import Path

import pytest

from examples.vllm_llm import batch_correctness as bc
from etx.tools.gpu_jobs import load_session
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
    assert len(jobs) == 6
    assert [j.gpu for j in jobs if j.lane == 'gpu'] == [0, 0, 0]
    assert 'qualify_b4/correctness.json' in next(j for j in jobs if j.name == 'qualify_b8').reads


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


def staged_model(tmp_path, batch=8):
    model = tmp_path / 'model'
    (model / 'hf').mkdir(parents=True)
    (model / 'hf/config.json').write_text((bc.ROOT / 'examples/llm/configs/qwen3-30b-a3b.json').read_text())
    (model / 'hf/REVISION').write_text('a' * 40)
    (model / 'vl.manifest').write_text('test fixture, not GPU inputs')
    (model / 'vl.bin').write_bytes(b'fixture')
    for r, name in enumerate(bc.golden_names(batch)):
        (model / name).write_text('prompt: ' + ' '.join([str(r)] * 1024) + '\ngen: ' + ' '.join(['1'] * 32) + '\n')
        (model / name.replace('golden', 'prefill', 1)).write_text('argmax: ' + ' '.join(['1'] * 1055) + '\nmargin: ' + ' '.join(['1.0'] * 1055))
    return model


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


def test_qualify_runs_all_five_processes_and_records_identity(tmp_path, monkeypatch):
    import sys
    model = staged_model(tmp_path, 2)
    build = tmp_path / 'build'
    build.mkdir()
    # A real subprocess fixture exercises log capture and the certificate parser.
    # This is deliberately labelled synthetic; it proves no GPU property.
    (build / 'vl_plan.txt').write_text('# synthetic\n0 attn_reduce 0 reduce 1 1 1 0 1 0 0 0 1 0 -1 0 1\n')
    (build / 'imp').mkdir()
    (build / 'imp/imports.json').write_text(json.dumps({'attn_reduce': {'block': [128, 1, 1]}}))
    script = build / 'fixture.py'
    script.write_text('import json, sys\n'
                      'if "--check" in sys.argv:\n'
                      ' p = int(sys.argv[sys.argv.index("--check-pos")+1])\n'
                      ' print("CHECK-INSTANCE " + json.dumps(dict(id=0, export="attn_reduce", layer=0, replicas=1, grid=[1,1,1], block=[128,1,1], position=p, bytes=4, result="identical", first_byte=-1)))\n'
                      ' print(f"CHECK-IMPORTED checked=1 planned=1 replicas=1 position={p}")\n'
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
    assert receipt['imported_per_position'] == 1 and receipt['greedy_tokens_per_mode'] == 128
    assert len(list(out.glob('*.log'))) == 5
