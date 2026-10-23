import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from examples.dsv2lite.arms import ARMS, build_env
from examples.dsv2lite import reference_prepare as refs
from examples.dsv2lite import x8_correctness as x8
from etx.experiments.jobs import load_session


def log(b=1, fail=None):
    lines = []
    for r in range(b):
        for l in range(27):
            bad = (r, l) == fail
            lines.append(f"LAYERGATE r={r} layer={l} rel={'.03' if bad else '.01'} cosine=.9999 pass={0 if bad else 1}")
        layers = fail[1] if fail and fail[0] == r else 27
        lines.append(f"BATCHREQ r={r} tokens=32/32 layers={layers} warmup=32/32 interval_median_ms=4.00000 device_median_ms=3.000000")
    lines.append(f"BATCH requests={b} tokens=32 step_median_ms=4.000000")
    return '\n'.join(lines)


def test_recipe_control():
    single, multi = ARMS['x1km'], ARMS['x8']
    assert single.levers == multi.levers
    a, b = build_env(single), build_env(multi)
    assert a['ETX_DSV2_D4'] == b['ETX_DSV2_D4'] == '0'
    assert a['ETX_DSV2_KVA'] == b['ETX_DSV2_KVA'] == 'shared'
    assert single.env['ETX_DSV2_REQUESTS'] == '1'
    assert '-DETX_SLOT_BY_CU' in single.flags


def test_diagnostic_strict_and_complete():
    assert x8.diagnostic(log(8), 8, 0)['status'] == 'PASS'
    assert x8.diagnostic(log(8, (5, 3)), 8, 3)['status'] == 'FAIL'
    for bad in (log().replace('rel=.01', 'rel=nan'), log().replace('tokens=32/32', 'tokens=31/32'),
                log().replace('LAYERGATE r=0 layer=0 ', 'ABSENT '), log()+'\n'+log().splitlines()[0]):
        assert x8.diagnostic(bad, 1, 0)['status'] != 'PASS'
    assert x8.diagnostic(log(), 1, 124)['status'] == 'INCOMPLETE'


def test_classification():
    out = {name: dict(status='PASS') for name, *_ in x8.CASES}
    assert x8.classify(out) == 'NOT_REPRODUCED_ON_CERTIFIED_INPUTS'
    out['x8_repeat0']['status'] = 'FAIL'
    assert x8.classify(out) == 'UNRESOLVED'
    out['x8_repeat1']['status'] = 'FAIL'
    assert x8.classify(out) == 'MULTI_REQUEST_PATH_DEPENDENT_NO_RACE_PROOF'
    out['x1km_r5']['status'] = 'FAIL'
    assert x8.classify(out) == 'SINGLE_REQUEST_FAILURE_CONCURRENCY_NOT_NECESSARY'
    out['kept_r7']['status'] = 'INCOMPLETE'
    assert x8.classify(out) == 'UNRESOLVED'


def tuple_files(d, prompt):
    import numpy as np
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d/'golden.npz', prompt_ids=prompt, token_ids=list(range(33)))
    (d/'golden_tokens.txt').write_text(' '.join(map(str, range(33))))
    (d/'fleet_cache.bin').write_bytes(b'cache')
    (d/'golden_hidden.bin').write_bytes(b'hidden')


def test_binding_alone_cannot_supply_producer_provenance(tmp_path):
    tuple_files(tmp_path, [1,2,3])
    refs.goldens([str(tmp_path), '--certify'])
    with pytest.raises(OSError):
        refs.verify(tmp_path, {})
    (tmp_path/'golden.producer.raw.log').write_text('producer log')
    receipt = dict(recipe={'rotation': 0}, binding=refs.binding(tmp_path), command=['producer'], exit_code=0,
                   producer_log_sha256=refs.sha(tmp_path/'golden.producer.raw.log'))
    (tmp_path/refs.RECEIPT).write_text(json.dumps(receipt))
    assert refs.verify(tmp_path, {'rotation':0}) == receipt
    (tmp_path/'fleet_cache.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='refused'):
        refs.verify(tmp_path, {'rotation':0})


def session(tmp_path):
    logs, data, fleet = (tmp_path/x for x in ('logs','data','fleet'))
    for p in (logs,data,fleet): p.mkdir()
    for f in ('weights.bin', 'weights.bin.manifest'): (data/f).write_bytes(b'weights')
    return SimpleNamespace(logs=logs, data=data, fleet=fleet)


def setup_refs(monkeypatch, s):
    monkeypatch.setattr(refs, 'fleet_export', lambda p:p)
    monkeypatch.setattr(refs, 'recipe', lambda *a:dict(rotation=0))
    monkeypatch.setattr(refs, 'PINNED_INPUTS', {f:refs.sha(s.data/f) for f in ('weights.bin','weights.bin.manifest')})


def test_missing_reference_refusal_writes_result(tmp_path, monkeypatch):
    s = session(tmp_path); setup_refs(monkeypatch,s)
    with pytest.raises(ValueError, match='regeneration required'): refs.prepare(s, Path('model'),4)
    out = json.loads((s.logs/'references_ready.json').read_text())
    assert out['status'] == 'INCOMPLETE' and not out['original_payload_recovered']


def test_regenerate_only_missing_and_never_certify_old(tmp_path, monkeypatch):
    s = session(tmp_path); setup_refs(monkeypatch,s)
    calls=[]
    prompt=list(range(1024))
    def run(name, cmd, **kw):
        calls.append(name)
        d=Path(cmd[-1]).parent; r=int(cmd[cmd.index('--rotate')+1])
        tuple_files(d,prompt[r:]+prompt[:r]); (s.logs/f'{name}.raw.log').write_text('producer completed')
    s.run=run
    result=refs.prepare(s, Path('model'),4,True)
    assert result['status']=='READY' and len(calls)==4
    assert not list(s.data.glob('golden*'))
    assert (s.logs/'references/weights.bin').is_symlink()
    assert all(v['mode']=='new_inputs' for v in result['requests'])
    # Reuse must call no producer, with both tuple and producer binding intact.
    new_logs=tmp_path/'reuse'; new_logs.mkdir()
    s.data=s.logs/'references'; s.logs=new_logs
    s.run=lambda *a,**kw:pytest.fail('producer ran for certified cache')
    assert all(v['mode']=='reused_producer_bound' for v in refs.prepare(s,Path('model'),4)['requests'])


def test_reference_producer_failure_is_durable(tmp_path, monkeypatch):
    s=session(tmp_path); setup_refs(monkeypatch,s)
    def fail(*a,**kw): raise RuntimeError('producer rc=9')
    s.run=fail
    with pytest.raises(RuntimeError): refs.prepare(s,Path('model'),4,True)
    out=json.loads((s.logs/'references_ready.json').read_text())
    assert 'rc=9' in out['error'] and not (s.logs/'references/golden.binding.json').exists()


def test_queue_dependency_and_correctness_only():
    jobs, cap=load_session('examples/sessions/x8_correctness.py')
    assert cap==3600
    assert [j.name for j in jobs if j.lane=='gpu']==['references','diagnostics']
    diag=next(j for j in jobs if j.name=='diagnostics')
    assert set(diag.reads)=={'references_ready.json', 'b_x1km/run','b_x8/run','b_alld4r0l1kvsmw/run'}
    assert not any('rounds' in j.cmd or 'vllm' in j.cmd for j in jobs)


def test_final_lane_reports_missing_dependencies(tmp_path):
    s=SimpleNamespace(logs=tmp_path)
    x8.finalize(s)
    out=json.loads((tmp_path/'x8_summary.json').read_text())
    assert out['status']=='INCOMPLETE' and out['references_ready']['status']=='MISSING'


def test_full_diagnostics_continue_after_hidden_failure(tmp_path, monkeypatch):
    s=session(tmp_path)
    real=x8.CorrectnessSession.__new__(x8.CorrectnessSession)
    real.__dict__.update(vars(s))
    ready=dict(status='READY',requests=[dict(receipt=dict(recipe={})) for _ in range(8)])
    (s.logs/'references_ready.json').write_text(json.dumps(ready))
    monkeypatch.setattr(x8,'verify',lambda *a:None)
    monkeypatch.setattr(x8,'sha',lambda *a:'a'*64)
    real.source_identity=lambda:'test archive'
    calls=[]
    def run(name,*a,**kw):
        calls.append(name)
        text=log(8,(5,3)) if name.startswith('x8') else log()
        (s.logs/f'{name}.raw.log').write_text(text)
        if name.startswith('x8'): raise RuntimeError('invocation failed rc=3')
        return text
    real.run=run
    real.diagnostics()
    out=json.loads((s.logs/'x8_correctness.json').read_text())
    assert len(calls)==6 and out['classification']=='MULTI_REQUEST_PATH_DEPENDENT_NO_RACE_PROOF'
    assert out['performance_verdict'] is None


def test_producer_receipt_requires_retained_raw_log(tmp_path):
    tuple_files(tmp_path,[1,2,3])
    refs.goldens([str(tmp_path),'--certify'])
    (tmp_path/refs.RECEIPT).write_text(json.dumps(dict(recipe={},binding=refs.binding(tmp_path),
        command=['producer'],exit_code=0,producer_log_sha256='a'*64)))
    with pytest.raises(OSError): refs.verify(tmp_path,{})


def test_embedded_offline_build_shell_syntax():
    import subprocess
    source=Path('examples/dsv2lite/rocm_build_check.sh').read_text()
    inner=source.split("-w /etx etx-rocm22 bash -c '\n",1)[1].split("\n' >",1)[0]
    inner=inner.replace("'\\''", "'")
    subprocess.run(['bash','-n'],input=inner,text=True,check=True)
