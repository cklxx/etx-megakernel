"""Source-faithful CPU evidence, plan reachability and fail-closed session guards."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
from etx.ops import producer
from etx.session.lease import Lease
from examples.vllm_llm import producer_replay as r, producer_session as session, router_study as s

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def arithmetic():
    return r.Arithmetic()


def test_real_production_arithmetic_and_sparse_near_tie(arithmetic):
    witness=json.loads((s.DEFAULT/'numerical.json').read_text())['counterexample']
    w=np.zeros((128,2048),np.float32);w[:7,2]=2;w[7,0]=witness['w7_0'];w[8,1]=witness['w8_1']
    o=np.ones(2048,np.float32);o[0]=3
    z,scale,x=s.norm(o,np.zeros(2048,np.float32),np.ones(2048,np.float32))
    logits,ss,partial=arithmetic.run(w,z,np.ones(2048,np.float32))
    assert ss.view(np.uint32)==scale.view(np.uint32)
    assert partial.shape==(256,129)
    assert s.routing(logits)[0].tolist()==list(range(7))+[8]
    assert s.routing(s.bf(s.dot(w,x)))[0].tolist()==list(range(8))
    assert not np.array_equal(s.routing(logits)[1],s.routing(s.bf(s.dot(w,x)))[1])
    with pytest.raises(ValueError): arithmetic.run(w[:1],z,z)
    z[0]=np.nan
    with pytest.raises(ValueError): arithmetic.run(w,z,z)


def test_completion_reorder_reuse_and_epoch_guards(arithmetic):
    rng=np.random.default_rng(4)
    z=s.bf(rng.normal(size=2048));g=s.bf(rng.normal(size=2048));w=s.bf(rng.normal(size=(128,2048)))
    _,_,partial=arithmetic.run(w,z,g)
    protocol=r.Protocol()
    for epoch in range(6):
        for i,owner in enumerate(rng.permutation(256)):
            assert protocol.arrive(int(owner),partial[owner],epoch=epoch)==(i==255)
        np.testing.assert_array_equal(protocol.finish(),partial)
    assert protocol.epoch==6
    with pytest.raises(ValueError,match='stale'): protocol.arrive(0,partial[0],epoch=0)
    protocol.arrive(0,partial[0])
    with pytest.raises(ValueError,match='duplicate'): protocol.arrive(0,partial[0])
    with pytest.raises(ValueError,match='visibility'): protocol.finish()
    protocol.reset()
    for i in range(256): protocol.arrive(i,partial[i],release=i!=99)
    with pytest.raises(ValueError,match='visibility'): protocol.finish()
    protocol.reset()
    for i in range(256): protocol.arrive(i,partial[i])
    for kwargs in (dict(acquire=False),dict(normalized=False),dict(fanout=False)):
        with pytest.raises(ValueError):protocol.finish(**kwargs)
    protocol.epoch=2**32-1
    with pytest.raises(ValueError,match='wrap'):protocol.finish()


def test_matching_schedule_endpoints_include_all_work():
    costs=dict(producer_us=1,interference_us=2,publication_us=3,readback_reduce_us=4,
               scale_us=5,topk_us=6,normalized_input_us=7,fanout_us=8)
    out=r.schedule([10,20],**costs)
    assert out['all_expert_inputs_and_routes_ready_us']==56
    for key in costs:
        extra=dict(costs);extra[key]+=1
        assert r.schedule([10,20],**extra)['all_expert_inputs_and_routes_ready_us']==57
    for done in ([],[float('nan')],[-1]):
        with pytest.raises(ValueError):r.schedule(done,**costs)


def fixture_capture(path, arithmetic):
    path.mkdir()
    o=np.ones(2048,np.float32);res=np.zeros(2048,np.float32);gamma=o.copy();w=np.zeros((128,2048),np.float32)
    z,_,x=s.norm(o,res,gamma);log=s.bf(s.dot(w,x));ids,mix=s.routing(log)
    values=dict(o=o,residual=res,gamma=gamma,router=w,norm=x,resid2=z,logits=log,
                ids=np.tile(ids.astype(np.int32),8),mix=np.tile(mix,8),tei=np.tile(np.arange(8,dtype=np.int32),8))
    for kind,x in values.items():
        if kind not in ('ids','mix','tei'): x=(s.bf(x).view(np.uint32)>>16).astype('<u2')
        x.tofile(path/f'L0_{kind}.bin')
    r.seal(path,identity='synthetic-fixture',provenance='synthetic-test-fixture',positions=[0],layers=1)


def test_capture_replay_never_mislabels_fixture_and_rejects_corruption(tmp_path,arithmetic):
    p=tmp_path/'capture';fixture_capture(p,arithmetic)
    assert r.replay(p,arithmetic)['actual_model_replay'] is False
    with pytest.raises(ValueError,match='provenance'):r.replay(p,arithmetic,actual=True)
    data=r.replay(p,arithmetic)['cases'][0]
    assert data['ordered_ids_equal'] and data['captured_reference_ids_equal']
    (p/'L0_o.bin').write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash/size'):r.read_capture(p)
    with pytest.raises(ValueError):r.seal(tmp_path,identity='',provenance='actual-model-check',positions=[])


def test_off_source_byte_identity_and_emitter_closure():
    env={k:v for k,v in os.environ.items() if not k.startswith(('ETX_','DSL_')) and k!='CFLAGS'}
    code='''
import subprocess
from etx.ops import gemv
old={"__name__":"old_gemv","__file__":gemv.__file__}
exec(compile(subprocess.check_output(['git','show','c432c77:etx/ops/gemv.py'],text=True),gemv.__file__,'exec'),old)
assert old['emit_tiles']()==gemv.emit_tiles()
assert old['import_entries']()==gemv.import_entries()
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,check=True,capture_output=True,text=True)
    env.update(ETX_VL_PRODUCER_SCORE='1',ETX_VL_CU='256',ETX_VL_ROUTER_NARROW='1')
    code='''
from examples.vllm_llm.plan_exports import exports,build_settings
from etx.ops import gemv,producer
cfg='examples/llm/configs/qwen3-30b-a3b.json'
d,names,phases=exports(cfg)
assert 'dsl_gemv_producer' in names
assert all(p['role'] not in ('router','topk') for p in phases)
assert len([p for p in phases if p['export']=='dsl_gemv_producer'])==48
assert build_settings(d,names,phases)['DSLOPS']=='attnprep,gemv'
assert gemv.import_entries()['dsl_gemv_producer']['arg_bytes']==192
source=gemv.emit_tiles()
assert 'etx_producer_dot8' in source and 'ETX_PRODUCER_INLINE float etx_producer_dot8' in source
assert 'dsl_topk_lds(d.logits,128,8,1' in source
from dataclasses import replace
from examples.vllm_llm import model
# Shape, not directory basename, establishes support for packed hf/config.json.
model.instances(replace(d,name='hf'))
for kw in ({'H':4096},{'L':1},{'norm_topk':False},{'MI':2048}):
 try: model.instances(replace(d,**kw))
 except (ValueError,SystemExit):pass
 else:raise AssertionError('unsupported shape accepted')
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,check=True,capture_output=True,text=True)


def test_emitted_plan_audit_actual_compiler_output():
    # Compile tests retain outputs locally. Unit suite does not require them.
    for name,on in [('producer-rocm-on-v2',True),('producer-rocm-off',False)]:
        build=ROOT/'build'/name
        if not (build/'plan.json').exists():pytest.skip('offline device output not present; local preflight runs this audit')
        assert session.audit_plan(build,on)['layers']==48


def test_plan_mutant_is_rejected(tmp_path):
    p=dict(types={f'o_proj_{i}':dict(symbol='etx_tile_dsl_gemv_producer',grid=[256]) for i in range(48)},args={'producer_scratch':2})
    (tmp_path/'plan.json').write_text(json.dumps(p));(tmp_path/'dsl_tiles.hip').write_text('etx_producer_dot8 dsl_topk_lds(d.logits,128,8,1')
    session.audit_plan(tmp_path,True)
    p['types']['router_0']={}
    (tmp_path/'plan.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='scheduling'):session.audit_plan(tmp_path,True)


def test_lease_excludes_duplicate_and_retry_does_not_extend(tmp_path):
    path=tmp_path/'lease'
    a=Lease(path,100,now=10)
    with pytest.raises(ValueError,match='active'):Lease(path,100,now=11)
    a.close()
    b=Lease(path,100,now=50)
    assert b.remaining==60;b.close()
    with pytest.raises(ValueError,match='extend'):Lease(path,200,now=50)
    with pytest.raises(ValueError,match='expired'):Lease(path,100,now=111)
    path.write_text('partial-write')
    with pytest.raises(ValueError):Lease(path,100,now=50)


def test_session_graph_resolution_and_cap(monkeypatch):
    from etx.tools.gpu_jobs import load_session,main
    jobs,cap=load_session(ROOT/'examples/sessions/qwen_producer.py')
    assert cap==3600
    assert [j.name for j in jobs if j.lane=='gpu']==['qualify','measure','diagnose']
    assert all(j.gpu==0 for j in jobs if j.lane=='gpu')
    assert next(j for j in jobs if j.name=='measure').reads==['correctness.json']
    for j in jobs:subprocess.run(['bash','-n','-c',j.cmd],check=True)
    assert main([str(ROOT/'examples/sessions/qwen_producer.py'),'--dry-run'])==0
    monkeypatch.setenv('ETX_SESS_MAX_SECONDS','3601')
    with pytest.raises(ValueError,match='cap'):load_session(ROOT/'examples/sessions/qwen_producer.py')


def test_missing_correctness_refuses_before_running_host(tmp_path,monkeypatch):
    monkeypatch.setattr(session.g6,'record',lambda:tmp_path)
    monkeypatch.setattr(session,'host',lambda *a,**k:pytest.fail('host launched without correctness'))
    with pytest.raises(FileNotFoundError):session.measure()
    session.report()
    out=json.loads((tmp_path/'report.json').read_text())
    assert out['status']=='NOT-COMPLETE' and not out['performance_claim']


def test_deadline_and_logs_are_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(session.g6,'record',lambda:tmp_path)
    monkeypatch.setattr(session,'DEADLINE',time.monotonic()+session.RESERVE+.2,raising=False)
    with pytest.raises(ValueError,match='exit=124'):
        session.run('timeout',[sys.executable,'-c','import time;time.sleep(3)'],seconds=.1)
    assert (tmp_path/'timeout.raw.log').exists()
    monkeypatch.setattr(session,'DEADLINE',time.monotonic()+1000)
    with pytest.raises(FileExistsError):session.run('timeout',[sys.executable,'-c','pass'],seconds=1)
    with pytest.raises(ValueError,match='exit=7'):session.run('failed',[sys.executable,'-c','raise SystemExit(7)'],seconds=1)
    monkeypatch.setattr(session,'DEADLINE',time.monotonic()+10)
    with pytest.raises(ValueError,match='reserve'):session.run('reserve',['false'],seconds=1)
    assert not (tmp_path/'reserve.raw.log').exists()


def test_timing_parser_requires_complete_tokens_and_correct_mode():
    text="""ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=
mode: ETX megakernel
run 0: 32/32 generated tokens match HF greedy
RESULT model=/m requests=1 matched=32 gen=32 matched_each=32 median_ms=4 wall_ms=4.1
CHECK-DSL OK
"""
    assert session.token_gate(text,dsl=True).wall_ms==4.1
    for bad in (text.replace('matched=32','matched=31'),text.replace('wall_ms=4.1','wall_ms=nan'),text.replace('CHECK-DSL OK','CHECK-DSL FAILED')):
        with pytest.raises(Exception):session.token_gate(bad,dsl=True)
    with pytest.raises(ValueError):session.token_gate(text,'graph')


def test_complete_report_recomputes_rule_and_refuses_partial(tmp_path,monkeypatch):
    from examples.vllm_llm.import_certificate import digest
    monkeypatch.setattr(session,'require_correctness',lambda *a,**k:None)
    session.save(tmp_path/'measurement.complete.json',dict(rounds=5,basis='host RESULT wall_ms',complete=True))
    identity={'test_fixture':True};tag=digest(identity)
    for arm in session.ARMS:
        d=tmp_path/f'certificate_{arm}';d.mkdir();session.save(d/'build_identity.json',identity)
    for rnd in range(5):
        for arm,mode,ms in [('baseline','fused',4.1),('producer','fused',4.0),('producer','graph',4.2)]:
            text=f'''BUILD-IDENTITY phase=before sha256={tag}
ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=
mode: {'ETX megakernel' if mode=='fused' else 'unfused HIP graph'}
run 0: 32/32 generated tokens match HF greedy
RESULT requests=1 matched=32 gen=32 matched_each=32 median_ms={ms} wall_ms={ms}
BUILD-IDENTITY phase=after sha256={tag}
'''
            (tmp_path/f'r{rnd}_{arm}_{mode}.raw.log').write_text(text)
        refdir=tmp_path/f's3_r{rnd}';refdir.mkdir()
        full=[31*4.3+10]*5;one=[10]*5;per=[(a-b)/31 for a,b in zip(full,one)]
        session.save(refdir/'bf16_bar.json',dict(correct=32,image=session.IMAGE,launch={'attention_backend':'ROCM_AITER_UNIFIED_ATTN'},
                    full_ms=full,one_ms=one,per_token_ms=per,median_ms=per[0]))
        (tmp_path/f'r{rnd}_s3.raw.log').write_text('CORRECTNESS OK 32/32 precision=bf16\n'*6+'BENCH-PROGRESS sample=5/5 correctness=32/32\n')
    report=session.result(tmp_path)
    assert report['incremental']=='KEEP' and report['fusion']['established'] is True
    for rnd in range(5):
        path=tmp_path/f'r{rnd}_producer_fused.raw.log'
        path.write_text(path.read_text().replace('wall_ms=4.0','wall_ms=4.07'))
    assert session.result(tmp_path)['incremental']=='KEEP' # exact 30 us boundary
    path=tmp_path/'r0_producer_fused.raw.log'
    path.write_text(path.read_text().replace('wall_ms=4.07','wall_ms=4.071'))
    assert session.result(tmp_path)['incremental']=='KEEP' # median still 30, spread 1
    (tmp_path/'r4_producer_graph.raw.log').unlink()
    with pytest.raises(FileNotFoundError):session.result(tmp_path)


def test_runner_failed_qualify_skips_measure_and_preserves_unique_records(tmp_path):
    """Execute the actual runner with CPU stand-ins. No GPU code is executed."""
    from etx.tools import gpu_jobs
    lease=tmp_path/'lease'
    jobfile=tmp_path/'session.py'
    jobfile.write_text(f'''MAX_SESSION_SECONDS=20
SESSION_LEASE={str(lease)!r}
JOBS=[dict(name='qualify',lane='gpu',gpu=0,cmd="exit 2",writes=['correctness.json'],timeout=2,cwd={str(ROOT)!r}),
      dict(name='measure',lane='gpu',gpu=0,cmd="exit 99",reads=['correctness.json'],timeout=2,cwd={str(ROOT)!r})]
''')
    args=[str(jobfile),'--gpus','0','--logdir',str(tmp_path/'records'),'--progress-s','1']
    assert gpu_jobs.main(args)!=0
    first=json.loads(lease.read_text())
    assert gpu_jobs.main(args)!=0
    assert json.loads(lease.read_text())==first
    records=list((tmp_path/'records').iterdir());assert len(records)==2
    for d in records:
        statuses=json.loads((d/'status.json').read_text())
        assert [j['status'] for j in statuses]==['fail','skipped']
        assert not (d/'measure.log').exists()
    with pytest.raises(SystemExit,match='complete graph'):gpu_jobs.main([*args,'--only','measure'])


def test_source_publication_guards_and_invalid_modes(monkeypatch):
    source=producer.emit()
    # Source-specific checks complement the logical protocol execution above.
    assert source.index('__threadfence_system();')<source.index('__hip_atomic_fetch_add(count')
    assert '__ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM' in source
    assert '__hip_atomic_load(tags+tid, __ATOMIC_ACQUIRE' in source
    assert 'score += dsl_a_ldf(scratch+p*129+tid)' in source
    assert 'if (xs[16] == 0.f) return' in source and 'while (' not in source
    monkeypatch.setenv('ETX_VL_PRODUCER_SCORE','bad')
    with pytest.raises(ValueError):producer.enabled()


def test_trace_endpoints_and_coverage_refusal():
    from examples.vllm_llm.producer_trace import summarize
    types={};rows=[]
    for layer in range(48):
        for name,ready,start,end in [(f'o_proj_{layer}',0,10,20),(f'moe_up_{layer}',25,26,30)]:
            ty=len(types);types[name]=dict(id=ty,grid=[1]);rows.append([ty,0,0,0,0,ready,start,end,0])
    raw=np.asarray(rows,dtype='<i8').tobytes()
    out=summarize({'types':types},raw,True)
    assert out['layers'][0]['o_start_to_all_expert_tasks_ready_us']==.15
    with pytest.raises(ValueError,match='truncated'):summarize({'types':types},raw[:-1],True)
    with pytest.raises(ValueError,match='coverage'):summarize({'types':types},raw[:-72],True)


def test_input_identity_binds_both_packed_and_reference_weights(tmp_path):
    model=tmp_path/'model';hf=model/'hf';hf.mkdir(parents=True)
    for name in ('vl.bin','vl.manifest','golden_c1024.txt'):(model/name).write_bytes(b'test-fixture')
    for name in ('config.json','REVISION','model.safetensors'):(hf/name).write_bytes(b'test-fixture')
    config={'model':str(model),'hf':str(hf)}
    a=session.input_identity(config)
    (hf/'model.safetensors').write_bytes(b'changed')
    b=session.input_identity(config)
    assert a['packed/vl.bin']==b['packed/vl.bin'] and a['hf/model.safetensors']!=b['hf/model.safetensors']
    (hf/'REVISION').unlink()
    with pytest.raises(ValueError,match='revision'):session.input_identity(config)


def test_raw_logit_nonidentity_is_diagnostic_but_mixing_is_exact(tmp_path,monkeypatch):
    # Isolate the acceptance policy. Capture coverage/hash validation has its
    # own tests above; these tiny records are explicitly synthetic fixtures.
    def metadata(path,actual=False):
        pos=int(Path(path).name.split('p')[-1])
        return dict(layers=48,positions=[pos],files={'L0_logits.bin':'fixture','L0_mix.bin':'fixture'})
    monkeypatch.setattr(session.replay,'read_capture',metadata)
    for pos in (0,1040):
        for arm in session.ARMS:
            path=tmp_path/f'capture_{arm}_p{pos}';path.mkdir()
            (path/'capture.json').write_text('{"synthetic_policy_fixture":true}')
            (path/'L0_logits.bin').write_bytes(b'baseline' if arm=='baseline' else b'different')
            (path/'L0_mix.bin').write_bytes(b'exact-mixing-weights')
    assert set(session.capture_gate(tmp_path))=={'0','1040'}
    (tmp_path/'capture_producer_p1040/L0_mix.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='performance blocked'):session.capture_gate(tmp_path)


def test_local_compile_rebases_only_real_include_paths(tmp_path,monkeypatch):
    from examples.vllm_llm.producer_local import rebase_includes
    fake_home=tmp_path/'fake-home'
    real=fake_home/'code/header.h';real.parent.mkdir(parents=True);real.write_text('// real fixture header\n')
    monkeypatch.setattr(Path,'home',lambda:fake_home)
    container='/code/header.h'
    text=f'#include "{container}"\nconst char* s="/code/leave-this-literal";\n'
    source=tmp_path/'source.hip';source.write_text(text)
    out=tmp_path/'native.hip'
    mapping=rebase_includes(source,out)
    assert source.read_text()==text and str(real) in out.read_text()
    assert '/code/leave-this-literal' in out.read_text() and len(mapping)==1
    source.write_text('#include "/code/does-not-exist-ETX-test/header.h"\n')
    with pytest.raises(ValueError,match='missing host include'):rebase_includes(source,tmp_path/'bad.hip')
