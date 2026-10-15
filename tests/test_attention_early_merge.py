"""C6 gates: missing correctness cannot become a timing result."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from examples.vllm_llm.attention_early_merge import evidence as e, session as s
from examples.vllm_llm.attention_early_merge.build import capture_host, hybrid_host, clean_env
from examples.vllm_llm.attention_early_merge.simulate import run
from etx.tools.gpu_jobs import load_session


def test_protocol_and_numeric_order():
    result=run()
    assert result['completed_epochs']==1440
    assert len(result['cancellation']['arrival_order_results'])>1
    assert result['gpu_visibility']=='NOT_TESTED'


def test_bf16_near_tie_and_cancellation_must_keep_partition_order():
    import itertools
    import struct
    from examples.vllm_llm.attention_early_merge.simulate import ordered_sum
    def bf16_bits(value):
        bits=struct.unpack('<I',struct.pack('<f',value))[0]
        return (bits+0x7fff+((bits>>16)&1))>>16
    # A halfway value rounds to even. One fp32 ULP crosses the bf16 boundary.
    assert bf16_bits(1+2**-8)==0x3f80
    assert bf16_bits(1+2**-8+2**-23)==0x3f81
    values=[2**24,-2**24,1+2**-8,2**-23,0.]
    orders=list(itertools.permutations(range(5)))
    assert {bf16_bits(ordered_sum(values,o)) for o in orders}=={0x3f80,0x3f81,0x4000}
    # Every election must use the original sequence, including bf16 rounding.
    for order in orders:
        elected=order[-1]
        assert elected in range(5)
        assert bf16_bits(ordered_sum(values,range(5)))==0x3f81


def measurement(gain=1000,check=True):
    lines=['C6-CHECK OK layer=0 position=1040 epochs=8'] if check else []
    for r in range(5):
        values={'control':100000,'candidate':100000-gain}
        for arm in (('control','candidate') if r%2==0 else ('candidate','control')):
            lines.append(f'C6-RESULT arm={arm} layer=0 position=1040 device_ms={values[arm]/1000000:.6f}')
    return '\n'.join(lines)


def test_exact_thresholds_and_noise():
    rows=e.parse_measurement(measurement(),0,1040)
    d=e.decision([rows]*48)
    assert d['median_ns']==48000 and d['combined_50us_target']=='CLOSE' and d['general_30us_rule']=='KEEP'
    assert e.decision([e.parse_measurement(measurement(1100),0,1040)]*48)['combined_50us_target']=='PASS'
    rows[0]['candidate']=rows[0]['control']
    assert e.decision([rows]*48)['combined_50us_target']=='CLOSE'


@pytest.mark.parametrize('text',[measurement(check=False),measurement()+'\n'+measurement(),measurement().replace('candidate','control'),measurement().rsplit('\n',1)[0]])
def test_incomplete_failed_or_wrong_pairs(text):
    with pytest.raises(ValueError): e.parse_measurement(text,0,1040)


@pytest.mark.parametrize('text',['','run 0: 31/32 generated tokens match HF greedy\nCHECK-DSL OK:',
    'run 0: 32/32 generated tokens match HF greedy; first divergence\nCHECK-DSL OK:',
    'run 0: 32/32 generated tokens match HF greedy\nCHECK-DSL FAILED'])
def test_missing_or_failed_tokens(text):
    with pytest.raises(ValueError): e.tokens(text)


def test_tokens_require_candidate_implementation():
    text='run 0: 32/32 generated tokens match HF greedy\nCHECK-DSL OK:'
    e.tokens(text)
    with pytest.raises(ValueError): e.tokens(text,True)
    e.tokens(text+'\nC6-HYBRID enabled=1 mode=combined purpose=correctness-only',True)


def test_stale_and_local_only_build(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'ROOT',tmp_path)
    (tmp_path/'binary').write_bytes(b'fake')
    record=dict(runnable=False,hashes={'binary':e.sha(tmp_path/'binary')},sources={'binary':e.sha(tmp_path/'binary')})
    (tmp_path/'identity.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='device binary'): e.identity(tmp_path)
    e.identity(tmp_path,runnable=False)
    (tmp_path/'binary').write_bytes(b'changed')
    with pytest.raises(ValueError,match='stale'): e.identity(tmp_path,runnable=False)


def test_resource_failure():
    e.resource_gate([dict(vgpr=128,lds_bytes=29752,static_blocks_per_cu=1)])
    for field,value in [('vgpr',129),('lds_bytes',29753),('static_blocks_per_cu',0)]:
        r=dict(vgpr=128,lds_bytes=29752,static_blocks_per_cu=1); r[field]=value
        with pytest.raises(ValueError): e.resource_gate([r])


def test_fixed_deadline_retry_and_duplicate(tmp_path):
    with s.locked(tmp_path,'model',now=lambda:100) as deadline:
        assert deadline==2500
        with pytest.raises(ValueError,match='duplicate'):
            with s.locked(tmp_path,'model',now=lambda:101): pass
    with s.locked(tmp_path,'model',now=lambda:120) as again: assert again==deadline
    with pytest.raises(ValueError,match='expired'):
        with s.locked(tmp_path,'model',now=lambda:2501): pass
    with pytest.raises(ValueError,match='model changed'):
        with s.locked(tmp_path,'other',now=lambda:120): pass


def test_timeout_retains_log_and_rc(tmp_path,monkeypatch):
    runner=s.Runner(tmp_path,tmp_path,99999999999); runner.attempt=tmp_path
    monkeypatch.setattr(s.proc,'run',lambda args,sink,**kw:(sink.write('partial\n'),124)[1])
    with pytest.raises(ValueError,match='exit=124'): runner.command('timeout',['unused'])
    assert (tmp_path/'timeout.log').read_text()=='partial\n'
    assert (tmp_path/'timeout.rc').read_text()=='124\n'
    with pytest.raises(ValueError,match='overwrite'): runner.command('timeout',['unused'])


def test_successful_command_and_cli(tmp_path):
    # Real process, CLI argument parsing and module resolution. No HIP required.
    runner=s.Runner(tmp_path,tmp_path,99999999999); runner.attempt=tmp_path
    assert 'usage:' in runner.command('help',[sys.executable,'-m','examples.vllm_llm.attention_early_merge.build','--help'])
    for module in ('offline','build_model','session','simulate'):
        subprocess.run([sys.executable,'-m','examples.vllm_llm.attention_early_merge.'+module,'--help'],check=True,capture_output=True)


def test_job_graph_and_generated_shell():
    jobs,limit=load_session('examples/sessions/attention_early_merge.py')
    assert limit==2400 and len(jobs)==2
    assert jobs[1].gpu==0 and jobs[1].reads==['c6_build']
    for job in jobs: subprocess.run(['bash','-n','-c',job.cmd],check=True)
    subprocess.run(['bash','-n','examples/vllm_llm/attention_early_merge/rocm_check.sh'],check=True)


def test_capture_missing_and_stale_inputs(tmp_path):
    with pytest.raises(FileNotFoundError): e.seal_capture(tmp_path,{})
    # Construct a labelled synthetic frame to exercise the capture parser only.
    rows=['C6CAP1 0 1040 15']
    for i,name in enumerate(e.BUFFERS):
        data=(1041).to_bytes(4,'little') if name=='seq_lens' else b'1234'
        rows.append(f'{name} {4096+i*16} 4')
        for suffix in ('before','after'): (tmp_path/(name+'.'+suffix)).write_bytes(data)
    imports={}
    for export in e.EXPORTS:
        imports[export]=dict(arg_bytes=8,params=[dict(ty='ptr addrspace(1)',offset=0)])
        rows.append(export+' 8'); (tmp_path/(export+'.args')).write_bytes((4096).to_bytes(8,'little'))
    (tmp_path/'frame.txt').write_text('\n'.join(rows)+'\n')
    e.frame(tmp_path,imports)
    (tmp_path/'seq_lens.before').write_bytes((1040).to_bytes(4,'little'))
    with pytest.raises(ValueError,match='sequence'): e.frame(tmp_path,imports)
    (tmp_path/'qkv.before').write_bytes(b'1')
    with pytest.raises(ValueError,match='byte count'): e.frame(tmp_path,imports)


def test_no_timing_when_precheck_fails(tmp_path,monkeypatch):
    (tmp_path/'current.json').write_text(json.dumps(dict(attempt='build')))
    (tmp_path/'build').mkdir()
    runner=s.Runner(tmp_path,tmp_path,99999999999)
    monkeypatch.setattr(runner,'verify',lambda:(_ for _ in ()).throw(ValueError('missing correctness')))
    commands=[]; monkeypatch.setattr(runner,'command',lambda *args,**kw:commands.append(args))
    with pytest.raises(ValueError,match='missing correctness'): runner.run()
    assert not commands


def test_generated_hooks_and_env(monkeypatch):
    monkeypatch.setenv('ETX_VL_HEAD_LOCAL_PROTOCOL','1'); monkeypatch.setenv('C6_HYBRID','1')
    assert 'C6_HYBRID' not in clean_env() and 'ETX_VL_HEAD_LOCAL_PROTOCOL' not in clean_env()
    assert capture_host().count('C6Capture c6_capture;')==1
    assert 'c6_capture.finish()' in capture_host()
    assert 'c6_hybrid.step(hst,insts)' in hybrid_host()

@pytest.mark.parametrize('fail', ['imports','tokens','payload',None])
def test_pipeline_gate_order_and_success_receipt(tmp_path,monkeypatch,fail):
    build=tmp_path/'attempt-000'; build.mkdir()
    (tmp_path/'current.json').write_text(json.dumps(dict(attempt=build.name)))
    for name in ('probe','diagnostic','model'):
        (build/name).mkdir(); (build/name/'run').write_bytes(name.encode())
    (build/'probe/imp').mkdir(); (build/'probe/imp/imports.json').write_text('{}')
    (build/'model/imp').mkdir(); (build/'model/imp/imports.json').write_text('{}')
    (build/'model/vl_plan.txt').write_text('test')
    (build/'probe/identity.json').write_text('{}'); (build/'model/c6_model_identity.json').write_text('{}')
    runner=s.Runner(tmp_path,tmp_path,99999999999)
    monkeypatch.setattr(runner,'verify',lambda:None); monkeypatch.setattr(runner,'inputs',lambda:{'fake':'synthetic'})
    monkeypatch.setattr(s.cert,'expected_instances',lambda *a:[])
    monkeypatch.setattr(s.cert,'verified_build',lambda *a:{})
    monkeypatch.setattr(s.cert,'stamped',lambda *a:None)
    def import_check(*a):
        if fail=='imports': raise ValueError('import failure')
    monkeypatch.setattr(s.cert,'check_output',import_check)
    def seal(path,imports):
        path.mkdir(); (path/'capture.json').write_text('{}')
    monkeypatch.setattr(e,'seal_capture',seal); monkeypatch.setattr(e,'verify_capture',lambda *a:None)
    calls=[]
    def command(name,args,seconds=120,extra=None):
        calls.append(name)
        if name.startswith('tokens_'):
            if fail=='tokens': return 'run 0: 31/32 generated tokens match HF greedy'
            return 'run 0: 32/32 generated tokens match HF greedy\nCHECK-DSL OK:\nC6-HYBRID enabled=1 mode=combined purpose=correctness-only'
        if '_fault_' in name: return 'C6-FAULT rejected='+name.split('_fault_')[1]
        if '_check_p' in name or name.startswith('measure_'):
            import re
            pos,layer=map(int,re.search(r'p(\d+)_L(\d+)',name).groups())
            if fail=='payload': return 'BLOCKER mismatched bytes'
            return measurement().replace('layer=0',f'layer={layer}').replace('position=1040',f'position={pos}')
        return 'synthetic successful imports or skew'
    monkeypatch.setattr(runner,'command',command)
    if fail:
        with pytest.raises(ValueError): runner.run()
        assert not any(n.startswith('measure_') for n in calls)
    else:
        runner.run()
        receipts=list(build.glob('checks-*/measurement.json')); assert len(receipts)==1
        record=json.loads(receipts[0].read_text()); assert len(record['positions'])==4
        assert record['wall_gain']=='UNPROVEN' and record['target_pass'] is False


def test_real_supervisor_timeout(tmp_path):
    with (tmp_path/'supervisor.log').open('w') as sink:
        rc=s.proc.run([sys.executable,'-c','import time; print("entered",flush=True); time.sleep(10)'],
                      sink=sink,deadline_s=.1,poll_s=.02)
    assert rc==s.proc.DEADLINE_RC
    assert (tmp_path/'supervisor.log').read_text()=='entered\n'


def test_retained_dispatcher_branch_and_signature_guard():
    from examples.vllm_llm.attention_early_merge.build import retained_source
    source='\n'.join([
        'extern "C" __global__ void __launch_bounds__(ETX_THREADS, ETX_MIN_WG_PER_CU) etx_megakernel_d0(etx_params p) {',
        'retained_real_dispatch(p); }',
        'extern "C" __global__ void __launch_bounds__(ETX_THREADS) etx_megakernel_d0(etx_params p) {',
        'retained_real_dispatch(p); }'])
    compiled=retained_source(source)
    assert compiled.count('retained_real_dispatch(p);')==2
    assert 'if (a.retained) c6_kept_dispatch(*a.retained);' in compiled
    assert 'else c6_run(a);' in compiled
    with pytest.raises(ValueError,match='signatures'): retained_source(source.replace('etx_megakernel_d0','changed'))


def test_retained_generated_header_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'ROOT',tmp_path)
    (tmp_path/'source.py').write_text('source')
    fp=tmp_path/'full'; fp.mkdir()
    (fp/'code.s').write_text('assembly')
    (fp/'plan_data.h').write_text('original header')
    full=dict(sources={'source.py':e.sha(tmp_path/'source.py')},hashes={'code.s':e.sha(fp/'code.s')})
    (fp/'offline.json').write_text(json.dumps(full))
    build=tmp_path/'probe'; build.mkdir(); (build/'code.s').write_text('probe assembly')
    record=dict(sources=full['sources'],hashes={'code.s':e.sha(build/'code.s')},runnable=False,
        retained_footprint=dict(directory=str(fp),identity_sha=e.sha(fp/'offline.json'),
                                headers={'plan_data.h':e.sha(fp/'plan_data.h')}))
    (build/'identity.json').write_text(json.dumps(record))
    e.identity(build,runnable=False)
    (fp/'plan_data.h').write_text('stale')
    with pytest.raises(ValueError,match='stale retained header'): e.identity(build,runnable=False)


def test_import_comparison_ignores_only_output_path(tmp_path):
    left=tmp_path/'left'; right=tmp_path/'right'; left.mkdir(); right.mkdir()
    body=b'define float @f(float %x) #0 {\n  %y = fadd float %x, 1.0\n  ret float %y\n}\nattributes #0 = { nounwind }\n'
    def ir(directory,payload=body):
        return (f"; ModuleID = '{directory}/source.hip'\nsource_filename = \"{directory}/source.hip\"\n".encode()+payload)
    (left/'f.ll').write_bytes(ir(left)); (right/'f.ll').write_bytes(ir(right))
    assert e.sha(left/'f.ll')!=e.sha(right/'f.ll')
    e.same_imports(left,right,['f.ll'])
    for changed in (body.replace(b'fadd',b'fsub'),body.replace(b'nounwind',b'noinline'),
                    body+b'!0 = !{i32 1}\n',body.replace(b'1.0',b'2.0')):
        (right/'f.ll').write_bytes(ir(right,changed))
        with pytest.raises(ValueError,match='IR mismatch'): e.same_imports(left,right,['f.ll'])
    (right/'f.ll').write_bytes(ir(right)+b'source_filename = "extra"\n')
    with pytest.raises(ValueError,match='provenance'): e.same_imports(left,right,['f.ll'])
    with pytest.raises(ValueError,match='coverage'): e.same_imports(left,right,[])


def test_raw_import_hash_is_still_required(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'ROOT',tmp_path)
    (tmp_path/'source.py').write_text('source'); (tmp_path/'imp').mkdir()
    (tmp_path/'imp/f.ll').write_text('original')
    record=dict(runnable=False,sources={'source.py':e.sha(tmp_path/'source.py')},
                hashes={'source.py':e.sha(tmp_path/'source.py')},imports={'f.ll':e.sha(tmp_path/'imp/f.ll')})
    (tmp_path/'identity.json').write_text(json.dumps(record))
    e.identity(tmp_path,runnable=False)
    (tmp_path/'imp/f.ll').write_text('changed')
    with pytest.raises(ValueError,match='stale import IR'): e.identity(tmp_path,runnable=False)


def test_import_unit_id_requires_inert_marker_only(tmp_path):
    left=tmp_path/'left'; right=tmp_path/'right'; left.mkdir(); right.mkdir()
    header=b'; ModuleID = \'file\'\nsource_filename = "file"\n'
    marker=(b'@__hip_cuid_abc123 = addrspace(1) global i8 0\n'
            b'@llvm.compiler.used = appending addrspace(1) global [1 x ptr] '
            b'[ptr addrspacecast (ptr addrspace(1) @__hip_cuid_abc123 to ptr)], section "llvm.metadata"\n')
    body=b'define void @f() { ret void }\n'
    (left/'f.ll').write_bytes(header+marker+body)
    other=marker.replace(b'abc123',b'def456')
    (right/'f.ll').write_bytes(header+other+body)
    e.same_imports(left,right,['f.ll'])
    for changed in (other.replace(b'global i8 0',b'global i8 1'),
                    other.replace(b'global i8 0',b'constant i8 0'),
                    other+ b'@alias = alias i8, ptr addrspace(1) @__hip_cuid_def456\n',
                    other.replace(b'llvm.metadata',b'changed')):
        (right/'f.ll').write_bytes(header+changed+body)
        with pytest.raises(ValueError,match='unit-ID'): e.same_imports(left,right,['f.ll'])
