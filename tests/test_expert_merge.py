"""Executable probe and fail-closed measurement gates. No GPU is used."""
import json
import os
from pathlib import Path
import struct
import sys
import time

import pytest
from examples.vllm_llm.expert_merge import capture, inputs, prepare, protocol, runner, source


def test_source_keeps_producers_and_exact_statistics():
    a=source.emit(False,False); b=source.emit(True,False)
    for text in (a,b):
        assert 'dsl_run_dsl_moe_up_fold' in text
        assert 'dsl_run_dsl_moe_down_fold' in text
        assert '#if EEM_DIAGNOSTIC' in text
    assert 'eem_append(c, j)' in b
    assert 'group*8 + (int)threadIdx.x' in b
    assert 'dsl_a_stf((float*)c->args[2] + i, ss)' in b
    assert 'threadIdx.x < 256 ? dsl_a_ldf' in b
    assert 'ETX_ARRIVE_DEVICE((int*)c->args[4] + 32)' in b
    assert b.count('ETX_RELEASE_DOMAIN();')==2
    with pytest.raises(ValueError,match='anchor changed'):
        source.once('new source','old source','replacement')


def test_generated_graph_and_identity(tmp_path,monkeypatch):
    # The caller's model switches must not label a different executable.
    monkeypatch.setenv('ETX_VL_NB','2')
    monkeypatch.setenv('ETX_VL_DSL_EXT','1')
    for arm in ('control','candidate'):
        out=tmp_path/arm
        identity=prepare.generate(out,arm,diagnostic=True,head=arm=='candidate')
        assert prepare.verify(out)==identity
        p=json.loads((out/'plan.json').read_text())
        assert len(p['descs'])==192+256+256
        assert sum(len(e['counts']) for e in p['events'])==9
        assert sorted(t['type'] for t in p['descs']).count(0)==192
        assert [(e['scope'],e['counts']) for e in p['events']]==[('DOMAIN',[24]*8),('DEVICE',[256])]
        for t in p['descs']:
            if t['type'] in (0,1): assert t['domain']==t['coord'][0]
        assert 'ETX_IMPORT_LDS 21120' in (out/'tiles.hip').read_text()
        assert 'full_model_default' in identity and identity['full_model_default']=='OFF'
        with pytest.raises(ValueError,match='build exists'):
            prepare.generate(out,arm)
        changed=dict(identity); changed['head']=not changed['head']
        (out/'identity.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='manifest identity'):
            prepare.verify(out)
        (out/'identity.json').write_text(json.dumps(identity))
        (out/'tiles.hip').write_text('stale')
        with pytest.raises(ValueError,match='generated identity'):
            prepare.verify(out)


def test_protocol_reset_order_and_publication():
    assert protocol.exercise(tokens=128)['row_owners']==4096
    p=protocol.Protocol()
    with pytest.raises(ValueError,match='drain'): p.produce(0,0,15)
    p=protocol.Protocol(); p.produce(0,0)
    with pytest.raises(ValueError,match='duplicate'): p.produce(0,0)
    with pytest.raises(ValueError,match='reset'): p.reset()
    with pytest.raises(ValueError,match='before full'): p.consume(0)
    for s in range(1,8): p.produce(s,0)
    with pytest.raises(ValueError,match='appended'): p.complete(7,0)
    with pytest.raises(ValueError,match='drain'): p.publish(0,15)
    # Remove a prior payload before the last counter arrival.
    q=protocol.Protocol()
    for s in range(7): q.produce(s,1)
    q.payload.remove((0,1))
    with pytest.raises(ValueError,match='stale payload'): q.produce(7,1)


@pytest.fixture
def fake_session(tmp_path,monkeypatch):
    root=tmp_path/'session'; root.mkdir()
    inp=tmp_path/'input'; inp.mkdir(); (inp/'input.json').write_text('{}')
    runner.dump(root/'session.json',dict(input=str(inp),input_sha256=prepare.sha(inp/'input.json'),
        deadline_unix=time.time()+300,head=False,kind='synthetic'))
    monkeypatch.setattr(inputs,'validate',lambda _:dict(M=5120,kind='synthetic',head=False))
    monkeypatch.setattr(runner,'check_build',lambda p:dict(id=Path(p).name,binary='test-only',head=False,
                                                       diagnostic=Path(p).name.endswith('_diag')))
    def fake_command(cmd,log,*args):
        cmd=list(map(str,cmd)); out=Path(cmd[2]); arm=Path(cmd[0]).parent.name
        diagnostic=arm.endswith('_diag'); candidate=arm.startswith('candidate')
        for r in range(12):
            fields={**runner.BASE_FIELDS,'output':5120*2,**(runner.DIAG_FIELDS if diagnostic else {})}
            for name,n in fields.items():
                value=bytes([r+1])
                (out/f'r{r}_{name}.bin').write_bytes(value*n)
            if diagnostic:
                (out/f'r{r}_staged.bin').write_bytes((out/f'r{r}_normalized.bin').read_bytes()*256)
            if candidate and diagnostic:
                (out/f'r{r}_owners.bin').write_bytes(struct.pack('<32i',*([r%8]*32)))
        return 'TEST-ONLY correctness subprocess fixture; no device execution'
    monkeypatch.setattr(runner,'command',fake_command)
    return root


def test_certificate_success_missing_failure_stale_retry(fake_session,monkeypatch):
    root=fake_session
    with pytest.raises(FileNotFoundError): runner.require_certificate(root)
    runner.certify(root); runner.require_certificate(root)
    pointer=json.loads((root/'certificate.json').read_text())
    certificate=json.loads(Path(pointer['path']).read_text())
    p=Path(certificate['path'])/'candidate_fused/r1_output.bin'
    p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='missing/short'): runner.require_certificate(root)
    runner.certify(root); runner.require_certificate(root)
    assert (root/'check-001/previous-certificate.json').exists()
    def fail(*a,**kw): raise ValueError('injected correctness failure')
    monkeypatch.setattr(runner,'command',fail)
    with pytest.raises(ValueError,match='injected'): runner.certify(root)
    assert not (root/'certificate.json').exists()
    assert (root/'check-002/previous-certificate.json').exists()
    with pytest.raises(FileNotFoundError): runner.require_certificate(root)


def test_byte_mismatch_blocks_certificate(fake_session,monkeypatch):
    old=runner.command
    def mismatch(cmd,*args):
        result=old(cmd,*args)
        if 'candidate_diag' in str(cmd[0]):
            (Path(cmd[2])/'r0_inverse.bin').write_bytes(b'FAIL')
        return result
    monkeypatch.setattr(runner,'command',mismatch)
    with pytest.raises(ValueError,match='BYTE CHECK FAILED'): runner.certify(fake_session)
    assert not (fake_session/'certificate.json').exists()


def test_timing_requires_certificate_and_interleaves(fake_session,monkeypatch):
    root=fake_session
    with pytest.raises(FileNotFoundError): runner.measure(root)
    runner.certify(root)
    calls=[]
    def timed(cmd,log,timeout,env):
        assert env['EEM_CHECKED_IDENTITY']
        arm=Path(cmd[0]).parent.name; mode=cmd[4]; calls.append((mode,arm))
        row=dict(identity=arm,arm=arm,mode=mode,n=200,us=10 if arm=='control' else 9.2)
        return 'EEM-TIME '+json.dumps(row)+'\n'
    monkeypatch.setattr(runner,'command',timed)
    report=runner.measure(root)
    assert len(calls)==20
    assert calls[:6]==[('fused','control'),('fused','candidate'),('graph','control'),
                       ('graph','candidate'),('fused','candidate'),('fused','control')]
    assert report['verdict']=='SYNTHETIC-PASS-REAL-INPUT-REQUIRED'
    assert (root/'time-000/report.json').exists()


def test_incomplete_build_and_timing_are_not_results(fake_session,monkeypatch):
    root=fake_session
    (root/'control_diag').mkdir()
    with pytest.raises(ValueError,match='incomplete build'):
        runner.build(root,runner.session(root)['input'])
    runner.certify(root)
    def short(cmd,*args):
        arm=Path(cmd[0]).parent.name
        return 'EEM-TIME '+json.dumps(dict(identity=arm,arm=arm,mode=cmd[4],n=199,us=10))+'\n'
    monkeypatch.setattr(runner,'command',short)
    with pytest.raises(ValueError,match='invalid timing'): runner.measure(root)
    assert not (root/'time-000/report.json').exists()


def test_nonfinite_output_blocks_timing(fake_session,monkeypatch):
    old=runner.command
    def nan(cmd,*args):
        result=old(cmd,*args)
        (Path(cmd[2])/'r0_output.bin').write_bytes(b'\xc0\x7f'*(5120))
        return result
    monkeypatch.setattr(runner,'command',nan)
    with pytest.raises(ValueError,match='nonfinite'): runner.certify(fake_session)


def test_session_deadline_stale_input_and_lock(fake_session):
    root=fake_session
    with runner.lock(root):
        with pytest.raises(ValueError,match='another'):
            with runner.lock(root): pass
    m=runner.session(root); m['deadline_unix']=time.time()-1
    runner.dump(root/'session.json',m)
    with pytest.raises(ValueError,match='expired'): runner.session(root)
    m['deadline_unix']=time.time()+100; runner.dump(root/'session.json',m)
    (Path(m['input'])/'input.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='stale input'): runner.session(root)


def test_command_logs_failure_timeout_and_preserves_attempts(tmp_path):
    log=tmp_path/'failure.log'
    with pytest.raises(ValueError,match='failed 7'):
        runner.command([sys.executable,'-c',"print('full output'); raise SystemExit(7)"],log)
    assert json.loads(Path(str(log)+'.status.json').read_text())['exit']==7
    assert 'full output' in log.read_text()
    with pytest.raises(ValueError,match='log exists'):
        runner.command(['true'],log)
    log=tmp_path/'timeout.log'
    with pytest.raises(ValueError,match='timeout'):
        runner.command([sys.executable,'-c','import time; print("started",flush=True); time.sleep(30)'],log,.1)
    assert json.loads(Path(str(log)+'.status.json').read_text())['timeout']
    assert runner.attempt(tmp_path,'retry')!=runner.attempt(tmp_path,'retry')


@pytest.mark.parametrize('gains,passed',[([.8,.81,.79,.8,.8],True),([.5]*5,False),
    ([.8,.8,.8,.8,-.1],False),([.625]*5,True),([.5,.8,.9,1,1.1],False)])
def test_registered_screen(gains,passed):
    result=runner.verdict(gains,'synthetic')
    assert (result['verdict']=='SYNTHETIC-PASS-REAL-INPUT-REQUIRED')==passed
    assert 'OFF' in result['full_model']
    with pytest.raises(ValueError): runner.verdict([float('nan')]*5,'synthetic')
    with pytest.raises(ValueError): runner.verdict(gains[:4],'synthetic')


def test_input_pack_hash_and_precision(tmp_path):
    out=tmp_path/'pack'; inputs.synthetic(out)
    assert inputs.validate(out)['kind']=='synthetic'
    assert inputs.bf16([1.00390625]).tolist()==[0x3f80]  # ties to even
    (out/'x.bin').write_bytes(b'\0'*4096)
    with pytest.raises(ValueError,match='hash mismatch'): inputs.validate(out)


def test_capture_entrypoint_and_failed_receipt(tmp_path):
    out=tmp_path/'capture-host'; capture.emit(out)
    script=(out/'build.sh').read_text()
    assert str(out/'capture_host.hip') in script
    assert 'capture attest' in script
    assert '; cd "$ROOT"' in script
    c=tmp_path/'captured'; c.mkdir(); b=tmp_path/'build'; b.mkdir()
    (c/'capture.txt').write_text('23 1040\n'); (c/'check.log').write_text('CHECK OK: 0 launch(es) differ\n')
    with pytest.raises(FileNotFoundError): capture.seal(c,b)
    runner.dump(c/'check.log.status.json',dict(exit=1,timeout=False))
    (b/'run').write_bytes(b'not a device binary'); (b/'vl_plan.txt').write_text('test')
    runner.dump(c/'build.json',dict(binary=prepare.sha(b/'run'),plan=prepare.sha(b/'vl_plan.txt')))
    with pytest.raises(ValueError,match='completed successful'): capture.seal(c,b)


@pytest.mark.parametrize('layer,consumer_layer,role',[(23,24,'qkv'),(47,-1,'head')])
def test_capture_complete_receipt_and_coverage(tmp_path,monkeypatch,layer,consumer_layer,role):
    c=tmp_path/'capture'; c.mkdir(); b=tmp_path/'build'; b.mkdir()
    (c/'capture.txt').write_text(f'{layer} 1040\n'); (b/'run').write_bytes(b'test-only')
    (b/'vl_plan.txt').write_text('test'); (c/'selected_ids.bin').write_bytes(bytes(32))
    identity=dict(binary=prepare.sha(b/'run'),plan=prepare.sha(b/'vl_plan.txt'),revision='test')
    runner.dump(c/'build.json',identity); runner.dump(c/'check.log.status.json',dict(exit=0,timeout=False))
    (c/'check.log').write_text('CHECK OK: 0 launch(es) differ\n')
    with pytest.raises(ValueError,match='coverage'): capture.seal(c,b)
    (c/'check.log').write_text('CHECK OK: 0 launch(es) differ\n'
        'CHECK-IMPORTED checked=97 planned=97 replicas=97 position=1040\n'
        f'EEM-CAPTURE up layer={layer} position=1040\nEEM-CAPTURE consumer={role} layer={consumer_layer} position=1040\n')
    sealed=[]; monkeypatch.setattr(inputs,'seal',lambda *a:sealed.append(a))
    capture.seal(c,b)
    assert sealed[0][1:3]==('real-capture',layer==47)
    assert json.loads((c/'capture.json').read_text())['revision']=='test'


def test_session_job_graph(monkeypatch):
    from etx.tools.gpu_jobs import load_session
    path=prepare.ROOT/'examples/sessions/qwen_expert_merge.py'
    monkeypatch.delenv('EEM_INPUT',raising=False)
    jobs,cap=load_session(path)
    assert cap==1800 and [j.name for j in jobs]==['synthetic','build','correctness','pairs']
    assert jobs[-1].reads==['correctness'] and jobs[-1].gpu==0
    monkeypatch.setenv('EEM_INPUT','/capture/pack')
    jobs,_=load_session(path)
    assert jobs[0].name=='build' and not jobs[0].reads
