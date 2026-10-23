"""Tiny CPU boundary regressions for RR1/RR2/RR3; no native execution.

RR1 keeps the real CLI, up helper, raw readmission, numerical outputs, timings,
telemetry and config generation. Only original native-producer certification,
sparse native qualification and unused export-space estimation are CPU stubs.
RR2 keeps aggregation/readmission and real decisive log parsing, with the native
children/diagnostic qualifier as explicit CPU stubs. RR3 stubs native provenance
only, retaining the real paired/queue/body/output decision.
"""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import pytest
from examples.experiment_program.protocol import sha,pair_order


def write(path,value):
    path.write_text(json.dumps(value));return path


@pytest.fixture
def tiny(tmp_path):
    # Per-case synthetic files are bounded; they are not reusable native evidence.
    with tempfile.TemporaryDirectory(prefix='CPU_ONLY_',dir=tmp_path) as p:yield Path(p)


@pytest.fixture
def row_boundary(tiny,monkeypatch):
    from examples.experiment_program.row_ops import build,source,diagnostics,replay,producer,results
    from examples.experiment_program.row_ops.protocol import invocation_orders
    from examples.experiment_program.telemetry import capture
    from etx.models.vllm_llm.bf16_300.build import preamble
    from etx.ops import gemv
    normal=tiny/'normal';diagnostic=tiny/'diagnostic';normal.mkdir();diagnostic.mkdir()
    # Actual canonical generated sources, but explicitly nonexecutable CPU bytes.
    for v in ('g10_up','g10_down'):
        (normal/(v+'.hip')).write_text(source.translation_unit(v,preamble(gemv.LDS_BYTES)))
    for mode in ('control','probe','empty'):
        (diagnostic/('g10_down_'+mode+'.hip')).write_text(diagnostics.unit('g10_down',mode,preamble(gemv.LDS_BYTES)))
    for d,names in [(normal,('replay','control.hsaco','g10_up.hsaco','g10_down.hsaco')),(diagnostic,('replay',)+tuple(v+'_'+m+'.hsaco' for v in ('g10_up','g10_down') for m in ('control','probe','empty')))]:
        for n in names:(d/n).write_bytes(b'CPU ONLY; never execute')
        hashes={p.name:sha(p) for p in d.iterdir()};identity=dict(hashes=hashes,files={str(d/n):h for n,h in hashes.items()})
        identity['math_flags' if d==normal else 'flags']=build.FLAGS
        write(d/'identity.json',identity)
    monkeypatch.setattr(producer,'verify',lambda *a:dict(synthetic_cpu_only=True))
    q=dict(qualified=True,synthetic_cpu_only=True)
    monkeypatch.setattr(diagnostics,'summarize',lambda *a:q)
    # The CLI is --dry-run; no export is written and no physical reserve is waived.
    monkeypatch.setattr(replay,'output_reservation',lambda *a:dict(synthetic_cpu_only=True,no_payload_writes=True))
    descriptors=[];up_packs=[]
    def pack(position,down=False):
        prefix=tiny/f'{"down" if down else "up"}-{position}';arg=Path(str(prefix)+'.args');raw=bytearray(160)
        M,K,rpt,count=(2048,768,64,256) if down else (768,2048,32,1)
        struct.pack_into('<q',raw,40,(1 if down else 2)*M*K);struct.pack_into('<3i',raw,48,M,K,rpt)
        struct.pack_into('<i',raw,96,8);struct.pack_into('<i',raw,108,128);struct.pack_into('<i',raw,120,1);struct.pack_into('<i',raw,156,count);arg.write_bytes(raw)
        gold=Path(str(prefix)+'.gold');gold.write_bytes(b'CPU!')
        p=dict(schema='etx-row-replay-v1',baseline_revision=producer.BASE,position=position,layer=24,role='dsl_moe_down' if down else 'dsl_moe_up',count=count,first=0,arguments=dict(file=str(arg),sha256=sha(arg)),buffers=[dict(file=str(gold),offset=0,bytes=4,sha256=sha(gold))],pointers=[],references=[dict(file=str(gold),bytes=4,sha256=sha(gold),buffer=0,offset=0)],reference_provenance='unchanged-original-host',producer_certificate=dict(synthetic_cpu_only=True))
        path=write(Path(str(prefix)+'.pack'),p);return path,p
    for position in (0,1040):
        path,p=pack(position);prefix=tiny/f'up-{position}-raw';rows=[]
        for sample,order in enumerate(invocation_orders(2,5)):
            for arm in order:
                rows.append(dict(sample=sample,arm=arm,exact=True,device_us=20-arm*2,host_wall_us=21-arm*2))
                Path(str(prefix)+f'.s{sample}.a{arm}.r0.bin').write_bytes(b'CPU!')
        for arm in (0,1):Path(str(prefix)+f'.a{arm}.intermediates').write_bytes(bytes(256))
        Path(str(prefix)+'.timing.jsonl').write_text('\n'.join(map(json.dumps,rows)))
        Path(str(prefix)+'.exit').write_text('0\n')
        command=['CPU_ONLY_NOT_EXECUTED']
        with capture(str(prefix)+'.telemetry',dict(command=command,pack_sha256=sha(path),build_sha256=sha(normal/'identity.json')),collector=lambda *a,**k:SimpleNamespace(returncode=0)):pass
        dp=tiny/f'up-{position}-diagnostic';Path(str(dp)+'.exit').write_text('0\n')
        dr=write(Path(str(dp)+'.result'),dict(build_identity_sha256=sha(diagnostic/'identity.json'),input_pack_sha256=sha(path),verdict=q))
        decision=write(Path(str(prefix)+'.result'),dict(gpu_executed=True,synthetic_cpu_only=True,build_identity_sha256=sha(normal/'identity.json'),input_pack_sha256=sha(path),command=command,verdict=results.summarize(prefix,p,{},'g10_up')))
        descriptors.append(dict(record='G10',variant='g10_up',build=str(normal),pack=str(path),output_prefix=str(prefix),decision=str(decision),diagnostic=dict(build=str(diagnostic),output_prefix=str(dp),decision=str(dr))))
        up_packs.append(path)
    evidence=write(tiny/'up-evidence.json',dict(schema='etx.row.up-evidence.v1',positions=descriptors));down,_=pack(0,True)
    return SimpleNamespace(root=tiny,normal=normal,diagnostic=diagnostic,evidence=evidence,descriptors=descriptors,down=down)


def row_command(f,*,diagnostic=True):
    cmd=['execute','--pack',str(f.down),'--build',str(f.diagnostic if diagnostic else f.normal),'--variant','g10_down','--output',str(f.root/'never-executed'),'--up-evidence',str(f.evidence),'--dry-run']
    return cmd+(['--diagnostic','--up-build',str(f.normal)] if diagnostic else [])


def test_rr1_real_cli_helper_and_raw_readmission_accept_distinct_builds(row_boundary):
    from examples.experiment_program.row_ops.__main__ import main
    f=row_boundary
    for diagnostic in (True,False):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):main(row_command(f,diagnostic=diagnostic))
        result=json.loads(output.getvalue());assert result['up_evidence']['sha256']==sha(f.evidence)
        assert result['command'][0]==str((f.diagnostic if diagnostic else f.normal)/'replay')
        assert not result['gpu_executed']


@pytest.mark.parametrize('fault',['normal_identity','diagnostic_source','diagnostic_flags','raw_output','failed_up','missing_raw','wrong_position'])
def test_rr1_invalid_normal_diagnostic_or_raw_up_refuses_before_command(row_boundary,fault,monkeypatch):
    from examples.experiment_program.row_ops import __main__ as cli
    f=row_boundary
    if fault=='normal_identity':
        p=f.normal/'identity.json';v=json.loads(p.read_text());v['math_flags']=[];write(p,v)
    elif fault=='diagnostic_source':
        p=f.diagnostic/'g10_down_probe.hip';p.write_text('// wrong variant source, newly resealed CPU fixture')
        identity=f.diagnostic/'identity.json';v=json.loads(identity.read_text());v['files'][str(p)]=sha(p);write(identity,v)
    elif fault=='diagnostic_flags':
        p=f.diagnostic/'identity.json';v=json.loads(p.read_text());v['flags']=[];write(p,v)
    elif fault=='raw_output':Path(f.descriptors[0]['output_prefix']+'.s0.a0.r0.bin').write_bytes(b'BAD!')
    elif fault=='failed_up':Path(f.descriptors[0]['output_prefix']+'.exit').write_text('1\n')
    elif fault=='missing_raw':Path(f.descriptors[0]['output_prefix']+'.timing.jsonl').unlink()
    else:
        p=Path(f.descriptors[1]['pack']);v=json.loads(p.read_text());v['position']=0;write(p,v)
    monkeypatch.setattr(cli,'config',lambda *a,**k:pytest.fail('rejected dependency must not construct command'))
    with pytest.raises((ValueError,OSError)):cli.main(row_command(f))


def router_child(root,build,*,winning=True):
    from examples.experiment_program.router.receipts import parse_native
    def child(argv):
        out=Path(argv[argv.index('--out')+1]);out.mkdir()
        lines=[f'CHECK G01 case={i} exact=1' for i in range(96)]+['G01 COMPLETE all96 exact']
        for i,order in enumerate(pair_order(19)):
            for ch in order:
                arm=0 if ch=='A' else 1;value=20000 if arm==0 else 18000 if winning else 20000
                lines.append(f'SAMPLE G01 ordinal={i} arm={arm} sample=0 device_ns={value} wall_ns={value}')
                lines.append(f'PAIR G01 ordinal={i} arm={arm} device_ns={value} wall_ns={value} population=1')
        text='\n'.join(lines);(out/'native.log').write_text(text)
        result=parse_native(text,population=1,seed=19);result.update(record='G01',build_identity_sha256=sha(build/'identity.json'),inputs={str(build/'identity.json'):sha(build/'identity.json')})
        write(out/'decision.json',result);write(out/'seal.json',dict(population=1,seed=19));return 0
    return child


@pytest.mark.parametrize('qualified,winning',[(False,True),(True,True),(True,False)])
def test_rr2_router_aggregation_and_independent_readmission(tiny,monkeypatch,qualified,winning):
    from examples.experiment_program.router import program,diagnostics
    from examples.experiment_program.validation import require_admission
    build=tiny/'build';build.mkdir();p=build/'CPU_ONLY';p.write_text('never executed');write(build/'identity.json',dict(files={str(p):sha(p)}))
    args=SimpleNamespace(out=tiny/'program',build=build,warmup=1,population=1,seed=19,deadline=1,capture=[])
    q=dict(qualified=qualified,status='QUALIFIED_DIAGNOSTIC' if qualified else 'UNQUALIFIED_DIAGNOSTIC',reasons=[] if qualified else ['perturbation exceeds500ns'],synthetic_cpu_only=True)
    monkeypatch.setattr(diagnostics,'qualify',lambda *a:q)
    result=program.run(args,router_child(tiny,build,winning=winning));assert result['admitted']==(qualified and winning)
    assert json.loads((args.out/'control/decision.json').read_text())['admitted']==winning
    assert len(result['samples'])==5
    c=dict(record='G01',build=str(build),decision=str(args.out/'decision.json'))
    if qualified and winning:assert require_admission(c)['record']=='G01'
    else:
        # Even an edited aggregate admitted flag cannot bypass raw qualification.
        result['admitted']=True;write(args.out/'decision.json',result)
        with pytest.raises(ValueError):require_admission(c)


@pytest.mark.parametrize('bad',['missing','nonboolean','malformed_raw'])
def test_rr2_missing_or_malformed_qualification_refuses(tiny,monkeypatch,bad):
    from examples.experiment_program.router import program,diagnostics
    from examples.experiment_program.validation import require_admission
    build=tiny/'build';build.mkdir();write(build/'identity.json',dict(files={'CPU_ONLY':'0'*64}))
    args=SimpleNamespace(out=tiny/'program',build=build,warmup=1,population=1,seed=19,deadline=1,capture=[])
    good=dict(qualified=True);monkeypatch.setattr(diagnostics,'qualify',lambda *a:good)
    program.run(args,router_child(tiny,build));p=args.out/'decision.json';result=json.loads(p.read_text())
    if bad=='malformed_raw':
        def broken(*a):raise ValueError('CPU native diagnostic missing/malformed')
        monkeypatch.setattr(diagnostics,'qualify',broken)
    else:
        badq={} if bad=='missing' else dict(qualified=1)
        result['diagnostic']=badq;write(p,result);monkeypatch.setattr(diagnostics,'qualify',lambda *a:badq)
    with pytest.raises(ValueError):require_admission(dict(record='G01',build=str(build),decision=str(p)))
    args.out=tiny/'second-program'
    with pytest.raises(ValueError):program.run(args,router_child(tiny,build))


@pytest.mark.parametrize('bodies,verdict',[
    ((10,10,20,10),'MIXED_CONTENTION_REQUIRES_ATTRIBUTION'),
    ((10,20,10,10),'MIXED_CONTENTION_REQUIRES_ATTRIBUTION'),
    ((10,10,20,20),'DOMINANT_NEW_ISOLATED'),
    ((10,10,10,10),'DOMINANT_NEW_ISOLATED')])
def test_rr3_body_qualification_uses_each_matched_control(tiny,monkeypatch,bodies,verdict):
    from examples.experiment_program.columns import runner
    monkeypatch.setattr(runner,'admit_native',lambda *a:({},b'CPU_ONLY'));monkeypatch.setattr(runner,'verify_native_probe',lambda *a:None)
    c=dict(record='G13',decode_position=0);build=tiny/'build';(build/'g13-ready-final').mkdir(parents=True)
    write(build/'g13-ready-final/plan.json',dict(types={'ready_r0':{'id':0},'up_r0':{'id':1}},descs=[dict(id=0,type=0,worker=0),dict(id=1,type=1,worker=0)],task_in=[[],[0]],ev_inst=[['up',[0],[0]]],static_queues={'0:0':[0,1]}))
    for order in ('request','both-up-first'):
        root=tiny/('G13-B2-'+order);root.mkdir();write(root/'report.json',dict(build=str(build)))
        for regime in ('simultaneous','archived'):
            d=root/f'p0-{regime}';d.mkdir();write(d/'exit.json',dict(exit=0));write(d/'capture.json',c);rows=[]
            for n in ('golden','control','candidate'):(d/(n+'.bin')).write_bytes(b'CPU_ONLY')
            for i in range(5):
                for arm in (('control','candidate') if i%2==0 else ('candidate','control')):
                    v=20 if arm=='control' else 18 if order=='both-up-first' else 12
                    rows.append(dict(pair=i,ordinal=i,arm=arm,host_ms=v,device_ms=v,request_ms=[dict(request=r,ready=0,done=v,latency=v) for r in range(2)]))
            (d/'stdout.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            for i,arm in enumerate(('control','candidate')):
                ticks=bodies[(0 if order=='both-up-first' else 2)+i]
                (d/(arm+'.trace.bin')).write_bytes(struct.pack('<8Q',1,2,2+ticks,0,60,61,61+ticks,0))
    result=runner.diagnose_g13(tiny,[c])[0];assert result['verdict']==verdict
    assert result['body_change_ratios']=={'both-up-first':[abs(bodies[1]-bodies[0])/bodies[0]]*2,'request':[abs(bodies[3]-bodies[2])/bodies[2]]*2}
    assert len(result['body_duration_ticks'])==8 and result['historical_attribution'].startswith('UNKNOWN')
