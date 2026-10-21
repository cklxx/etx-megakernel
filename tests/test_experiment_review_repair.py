from pathlib import Path
import json, tempfile,struct
from examples.experiment_program.routing import decode_records
from examples.experiment_program.validation_prepare import executable_build
from examples.experiment_program.protocol import sha
from examples.experiment_program.columns.program import run_program
# Small CPU protocol fixtures only, never native output evidence.
import pytest

def test_json_roundtrip():
 raw=bytearray(32*26*8*64)
 for i in range(32*26*8):struct.pack_into('<16I',raw,i*64,33+i//(26*8),1,*range(6),*([0x3e800000]*6),0,0)
 for b in (2,4):
  actual=decode_records([raw]*b,batch=b);assert actual==json.loads(json.dumps(actual))

def test_selected_executable(tmp_path):
 old=tmp_path/'old';new=tmp_path/'selected';old.mkdir();new.mkdir()
 for p in (old/'run',new/'run',old/'model.hsaco',new/'model.hsaco'):p.write_text('unchanged');p.chmod(0o700)
 (new/'build-identity.json').write_text(json.dumps(dict(host_link_exit=0,hashes={str(old/n):sha(old/n) for n in ('run','model.hsaco')})))
 assert executable_build(new,'deepseek')
 (new/'run').write_text('substituted')
 with pytest.raises(ValueError):executable_build(new,'deepseek')

def test_populated_program(tmp_path):
 root=tmp_path/'captures'
 for b in (2,4):
  for pos in (1024,1040):
   p=root/f'qkv-B{b}'/f'p{pos}'/'capture.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(record='G02',batch=b,role='qkv')))
 cfg=tmp_path/'program.json';cfg.write_text(json.dumps(dict(schema='etx.columns.program.v1',record='G02',build=str(tmp_path/'unused'),capture_root=str(root))))
 calls=[]
 result=run_program(cfg,'G02',None,execute=False,case_main=lambda argv:calls.append(argv) or 0)
 assert len(calls)==2 and result['cases'][1]['pending_winners']==['qkv-B2']

@pytest.mark.parametrize('role,K,rpt,values',[('dsl_moe_up',2048,32,64),('dsl_moe_down',768,64,0),('dsl_gemv_fold',2048,20,2048)])
def test_native_compact_intermediate_extent(tmp_path,role,K,rpt,values):
 from examples.experiment_program.row_ops.export_contract import intermediate_extent
 a=tmp_path/'args';raw=bytearray(160);struct.pack_into('<2i',raw,52,K,rpt);a.write_bytes(raw)
 pack=dict(role=role,count=3,arguments=dict(file=str(a)))
 assert intermediate_extent(pack)==3*values*4

@pytest.mark.parametrize('batch,x,role',[(b,x,r) for b in (2,4) for x in (6,7) for r in ('gate_up','down')])
def test_full_vs_selected_population(batch,x,role):
 from examples.experiment_program.columns.capture_pack import replay_rows,task_identity
 rows=[]
 for request in range(batch):
  for domain in range(8):
   for worker in range(37):
    h=[0]*64;h[2]=request;h[6]=2 if role=='gate_up' else 3;h[7]=domain;h[8]=worker;h[18]=(1<<64)-1 if domain in (6,7) else domain
    rows.append(dict(h=h))
 selected=replay_rows(rows,dict(record='G02',role=role,batch=batch,shared_xcd=x))
 assert len(rows)==batch*296 and len(selected)==batch*37
 assert len({tuple(task_identity(r,'fleet')) for r in selected})==len(selected)
 assert {r['h'][7] for r in selected}=={x}
 with pytest.raises(ValueError):replay_rows([r for r in rows if not(r['h'][2]==0 and r['h'][7]==x)],dict(record='G02',role=role,batch=batch,shared_xcd=x))

def test_untimed_telemetry(tmp_path):
 from examples.experiment_program.telemetry import capture,verify
 from types import SimpleNamespace
 phases=[]
 def stub(argv,stdout,**kwargs):
  assert kwargs['timeout']==10;stdout.write('CPU SYNTHETIC clocks/power\n');phases.append('telemetry');return SimpleNamespace(returncode=0)
 with capture(tmp_path/'telemetry',{'raw_run':'CPU_ONLY'},collector=stub):phases.append('native group')
 assert phases==['telemetry','native group','telemetry']
 assert verify(tmp_path/'telemetry',{'raw_run':'CPU_ONLY'})
 (tmp_path/'telemetry/after.log').write_text('edited')
 with pytest.raises(ValueError):verify(tmp_path/'telemetry',{'raw_run':'CPU_ONLY'})

@pytest.mark.parametrize('variant,role,K,rpt,values',[('g09','dsl_moe_up',2048,32,64),('g10_up','dsl_moe_up',2048,32,64),('g10_down','dsl_moe_down',768,64,0),('g11_probe','dsl_gemv_fold',2048,20,2048)])
def test_producer_compact_export_roundtrip(tmp_path,variant,role,K,rpt,values):
 from examples.experiment_program.row_ops.results import summarize,RECORD
 a=tmp_path/'args';raw=bytearray(160);struct.pack_into('<2i',raw,52,K,rpt);a.write_bytes(raw)
 golden=tmp_path/'golden';golden.write_bytes(b'original')
 pack=dict(count=1,role=role,arguments=dict(file=str(a)),references=[dict(file=str(golden),bytes=8)])
 prefix=str(tmp_path/'CPU_ONLY');arms=3 if variant=='g11_probe' else 2;population=16 if arms==3 else 5
 rows=[]
 for sample in range(population):
  from examples.experiment_program.row_ops.protocol import invocation_orders
  for arm in invocation_orders(arms,population)[sample]:
   rows.append(dict(sample=sample,arm=arm,exact=True,device_us=20,host_wall_us=21))
   Path(prefix+f'.s{sample}.a{arm}.r0.bin').write_bytes(b'original')
  if arms==3:Path(prefix+f'.s{sample}.a1.records').write_bytes(b''.join(RECORD.pack(1,2,3,4,5,6,7,0,w,0,0) for w in range(16)))
 for arm in range(arms):Path(prefix+f'.a{arm}.intermediates').write_bytes(bytes(values*4))
 Path(prefix+'.timing.jsonl').write_text('\n'.join(map(json.dumps,rows)))
 identity=dict(resources={n:[dict(kernel='row_dsl_gemv_fold',physical_signature={})] for n in ('g11_control','g11_probe','g11_empty')})
 assert summarize(prefix,pack,identity,variant)
 changed=Path(prefix+'.a1.intermediates');changed.write_bytes(bytes(values*4+1))
 with pytest.raises(ValueError,match='intermediate'):summarize(prefix,pack,identity,variant)
 if values:
  changed.write_bytes(bytes(values*4-1))
  with pytest.raises(ValueError,match='intermediate'):summarize(prefix,pack,identity,variant)
  changed.write_bytes(b'x'+bytes(values*4-1))
  with pytest.raises(ValueError,match='intermediate'):summarize(prefix,pack,identity,variant)

def test_row_coverage_schema_and_immutable_derived_recipe(tmp_path):
 from examples.experiment_program.row_ops.program import configuration,cases
 assert len(cases('G11'))==6 and {x['position'] for x in cases('G09')}=={0,1040}
 packs={};inputs={}
 for r in cases('G10'):
  packs[r['id']]={}
  for variant in ('g10_up','g10_down'):
   p=tmp_path/(r['id']+variant+'.json');p.write_text(json.dumps(dict(position=r['position'],layer=r['layer'])))
   packs[r['id']][variant]=str(p);inputs[str(p)]=sha(p)
 build=tmp_path/'build';diag=tmp_path/'diag'
 for d in (build,diag):
  d.mkdir();(d/'identity.json').write_text('{}');inputs[str(d/'identity.json')]=sha(d/'identity.json')
 path=tmp_path/'program';path.write_text(json.dumps(dict(schema='etx.row.program.v1',record='G10',packs=packs,inputs=inputs,build=str(build),diagnostic_build=str(diag))))
 assert len(configuration(path,'G10')[1])==2
 p.write_text('edited')
 with pytest.raises(ValueError):configuration(path,'G10')

def test_standalone_up_summary_cannot_unlock_down(tmp_path):
 from examples.experiment_program.row_ops.program import readmit
 fake=tmp_path/'fake.json';fake.write_text(json.dumps(dict(gpu_executed=True,exact=True,admitted=True)))
 # A fabricated summary does not even carry a raw evidence descriptor.
 with pytest.raises((ValueError,KeyError,OSError)):
  readmit(dict(record='G10',decision=str(fake),build=str(tmp_path/'absent'),variant='g10_up',pack='/absent',output_prefix='/absent'))

@pytest.mark.parametrize('position,nb',[(p,b) for p in (0,1040) for b in (1,2)])
def test_historical_native_pack_to_launch(bounded_pack_directory,position,nb):
 """Complete native-layout CPU zeros, not model captures or device evidence."""
 tmp_path=bounded_pack_directory
 from examples.experiment_program.columns.capture_pack import pack,ROW_BYTES,MAGIC
 from examples.experiment_program.columns.protocol import DAG,REVISIONS
 from examples.experiment_program.columns.runner import write_launch
 src=tmp_path/'CPU_ONLY_source.py';src.write_text('# synthetic protocol fixture, never executed\n');hsh=sha(src)
 weights=tmp_path/'weights';weights.touch();weights.write_bytes(b'')
 with weights.open('r+b') as f:f.truncate(5120*2048*2)
 log=tmp_path/'original.log';log.write_text('CPU SYNTHETIC ONLY\nCAPTURE COMPLETE\n'+'32/32 generated tokens match HF greedy\n'*2)
 def producer(columns):
  rows=[]
  for request in range(2 if columns==1 else 1):
   for block in range(256):
    row=bytearray(ROW_BYTES);h=[0]*64
    for k,v in {0:MAGIC,1:1,2:request,4:position,7:block//32,8:block%32,10:100+request*10+block,11:1000+request*10+block,12:2048,13:5120,14:2048,20:2,21:20*columns,22:16,23:1,31:100000,32:columns,35:123+request,36:block,40:block//32}.items():h[k]=v
    struct.pack_into('<64Q',row,0,*h)
    struct.pack_into('<'+'I'*(20*columns),row,16896,*[b*5120+block*20+i for b in range(columns) for i in range(20)])
    rows.append(row)
  raw=tmp_path/f'NB{columns}.bin';raw.write_bytes(b''.join(rows))
  seal=dict(schema='etx.columns.native-producer.v1',record='G03',family='dsl',batch=2,role='qkv',decode_position=position,native_task_count=len(rows),session_id='CPU-SYNTHETIC-SESSION',hardware=dict(path=str(src),sha256=hsh),source_path=str(src),source_files={str(src):hsh},input_files={str(weights):sha(weights)},build_files={str(src):hsh},raw_files={str(raw):sha(raw)},weights=str(weights),log=str(log),log_sha256=sha(log),full_greedy_tokens=32,per_request_greedy=[32,32],payload_upper_bound_bytes=24<<20)
  if columns==2:
   prior=producer(1);seal['readiness_producer']=dict(path=str(prior),sha256=sha(prior))
  p=tmp_path/f'NB{columns}.seal.json';p.write_text(json.dumps(seal));return p
 receipt=producer(nb);out=tmp_path/'pack';c=pack(receipt,out)
 cert=json.loads((out/'native-certificate.json').read_text())
 assert cert['native_task_count']==cert['selected_task_count']==(512 if nb==1 else 256)
 assert len({tuple(x) for x in cert['selected_task_ids']})==cert['selected_task_count']
 assert c['arrivals_ticks']==[0,10]
 build=tmp_path/'build';build.mkdir();dummy=build/'CPU_ONLY';dummy.write_text('not native code')
 (build/'SHA256SUMS').write_text(sha(dummy)+' CPU_ONLY\n')
 (build/'identity.json').write_text(json.dumps(dict(files={str(dummy):sha(dummy)},local_device_compile_complete=True,resources=dict(b5eaa67=[{'CPU_SYNTHETIC_ONLY':True}]))))
 (build/'sources.json').write_text(json.dumps(dict(fleet=dict(gemv_sha256=hsh),b5eaa67=dict(sha256=hsh),ff9ef92=dict(sha256=hsh))))
 argv=write_launch(c,build,tmp_path/'launch','archived')
 assert argv[1]=='--execute' and str(position) in str(c['decode_position'])
 assert (tmp_path/'launch/launch.txt').is_file()

@pytest.mark.parametrize('branch',['positive','miss','contention','incomplete'])
def test_actual_g13_four_cell_decision(tmp_path,monkeypatch,branch):
 from examples.experiment_program.columns import runner
 monkeypatch.setattr(runner,'admit_native',lambda *a:({},b'CPU-ONLY'));monkeypatch.setattr(runner,'verify_native_probe',lambda *a:None)
 c=dict(record='G13',decode_position=0);build=tmp_path/'build';(build/'g13-ready-final').mkdir(parents=True)
 (build/'g13-ready-final/plan.json').write_text(json.dumps(dict(types={'ready_r0':{'id':0},'up_r0':{'id':1}},descs=[dict(id=0,type=0,worker=0),dict(id=1,type=1,worker=0)],task_in=[[],[0]],ev_inst=[['up',[0],[0]]],static_queues={'0:0':[0,1]})))
 for order in ('request','both-up-first'):
  root=tmp_path/('G13-B2-'+order);root.mkdir();(root/'report.json').write_text(json.dumps(dict(build=str(build))))
  for regime in ('simultaneous','archived'):
   d=root/f'p0-{regime}';d.mkdir();(d/'exit.json').write_text('{"exit":0}');(d/'capture.json').write_text(json.dumps(c));rows=[]
   for name in ('golden','control','candidate'):(d/(name+'.bin')).write_bytes(b'CPU-ONLY')
   for i in range(5):
    for arm in (('control','candidate') if i%2==0 else ('candidate','control')):
     v=20 if arm=='control' else 18 if order=='both-up-first' else 19.5 if branch=='miss' else 12
     rows.append(dict(pair=i,ordinal=i,arm=arm,host_ms=v,device_ms=v,request_ms=[dict(request=r,ready=0,done=v,latency=v) for r in range(2)]))
   if branch=='incomplete' and order=='request' and regime=='simultaneous':rows.pop()
   (d/'stdout.jsonl').write_text('\n'.join(map(json.dumps,rows)))
   for arm in ('control','candidate'):
    ticks=20 if branch=='contention' and order=='request' and arm=='candidate' else 10
    (d/(arm+'.trace.bin')).write_bytes(struct.pack('<8Q',1,2,2+ticks,0,30,31,31+ticks,0))
 if branch=='incomplete':
  with pytest.raises(ValueError):runner.diagnose_g13(tmp_path,[c])
 else:
  r=runner.diagnose_g13(tmp_path,[c])[0]
  assert r['verdict']=={'positive':'DOMINANT_NEW_ISOLATED','miss':'PARTIAL_OR_UNKNOWN','contention':'MIXED_CONTENTION_REQUIRES_ATTRIBUTION'}[branch]
  assert len(r['body_duration_ticks'])==8 and r['historical_attribution'].startswith('UNKNOWN')


@pytest.fixture
def bounded_pack_directory(tmp_path):
 import tempfile
 # Synthetic payload lifetime is one test, not pytest's retained multi-run cache.
 with tempfile.TemporaryDirectory(prefix='CPU_SYNTHETIC_',dir=tmp_path) as d:
  yield Path(d)

@pytest.mark.parametrize('batch',[2,4])
def test_actual_g04_session_json_dependency(tmp_path,batch):
 from examples.experiment_program.routing import decode_records
 from examples.experiment_program.session import decision_for_dependencies
 from examples.experiment_program.routing.__main__ import ROLES
 raw=bytearray(32*26*8*64)
 for i in range(32*26*8):struct.pack_into('<16I',raw,i*64,33+i//(26*8),1,*range(6),*([0x3e800000]*6),0,0)
 files={}
 for i in range(batch):
  p=tmp_path/f'routes_r{i}.bin';p.write_bytes(raw);files[str(p)]=sha(p)
 identity=tmp_path/'CPU_ONLY_identity';identity.write_text('CPU SYNTHETIC ONLY ETX_PROGRAM_ROUTE_CAPTURE=1')
 roles={k:{str(identity):sha(identity)} for k in ROLES};cfg=tmp_path/'config.json';cfg.write_text(json.dumps(dict(batch=batch,roles=roles,command=[str(identity)],deadline_seconds=10)))
 log=tmp_path/'capture.log';log.write_text(f'G04 CAPTURE COMPLETE requests={batch} steps=32 routed_layers=26 replicas=8 slots=6 dense_layer=0\n'+''.join(f'BATCHREQ r={i} tokens=32/32 layers=27 warmup=32/32\n' for i in range(batch)))
 value=decode_records([raw]*batch,batch=batch);value.update(record='G04',complete=True,config_sha256=sha(cfg),roles=roles,raw_files=files,log_sha256=sha(log))
 p=tmp_path/'capture.json';stage=dict(command=['capture','--config',str(cfg)])
 def write(v):p.write_text(json.dumps(v))
 write(value);assert decision_for_dependencies('G04',stage,p)['complete']
 import copy
 for field in ('expert_groups','slot_ids','effective_batch_histogram','config_sha256'):
  bad=copy.deepcopy(value)
  if field=='slot_ids':bad[field][0]['mix_bits']^=1
  elif field=='effective_batch_histogram':bad[field][str(batch)]+=1
  elif field=='config_sha256':bad[field]='0'*64
  else:bad[field]+=1
  write(bad)
  with pytest.raises(ValueError):decision_for_dependencies('G04',stage,p)
 write(value);first=tmp_path/'routes_r0.bin';first.write_bytes(bytes(raw[:-1]))
 with pytest.raises(ValueError):decision_for_dependencies('G04',stage,p)

@pytest.mark.parametrize('won',[False,True])
def test_populated_column_execute_dependency_branch(tmp_path,won):
 from examples.experiment_program.columns.program import run_program
 root=tmp_path/'captures';calls=[]
 for b in (2,4):
  for pos in (1024,1040):
   p=root/f'qkv-B{b}'/f'p{pos}/capture.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(record='G02',batch=b,role='qkv')))
 cfg=tmp_path/'program.json';cfg.write_text(json.dumps(dict(schema='etx.columns.program.v1',record='G02',build=str(tmp_path/'build'),capture_root=str(root))))
 def child(argv):
  calls.append(argv);out=Path(argv[argv.index('--out')+1]);out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(dict(status='MICRO_PASS_BOTH_REGIMES' if won else 'NO_MODEL_ADMISSION',admitted=won)));return 0
 r=run_program(cfg,'G02',tmp_path/'output',execute=True,case_main=child)
 assert len(calls)==(2 if won else 1)
 assert r['cases'][1]['status']==('MICRO_PASS_BOTH_REGIMES' if won else 'SKIPPED_DEPENDENCY')

@pytest.mark.parametrize('batch,x,role',[(b,x,r) for b in (2,4) for x in (6,7) for r in ('gate_up','down')])
def test_full_native_shared_pack_admission(bounded_pack_directory,batch,x,role):
 from examples.experiment_program.columns.capture_pack import pack,ROW_BYTES,MAGIC
 from examples.experiment_program.columns.runner import admit_native
 tmp_path=bounded_pack_directory;src=tmp_path/'CPU_ONLY_source.py';src.write_text('# CPU synthetic source, no native execution\n');hsh=sha(src)
 weights=tmp_path/'weights';weights.write_bytes(bytes(37*2048*(4 if role=='gate_up' else 2)))
 log=tmp_path/'capture.log';log.write_text('CPU SYNTHETIC ONLY\nCAPTURE COMPLETE\n'+''.join(f'BATCHREQ r={r} tokens=32/32 layers=27 warmup=32/32\n' for r in range(batch)))
 raw=bytearray(batch*296*ROW_BYTES)
 for request in range(batch):
  for domain in range(8):
   for worker in range(37):
    offset=(request*296+domain*37+worker)*ROW_BYTES;h=[0]*64
    for k,v in {0:MAGIC,1:1,2:request,4:1040,6:2 if role=='gate_up' else 3,7:domain,8:worker,10:100+10*request,11:1000+10*request,12:2048,13:37,14:2048,18:(1<<64)-1 if domain>=6 else domain,20:4 if role=='gate_up' else 2,21:1,22:4,23:1,31:100000,40:domain}.items():h[k]=v
    struct.pack_into('<64Q',raw,offset,*h);struct.pack_into('<I',raw,offset+16896,worker)
 rp=tmp_path/'rows';rp.write_bytes(raw)
 seal=dict(schema='etx.columns.native-producer.v1',record='G02',family='fleet',batch=batch,role=role,shared_xcd=x,decode_position=1040,native_task_count=batch*296,session_id='CPU-SYNTHETIC-SESSION',hardware=dict(path=str(src),sha256=hsh),source_path=str(src),source_files={str(src):hsh},input_files={str(weights):sha(weights)},build_files={str(src):hsh},raw_files={str(rp):sha(rp)},weights=str(weights),log=str(log),log_sha256=sha(log),full_greedy_tokens=32,per_request_greedy=[32]*batch,payload_upper_bound_bytes=1<<20)
 sp=tmp_path/'seal';sp.write_text(json.dumps(seal));out=tmp_path/'pack';c=pack(sp,out)
 cert=out/'native-certificate.json';saved=json.loads(cert.read_text())
 assert saved['native_task_count']==batch*296 and saved['selected_task_count']==batch*37
 build=tmp_path/'build';build.mkdir();dummy=build/'CPU_ONLY';dummy.write_text('not a native module')
 (build/'SHA256SUMS').write_text(sha(dummy)+' CPU_ONLY\n');(build/'identity.json').write_text(json.dumps(dict(files={str(dummy):sha(dummy)},local_device_compile_complete=True)))
 (build/'sources.json').write_text(json.dumps(dict(fleet=dict(gemv_sha256=hsh),b5eaa67=dict(sha256=hsh),ff9ef92=dict(sha256=hsh))))
 assert admit_native(c,build)
 for key,value in [('selected_task_count',1),('selected_role','qkv'),('selected_native_xcd',5),('selected_task_ids',saved['selected_task_ids'][:-1])]:
  changed=dict(saved,**{key:value});cert.write_text(json.dumps(changed))
  with pytest.raises(ValueError,match='selected'):admit_native(c,build)
 cert.write_text(json.dumps(saved));rp.write_bytes(raw[:-ROW_BYTES])
 with pytest.raises(ValueError):admit_native(c,build)

def test_old_editable_up_summary_is_refused_before_command(tmp_path,monkeypatch):
 from examples.experiment_program.row_ops import __main__ as cli
 import sys
 build=tmp_path/'build';build.mkdir();(build/'identity.json').write_text('{}')
 pack=tmp_path/'down-pack';pack.write_text(json.dumps(dict(position=0,layer=24)))
 admission=tmp_path/'editable';admission.write_text(json.dumps(dict(gpu_executed=True,arms=['control','g10_up'],position=0,layer=24,build_identity_sha256=sha(build/'identity.json'),verdict=dict(exact=True,measurements=dict(device_us=dict(savings_us=[1]*5,control_us=[20]*5))))))
 monkeypatch.setattr(cli,'validate_build',lambda *a:None)
 monkeypatch.setattr(cli,'config',lambda *a,**k:('not executed',{}))
 monkeypatch.setattr(sys,'argv',['row','execute','--pack',str(pack),'--build',str(build),'--variant','g10_down','--output',str(tmp_path/'unused'),'--dry-run','--up-admission',str(admission)])
 with pytest.raises(SystemExit) as e:cli.main()
 assert e.value.code==2

@pytest.mark.parametrize('branch',['winner','miss','failure','unqualified'])
def test_finite_row_program_conditional_down(tmp_path,monkeypatch,branch):
 from examples.experiment_program.row_ops import program
 build=tmp_path/'build';build.mkdir();diag=tmp_path/'diag';diag.mkdir();files={}
 for variant in ('g10_up','g10_down'):
  for mode in ('control','probe','empty'):
   p=diag/(variant+'_'+mode+'.hsaco');p.write_text('CPU protocol stub, not ELF');files[str(p)]=sha(p)
 p=diag/'replay';p.write_text('CPU protocol stub, never executed');files[str(p)]=sha(p)
 (diag/'identity.json').write_text(json.dumps(dict(files=files)))
 packs={};inputs={}
 for r in program.cases('G10'):
  packs[r['id']]={}
  for v in ('g10_up','g10_down'):
   p=tmp_path/(r['id']+v+'.json');p.write_text(json.dumps(dict(position=r['position'],layer=r['layer'],role='dsl_moe_up' if v=='g10_up' else 'dsl_moe_down')));packs[r['id']][v]=str(p);inputs[str(p)]=sha(p)
 (build/'identity.json').write_text('{}');inputs[str(build/'identity.json')]=sha(build/'identity.json');inputs[str(diag/'identity.json')]=sha(diag/'identity.json')
 cfg=tmp_path/'program';cfg.write_text(json.dumps(dict(schema='etx.row.program.v1',record='G10',build=str(build),diagnostic_build=str(diag),packs=packs,inputs=inputs)))
 calls=[];admissions=[]
 monkeypatch.setattr(program,'readmit',lambda d,**k:admissions.append(d) or d)
 def cpu_child(argv):
  calls.append(argv)
  if branch=='failure':return 2
  out=argv[argv.index('--output')+1]
  verdict=dict(qualified=branch!='unqualified') if '--diagnostic' in argv else dict(status='MICRO_MISS' if branch=='miss' else 'MICRO_PASS')
  Path(out+'.result.json').write_text(json.dumps(dict(verdict=verdict,synthetic_cpu_only=True)));return 0
 if branch=='failure':
  with pytest.raises(ValueError):program.run(cfg,'G10',tmp_path/'output',execute=True,case_main=cpu_child)
  assert json.loads((tmp_path/'output/program-error.json').read_text())['status']=='NOT_COMPLETE'
 else:
  r=program.run(cfg,'G10',tmp_path/'output',execute=True,case_main=cpu_child)
  assert r['admitted']==(branch=='winner')
  downs=[c for c in calls if c[c.index('--variant')+1]=='g10_down']
  assert len(downs)==(4 if branch=='winner' else 0)
  if branch=='winner':
   deps=json.loads((tmp_path/'output/up-evidence.json').read_text());assert {json.loads(Path(d['pack']).read_text())['position'] for d in deps['positions']}=={0,1040}
   for cmd in downs:
    if '--diagnostic' in cmd:assert cmd[cmd.index('--up-build')+1]==str(build)
  else:assert r['down_status']=='CLOSED_UP_MISS'

def test_same_session_derived_pack_keeps_presealed_base(tmp_path,monkeypatch):
 from examples.experiment_program.row_ops.program import derive
 from examples.experiment_program import validation_trace as trace
 p=tmp_path/'base';p.write_text(json.dumps(dict(position=1040,layer=24)));before=sha(p)
 monkeypatch.setattr(trace,'verify',lambda m:dict(session_id='CPU-SESSION-A',position=1040))
 monkeypatch.setattr(trace,'bind_g11',lambda p,m:dict(CPU_SYNTHETIC_ONLY=True))
 meta=dict(proof_files={'CPU-stub':'0'*64})
 result=derive(p,meta,tmp_path/'derived',expected_session='CPU-SESSION-A')
 assert sha(p)==before and json.loads(result.read_text())['derived_from'][str(p)]==before
 with pytest.raises(ValueError):derive(p,meta,tmp_path/'wrong',expected_session='CPU-SESSION-B')
 with pytest.raises(FileExistsError):derive(p,meta,result,expected_session='CPU-SESSION-A')

def test_native_export_header_is_generated_from_single_abi_contract():
 from examples.experiment_program.row_ops.export_contract import cpp
 from examples.experiment_program.row_ops import export_contract
 assert Path(export_contract.__file__).with_name('row_export_contract.h').read_text()==cpp()

@pytest.mark.parametrize('fault',['baseline','candidate','missing_log','changed_output','failed_exit'])
def test_original_producer_proof_refuses_broken_execution(tmp_path,fault):
 from examples.experiment_program.row_ops.producer import verify,BASE
 output=tmp_path/'reference';output.write_bytes(b'CPU-only original placeholder')
 log=tmp_path/'original.log';log.write_text('CPU ONLY CHECK OK: 0 launch(es) differ\n')
 exit=tmp_path/'original.exit';exit.write_text('0\n')
 c=dict(schema='etx.row.original-producer.v1',baseline_revision=BASE,producer_kind='unchanged-original-checker',exit=0,exit_file=str(exit),check_log=str(log),files={str(log):sha(log),str(exit):sha(exit)},inputs={str(output):sha(output)},exports={str(output):sha(output)})
 if fault=='baseline':c['baseline_revision']='wrong-baseline'
 if fault=='candidate':c['producer_kind']='candidate'
 if fault=='missing_log':log.unlink()
 if fault=='changed_output':output.write_bytes(b'changed')
 if fault=='failed_exit':exit.write_text('1\n')
 p=tmp_path/'producer';p.write_text(json.dumps(c))
 with pytest.raises((ValueError,OSError)):
  verify(dict(path=str(p),sha256=sha(p)),dict(position=0,layer=24,role='dsl_moe_up',references=[dict(file=str(output))],buffers=[]))

@pytest.mark.parametrize('fault',['none','perturbation','resource','missing_marker','changed_output','nan'])
def test_actual_sparse_row_qualification(bounded_pack_directory,monkeypatch,fault):
 from examples.experiment_program.row_ops import diagnostics as diag,elf
 from examples.experiment_program.row_ops.results import RECORD
 tmp_path=bounded_pack_directory;variant='g09';a=tmp_path/'args';args=bytearray(160);struct.pack_into('<2i',args,52,2048,32);a.write_bytes(args)
 golden=tmp_path/'golden';golden.write_bytes(b'CPU_ONLY_ORIGINAL')
 pack=dict(role='dsl_moe_up',count=1,arguments=dict(file=str(a)),references=[dict(file=str(golden))]);prefix=str(tmp_path/'CPU_SYNTHETIC')
 items=[]
 for sample in range(128):
  from examples.experiment_program.row_ops.protocol import invocation_orders
  for arm in invocation_orders(3,128)[sample]:
   value=10+arm*.1
   if fault=='perturbation' and arm==1:value=10.6
   if fault=='nan' and arm==1:value=float('nan')
   items.append(dict(sample=sample,arm=arm,exact=True,device_us=value,host_wall_us=value+1))
   Path(prefix+f'.s{sample}.a{arm}.r0.bin').write_bytes(golden.read_bytes())
  rows=[RECORD.pack(1,0 if fault=='missing_marker' and w==sample%16 else 2,3,0,0,5,6,0,w,0,32+w) for w in range(16)]
  Path(prefix+f'.s{sample}.a1.records').write_bytes(b''.join(rows))
 for arm in range(3):Path(prefix+f'.a{arm}.intermediates').write_bytes(bytes(256))
 Path(prefix+'.timing.jsonl').write_text('\n'.join(map(json.dumps,items)))
 if fault=='changed_output':Path(prefix+'.s0.a1.r0.bin').write_bytes(b'changed')
 identity=dict(_directory=str(tmp_path),resources={variant+'_'+mode:[dict(kernel='row_dsl_moe_up',agpr=0)] for mode in ('control','probe','empty')})
 monkeypatch.setattr(elf,'descriptors',lambda path:{'row_dsl_moe_up':dict(allocation=2 if fault=='resource' and 'probe' in path.name else 1)})
 monkeypatch.setattr(elf,'allocation_signature',lambda d,a:d)
 if fault in ('missing_marker','changed_output','nan'):
  with pytest.raises(ValueError):diag.summarize(prefix,pack,identity,variant)
 else:
  r=diag.summarize(prefix,pack,identity,variant)
  assert r['qualified']==(fault=='none') and len(r['observations'])==128
  assert {x['phase'] for x in r['observations']}==set(diag.PHASES)

@pytest.mark.parametrize('failure',['exit','timeout'])
def test_telemetry_failure_is_retained_and_stops_trial(tmp_path,failure):
 from examples.experiment_program.telemetry import capture,verify
 import subprocess
 from types import SimpleNamespace
 def stub(argv,**kwargs):
  if failure=='timeout':raise subprocess.TimeoutExpired(argv,10)
  return SimpleNamespace(returncode=3)
 with pytest.raises(ValueError):
  with capture(tmp_path/'telemetry',dict(CPU_SYNTHETIC=True),collector=stub):pytest.fail('failed collection must precede any native work')
 r=json.loads((tmp_path/'telemetry/receipt.json').read_text());assert len(r['observations'])==2 and all(x['exit']!=0 for x in r['observations'])
 with pytest.raises(ValueError):verify(tmp_path/'telemetry',dict(CPU_SYNTHETIC=True))

@pytest.mark.parametrize('fault',['none','missing_wave','owner','epoch','physical','hwid','reader'])
def test_router_native_landmark_identity_population(fault):
 from examples.experiment_program.router.diagnostics import validate_landmarks
 rows=[]
 for task in range(264):
  for wave in range(16):
   domain=task//32 if task<256 else task-256
   row=[1,2,3,4 if task==0 else 0,5 if task==0 else 0,6 if task==0 else 0,7,8,domain,domain,0 if task>=256 else task%32,55,123+wave]
   rows.append(row)
 if fault=='missing_wave':rows[15][0]=0
 if fault=='owner':rows[0][3]=0
 if fault=='epoch':rows[20][11]=54
 if fault=='physical':rows[40][8]=15
 if fault=='hwid':rows[20][12]=1<<32
 if fault=='reader':rows[-1][6]=0
 raw=b''.join(struct.pack('<13Q',*r) for r in rows)
 if fault=='none':assert validate_landmarks(raw,epoch=55,arm=1,winner=0)['tasks']==256
 else:
  with pytest.raises(ValueError):validate_landmarks(raw,epoch=55,arm=1,winner=0)

def test_publication_resource_terminal_precedes_any_device_or_capture(tmp_path,monkeypatch):
 from examples.experiment_program import stage,session,publication
 proof=tmp_path/'CPU_ONLY_publication_identity';proof.write_text('CPU SYNTHETIC gate fixture, not native evidence')
 cfg=tmp_path/'session-config';out=tmp_path/'session'
 stage.prepare(dict(staging_root=str(tmp_path/'staged'),records={'G11':dict(publication_preparation=str(proof))}),cfg,out)
 gate=dict(status='CLOSED_NATIVE_DIAGNOSTIC_RESOURCE',admitted=False,synthetic_cpu_only=True)
 monkeypatch.setattr(publication,'admission',lambda path:gate)
 monkeypatch.setattr(session,'run_logged',lambda *a,**k:pytest.fail('closed publication gate may not launch a child'))
 assert session.main(['preflight','--config',str(cfg),'--out',str(out),'--record','G11'])==0
 r=json.loads((out/'session.json').read_text())
 assert r['stages']['G11']==dict(status='CLOSED_LOCAL_DIAGNOSTIC',evidence=gate)

def test_copied_qwen_selected_import_metadata_is_bound(tmp_path):
 old=tmp_path/'old';new=tmp_path/'new'
 for d in (old,new):
  (d/'imp').mkdir(parents=True)
  for name in ('run','model.hsaco'):(d/name).write_text('CPU protocol placeholder, never executed')
  (d/'run').chmod(0o700)
  (d/'imp/imports.json').write_text(json.dumps({n:{} for n in ('attn_mfma16_g8','attn_reduce','rms_norm_2d')}))
 original=old/'original.o';original.write_text('CPU placeholder')
 identity=dict(host_link_exit=0,check_enabled=True,hashes={str(old/n):sha(old/n) for n in ('run','model.hsaco')},metadata_files={str(old/'imp/imports.json'):sha(old/'imp/imports.json')},original_reference_objects={str(original):sha(original)})
 (new/'build-identity.json').write_text(json.dumps(identity))
 assert executable_build(new,'qwen')
 (new/'imp/imports.json').write_text('{"wrong-selected-metadata":{}}')
 with pytest.raises(ValueError,match='selected artifact'):executable_build(new,'qwen')

@pytest.mark.parametrize('relocated',[False,True])
@pytest.mark.parametrize('fault',['none','run_bytes','module_bytes','missing_run','missing_module','ambiguous_identity'])
def test_selected_executable_and_module_under_copy_or_relocation(tmp_path,monkeypatch,relocated,fault):
 old=tmp_path/'origin';selected=tmp_path/'selected';old.mkdir();selected.mkdir()
 for d in (old,selected):
  for name in ('run','model.hsaco'):(d/name).write_bytes(b'CPU fixture, no binary execution')
  (d/'run').chmod(0o700)
 identity=dict(host_link_exit=0,hashes={str(old/n):sha(old/n) for n in ('run','model.hsaco')})
 if fault=='ambiguous_identity':identity['hashes']['/different/origin/run']=sha(old/'run')
 (selected/'build-identity.json').write_text(json.dumps(identity))
 if fault=='run_bytes':(selected/'run').write_bytes(b'substituted selected executable')
 if fault=='module_bytes':(selected/'model.hsaco').write_bytes(b'substituted selected module')
 if fault=='missing_run':(selected/'run').unlink()
 if fault=='missing_module':(selected/'model.hsaco').unlink()
 if relocated:
  mapping=tmp_path/'relocations.json';mapping.write_text(json.dumps(dict(schema='etx.path-map.v1',paths={str(old):str(selected)})));monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping));directory=old
 else:directory=selected
 if fault=='none':assert executable_build(directory,'deepseek')
 else:
  with pytest.raises(ValueError):executable_build(directory,'deepseek')

def test_issue_guard_refresh_preserves_nested_hashed_artifacts(tmp_path,monkeypatch):
 from examples.experiment_program import review_guard_refresh as refresh
 source=tmp_path/'native';source.mkdir();(source/'candidate').mkdir()
 nested=source/'candidate/body.hip';nested.write_text('// CPU fixture: original nested native source')
 (source/'identity.json').write_text(json.dumps(dict(hashes={'candidate/body.hip':sha(nested)},dependency_files={},source_hashes={})))
 monkeypatch.setattr(refresh,'unchanged_generators',lambda:dict(cpu_fixture=True))
 out=tmp_path/'refreshed';value=refresh.refresh_issue(source,out)
 assert sha(out/'candidate/body.hip')==value['hashes']['candidate/body.hip']==sha(nested)
