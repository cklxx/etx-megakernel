"""Tiny CPU-only boundary fixtures. None is a GPU observation/certificate."""
import copy,json,struct,subprocess
from pathlib import Path
import pytest
from examples.experiment_program import publication_subgraph as sg,publication_run as run,publication_capture as capture
W=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (W/'build/publication-gap-subgraph4/identity.json').is_file(), reason="requires retained native build artifacts; not a portable CPU fixture")
def test_compiler_spill_counters_are_evidence_not_physical_equality():
    old=json.loads((W/'build/publication-gap-subgraph4/identity.json').read_text())
    profiles={m:next(r for r in old['resources'][m] if r['kernel']=='etx_megakernel_d0') for m in sg.MODES}
    physical={m:dict(old['physical_resources']['probe']['etx_megakernel_d0']) for m in sg.MODES}
    assert [profiles[m]['sgpr_spills'] for m in sg.MODES]==[14,46,44]
    assert sg.resource_gate(profiles,physical)['resources_equal']
    physical['empty']['rsrc1']+=1
    assert not sg.resource_gate(profiles,physical)['resources_equal']


def test_actual_original_capture_hook_is_complete_not_row_zero():
    original=subprocess.check_output(['git','show',capture.BASE+':examples/vllm_llm/host.hip'],cwd=W,text=True)
    text=capture.checking_source(original)
    assert text.count('launch_imported(x.n + 1);')==original.count('launch_imported(x.n + 1);')
    assert 'rr*x.stride' in text and 'down?rr*32:0,down?32:256,256' in text
    assert 'for(auto& capture:g11_captures)capture.after();' in text
    with pytest.raises(ValueError,match='original DSL capture'):capture.checking_source(original.replace('        launch_imported(x.n + 1);',''))
    with pytest.raises(ValueError,match='not an original'):capture.verify_producer(dict(schema='candidate',baseline_revision=capture.BASE,exit=0),1,0)


def raw_fixture(tmp_path,selector=0,arm=1,coverage=False):
    # Synthetic 512-task plan: parser coverage, not a native placement receipt.
    descs=[dict(worker=t%304,domain=(t%304)//38) for t in range(512)]
    plan=dict(descs=descs,static_queues={str(w):[t for t in range(512) if descs[t]['worker']==w] for w in range(304)})
    words=[run.MAGIC,1,1,selector,arm,100000,int(coverage)];legacy=[];wave=[]
    for t,d in enumerate(plan['descs']):
        start=10+t*20 if t<256 else 7000+(t-256)*20
        legacy.append([start,start+1,start+18,d['worker']])
        for w in range(16):wave.extend([start+15,start+16,(d['domain']<<32)|((d['worker']%38)<<8)|w])
    words += [v for row in legacy for v in row];words += [0 if t<256 else legacy[t][0]+1 for t in range(512)];words += [6000,6010,255];words += wave
    phase=[0]*400;selected=selector%16;target=256+selector//16;start=legacy[target][1];phase[selected*25]=5
    for i in range(5):
        if selected>=4 and i<4:continue
        off=selected*25+1+i*3;phase[off:off+3]=[start+2+i*2,start+3+i*2,(plan['descs'][target]['domain']<<32)|((plan['descs'][target]['worker']%38)<<8)|selected]
    words += phase;slots=[38]*8+[0]*(run.SLOT_WORDS-8);counts=[1]*512
    path=tmp_path/'CPU_ONLY.raw';path.write_bytes(struct.pack('<'+str(len(words))+'Q',*words)+struct.pack('<'+str(len(slots))+'i',*slots)+struct.pack('<512i',*counts));return path,plan


@pytest.mark.parametrize('selector',[0,4095])
def test_real_typed_reader_covers_final_task_and_inactive_operand_waves(tmp_path,selector):
    path,plan=raw_fixture(tmp_path,selector)
    r=run.read_raw(path,plan,selector,1)
    assert r['consumer']==256+selector//16 and r['wave']==selector%16 and r['final_consumer']==511
    assert (r['phase_us']['slot_read'] is None)==(selector%16>=4)
    assert r['publication_bounds']==[6000,6010]


@pytest.mark.parametrize('fault',['truncated','last_wave','owner','placement','execution','phase','selector'])
def test_real_raw_rejects_boundary_faults(tmp_path,fault):
    p,plan=raw_fixture(tmp_path);data=bytearray(p.read_bytes())
    def word(index,value):struct.pack_into('<Q',data,index*8,value)
    if fault=='truncated':data.pop()
    if fault=='owner':word(7+512*4+512+2,256)
    if fault=='last_wave':word(7+512*4+512+3+(512*16-1)*3,0)
    if fault=='placement':word(7+512*4+512+3+(512*16-1)*3+2,0)
    if fault=='execution':struct.pack_into('<i',data,len(data)-4,0)
    if fault=='phase':word(7+512*4+512+3+512*16*3+14,0)
    if fault=='selector':word(3,1)
    p.write_bytes(data)
    with pytest.raises(ValueError):run.read_raw(p,plan,0,1)


def test_no_placeholder_interface_can_admit(tmp_path):
    with pytest.raises((KeyError,ValueError)):run.verify_interface(dict(interface=dict(sources=run.interface_sources(),files={},host_layers=[1,24,47],commands=[],completed=[])))


def test_missing_or_duplicate_case_cannot_stage_as_prepared(tmp_path):
    p=tmp_path/'recipe.json';p.write_text(json.dumps(dict(schema='etx.g11.program.v1',inputs={},publication_preparation=str(tmp_path/'missing'),packs={})))
    with pytest.raises((ValueError,OSError)):run.configuration(p)

def captured_cpu_fixture(tmp_path,monkeypatch):
    """Stub only the unavailable original native checker authority, not merging."""
    monkeypatch.setattr(capture,'verify_producer',lambda c,l,p:None)
    from examples.experiment_program.protocol import sha
    root=tmp_path/'CPU_ONLY_ORIGINAL_BOUNDARY';root.mkdir();exports={};inputs={}
    datafiles={}
    def buffer(address,n):
        if address not in datafiles:
            f=root/f'pointer-{address}.bin';f.write_bytes(bytes([address%251])*n);datafiles[address]=f;inputs[str(f)]=sha(f)
        return dict(file=str(datafiles[address]),offset=0,bytes=n)
    for i,name in enumerate([f'down{i}' for i in range(8)]+['fold0']):
        d=root/name;d.mkdir();block=bytearray(160)
        offsets=(0,16,64,72,80,88) if i==8 else (0,8,16,112)
        addresses=(20000,30000,40000,50000,60000,70000) if i==8 else (10000,80000,40000,90000+i*100)
        buffers=[];pointers=[]
        for off,addr in zip(offsets,addresses):struct.pack_into('<Q',block,off,addr);pointers.append(dict(offset=off,buffer=len(buffers),add=0));buffers.append(buffer(addr,32))
        args=d/'arguments.bin';args.write_bytes(block);exports[str(args)]=sha(args)
        refs=[]
        for off in ((16,80) if i==8 else (16,)):
            ptr=next(x for x in pointers if x['offset']==off);r=d/f'ref-{off}.bin';r.write_bytes(b'R'*32);exports[str(r)]=sha(r);refs.append(dict(buffer=ptr['buffer'],offset=0,file=str(r),bytes=32))
        p=dict(position=0,layer=1 if i==8 else 0,first=0 if i==8 else i*32,count=256 if i==8 else 32,total_tasks=256,arguments=dict(file=str(args)),buffers=buffers,pointers=pointers,references=refs)
        meta=d/'capture.json';meta.write_text(json.dumps(p));exports[str(meta)]=sha(meta)
    certificate=tmp_path/'CPU_ONLY-producer.json';certificate.write_text(json.dumps(dict(layer=1,position=0,capture_directory=str(root),exports=exports,inputs=inputs,build='CPU_ONLY')))
    return certificate


def test_real_capture_assembly_and_relocation_aliases_then_mutation_refusal(tmp_path,monkeypatch):
    certificate=captured_cpu_fixture(tmp_path,monkeypatch);p=capture.assemble(certificate,tmp_path/'CPU_ONLY-pack.json')
    assert len(p['arguments'])==9
    mapping={(x['block'],x['offset']):x for x in p['pointers']}
    assert all(mapping[i,16]['buffer']==mapping[8,64]['buffer'] for i in range(8))
    mutated=copy.deepcopy(p);mutated['pointers'][0]['add']=1
    with pytest.raises(ValueError,match='differs from original'):capture.validate(mutated)
    mutated=copy.deepcopy(p);mutated['references'].pop()
    with pytest.raises(ValueError,match='differs from original'):capture.validate(mutated)
    # Earliest pre-down slot snapshot survives, not the post-down fold snapshot.
    assert p['buffers'][mapping[8,64]['buffer']]['file'].endswith('pointer-40000.bin')


def test_real_native_config_preserves_live_operands_and_all_blocks(tmp_path,monkeypatch):
    certificate=captured_cpu_fixture(tmp_path,monkeypatch);p=capture.assemble(certificate,tmp_path/'CPU_ONLY-pack.json')
    identity=tmp_path/'identity.json';identity.write_text(json.dumps(dict(interface=dict(directory=str(tmp_path)))))
    monkeypatch.setattr(sg,'admission',lambda p:dict(admitted=True)) # unavailable native build boundary only
    text,argv=run.config(p,identity,tmp_path/'output',600)
    assert argv==[str(tmp_path/'host-L1'),str(tmp_path/'output.config')]
    assert '4096 600' in text and all(x['file'] in text for x in p['arguments'])
    assert str(tmp_path/'probe.hsaco') in text
    assert len(text.splitlines())==1+3+9+len(p['buffers'])+len(p['pointers'])+len(p['references'])


def test_integrated_recipe_executes_six_cases_and_rechecks_raw_verdict(tmp_path,monkeypatch):
    """Stub the actual HIP child and G12 authority only; retain orchestration."""
    from examples.experiment_program.protocol import sha
    identity=tmp_path/'identity.json';identity.write_text('{}');packs={};inputs={str(identity):sha(identity)}
    for l in sg.LAYERS:
        for pos in (0,1040):
            p=tmp_path/f'L{l}-p{pos}.json';p.write_text(json.dumps(dict(layer=l,position=pos)));packs[p.stem]=str(p);inputs[str(p)]=sha(p)
    recipe=tmp_path/'program.json';recipe.write_text(json.dumps(dict(schema='etx.g11.program.v1',publication_preparation=str(identity),packs=packs,inputs=inputs)))
    traces=[];trace_root=tmp_path/'G12';trace_root.mkdir()
    for pos in (0,1040):
        p=tmp_path/f'trace{pos}.json';p.write_text(json.dumps(dict(position=pos,session_directory=str(trace_root))));traces.append(dict(proof=str(p)))
    (trace_root/'session-config.json').write_text(json.dumps(dict(session_id='CPU_ONLY',fold_decision_active=True,fold_traces=traces)))
    from examples.experiment_program import validation_run
    monkeypatch.setattr(validation_run,'transcript',lambda *a:None)
    monkeypatch.setattr(run,'config',lambda p,i,o,d:('real-config-placeholder-at-HIP-boundary',['CPU_ONLY_HIP',str(o)]))
    monkeypatch.setattr(run,'bind_trace',lambda p,m,s:dict(proof=m,session_id=s))
    # Supply the fields used by the real write-budget calculation.
    for path in packs.values():
        p=json.loads(Path(path).read_text());p['references']=[];Path(path).write_text(json.dumps(p));inputs[path]=sha(path)
    c=json.loads(recipe.read_text());c['inputs']=inputs;recipe.write_text(json.dumps(c))
    monkeypatch.setattr(run.shutil,'disk_usage',lambda p:type('D',(),dict(free=1<<40))())
    children=[]
    def child(argv,**kwargs):children.append(argv);return type('Result',(),dict(returncode=0))()
    monkeypatch.setattr(run.subprocess,'run',child)
    reread=[]
    def summary(prefix,p,i):reread.append(str(prefix));return dict(qualified=True,status='CPU_ONLY')
    monkeypatch.setattr(run,'summarize',summary)
    out=tmp_path/'native';r=run.program(recipe,out,True,trace_root)
    assert len(children)==len(reread)==len(r['cases'])==6 and r['qualified']
    stage=dict(command=['python','program','--config',str(recipe),'--trace-session',str(trace_root)])
    assert run.verify_report(out/'program-report.json',stage)['qualified']
    assert len(reread)==12

@pytest.mark.skipif(not (W/'build/publication-completion-final/identity-final.json').is_file(), reason="requires retained native build artifacts; not a portable CPU fixture")
def test_integrated_staging_uses_real_parser_and_fixed_module_guard(tmp_path,monkeypatch):
    from examples.experiment_program import stage,session
    identity=W/'build/publication-completion-final/identity-final.json';recipe=tmp_path/'program.json';recipe.write_text(json.dumps(dict(schema='etx.g11.program.v1',publication_preparation=str(identity),inputs={},packs={})))
    cfg=tmp_path/'session.json';out=tmp_path/'out'
    c=stage.prepare(dict(staging_root=str(tmp_path),records={'G11':dict(build=str(identity.parent),program=str(recipe),publication_preparation=str(identity))}),cfg,out)
    argv=c['records']['G11']['preflight'];assert argv[2]=='examples.experiment_program.publication_run'
    observed=[]
    monkeypatch.setattr(run,'program',lambda path,out,execute,trace_session:observed.append((path,out,execute,trace_session)) or dict(CPU_ONLY=True))
    assert run.main(argv[3:])==0 and len(observed)==1 and observed[0][2] is False
    bad=copy.deepcopy(c);bad['records']['G09']['preflight']=list(argv);p=tmp_path/'bad.json';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='exact prepared module'):session.configuration(p)
