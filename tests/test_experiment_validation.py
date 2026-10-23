import os
import pytest
from examples.experiment_program.validation import model_rule,system_rule,require_admission


def test_qwen_model_wall_rule():
    c=[4_100_000]*5;x=[4_060_000,4_059_900,4_060_100,4_060_000,4_060_000]
    assert model_rule(c,x)['keep']
    assert not model_rule(c,[4_080_000]*5)['keep']
    assert not model_rule(c,[4_060_000]*4+[4_100_000])['keep']
    with pytest.raises(ValueError):model_rule(c[:-1],x)


def test_system_step_and_request_latency_separate():
    v=[6_000_000]*5;e=[5_300_000]*5
    r=system_rule(e,v,[6_500_000]*5,v,batch=4)
    assert r['throughput_verdict']=='WIN' and r['request_latency_verdict']=='LOSS'
    assert r['pursuit_target_pass']
    assert r['candidate_tokens_per_second']==pytest.approx(4e9/5_300_000)


def test_missing_or_forged_winner_refused(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):require_admission(dict(record='G01',decision=str(tmp_path/'missing')))
    import json
    p=tmp_path/'fake';p.write_text(json.dumps(dict(admitted=True,record='G01')))
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):require_admission(dict(record='G01',decision=str(p)))


def full_system_log(batch=2,label='S3'):
    rows=['VLLM-BATCH-MODE label='+label+' version=0.27.1 context=1024 batches='+str(batch)]
    for phase in ['warmup','warmup_short']+[p for k in range(5) for p in (f'sample{k}',f'sample{k}_short')]:
        n=1 if phase.endswith('_short') else 32
        for request in range(batch):rows.append(f'VLLMSEQ label={label} batch={batch} phase={phase} seq={request} first_divergence={n}/{n} tokens={n}')
    rows.append(f'VLLMB label={label} batch={batch} per_token_ms=5,5,5,5,5')
    return '\n'.join(rows)+'\n'


def test_system_exact_full_short_population_rejects_duplicates_and_extras():
    from examples.experiment_program.validation import system_population
    text=full_system_log();assert system_population(text,'S3',2)['legal']
    for changed in (text+text.splitlines()[1]+'\n',text.replace('sample4_short seq=1','sample3_short seq=1'),text+'VLLMB label=other batch=2 per_token_ms=5,5,5,5,5\n',text.replace('version=0.27.1','version=0.28.0')):
        with pytest.raises(ValueError):system_population(changed,'S3',2)


def test_applied_system_proof_must_itself_be_sealed(tmp_path):
    import json
    from examples.experiment_program.protocol import sha
    from examples.experiment_program.validation import applied_system
    proof=tmp_path/'applied.json';proof.write_text(json.dumps(dict(label='S3',batch=2,applied_counters={'attention':1})))
    other=tmp_path/'other';other.write_text('synthetic CPU proof test')
    with pytest.raises(ValueError,match='proof'):applied_system(dict(applied_proof=str(proof),proof_files={str(other):sha(other)}),'S3',2)
    actual=applied_system(dict(applied_proof=str(proof),proof_files={str(proof):sha(proof)}),'S3',2)
    assert actual['applied_counters']=={'attention':1}


def test_model_schedule_interleaves_every_comparison_and_keeps_deepseek_graph_unavailable():
    from examples.experiment_program.validation import schedule
    from examples.experiment_program.protocol import pair_order
    for model,batch in [('qwen',1),('deepseek',4)]:
        s=schedule(model,batch,['S0','S3'],'S3',19)
        for phase,seed,arms in [('model',19,{'control','candidate'}),('system',21,{'candidate','S3'})]:
            rows=[r for r in s if r['phase']==phase];assert len(rows)==10
            for i in range(5):assert {x['arm'] for x in rows[2*i:2*i+2]}==arms and all(x['ordinal']==i and x['order']==pair_order(seed)[i] for x in rows[2*i:2*i+2])
        assert len([r for r in s if r['phase']=='fusion'])==(10 if model=='qwen' else 0)


def test_missing_runtime_never_becomes_an_executable_or_model_admission(tmp_path):
    import json
    from examples.experiment_program.validation_prepare import executable_build
    p=tmp_path/'build';p.mkdir();(p/'build-identity.json').write_text(json.dumps(dict(host_link_exit=77,hashes={})))
    with pytest.raises(ValueError,match='runtime|link'):executable_build(p,'qwen')


def test_local_sources_are_real_disabled_mechanisms_without_fake_winner(tmp_path):
    from examples.experiment_program.validation_prepare import prepare_sources
    from examples.experiment_program.protocol import sha
    from pathlib import Path
    # The retained closure is read only. No model, native compiler or GPU runs.
    reference=Path(os.environ.get('ETX_VALIDATION_REFERENCE','/nonexistent'))
    if not reference.is_dir():pytest.skip('local retained closure unavailable')
    for record,variant,marker in [('G09','g09','float retained[4][8]'),('G10','g10_up','row_single_up')]:
        result=prepare_sources(dict(record=record,variant=variant,model='qwen',batch=1,reference=str(reference)),tmp_path/record,local_validation=True)
        assert result['execution_enabled'] is False and result['winner'] is None
        assert marker in Path(result['candidate']['dsl_source']).read_text()
        assert Path(result['control']['reference']).resolve()==reference.resolve()


def test_registered_commands_fix_gates_graph_and_system_populations(tmp_path):
    from examples.experiment_program.validation_run import native_command
    c=dict(model='qwen',batch=1,device_index=0,model_dir='/sealed/packed',hf_model='/sealed/hf',goldens=['/sealed/packed/golden_c1024.txt'])
    models={'arms':{arm:dict(directory='/sealed/'+arm) for arm in ('control','candidate')}}
    cmd=native_command(c,models,dict(phase='imports',ordinal=1040,arm='control'),tmp_path/'bench.py')
    assert cmd[-5:]==['--','--check','48','--check-pos','1040']
    assert 'HF_HUB_OFFLINE=1' in cmd and 'HIP_VISIBLE_DEVICES=0' in cmd
    graph=native_command(c,models,dict(phase='fusion',ordinal=0,arm='graph'),tmp_path/'bench.py')
    assert graph[-1]=='--unfused' and '/sealed/candidate' in graph
    system=native_command(c,models,dict(phase='system',ordinal=0,arm='S3'),tmp_path/'bench.py')
    assert system[system.index('--batches')+1]=='1' and system[system.index('--repeats')+1]=='5'
    assert 'ROCM_AITER_UNIFIED_ATTN' in system[-1]
    with pytest.raises(ValueError):native_command(dict(c,runtime_env={'ETX_TRACE':'1'}),models,dict(phase='model',arm='candidate'),tmp_path/'bench.py')


def test_native_timestamp_population_is_exact_and_preserves_request_intervals():
    from examples.experiment_program.validation import native_population
    # Explicit synthetic CPU parser fixture; no timings are submitted as GPU evidence.
    lines=[]
    for r in range(2):
        for t in range(32):lines.append(f'G12TOKEN run=1 request={r} token={t} start_ticks={1+t*500000} end_ticks={1+t*500000+1000} wall_ms=5.000000000')
        lines.append(f'BATCHREQ r={r} tokens=32/32 layers=27 warmup=32/32 interval_median_ms=5.00000 device_median_ms=0.010000')
    text='\n'.join(lines)+'\nBATCH requests=2 tokens=32 step_median_ms=5.000000\n'
    assert len(native_population(text,'deepseek',2))==64
    for bad in (text+lines[0]+'\n',text.replace('end_ticks=1001','end_ticks=0',1),text.replace('interval_median_ms=5.00000','interval_median_ms=6.00000',1)):
        with pytest.raises(ValueError):native_population(bad,'deepseek',2)


def test_transcript_rejects_reordered_missing_and_foreign_session_calls(tmp_path,monkeypatch):
    import json
    from examples.experiment_program import validation_run as runner
    from examples.experiment_program.validation import schedule
    from examples.experiment_program.protocol import sha
    # Synthetic CPU orchestration fixture only. Native source/link admission is
    # separately fail-closed; this fixture cannot produce a G12 performance verdict.
    models=tmp_path/'models.json';models.write_text('{}')
    input_file=tmp_path/'CPU_SYNTHETIC';input_file.write_text('parser regression only')
    c=dict(model='deepseek',batch=2,seed=19,session_id='CPU_SYNTHETIC',requested_systems=['S0','S3'],selected_system='S3',systems={},input_files={str(input_file):sha(input_file)},models=str(models))
    monkeypatch.setattr(runner,'configuration',lambda value:({},{}))
    monkeypatch.setattr(runner,'native_command',lambda *args:['CPU_SYNTHETIC_NO_GPU'])
    (tmp_path/'hardware.json').write_text('{"synthetic":true}')
    registered=dict(c,systems=c['requested_systems']);(tmp_path/'registration.json').write_text(json.dumps(registered))
    monkeypatch.setattr(runner,'resolved_configuration',lambda *args:dict(synthetic=True))
    (tmp_path/'normalized-config.json').write_text('{"synthetic":true}')
    planned=schedule('deepseek',2,c['requested_systems'],'S3',19);(tmp_path/'run-plan.json').write_text(json.dumps(planned))
    rows=[]
    for i,row in enumerate(planned):
        log=tmp_path/runner.log_name(row);log.parent.mkdir(exist_ok=True,parents=True);log.write_text('CPU_SYNTHETIC_NO_GPU\n')
        rows.append(dict(row,log=str(log.relative_to(tmp_path)),log_sha256=sha(log),argv=['CPU_SYNTHETIC_NO_GPU'],returncode=0,session_id=c['session_id'],registration_sha256=sha(tmp_path/'registration.json'),normalized_config_sha256=sha(tmp_path/'normalized-config.json'),input_digest=runner.digest(c['input_files']),hardware_sha256=sha(tmp_path/'hardware.json'),models_sha256=sha(models),begin_monotonic_ns=i*10+2,end_monotonic_ns=i*10+3,telemetry_before=dict(monotonic_ns=i*10+1,raw_sysfs={'synthetic':'1'}),telemetry_after=dict(monotonic_ns=i*10+4,raw_sysfs={'synthetic':'1'})))
    def seal(population):
        (tmp_path/'transcript.json').write_text(json.dumps(population));c['output_files']={str(p):sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    seal(rows);assert runner.transcript(c,tmp_path)['invocations']==22
    for changed in ([rows[1],rows[0],*rows[2:]],rows[:-1],[dict(rows[0],session_id='other'),*rows[1:]]):
        seal(changed)
        with pytest.raises(ValueError):runner.transcript(c,tmp_path)


def test_private_native_reporting_is_outside_existing_timers(tmp_path):
    from pathlib import Path
    from examples.experiment_program.validation_prepare import host_reporting,ROOT,benchmark_source
    for model,source,anchor in [('qwen','examples/vllm_llm/host.hip','    step_ms.clear();'),('deepseek','examples/dsv2lite/host.hip','  std::printf("BATCH requests=')]:
        p=tmp_path/(model+'.hip');host_reporting(ROOT/source,p,model);text=p.read_text()
        assert text.count('G12TOKEN run=')==1
        assert text.index('G12TOKEN run=')<text.rindex(anchor)
        assert 'wall_ms=%.9f' in text
    p=tmp_path/'bench.py';benchmark_source(p);compile(p.read_text(),str(p),'exec')
    assert 'async_scheduling' in p.read_text() and 'kernel_config' in p.read_text()


def test_device_identity_rejects_mi300a_and_wrong_topology_without_touching_hip(monkeypatch):
    from examples.experiment_program import validation_run as runner
    class SyntheticHip:
        name=b'AMD Instinct MI300X';cus=304
        def hipGetDeviceCount(self,p):p._obj.value=1;return 0
        def hipDeviceGetName(self,p,n,i):p.value=self.name;return 0
        def hipDeviceGetPCIBusId(self,p,n,i):p.value=b'0000:01:00.0';return 0
        def hipDeviceGetAttribute(self,p,a,i):p._obj.value={63:self.cus,87:64,23:9,61:4}[a];return 0
    synthetic=SyntheticHip();monkeypatch.setattr(runner.ctypes,'CDLL',lambda name:synthetic);monkeypatch.setattr(runner.sys,'platform','linux')
    assert runner.device_identity()['attributes']['compute_units']==304
    synthetic.name=b'AMD Instinct MI300A'
    with pytest.raises(ValueError,match='MI300X'):runner.device_identity()
    synthetic.name=b'AMD Instinct MI300X';synthetic.cus=228
    with pytest.raises(ValueError,match='topology'):runner.device_identity()


def test_explicit_staging_resolves_commands_without_rewriting_origin_metadata(tmp_path,monkeypatch):
    import json
    from examples.experiment_program.validation_run import native_command,resolved_configuration
    mapping=tmp_path/'map.json';target=tmp_path/'staged';target.mkdir()
    mapping.write_text(json.dumps(dict(schema='etx.path-map.v1',paths={'/ORIGIN':str(target)})))
    monkeypatch.setenv('ETX_PROGRAM_RELOCATIONS',str(mapping))
    c=dict(model='qwen',batch=1,device_index=0,model_dir='/ORIGIN/packed',hf_model='/ORIGIN/hf',goldens=['/ORIGIN/golden'],preparation='/ORIGIN/prep',models='/ORIGIN/models')
    original=json.dumps(c,sort_keys=True);models={'arms':{arm:dict(directory='/ORIGIN/'+arm) for arm in ('control','candidate')}}
    cmd=native_command(c,models,dict(phase='model',ordinal=0,arm='candidate'),tmp_path/'bench.py')
    assert str(target/'candidate') in cmd and str(target/'packed') in cmd
    assert 'ETX_PROGRAM_RELOCATIONS='+str(mapping) in cmd
    normalized=resolved_configuration(c,models)
    assert normalized['resolved_paths']['model_dir']==str(target/'packed') and normalized['relocation_files']
    assert json.dumps(c,sort_keys=True)==original


def test_trace_phase_is_separate_post_pair_and_env_is_default_off(tmp_path):
    from examples.experiment_program.validation import schedule
    from examples.experiment_program.validation_run import native_command
    plain=schedule('qwen',1,['S0'],'S0',19)
    rows=schedule('qwen',1,['S0'],'S0',19,fold_decision_active=True,trace_positions=(0,1040))
    assert rows[:len(plain)]==plain and [r['phase'] for r in rows[len(plain):]]==['trace']*4
    with pytest.raises(ValueError):schedule('deepseek',2,['S0'],'S0',19,fold_decision_active=True)
    c=dict(model='qwen',batch=1,device_index=0,model_dir='/sealed/model',fold_decision_active=True)
    models={'arms':{'control':dict(directory='/sealed/control'),'candidate':dict(directory='/sealed/candidate')}}
    trace=native_command(c,models,rows[-1],tmp_path/'bench.py')
    assert 'ETX_TRACE=1' in trace and 'ETX_PROGRAM_TRACE_POS=1040' in trace
    assert 'ETX_TRACE_RAW='+str(tmp_path/'trace/candidate_p1040.bin') in trace
    timed=native_command(c,models,dict(phase='model',ordinal=0,arm='candidate'),tmp_path/'bench.py')
    assert not any(x.startswith(('ETX_TRACE=','ETX_TRACE_RAW=','ETX_PROGRAM_TRACE_POS=')) for x in timed)


def test_trace_population_and_dependency_brackets_are_exact(tmp_path):
    import json,struct
    from examples.experiment_program.validation_trace import validate_raw
    plan=dict(types={'producer':dict(id=0),'qkv_1_x':dict(id=1)},descs=[dict(id=0,type=0,coord=[0],mode='static',domain=0,worker=0),dict(id=1,type=1,coord=[0],mode='static',domain=0,worker=0)],domains=1,workers_per_domain=1,task_in=[[],[0]],ev_inst=[['event',[0],[0]]],static_queues={'0:0':[0,1]})
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan));raw=tmp_path/'trace.bin'
    rows=[(0,0,0,0,0,10,11,20,0),(1,0,0,0,0,21,22,30,0)]
    raw.write_bytes(b''.join(struct.pack('<9q',*r) for r in rows))
    result=validate_raw(p,raw);assert result['tasks']==2 and result['publication_timestamp_available'] is False
    raw.write_bytes(raw.read_bytes()[:-8])
    with pytest.raises(ValueError):validate_raw(p,raw)


def test_qwen_wall_basis_matches_original_upper_median(monkeypatch):
    from examples.experiment_program.validation import qwen_correct
    from examples.vllm_llm import import_certificate as cert
    monkeypatch.setattr(cert,'stamped',lambda *a:None)
    lines=['run 0: 32/32 generated tokens match HF greedy','run 1: 32/32 generated tokens match HF greedy']
    for run in range(2):
        for token in range(32):lines.append(f'G12TOKEN run={run} request=0 token={token} start_ticks=1 end_ticks=2 wall_ms={1 if token<16 else 10}')
    lines.append('RESULT model=CPU_SYNTHETIC requests=1 matched=32 gen=32 median_ms=1.000 wall_ms=10.000 matched_each=32')
    assert qwen_correct('\n'.join(lines),{})==10_000_000


def test_reuse_control_rejects_changed_host_before_any_generation(tmp_path):
    import json
    from examples.experiment_program.validation_prepare import verify_reusable_control
    from examples.experiment_program.protocol import sha
    from examples.vllm_llm.kept_build import BASE_FLAGS,KEPT_ENV
    d=tmp_path/'control';d.mkdir();f=d/'sealed';f.write_text('synthetic test, no native object')
    host=tmp_path/'host';host.write_text('new guarded private host')
    identity=dict(hashes={str(f):sha(f)},dependency_files={str(f):sha(f)},original_reference_objects={str(f):sha(f)},reused_imports={str(f):sha(f)},flags=BASE_FLAGS.split()+['-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_TOPK_KEY32','-DETX_VL_CHECK'],check_enabled=True,environment=dict(KEPT_ENV,ETX_VL_LAST_ROUTER='0'),host_source_sha256='0'*64)
    (d/'build-identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='host differs'):verify_reusable_control(d,dict(model='qwen',control=dict(host=str(host),reference=str(tmp_path))),tmp_path/'new-proof')
    assert not (tmp_path/'new-proof').exists()


def test_current_dsl_overlay_keeps_narrow_router_wrappers(tmp_path):
    from pathlib import Path
    from examples.experiment_program.validation_prepare import prepare_sources
    from examples.experiment_program.protocol import sha
    reference=Path(os.environ.get('ETX_VALIDATION_REFERENCE','/nonexistent'))
    if not reference.is_dir():pytest.skip('read-only retained closure unavailable')
    before={str(p):sha(p) for p in (reference/'imp/imports.json',reference/'dsl_tiles.hip',reference/'imports.bc',reference/'imp_tiles.hip')}
    for record in ('G06','G09','G10'):
        value=prepare_sources(dict(record=record,model='qwen',batch=1,reference=str(reference)),tmp_path/record,local_validation=True)
        text=Path(value['candidate']['dsl_source']).read_text()
        assert 'etx_tile_dsl_gemv_router_fold' in text
        assert Path(value['current_kept_dsl']['manifest']).is_file()
        assert value['current_kept_dsl']['original_non_dsl_imports_unchanged']
    assert before=={p:sha(p) for p in before}


def test_original_import_abi_ignores_only_dsl_additions():
    from examples.experiment_program.validation_prepare import original_import_abi
    old={'native':{'slots':[1,2]},'old_dsl':{'dsl':True}}
    new={'native':{'slots':[1,2]},'new_dsl':{'dsl':True}}
    assert original_import_abi(old)==original_import_abi(new)
    new['native']['slots']=[2,1]
    assert original_import_abi(old)!=original_import_abi(new)


def test_guard_only_attestation_rejects_non_guard_source_change(tmp_path):
    import json
    from examples.experiment_program.validation_prepare import attest_guard_only_models
    from examples.experiment_program.protocol import sha
    original=tmp_path/'native-math.hip';original.write_text('old source')
    p=tmp_path/'old-prep.json';value=dict(record='G01',model='qwen',batch=1,source_files={str(original):sha(original)})
    p.write_text(json.dumps(value));m=tmp_path/'models.json';m.write_text(json.dumps(dict(preparation_sha256=sha(p))))
    original.write_text('changed math')
    n=tmp_path/'new-prep.json';n.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='non-guard source'):
        attest_guard_only_models(m,p,n,tmp_path/'new-attestation')
    assert not (tmp_path/'new-attestation').exists()


def test_guard_attestation_consumer_rejects_different_current_preparation(tmp_path):
    import json
    from examples.experiment_program.validation_prepare import guard_attestation_required
    from examples.experiment_program.protocol import sha
    prep=tmp_path/'preparation.json';prep.write_text('{}')
    proof=tmp_path/'attestation.json';proof.write_text(json.dumps(dict(schema='etx.g12.guard-only-native-attestation.v1',current_preparation_sha256='0'*64)))
    models=dict(guard_only_attestation=str(proof),guard_only_attestation_sha256=sha(proof))
    with pytest.raises(ValueError,match='current preparation'):
        guard_attestation_required(models,prep)


def test_actual_g04_b4_unified_register_descriptor_is_not_double_counted():
    from pathlib import Path
    from examples.experiment_program.native_resources import persistent_capacity
    d=Path('build/experiment_model/G04-B4-final01')
    if not d.is_dir():pytest.skip('read-only retained G04B4 native closure unavailable')
    r=persistent_capacity(d)
    assert r['raw_metadata']['vgpr']==254 and r['raw_metadata']['agpr']==6
    assert r['next_free_vgpr']==254 and r['allocated_unified_registers_per_lane']==256
    assert r['threads']==256 and r['workers_per_domain']==75
    assert r['resident_capacity']==2 and r['required_persistent_blocks']==2


def conditional_fixture(tmp_path):
    import json
    from examples.experiment_program.protocol import sha
    rows=[];files={}
    for record in ('G09','G01'):
        d=tmp_path/record;d.mkdir();prep=d/'prep.json';prep.write_text(json.dumps(dict(record=record,local_validation=True,execution_enabled=False)))
        models=d/'models.json';models.write_text('{}');template=d/'template.json';template.write_text(json.dumps(dict(preparation=str(prep),models=str(models),batch=1,input_files={})))
        micro=d/'micro';micro.mkdir();identity=micro/'identity.json';identity.write_text('{}')
        rows.append(dict(preparation=str(prep),models=str(models),template=str(template),admission=dict(record=record,decision=str(tmp_path/'session'/record/'decision.json'),build=str(micro))))
        files.update({str(p):sha(p) for p in (prep,models,template,identity)})
    return dict(conditional_candidates=rows,input_files=files,source_snapshot=dict(files),readiness_files=dict(files),source_revision='CPU_SYNTHETIC_REVISION')


def test_conditional_preflight_dormant_without_invented_winner(tmp_path,monkeypatch):
    from examples.experiment_program import validation_dispatch as dispatch,validation_run
    c=conditional_fixture(tmp_path);calls=[]
    def dormant(value,**kw):
        assert 'winner' not in value and kw==dict(admission=False,dormant=True);calls.append(value);return {},{}
    monkeypatch.setattr(validation_run,'configuration',dormant)
    result=dispatch.preflight(c)
    assert result['status']=='PASS' and not result['enabled'] and not result['winner_invented'] and len(calls)==2


def test_conditional_select_uses_only_current_sealed_completed_micro_and_priority(tmp_path,monkeypatch):
    import json
    from examples.experiment_program import validation_dispatch as dispatch,validation_run,validation
    from examples.experiment_program.protocol import sha
    c=conditional_fixture(tmp_path);root=tmp_path/'session';root.mkdir();sealed={};stages={}
    for i,entry in enumerate(c['conditional_candidates']):
        p=tmp_path/'session'/entry['admission']['record']/'decision.json';p.parent.mkdir();p.write_text(json.dumps(dict(status='MICRO_MISS' if i==0 else 'MICRO_PASS')))
        sealed[str(p)]=sha(p);stages[entry['admission']['record']]=dict(status='COMPLETE')
    state=dict(source_revision=c['source_revision'],sealed=sealed,stages=stages);(root/'session.json').write_text(json.dumps(state))
    monkeypatch.setattr(validation_run,'configuration',lambda *a,**kw:({},{}))
    calls=[]
    def admitted(value):calls.append(value['record']);return dict(record=value['record'],cpu_synthetic=True)
    monkeypatch.setattr(validation,'require_admission',admitted)
    entry,_,proof,session=dispatch.select(c,root)
    assert entry['admission']['record']=='G01' and proof['cpu_synthetic'] and calls==['G01']
    # Edited or externally transplanted decision cannot be selected.
    chosen=tmp_path/'session/G01/decision.json';chosen.write_text('{"status":"MICRO_PASS","edited":true}')
    with pytest.raises(ValueError,match='sealed prior result'):dispatch.select(c,root)


def test_conditional_known_model_resource_rejection_does_not_block_another_winner(tmp_path,monkeypatch):
    import json
    from examples.experiment_program import validation_dispatch as dispatch,validation_run,validation
    from examples.experiment_program.protocol import sha
    c=conditional_fixture(tmp_path);root=tmp_path/'session';root.mkdir();sealed={};stages={}
    for entry in c['conditional_candidates']:
        p=root/entry['admission']['record']/'decision.json';p.parent.mkdir();p.write_text('{"status":"MICRO_PASS"}')
        sealed[str(p)]=sha(p);stages[entry['admission']['record']]=dict(status='COMPLETE')
    (root/'session.json').write_text(json.dumps(dict(source_revision=c['source_revision'],sealed=sealed,stages=stages)))
    monkeypatch.setattr(validation,'require_admission',lambda value:dict(record=value['record'],cpu_synthetic=True))
    def ready(value,**kw):
        if '/G09/' in value['preparation']:raise ValueError('synthetic static residency rejection')
        return {},{}
    monkeypatch.setattr(validation_run,'configuration',ready)
    chosen=dispatch.select(c,root)
    assert chosen[0]['admission']['record']=='G01'
