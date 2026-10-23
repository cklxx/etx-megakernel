"""Registered G12 execution: bounded native calls, current inputs, one GPU.

Preparation and --help never initialize HIP. Only explicit run --execute calls
this module's device identity routine. No arbitrary stage/command list exists.
"""
from __future__ import annotations
import ctypes,json,os,shutil,sys,time
from pathlib import Path
from .protocol import require,sha,verify_files,resolve_path
from .validation_prepare import executable_build,benchmark_source,ROOT


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def resolved_configuration(c,models):
    from .protocol import relocation_seal
    paths={key:str(resolve_path(c[key])) for key in ('preparation','models','hf_model','model_dir','weights','golden_manifest') if key in c}
    paths.update({key:[str(resolve_path(p)) for p in c[key]] for key in ('goldens','requests') if key in c})
    return dict(origin_config_digest=digest(c),resolved_paths=paths,model_directories={name:str(resolve_path(arm['directory'])) for name,arm in models['arms'].items()},relocation_files=relocation_seal())


def configuration(c,*,admission=True,dormant=False):
    """Resolve a concrete prepared pair; reject partial/unsealed input closures."""
    from .validation import require_admission
    require('stages' not in c,'arbitrary stage commands are not a G12 preparation')
    p=resolve_path(c['preparation']);v=json.loads(p.read_text());m=resolve_path(c['models']);models=json.loads(m.read_text())
    require(v['schema']=='etx.g12.preparation.v1','unknown G12 preparation')
    require((v['execution_enabled'] and not v['local_validation']) or (dormant and not admission and v['local_validation'] and not v['execution_enabled']),'local source validation is execution disabled; activate with a real winner')
    if admission:require(require_admission(c['winner'])==v['winner'],'winner differs from source activation')
    require(models['preparation_sha256'] in (sha(p),v.get('activation_source_preparation_sha256')),'different model source preparation')
    require((c['model'],c['batch'])==(v['model'],v['batch'])==(models['model'],models['batch']),'model/batch mismatch')
    require(models['record']==v['record'] and models['resource_admission']['admitted'],'whole-model resources not admissible')
    from .validation_prepare import model_resource_admission
    require(model_resource_admission({name:resolve_path(arm['directory']) for name,arm in models['arms'].items()},c['model'],c['batch'])==models['resource_admission'],'raw whole-model resource admission differs')
    verify_files(v['source_files']);verify_files(c['input_files'])
    require(isinstance(c['session_id'],str) and c['session_id'] and type(c['seed']) is int,'session/seed missing')
    require(type(c['device_index']) is int and c['device_index']>=0,'one explicit physical GPU index required')
    require(c['vllm_version']=='0.27.1','registered vLLM 0.27.1 proof contract required')
    require(type(c.get('fold_decision_active',False)) is bool,'explicit fold decision boolean required')
    if c.get('fold_decision_active'):
        from .validation_trace import options
        from .validation import schedule
        schedule(c['model'],c['batch'],c['systems'],c['systems'][0],c['seed'],**options(c))
    labels=c['systems'];require(isinstance(labels,list) and labels and len(set(labels))==len(labels),'registered unique system configurations required')
    # These are implemented by the checked-in registered proof parser, not
    # unchecked engine kwargs or self-authored applied counters.
    require(set(labels)<={'S0','S2','S3','S4f','S5f','S6f','S7','S8'},'unsupported/unprovisioned system lever')
    sealed={resolve_path(x).resolve() for x in c['input_files']}
    required={p.resolve(),m.resolve()}
    from .validation_prepare import guard_attestation_required
    required.update(guard_attestation_required(models,p))
    if (models.get('control_reuse') or {}).get('status')=='REUSE_EXACT_IMMUTABLE_NATIVE_CONTROL':
        verify_files(models['control_reuse']['proof_files']);required.update(resolve_path(p).resolve() for p in models['control_reuse']['proof_files'])
    require(set(models['arms'])=={'control','candidate'},'exactly one unchanged control and one single-mechanism candidate required')
    mapping=os.environ.get('ETX_PROGRAM_RELOCATIONS')
    if mapping:required.add(resolve_path(mapping).resolve())
    if c.get('runtime_env',{}).get('ETX_PROGRAM_RELOCATIONS'):
        require(mapping and resolve_path(mapping)==resolve_path(c['runtime_env']['ETX_PROGRAM_RELOCATIONS']),'activate relocation with the same sealed ETX_PROGRAM_RELOCATIONS mapping')
    for name in ('control','candidate'):
        arm=models['arms'][name];d=resolve_path(arm['directory']);required.add((d/'build-identity.json').resolve())
        require(sha(d/'build-identity.json')==arm['identity_sha256'],'model native identity changed')
        native=executable_build(d,c['model'])
        verify_files(native['dependency_files'])
        if native.get('reused_imports'):verify_files(native['reused_imports'])
    hf=resolve_path(c['hf_model']).resolve();require(hf.is_dir(),'existing local HF checkpoint required; no download')
    # Seal the complete local HF checkpoint, including tokenizer/custom model
    # code. This does not copy the checkpoint or read credentials/cache files.
    required.update(x.resolve() for x in hf.rglob('*') if x.is_file())
    goldens=list(map(resolve_path,c['goldens']));require(len(goldens)==c['batch'],'one real golden per request')
    from examples.llm.vllm_bench import read_golden
    parsed=[read_golden(x) for x in goldens]
    require(all(len(prompt)==1024 and len(tokens)>=32 for prompt,tokens in parsed),'context1024/full32 goldens required')
    require(len({tuple(prompt) for prompt,_ in parsed})==c['batch'],'duplicate request prompts')
    required.update(x.resolve() for x in goldens)
    if c['model']=='qwen':
        model=resolve_path(c['model_dir']);required.update((model/x).resolve() for x in ('vl.bin','vl.manifest','golden_c1024.txt'))
        require(goldens[0].resolve()==(model/'golden_c1024.txt').resolve(),'Qwen ETX/system golden mismatch')
    else:
        reqs=c['requests'];require(len(reqs)==c['batch'],'exact DeepSeek request directories required')
        from examples.dsv2lite.batch_goldens import binding,read_binding,check
        # The producer validation binds distinct prompts and native references.
        prompts=[];gens=[];tokens=[];hashes={};bindings=[]
        import numpy as np
        for index,r in enumerate(reqs):
            r=resolve_path(r);required.update((r/x).resolve() for x in ('fleet_cache.bin','golden_tokens.txt','golden_hidden.bin','golden.npz','golden.binding.json'))
            actual=binding(r);require(actual==read_binding(r),'native producer binding differs from execution bytes')
            with np.load(r/'golden.npz') as z:gen=[int(x) for x in z['token_ids'].reshape(-1)]
            prompts.append(actual['prompt_ids']);gens.append(gen);tokens.append(list(map(int,(r/'golden_tokens.txt').read_text().split())))
            hashes[str(index)]=actual['sha256'];bindings.append(actual)
            require(parsed[index]==(prompts[-1],gen),'ETX/vLLM request reference mismatch')
        weights=resolve_path(c['weights']);required.update((weights.resolve(),resolve_path(str(weights)+'.manifest').resolve()))
        # Explicit producer manifest is required rather than inferring equality
        # from unrelated golden token files.
        producer=resolve_path(c['golden_manifest']);required.add(producer.resolve())
        require(not check(prompts,gens,tokens,bindings=bindings,hashes=hashes),'native request rotations/goldens differ')
        manifest=json.loads(producer.read_text());require(manifest['requests']>=c['batch'] and all(manifest['sha256'][str(i)]==hashes[str(i)] for i in range(c['batch'])),'producer manifest differs')
    require(required<=sealed,'incomplete current input closure: '+str(sorted(map(str,required-sealed))[:8]))
    require(0<c['timeout_seconds']<=900 and 0<c['budget_seconds']<=13500,'bounded stage/session deadlines required')
    return v,models


def device_identity():
    """Actual visible HIP device identity; called only inside authorized run."""
    require(sys.platform.startswith('linux'),'native GPU session requires Linux')
    hip=ctypes.CDLL('libamdhip64.so');count=ctypes.c_int()
    require(hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==1,'exactly one HIP-visible GPU required')
    name=ctypes.create_string_buffer(256);pci=ctypes.create_string_buffer(64)
    require(hip.hipDeviceGetName(name,256,0)==0 and hip.hipDeviceGetPCIBusId(pci,64,0)==0,'actual HIP identity unavailable')
    require(b'MI300X' in name.value,'registered MI300X only')
    # Exact hipDeviceAttribute_t values in the pinned ROCm 7.2.4 header:
    # MultiprocessorCount63, WarpSize87, ComputeCapabilityMajor23/Minor61.
    attributes={}
    for key,enum in [('compute_units',63),('wave_size',87),('compute_major',23),('compute_minor',61)]:
        value=ctypes.c_int();require(hip.hipDeviceGetAttribute(ctypes.byref(value),enum,0)==0,'HIP device attribute unavailable');attributes[key]=value.value
    require(attributes==dict(compute_units=304,wave_size=64,compute_major=9,compute_minor=4),'registered 304CU/wave64 MI300X topology required')
    return dict(host=os.uname().nodename,device_name=name.value.decode(),pci_bus_id=pci.value.decode(),visible_devices=1,attributes=attributes)


def certificate_files(directory,model,reference):
    """Issue a fresh original checker identity from this linked native closure."""
    from examples.vllm_llm import import_certificate as cert
    d=resolve_path(directory);native=executable_build(d,'qwen');ref=resolve_path(reference)
    for name,source in [('imports.bc',next(iter(native['reused_imports']))),('imp_tiles.hip',ref/'imp_tiles.hip')]:
        destination=d/name
        if destination.exists():require(sha(destination)==sha(source),'certificate convenience source differs')
        else:shutil.copyfile(resolve_path(source),destination)
    (d/'hipcc.log').write_text(''.join((d/name).read_text() for name in ('compiler.log','device.log','device-object.log','device-link.log','host-object.log','host-link.log')))
    (d/'compiler_version.txt').write_text((d/'compiler.log').read_text())
    (d/'hipcc_command.json').write_text(json.dumps(native['commands'],indent=2)+'\n')
    # write_build reads only these controlled build flags. Preserve the caller's
    # process environment; benchmark subprocesses receive a separate clean env.
    prior=dict(os.environ)
    try:
        for key in list(os.environ):
            if key.startswith(('ETX_','DSL_ABLATE_')):os.environ.pop(key)
        os.environ.update(ETX_VL_CHECK='1',ETX_HIPCC_FLAGS=' '.join(native['flags']))
        return cert.write_build(d,resolve_path(model))
    finally:os.environ.clear();os.environ.update(prior)


def log_name(row):
    phase,i,arm=row['phase'],row['ordinal'],row['arm']
    if phase=='imports':return f'{arm}/check_p{i}.txt'
    if phase=='dsl':return f'{arm}/gate_chkx.log'
    if phase=='graph-gate':return 'candidate/gate_unf.log'
    if phase=='selection':return f'select_{arm}.log'
    if phase=='model':return f'{arm}/model_r{i}.log'
    if phase=='fusion':return f'candidate/{"etx" if arm=="fused" else "etxu"}_r{i+1}.log'
    if phase=='system':return f'system_{arm}_r{i}.log'
    if phase=='trace':return f'trace/{arm}_p{i}.log'
    raise ValueError('unregistered phase')


def native_command(c,models,row,benchmark):
    phase,arm=row['phase'],row['arm'];system=phase=='selection' or (phase=='system' and arm!='candidate')
    allowed={'PATH','HOME','PYTHONPATH','LD_LIBRARY_PATH','TMPDIR','TRITON_CACHE_DIR','VLLM_CACHE_ROOT','ETX_PROGRAM_RELOCATIONS'}
    require(set(c.get('runtime_env',{}))<=allowed,'unregistered environment override')
    env=dict(PATH='/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin',PYTHONPATH=str(ROOT));env.update(c.get('runtime_env',{}))
    if os.environ.get('ETX_PROGRAM_RELOCATIONS'):env['ETX_PROGRAM_RELOCATIONS']=str(resolve_path(os.environ['ETX_PROGRAM_RELOCATIONS']))
    env.update(HIP_VISIBLE_DEVICES=str(c['device_index']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    if phase=='trace':
        require(c.get('fold_decision_active') is True and c['model']=='qwen' and row['ordinal'] in (0,1040),'trace requires active Qwen fold decision')
        env.update(ETX_TRACE='1',ETX_TRACE_RAW=str(resolve_path(benchmark).parent/'trace'/f'{arm}_p{row["ordinal"]}.bin'),ETX_PROGRAM_TRACE_POS=str(row['ordinal']))
    if system:
        from examples.vllm_llm.vlbest_report import config
        cfg=config(arm);env.update(cfg['env'])
        argv=[sys.executable,str(benchmark),'--model',str(resolve_path(c['hf_model'])),'--goldens',*[str(resolve_path(p)) for p in c['goldens']],'--batches',str(c['batch']),'--tokens','32','--repeats','5','--label',arm,'--echo-config','--engine-kw',json.dumps(cfg['kw']),*cfg['flags']]
    else:
        which=arm if phase=='model' else 'candidate' if arm in ('fused','graph') or phase in ('graph-gate','system') else arm
        directory=resolve_path(models['arms'][which]['directory'])
        if c['model']=='qwen':
            opts=['--check','48','--check-pos',str(row['ordinal'])] if phase=='imports' else ['--check-dsl'] if phase=='dsl' else ['--unfused'] if phase=='graph-gate' or arm=='graph' else []
            argv=[sys.executable,'-m','examples.vllm_llm.import_certificate','--run-build',str(directory),'--model',str(resolve_path(c['model_dir'])),'--',*opts]
        else:
            r=resolve_path(c['requests'][0]);argv=[str(directory/'run'),'--weights',str(resolve_path(c['weights'])),'--cache',str(r/'fleet_cache.bin'),'--golden',str(r/'golden_tokens.txt'),'--hidden',str(r/'golden_hidden.bin'),'--context','1024','--tokens','32','--repeat','2']
            for request in c['requests'][1:]:argv+=['--req-dir',str(resolve_path(request))]
    # run_logged intentionally has no environment mutation hook; /usr/bin/env
    # supplies this exact per-command environment as argv without a shell.
    return ['/usr/bin/env','-i',*[k+'='+v for k,v in sorted(env.items())],*argv]


def prove_system(log,label,c,out):
    from .validation import system_population
    from examples.vllm_llm.vlbest_report import proof,config,expected_replacements
    text=resolve_path(log).read_text();system_population(text,label,c['batch'])
    expected=None
    if c['model']=='qwen':expected=expected_replacements((out/'candidate/vl_plan.txt').read_text())
    require(proof(text,config(label),expected) is None,'system lever not proven in actual invocation')
    p=resolve_path(str(log)+'.applied.json')
    p.write_text(json.dumps(dict(label=label,batch=c['batch'],session_id=c['session_id'],benchmark_log_sha256=sha(log),applied_counters=dict(registered_configuration=1),expected_full_graph_replacements=expected),indent=2)+'\n')
    return dict(applied_proof=str(p.resolve()),proof_files={str(p.resolve()):sha(p),str(resolve_path(log).resolve()):sha(log)})


def telemetry(hardware):
    root=resolve_path('/sys/bus/pci/devices')/hardware['pci_bus_id']
    paths=[root/'pp_dpm_sclk',root/'pp_dpm_mclk',*root.glob('hwmon/hwmon*/power1_average'),*root.glob('hwmon/hwmon*/power1_input')]
    values={str(p):p.read_text().strip() for p in paths if p.is_file()}
    require(any('power1_' in p for p in values) and any('pp_dpm_' in p for p in values),'actual GPU clocks/power telemetry unavailable')
    return dict(monotonic_ns=time.monotonic_ns(),raw_sysfs=values)


def run(c,out):
    from .validation import schedule,system_population,replay
    from etx.experiments.sealed import run_logged
    from examples.vllm_llm import import_certificate as cert
    from etx.session.batch import ns
    from .validation_trace import options,reserve,create
    v,models=configuration(c);out=resolve_path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    benchmark=resolve_path(benchmark_source(out/'registered-vllm-bench.py'))
    before=dict(os.environ)
    try:
        for k in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):os.environ.pop(k,None)
        os.environ['HIP_VISIBLE_DEVICES']=str(c['device_index']);hardware=device_identity()
    finally:os.environ.clear();os.environ.update(before)
    (out/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
    (out/'registration.json').write_text(json.dumps(c,indent=2)+'\n')
    (out/'normalized-config.json').write_text(json.dumps(resolved_configuration(c,models),indent=2)+'\n')
    if c['model']=='qwen':
        for arm in ('control','candidate'):
            d=resolve_path(models['arms'][arm]['directory']);certificate_files(d,c['model_dir'],v[arm]['reference']);cert.export_build(d,resolve_path(c['model_dir']),out/arm)
            shutil.copyfile(d/'build-identity.json',out/arm/'native-build-identity.json')
    ledger=[];proofs={};trace_proofs=[];start=time.monotonic();input_digest=digest(c['input_files'])
    if c.get('fold_decision_active'):reserve(models,out,options(c)['trace_positions']);(out/'trace').mkdir()
    stat_snapshot={p:(resolve_path(p).stat().st_size,resolve_path(p).stat().st_mtime_ns,resolve_path(p).stat().st_ino) for p in c['input_files']}
    def invoke(row):
        require(all((resolve_path(p).stat().st_size,resolve_path(p).stat().st_mtime_ns,resolve_path(p).stat().st_ino)==value for p,value in stat_snapshot.items()),'input changed during session')
        require(device_identity()==hardware,'physical GPU changed during session')
        log=out/log_name(row);cmd=native_command(c,models,row,benchmark)
        row=dict(row,log=str(log.relative_to(out)),argv=cmd,session_id=c['session_id'],registration_sha256=sha(out/'registration.json'),normalized_config_sha256=sha(out/'normalized-config.json'),input_digest=input_digest,hardware_sha256=sha(out/'hardware.json'),models_sha256=sha(c['models']),telemetry_before=telemetry(hardware),begin_monotonic_ns=time.monotonic_ns())
        remaining=c['budget_seconds']-(time.monotonic()-start)
        rc=run_logged(cmd,log,seconds=min(c['timeout_seconds'],remaining))
        row.update(returncode=rc,end_monotonic_ns=time.monotonic_ns(),telemetry_after=telemetry(hardware),log_sha256=sha(log));ledger.append(row)
        (out/'transcript.json').write_text(json.dumps(ledger,indent=2)+'\n');require(rc==0,'native invocation failed; no G12 verdict')
        if c['model']=='qwen' and row['phase']=='fusion':
            # Existing pairs parser binds all raw graph/fused results to the
            # single physical identity, not runner-local numeric GPU aliases.
            text=log.read_text();log.write_text('# job G12_'+str(len(ledger))+' gpu='+digest(hardware)+'\n'+text);row['log_sha256']=sha(log)
            (out/'transcript.json').write_text(json.dumps(ledger,indent=2)+'\n')
        if row['phase'] not in ('selection',) and not (row['phase']=='system' and row['arm']!='candidate'):
            from .validation import qwen_correct,native_population
            if c['model']=='qwen':
                which=row['arm'] if row['arm'] in ('control','candidate') else 'candidate'
                identity=json.loads((out/which/'build_identity.json').read_text())
                if row['phase']=='imports':
                    expected=cert.expected_instances((out/which/'vl_plan.txt').read_text(),json.loads((out/which/'imports.json').read_text()))
                    require(len(expected)==97,'current registered Qwen import population changed')
                    cert.check_output(log.read_text(),expected,row['ordinal'],identity)
                else:
                    qwen_correct(log.read_text(),identity)
                    if row['phase']=='dsl':require('CHECK-DSL OK' in log.read_text(),'DSL checker failed/missing')
            else:native_population(log.read_text(),'deepseek',c['batch'])
        return log
    # Selection must finish before the strongest legal system is chosen. The
    # final model/system pair populations are independent of selection samples.
    initial=schedule(c['model'],c['batch'],c['systems'],c['systems'][0],c['seed'],**options(c))
    for row in initial:
        if row['phase'] in ('model','fusion','system'):break
        log=invoke(row)
        if row['phase']=='selection':proofs[row['arm']]=prove_system(log,row['arm'],c,out)
    selected=min(c['systems'],key=lambda label:ns(system_population((out/f'select_{label}.log').read_text(),label,c['batch'])['step_ms']))
    planned=schedule(c['model'],c['batch'],c['systems'],selected,c['seed'],**options(c))
    (out/'run-plan.json').write_text(json.dumps(planned,indent=2)+'\n')
    trace_gates_checked=False
    for row in planned[len(ledger):]:
        if row['phase']=='trace' and not trace_gates_checked:
            for arm in ('control','candidate'):cert.validate(out/arm,timing=False)
            cert.validate(out/'candidate',timing=True)
            trace_gates_checked=True
        log=invoke(row)
        if row['phase']=='system' and row['arm']!='candidate':prove_system(log,selected,c,out)
        if row['phase']=='trace':trace_proofs.append(create(c,models,out,row))
    sealed=dict(c,systems=proofs,requested_systems=list(c['systems']),selected_system=selected,fold_traces=trace_proofs)
    sealed['output_files']={str(p.resolve()):sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'session-config.json').write_text(json.dumps(sealed,indent=2)+'\n')
    return replay(sealed,out)


def transcript(c,out):
    """Raw completeness, order and immutable session-input/build binding."""
    from .validation import schedule
    from .validation_trace import options
    out=resolve_path(out);verify_files(c['output_files']);verify_files(c['input_files'])
    _,models=configuration(dict(c,systems=c['requested_systems']))
    registered=json.loads((out/'registration.json').read_text())
    require(all((c.get(key) if key!='systems' else c['requested_systems'])==value for key,value in registered.items()),'registered session config changed')
    require(json.loads((out/'normalized-config.json').read_text())==resolved_configuration(registered,models),'normalized staging configuration differs')
    rows=json.loads((out/'transcript.json').read_text());planned=schedule(c['model'],c['batch'],c['requested_systems'],c['selected_system'],c['seed'],**options(c))
    require(json.loads((out/'run-plan.json').read_text())==planned and len(rows)==len(planned),'missing/extra session invocation')
    previous=0
    for row,want in zip(rows,planned):
        require({k:row.get(k) for k in want}==want,'session pair/order differs from registration')
        require(row['log']==log_name(want) and row['log_sha256']==sha(out/row['log']),'transcript/raw log differs')
        require(row['returncode']==0 and row['begin_monotonic_ns']>=previous and row['end_monotonic_ns']>row['begin_monotonic_ns'],'incomplete/concurrent invocation')
        require(row['telemetry_before']['monotonic_ns']<=row['begin_monotonic_ns']<row['end_monotonic_ns']<=row['telemetry_after']['monotonic_ns'] and all(row[x]['raw_sysfs'] for x in ('telemetry_before','telemetry_after')),'missing paired actual clocks/power readings')
        require(row['session_id']==c['session_id'] and row['input_digest']==digest(c['input_files']) and row['hardware_sha256']==sha(out/'hardware.json') and row['models_sha256']==sha(c['models']),'session input/build/hardware identity differs')
        require(row['registration_sha256']==sha(out/'registration.json'),'session registration changed')
        require(row['normalized_config_sha256']==sha(out/'normalized-config.json'),'staging mapping receipt changed')
        require(row['argv']==native_command(dict(c,systems=c['requested_systems']),models,want,out/'registered-vllm-bench.py'),'unregistered native command')
        previous=row['end_monotonic_ns']
    return dict(invocations=len(rows),hardware=json.loads((out/'hardware.json').read_text()),inputs_digest=digest(c['input_files']),models_sha256=sha(c['models']))
