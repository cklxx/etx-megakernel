"""Concrete disabled source/target-build preparation for each G12 winner family.

No invented winner enables a build. Local validation prepares real source
interventions and complete build recipes with execution explicitly disabled.
"""
from __future__ import annotations
import json,os,shutil,sys
from pathlib import Path
from .protocol import require,sha,verify_files,resolve_path,selected_artifacts
ROOT=resolve_path(__file__).resolve().parents[2]
QWEN_BASE='b573664faa4e7988dff353bc03aca4ce8561a729'
DEEPSEEK_BASE='554133d744b4591c9cf87c68b6817df781137c9b'


def original_import_abi(manifest):
    """Only DSL entries may be added by the current clean kept emitter."""
    return {name:entry for name,entry in manifest.items() if not entry.get('dsl')}


def current_kept_dsl(reference,out):
    from .model_build import run
    from examples.vllm_llm.kept_build import KEPT_ENV
    out=resolve_path(out);out.mkdir();(out/'tmp').mkdir()
    manifest=out/'imports.json';shutil.copyfile(reference/'imp/imports.json',manifest)
    env={k:v for k,v in os.environ.items() if not k.startswith(('ETX_','DSL_ABLATE_','BF16_'))}
    env.update(KEPT_ENV,ETX_VL_LAST_ROUTER='0',ETX_VL_IMPORTS=str(manifest),ETX_VL_DSL_TILES=str(out/'dsl_tiles.hip'),ETX_VL_ADAPTERS=str(reference/'imp_tiles.hip'),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'))
    run([sys.executable,'-m','etx.ops.emit',manifest,'--tiles',out/'dsl_tiles.hip','--ops','gemv,topk,attnprep'],out,'emit-current-kept-dsl',env=env)
    require(original_import_abi(json.loads(manifest.read_text()))==original_import_abi(json.loads((reference/'imp/imports.json').read_text())),'emitter changed original non-DSL ABI')
    return dict(source=str(out/'dsl_tiles.hip'),manifest=str(manifest),original_non_dsl_imports_unchanged=True,files={str(p):sha(p) for p in (manifest,out/'dsl_tiles.hip')})


def prepare_sources(c,out,*,local_validation=False):
    from .validation import require_admission
    record=c.get('record',c.get('winner',{}).get('record'));model=c['model'];batch=c['batch']
    require(record in ('G01','G02','G05','G06','G07','G08','G09','G10'),'only registered optimization mechanism')
    require((model=='qwen' and batch==1 and record not in ('G02','G05')) or (model=='deepseek' and batch in (2,4) and record in ('G02','G05')),'mechanism/model/batch mismatch')
    winner=None if local_validation else require_admission(c['winner'])
    out=resolve_path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    control={};candidate={};source_files={};current_dsl=None
    if model=='qwen':
        reference=resolve_path(c['reference']).resolve();require((reference/'imp/imports.json').is_file() and (reference/'dsl_tiles.hip').is_file(),'complete read-only kept import/DSL closure required')
        original_imports=json.loads((reference/'imp/imports.json').read_text())
        require({n for n,d in original_imports.items() if not d.get('dsl')}=={'attn_mfma16_g8','attn_reduce','rms_norm_2d'},'registered three original HIP imports only; Triton originals unavailable in this checker closure')
        control=dict(reference=str(reference),last_router=False)
        candidate=dict(reference=str(reference),last_router=record=='G01')
        if record in ('G06','G09','G10'):
            current_dsl=current_kept_dsl(reference,out/'current-kept-dsl');source_files.update(current_dsl['files'])
        if record in ('G09','G10'):
            from .row_ops.integration import emit_winner
            variant=c.get('variant','g09' if record=='G09' else 'g10_up')
            require(variant in (('g09',) if record=='G09' else ('g10_up','g10_down')),'wrong row winner variant')
            value=emit_winner(variant,current_dsl['source'],out/'candidate-dsl.hip');candidate['dsl_source']=value['source']
            source_files[value['source']]=sha(value['source'])
        elif record in ('G06','G07','G08'):
            from .issue_order.integration import emit_sources
            from examples.vllm_llm.kept_build import BASE_FLAGS
            micro=None;raw=None
            if not local_validation:
                raw=json.loads(resolve_path(c['winner']['decision']).read_text());raw['_evidence_directory']=str(resolve_path(c['winner']['decision']).parent.resolve())
                from .issue_order.__main__ import verify_build
                micro=verify_build(c['winner']['build']);micro['_identity_sha256']=sha(resolve_path(c['winner']['build'])/'identity.json')
            value=emit_sources(record,reference,out/'issue-overlay',BASE_FLAGS.split()+['-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_TOPK_KEY32'],raw,micro,local_validation,dsl_source=current_dsl['source'] if current_dsl else None)
            candidate['issue_overlay']=str(out/'issue-overlay/overlay.json');source_files[value['candidate_source']]=sha(value['candidate_source'])
            if record=='G06':candidate['dsl_source']=value['candidate_source']
            else:candidate['imports_source']=value['candidate_source']
        for name in ('imp/imports.json','dsl_tiles.hip','imports.bc','imp_tiles.hip'):
            source_files[str(reference/name)]=sha(reference/name)
    else:
        from .columns.integration import prepare,prepare_grouped
        fleet=resolve_path(c.get('fleet_export',Path.home()/'code/etx-cache/fleet-3fb67a6')).resolve()
        from examples.dsv2lite.pinned import source_sha,FLEET_SRC_SHA256
        require((fleet/'src').is_dir() and source_sha(fleet/'src')==FLEET_SRC_SHA256,'existing frozen Fleet export required; no fetch or mutable checkout')
        control_graph=out/'control-model.py';control_graph.write_text('from examples.dsv2lite.graph import bindings,runtime\nfrom examples.experiment_program.columns.integration import frozen_graph\n'+f'def build():\n    return frozen_graph({batch})\n')
        control=dict(graph=str(control_graph),fleet_export=str(fleet),host=str(ROOT/'examples/dsv2lite/host.hip'))
        raw=None
        if not local_validation:
            from .columns.runner import admit_model_winner
            raw=admit_model_winner(c['winner']['decision'],c['winner']['build'])
        if record=='G02':value=prepare(raw,out/'column-overlay',batch=batch,baseline_revision=DEEPSEEK_BASE,local_validation=local_validation)
        else:value=prepare_grouped(raw,out/'column-overlay',batch=batch,baseline_revision=DEEPSEEK_BASE,host_source=control['host'],local_validation=local_validation)
        candidate=dict(graph=value['graph'],host=value.get('host',control['host']),fleet_export=str(fleet))
        for source in (control_graph,resolve_path(value['graph']),resolve_path(value['source']),resolve_path(candidate['host'])):source_files[str(source)]=sha(source)
    # Reporting uses the already recorded timestamps/vectors after each decode
    # completes. No device probe or additional timed operation is introduced.
    for name,arm in (('control',control),('candidate',candidate)):
        original=resolve_path(arm.get('host',ROOT/'examples/vllm_llm/host.hip'))
        host=out/(name+'-host.hip');host_reporting(original,host,model);arm['host']=str(host)
        source_files[str(original)]=sha(original);source_files[str(host)]=sha(host)
    for path in (resolve_path(__file__),resolve_path(__file__).with_name('validation.py'),resolve_path(__file__).with_name('validation_run.py'),resolve_path(__file__).with_name('validation_trace.py'),resolve_path(__file__).with_name('native_resources.py'),resolve_path(__file__).with_name('validation_dispatch.py')):source_files[str(path)]=sha(path)
    value=dict(schema='etx.g12.preparation.v1',record=record,model=model,batch=batch,baseline_revision=QWEN_BASE if model=='qwen' else DEEPSEEK_BASE,execution_enabled=not local_validation,local_validation=local_validation,winner=winner,control=control,candidate=candidate,source_files=source_files,graph_status='PENDING_EXACT_SAME_KERNEL_GRAPH' if model=='qwen' else 'UNAVAILABLE_LDS_LIFETIME',future_build_command=[sys.executable,'-m','examples.experiment_program.validation','build-models','--preparation',str(out/'preparation.json'),'--out',str(out/'native')],original_config=c)
    if current_dsl:value['current_kept_dsl']=current_dsl
    (out/'preparation.json').write_text(json.dumps(value,indent=2)+'\n');return value


def executable_build(directory,model):
    directory=resolve_path(directory);p=directory/'build-identity.json';identity=json.loads(p.read_text())
    require(identity.get('host_link_exit')==0,'genuine HIP runtime/host link unavailable; no model executable')
    selected_artifacts(directory,identity['hashes'],list(dict.fromkeys(['run','model.hsaco']+[Path(p).name for p in identity['hashes']])));require((directory/'run').is_file() and os.access(directory/'run',os.X_OK),'linked model executable missing')
    if model=='qwen':
        verify_files(identity.get('metadata_files',{}))
        selected_artifacts(directory/'imp',identity['metadata_files'],['imports.json'])
        require(identity.get('check_enabled') is True and bool(identity.get('original_reference_objects')),'full unchanged original checker closure required')
        verify_files(identity['original_reference_objects'])
        manifest=json.loads((directory/'imp/imports.json').read_text())
        require({n for n,d in manifest.items() if not d.get('dsl')}=={'attn_mfma16_g8','attn_reduce','rms_norm_2d'},'unexpected Triton/original import closure')
    return identity


def _compile_import(overlay):
    from .model_build import run
    from .build_support import HOME_CODE,DEPS,cpath
    out=resolve_path(overlay).parent
    cmd=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(HOME_CODE)+':/code','-v',str(HOME_CODE)+':'+str(HOME_CODE)+':ro','-v',str(DEPS)+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu','-e','TMPDIR='+cpath(out/'tmp'),'-w',cpath(ROOT),'etx-rocm22','python3','-m','examples.experiment_program.validation','compile-import','--preparation',cpath(overlay),'--out',cpath(out)]
    (out/'tmp').mkdir(exist_ok=True);run(cmd,out,'selected-import',seconds=240)
    return out/'imports.bc'


def prepare_deepseek(out,arm,batch,source_root):
    """Compile the exact registered K1/M1/l1 graph with a private source overlay."""
    from examples.dsv2lite.arms import ARMS,build_env
    from .model_build import run,compile_closure
    from .build_support import cpath
    out=resolve_path(out);require(shutil.disk_usage(out.parent).free>=2*(1<<30),'storage free floor2GiB');out.mkdir(parents=True,exist_ok=False);(out/'tmp').mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('ETX_','FLEET_'))};e=build_env(ARMS['x'+str(batch)]);env.update(e,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'))
    run([sys.executable,'-m','etx','compile',arm['graph'],'--arch','gfx942','--out',out,'--inline-tiles','--relay','on'],out,'plan',env=env)
    flags=e['ETX_CFLAGS'].split()+['-DFLEET_NT_WEIGHTS=1','-I'+cpath(source_root)]
    return compile_closure(out,out,resolve_path(arm['host']),flags,env_record=e)


def verify_reusable_control(directory,preparation,out,*,arm='control'):
    """Regenerate the exact current control closure without compiling duplicates."""
    from .model_build import run
    from examples.vllm_llm.kept_build import KEPT_ENV,BASE_FLAGS
    d=resolve_path(directory);v=preparation;out=resolve_path(out)
    require(v['model']=='qwen','control reuse currently implemented only for exact Qwen closure')
    require(arm=='control' or (arm=='candidate' and v['record']=='G01'),'only unchanged control or exact G01 candidate guard attestation')
    selected=v[arm];last_router=str(int(selected.get('last_router',False)))
    native=json.loads((d/'build-identity.json').read_text());verify_files(native['hashes']);verify_files(native['dependency_files']);verify_files(native['original_reference_objects']);verify_files(native['reused_imports'])
    reference=resolve_path(selected['reference']);env=native['environment']
    expected=BASE_FLAGS.split()+['-DETX_DSL_NT','-DETX_DSL_VECNORM','-DETX_TOPK_KEY32','-DETX_VL_CHECK']
    require(native['flags']==expected and native['check_enabled'] is True,'unchanged control complete math/check flags differ')
    require(all(env.get(k)==value for k,value in KEPT_ENV.items()) and env.get('ETX_VL_LAST_ROUTER')==last_router,'control is not the unchanged complete kept recipe')
    require(native['host_source_sha256']==sha(selected['host']),'compiled private control host differs')
    require(len(native['reused_imports'])==1 and next(iter(native['reused_imports'].values()))==sha(reference/'imports.bc'),'control original import bundle differs')
    require(original_import_abi(json.loads((d/'imp/imports.json').read_text()))==original_import_abi(json.loads((reference/'imp/imports.json').read_text())) and sha(env['ETX_VL_ADAPTERS'])==sha(reference/'imp_tiles.hip'),'control original import ABI/adapters differ')
    out.mkdir(exist_ok=False);(out/'tmp').mkdir();(out/'imp').mkdir()
    shutil.copyfile(d/'imp/imports.json',out/'imp/imports.json')
    fresh={k:value for k,value in os.environ.items() if not k.startswith(('ETX_','DSL_ABLATE_','BF16_'))}
    fresh.update(KEPT_ENV,ETX_VL_LAST_ROUTER=last_router,ETX_VL_IMPORTS=str(out/'imp/imports.json'),ETX_VL_ADAPTERS=str(reference/'imp_tiles.hip'),ETX_VL_DSL_TILES=str(out/'dsl_tiles.hip'),ETX_VL_CHAINS=str(out/'chain_tiles.hip'),ETX_VL_PLAN=str(out/'vl_plan.txt'),ETX_LLM_CONFIG=str(ROOT/'examples/llm/configs/qwen3-30b-a3b.json'),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'))
    run([sys.executable,'-m','etx.ops.emit',out/'imp/imports.json','--tiles',out/'dsl_tiles.hip','--ops','gemv,topk,attnprep'],out,'current-control-dsl',env=fresh)
    require(sha(out/'imp/imports.json')==sha(d/'imp/imports.json'),'current original ABI metadata differs')
    require(sha(out/'dsl_tiles.hip')==sha(d/'dsl_tiles.hip'),'current emitted full control DSL differs')
    # Refer to that identical immutable DSL path so generated source equality
    # includes every include directive. No path/body normalization is used.
    fresh['ETX_VL_DSL_TILES']=str(d/'dsl_tiles.hip')
    run([sys.executable,'-m','etx','compile','examples/vllm_llm/model.py','--arch','gfx942','--out',out,'--inline-tiles'],out,'current-control-plan',env=fresh,seconds=180)
    exact={}
    for name in ('dsl_tiles.hip','megakernel_d0.hip','plan_data.h','etx_lowering.h','plan.json','vl_plan.txt'):
        require(sha(out/name)==sha(d/name),'current generated full control closure differs: '+name);exact[name]=sha(d/name)
    verify_files(native['hashes'])
    value=dict(status='REUSE_EXACT_IMMUTABLE_NATIVE_CONTROL',native_directory=str(d),native_identity_sha256=sha(d/'build-identity.json'),exact_current_generated_files=exact,host_source_sha256=native['host_source_sha256'],complete_flags=expected,output_location_only_exceptions={key:fresh[key] for key in ('ETX_VL_IMPORTS','ETX_VL_CHAINS','ETX_VL_PLAN')},readonly_identical_dsl_input=fresh['ETX_VL_DSL_TILES'],no_recompile_or_relabel=True,gpu_certificate_or_timing_transferred=False,proof_files={str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'reuse.json').write_text(json.dumps(value,indent=2)+'\n');return native,value


def attest_guard_only_models(previous_models,previous_preparation,current_preparation,out):
    """Bind unchanged real G01 objects to fresh runtime guards without relabeling.

    Both complete native arms must still verify; each complete current source
    closure is freshly emitted and byte-compared, without target recompilation.
    Missing host runtime and unmeasured GPU gates are copied unchanged.
    """
    oldp=resolve_path(previous_preparation);newp=resolve_path(current_preparation);mp=resolve_path(previous_models)
    old=json.loads(oldp.read_text());new=json.loads(newp.read_text());models=json.loads(mp.read_text())
    require(models['preparation_sha256']==sha(oldp),'previous native preparation seal differs')
    require(all(v.get('record')=='G01' and v.get('model')=='qwen' and v.get('batch')==1 for v in (old,new)),'guard-only attestation restricted to complete G01 pair')
    guard_names={'validation_prepare.py','validation.py','validation_run.py','validation_trace.py','native_resources.py','validation_dispatch.py'}
    changed={}
    for path,digest in old['source_files'].items():
        p=resolve_path(path)
        if sha(p)!=digest:
            require(p.parent==ROOT/'examples/experiment_program' and p.name in guard_names,'changed non-guard source forbids native reuse: '+str(p))
            changed[str(p)]={'previous_sha256':digest,'current_sha256':sha(p)}
    verify_files(new['source_files'])
    for arm in ('control','candidate'):
        require({k:v for k,v in old[arm].items() if k!='host'}=={k:v for k,v in new[arm].items() if k!='host'},'native arm arguments changed')
        require(sha(old[arm]['host'])==sha(new[arm]['host']),'native host bytes changed')
    out=resolve_path(out);out.mkdir(parents=True,exist_ok=False);proofs={}
    for arm in ('control','candidate'):
        d=resolve_path(models['arms'][arm]['directory'])
        require(sha(d/'build-identity.json')==models['arms'][arm]['identity_sha256'],'native identity seal differs')
        _,proofs[arm]=verify_reusable_control(d,new,out/(arm+'-current-proof'),arm=arm)
    resource=model_resource_admission({arm:resolve_path(models['arms'][arm]['directory']) for arm in ('control','candidate')},'qwen',1)
    attestation=dict(schema='etx.g12.guard-only-native-attestation.v1',previous_models=str(mp),previous_models_sha256=sha(mp),previous_preparation=str(oldp),previous_preparation_sha256=sha(oldp),current_preparation=str(newp),current_preparation_sha256=sha(newp),changed_guard_sources=changed,arm_proofs=proofs,native_objects_recompiled=False,native_identities_unchanged=True,gpu_certificate_or_timing_transferred=False)
    ap=out/'guard-attestation.json';ap.write_text(json.dumps(attestation,indent=2)+'\n')
    result=dict(models,preparation_sha256=sha(newp),resource_admission=resource,guard_only_attestation=str(ap),guard_only_attestation_sha256=sha(ap))
    (out/'models.json').write_text(json.dumps(result,indent=2)+'\n');return result


def guard_attestation_required(models,preparation):
    """Replay the exact guard-only proof and return every mandatory input seal."""
    if models.get('linked_guard_reuse'):
        from .review_guard_refresh import verify_models
        return verify_models(models,preparation)
    if not models.get('guard_only_attestation'):return set()
    p=resolve_path(preparation);v=json.loads(p.read_text());ap=resolve_path(models['guard_only_attestation'])
    require(sha(ap)==models['guard_only_attestation_sha256'],'guard attestation seal differs')
    a=json.loads(ap.read_text())
    require(a['schema']=='etx.g12.guard-only-native-attestation.v1','unknown guard attestation')
    require(a['current_preparation_sha256'] in (sha(p),v.get('activation_source_preparation_sha256')),'guard attestation current preparation differs')
    require(v['record']=='G01' and v['model']=='qwen' and v['batch']==1,'guard attestation mechanism differs')
    oldm=resolve_path(a['previous_models']);oldp=resolve_path(a['previous_preparation'])
    require(sha(oldm)==a['previous_models_sha256'] and sha(oldp)==a['previous_preparation_sha256'],'original guard attestation metadata changed')
    old=json.loads(oldm.read_text())
    require(old['preparation_sha256']==sha(oldp) and models['arms']==old['arms'],'guard attestation native identities changed')
    require(a['native_objects_recompiled'] is False and a['native_identities_unchanged'] is True and a['gpu_certificate_or_timing_transferred'] is False,'guard attestation scope differs')
    required={ap.resolve(),oldm.resolve(),oldp.resolve()}
    require(set(a['arm_proofs'])=={'control','candidate'},'guard attestation requires both complete arms')
    for arm,proof in a['arm_proofs'].items():
        d=resolve_path(models['arms'][arm]['directory']);ip=d/'build-identity.json'
        require(sha(ip)==models['arms'][arm]['identity_sha256']==proof['native_identity_sha256'],'guard proof native identity changed')
        native=json.loads(ip.read_text())
        for key in ('hashes','dependency_files','original_reference_objects','reused_imports'):verify_files(native[key])
        require(resolve_path(proof['native_directory']).resolve()==d.resolve(),'guard proof arm directory differs')
        require(proof['complete_flags']==native['flags'] and proof['host_source_sha256']==native['host_source_sha256']==sha(v[arm]['host']),'guard proof flags/host changed')
        verify_files(proof['proof_files'])
        normalized_proofs={resolve_path(path).resolve():digest for path,digest in proof['proof_files'].items()}
        rp=ap.parent/(arm+'-current-proof/reuse.json')
        require(json.loads(rp.read_text())==proof,'guard proof JSON differs')
        for name in ('dsl_tiles.hip','megakernel_d0.hip','plan_data.h','etx_lowering.h','plan.json','vl_plan.txt'):
            fresh=rp.parent/name
            require(fresh.resolve() in normalized_proofs and proof['exact_current_generated_files'][name]==sha(d/name)==sha(fresh),'guard proof generated closure differs: '+name)
        required.update((ip.resolve(),rp.resolve()))
        required.update(resolve_path(path).resolve() for path in proof['proof_files'])
    for path,change in a['changed_guard_sources'].items():
        require(v['source_files'].get(path)==change['current_sha256']==sha(path),'current runtime guard source differs')
    return required


def build_models(preparation,out,*,reuse_control=None):
    from .model_build import prepare_qwen,run
    p=resolve_path(preparation);v=json.loads(p.read_text());verify_files(v['source_files'])
    out=resolve_path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    result={};directories={};reuse=None
    if v['model']=='deepseek':
        source_root=out/'fleet_src';shutil.copytree(resolve_path(v['control']['fleet_export'])/'src',source_root)
        from examples.dsv2lite.arms import ARMS,build_env
        e=build_env(ARMS['x'+str(v['batch'])]);run([sys.executable,'-m','examples.dsv2lite.patch_fleet',source_root,*e['ETX_FLEET_PATCH'].split()],out,'registered-fleet-patches')
    for name in ('control','candidate'):
        arm=v[name]
        if name=='control' and reuse_control is not None:
            try:
                result[name],reuse=verify_reusable_control(reuse_control,v,out/'control-reuse-proof');directories[name]=resolve_path(reuse_control);continue
            except ValueError as error:
                reuse=dict(status='REUSE_REFUSED_FRESH_CONTROL_REQUIRED',reason=str(error),original_native_directory=str(resolve_path(reuse_control)))
                (out/'control-reuse-refused.json').write_text(json.dumps(reuse,indent=2)+'\n')
        directories[name]=out/name
        if v['model']=='qwen':
            imports=_compile_import(arm['issue_overlay']) if arm.get('imports_source') else None
            result[name]=prepare_qwen(out/name,resolve_path(arm['reference']),last_router=arm.get('last_router',False),overlay=resolve_path(arm['dsl_source']) if arm.get('dsl_source') else None,imports_override=imports,host_override=resolve_path(arm['host']))
        else:result[name]=prepare_deepseek(out/name,arm,v['batch'],source_root)
    resource=model_resource_admission(directories,v['model'],v['batch'])
    (out/'resource-admission.json').write_text(json.dumps(resource,indent=2)+'\n')
    value=dict(resource_admission=resource,preparation_sha256=sha(p),model=v['model'],record=v['record'],batch=v['batch'],local_validation=v['local_validation'],control_reuse=reuse,arms={name:dict(directory=str(directories[name]),identity_sha256=sha(directories[name]/'build-identity.json'),host_link_exit=result[name]['host_link_exit']) for name in result},executable_models=all(r['host_link_exit']==0 for r in result.values()),gpu_executed=False,graph_status=v['graph_status'])
    (out/'models.json').write_text(json.dumps(value,indent=2)+'\n');return value


def activate(preparation,winner,out):
    from .validation import require_admission
    path=resolve_path(preparation);v=json.loads(path.read_text());verify_files(v['source_files']);receipt=require_admission(winner)
    require(receipt['record']==v['record'],'winner differs from prepared mechanism')
    expected=v['original_config'].get('variant','g09' if v['record']=='G09' else 'g10_up')
    if v['record'] in ('G09','G10'):require(winner.get('variant')==expected,'row variant differs from prepared source')
    if v['record'] in ('G02','G05'):
        from .columns.runner import admit_model_winner
        actual=admit_model_winner(winner['decision'],winner['build'])
        require(actual['batch']==v['batch'],'column winner batch differs from prepared model')
        if v['record']=='G02':require(actual['role']=='qkv','only QKV is prepared')
        else:require(actual['role']=='gate_up_down' and actual['matched_up_down'],'both actual routed up/down roles required')
    v.update(local_validation=False,execution_enabled=True,winner=receipt,activation_source_preparation_sha256=sha(path))
    v['original_config']['winner']=winner
    out=resolve_path(out)
    with out.open('x') as f:f.write(json.dumps(v,indent=2)+'\n')
    return v


def benchmark_source(out):
    """Private batch benchmark with the existing registered lever echo contract."""
    old=(ROOT/'examples/llm/vllm_bench.py').read_text()
    start=old.index('        print(encode_config({"cudagraph_mode":')
    end=old.index('    print(f"VLLM-BATCH-MODE',start)
    new='''        print(encode_config({"async_scheduling":config.scheduler_config.async_scheduling,
            "cudagraph_mode":config.compilation_config.cudagraph_mode,
            "pass_config":config.compilation_config.pass_config,"kernel_config":config.kernel_config,
            "attention_backend":config.attention_config.backend,
            "env":{k:v for k,v in sorted(os.environ.items()) if k.startswith(("VLLM_","HIP_"))}}),flush=True)
    print(f"VLLM-SAMPLING detok={int(not a.no_detok)}",flush=True)
'''
    resolve_path(out).write_text(old[:start]+new+old[end:]);return str(resolve_path(out).resolve())


def host_reporting(original,out,model):
    text=resolve_path(original).read_text()
    if model=='qwen':
        setup='  const int progress = getenv("ETX_VL_PROGRESS")'
        require(text.count(setup)==1,'Qwen trace guard setup anchor')
        guard='''  int g12_trace_position=-1; bool g12_trace_done=false; std::string g12_trace_file;
  if(const char* selected=getenv("ETX_PROGRAM_TRACE_POS")) {
    if(strcmp(selected,"0") && strcmp(selected,"1040")) die("G12 trace position must be0 or1040");
    if(!getenv("ETX_TRACE") || !getenv("ETX_TRACE_RAW") || nreq!=1 || repeat!=2 || gen!=32 || teacher || unfused || check_layers || check_dsl)
      die("G12 trace requires separate fused full32 invocation,repeat2 and exact runtime trace output");
    g12_trace_position=atoi(selected);g12_trace_file=getenv("ETX_TRACE_RAW");
  }
'''
        text=text.replace(setup,guard+setup,1)
        after='      if (t + 1 >= (int)prompt.size()) wall_ms.push_back(wms);'
        require(text.count(after)==1,'Qwen post-completion trace anchor')
        text=text.replace(after,after+'''
      if(g12_trace_position>=0 && run==repeat-1 && t==g12_trace_position) {
        hst.report_phases(etx_type_names,ETX_N_TYPES);
        unsetenv("ETX_TRACE_RAW"); // the end-of-run phase report must not overwrite this exact step
        g12_trace_done=true;
        printf("G12TRACE position=%d run=%d tasks=%d completed=1\\n",t,run,ETX_N_TASKS);
      }
''',1)
        finish='  // --check-dsl: the last layer'
        require(text.count(finish)==1,'Qwen full32 trace finish anchor')
        text=text.replace(finish,'  if(g12_trace_position>=0 && !g12_trace_done) die("G12 selected trace step not exported");\n'+finish,1)
        anchor='    step_ms.clear();\n    for (int i = 0; i < gen; ++i)'
        require(text.count(anchor)==1,'Qwen native reporting source guard')
        emit='''    for(int i=0;i<gen;++i) { const int t=(int)prompt.size()-1+i;
      printf("G12TOKEN run=%d request=0 token=%d start_ticks=%llu end_ticks=%llu wall_ms=%.9f\\n",run,i,
             (unsigned long long)t0[t],(unsigned long long)t1[t],wall_ms.at(i));
    }
'''
    else:
        anchor='  std::printf("BATCH requests=%d tokens=%d step_median_ms=%.6f\\n", R, tokens, step_med);'
        require(text.count(anchor)==1,'DeepSeek native reporting source guard')
        emit='''  for(int r=0;r<R;++r)for(int i=0;i<tokens;++i)
    std::printf("G12TOKEN run=1 request=%d token=%d start_ticks=%llu end_ticks=%llu wall_ms=%.9f\\n",r,i,
      (unsigned long long)reqs[r].times[2*i],(unsigned long long)reqs[r].times[2*i+1],step_ms.at(i));
'''
    resolve_path(out).write_text(text.replace(anchor,emit+anchor,1));return str(resolve_path(out).resolve())


def model_resource_admission(builds,model,batch):
    """Actual whole persistent kernel capacity; no operator-allocation transfer."""
    from .native_resources import persistent_capacity
    data={}
    for name,directory in builds.items():
        proof=persistent_capacity(directory)
        data[name]=dict(resources=proof['raw_metadata'],threads=proof['threads'],resident_capacity=proof['resident_capacity'],required_persistent_blocks=proof['required_persistent_blocks'],plan_sha256=proof['plan_sha256'],assembly_sha256=proof['assembly_sha256'],physical_capacity_proof=proof)
    c=data['control'];x=data['candidate'];passed=all(d['resident_capacity']>=d['required_persistent_blocks'] for d in data.values())
    passed &= x['resources']['scratch_bytes_lane']<=c['resources']['scratch_bytes_lane'] and x['resources']['vgpr_spills']<=c['resources']['vgpr_spills'] and x['resources']['sgpr_spills']<=c['resources']['sgpr_spills']
    return dict(admitted=bool(passed),arms=data,gpu_residency='UNMEASURED',scope='static whole-model capacity and no new spills; not device correctness')
