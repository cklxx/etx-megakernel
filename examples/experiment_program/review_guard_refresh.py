"""Bounded review repair: rebind Python guards, never relabel compiled kernels."""
import ast,json,os,subprocess
from pathlib import Path
from .protocol import require,sha,verify_files,resolve_path
from .build_support import ROOT
REVIEW='94a29c0f3e51e271ea884eb216de5b90c4a7b25b'
GUARDS={'validation_prepare.py','validation.py','validation_run.py','validation_trace.py','validation_dispatch.py','native_resources.py'}

def unchanged_generators():
    require(not subprocess.check_output(['git','diff',REVIEW,'--name-only','--','etx','examples/vllm_llm','examples/dsv2lite','examples/llm'],cwd=ROOT,text=True),'native core source changed; review guard refresh forbidden')
    path=ROOT/'examples/experiment_program/validation_prepare.py'
    original=ast.parse(subprocess.check_output(['git','show',REVIEW+':examples/experiment_program/validation_prepare.py'],cwd=ROOT,text=True));current=ast.parse(path.read_text())
    names={'prepare_sources','current_kept_dsl','original_import_abi','host_reporting','_compile_import','prepare_deepseek','model_resource_admission'}
    functions=lambda tree:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names}
    require(functions(original)==functions(current) and set(functions(current))==names,'native generator implementation changed')
    return dict(reviewed_revision=REVIEW,native_core_diff=[],generator_functions=sorted(names))

def refresh_issue(source,out):
    from .issue_order.paths import dependency_path
    source=resolve_path(source);out=resolve_path(out);old=json.loads((source/'identity.json').read_text())
    for name,digest in old['hashes'].items():require(sha(source/name)==digest,'original issue artifact changed')
    for name,digest in old['dependency_files'].items():require(sha(dependency_path(name))==digest,'original native dependency changed')
    changed={p:dict(previous=h,current=sha(ROOT/p)) for p,h in old['source_hashes'].items() if sha(ROOT/p)!=h}
    allowed={'examples/experiment_program/issue_order/__main__.py','examples/experiment_program/protocol.py'}
    require(set(changed)<=allowed,'changed compiled issue source; no guard-only refresh')
    core=unchanged_generators();out.mkdir(exist_ok=False)
    for p in source.rglob('*'):
        if p.is_file() and p!=source/'identity.json':
            target=out/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            os.link(p,target)
    proof=dict(schema='etx.review.issue-guard-reuse.v1',previous_identity=str(source/'identity.json'),previous_identity_sha256=sha(source/'identity.json'),changed_python_guards=changed,native_artifacts_recompiled=False,native_artifact_hashes=old['hashes'],core=core,gpu_evidence_transferred=False)
    (out/'guard-reuse.json').write_text(json.dumps(proof,indent=2)+'\n')
    value=dict(old,source_hashes={p:sha(ROOT/p) for p in old['source_hashes']},review_guard_reuse=str(out/'guard-reuse.json'))
    value['source_hashes']['examples/experiment_program/telemetry.py']=sha(ROOT/'examples/experiment_program/telemetry.py')
    value['hashes']=dict(old['hashes'],**{'guard-reuse.json':sha(out/'guard-reuse.json')})
    (out/'identity.json').write_text(json.dumps(value,indent=2)+'\n');return value

def refresh_models(previous_preparation,previous_models,linked_arms,out):
    core=unchanged_generators();pp=resolve_path(previous_preparation);mp=resolve_path(previous_models);old=json.loads(pp.read_text());models=json.loads(mp.read_text())
    require(models['preparation_sha256']==sha(pp) and models['record'] in ('G01','G06','G07'),'exact eligible reviewed model pair required')
    changes={};source_files={}
    for path,digest in old['source_files'].items():
        p=resolve_path(path);current=sha(p)
        if current!=digest:
            require(p.parent==ROOT/'examples/experiment_program' and p.name in GUARDS,'non-guard prepared source changed')
            changes[path]=dict(previous=digest,current=current)
        source_files[path]=current
    proofs={};arms={}
    for arm,linked in linked_arms.items():
        d=resolve_path(linked);n=json.loads((d/'build-identity.json').read_text());verify_files(n['hashes']);verify_files(n['dependency_files']);verify_files(n['runtime_files'])
        prior=resolve_path(models['arms'][arm]['directory']);previous=json.loads((prior/'build-identity.json').read_text());verify_files(previous['hashes'])
        require(sha(prior/'build-identity.json')==models['arms'][arm]['identity_sha256'],'reviewed native arm changed')
        require(n['host_link_exit']==0 and n['flags']==previous['flags'] and n['host_source_sha256']==previous['host_source_sha256']==sha(old[arm]['host']),'native math/host changed')
        names=('model.hsaco','mk.s','dsl_tiles.hip','megakernel_d0.hip','plan_data.h','plan.json','vl_plan.txt')
        require(all(sha(d/name)==sha(prior/name) for name in names),'reviewed device closure changed')
        proofs[arm]=dict(previous_directory=str(prior),previous_identity_sha256=sha(prior/'build-identity.json'),linked_directory=str(d),linked_identity_sha256=sha(d/'build-identity.json'),unchanged_files={name:sha(d/name) for name in names},host_source_sha256=n['host_source_sha256'],flags=n['flags'])
        arms[arm]=dict(directory=str(d),identity_sha256=sha(d/'build-identity.json'),host_link_exit=0)
    require(set(arms)=={'control','candidate'},'both linked arms required')
    out=resolve_path(out);out.mkdir(exist_ok=False);prep=dict(old,source_files=source_files)
    (out/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    proof=dict(schema='etx.review.linked-guard-reuse.v1',previous_preparation=str(pp),previous_preparation_sha256=sha(pp),previous_models=str(mp),previous_models_sha256=sha(mp),core=core,changed_guards=changes,arms=proofs,gpu_evidence_transferred=False)
    ap=out/'linked-guard-reuse.json';ap.write_text(json.dumps(proof,indent=2)+'\n')
    from .validation_prepare import model_resource_admission
    result={k:v for k,v in models.items() if k not in ('control_reuse','guard_only_attestation','guard_only_attestation_sha256')}
    result.update(preparation_sha256=sha(out/'preparation.json'),arms=arms,executable_models=True,linked_guard_reuse=str(ap),linked_guard_reuse_sha256=sha(ap),resource_admission=model_resource_admission({a:resolve_path(v['directory']) for a,v in arms.items()},'qwen',1))
    (out/'models.json').write_text(json.dumps(result,indent=2)+'\n');return result

def verify_models(models,preparation):
    p=resolve_path(models['linked_guard_reuse']);require(sha(p)==models['linked_guard_reuse_sha256'],'linked guard reuse receipt changed');proof=json.loads(p.read_text());v=json.loads(resolve_path(preparation).read_text())
    require(proof['schema']=='etx.review.linked-guard-reuse.v1' and proof['core']==unchanged_generators() and proof['gpu_evidence_transferred'] is False,'linked guard reuse scope differs')
    verify_files(v['source_files']);required={p,resolve_path(proof['previous_models']),resolve_path(proof['previous_preparation'])}
    require(sha(proof['previous_models'])==proof['previous_models_sha256'] and sha(proof['previous_preparation'])==proof['previous_preparation_sha256'],'original guard/native preparation changed')
    for arm,row in proof['arms'].items():
        d=resolve_path(row['linked_directory']);require(models['arms'][arm]['directory']==str(d) and sha(d/'build-identity.json')==row['linked_identity_sha256']==models['arms'][arm]['identity_sha256'],'selected linked model changed')
        prior=resolve_path(row['previous_directory']);require(sha(prior/'build-identity.json')==row['previous_identity_sha256'],'original native identity changed')
        for name,digest in row['unchanged_files'].items():require(sha(d/name)==sha(prior/name)==digest,'native device reuse differs')
        native=json.loads((d/'build-identity.json').read_text());require(native['host_source_sha256']==sha(v[arm]['host'])==row['host_source_sha256'] and native['flags']==row['flags'],'linked host/flags changed')
        required|={d/'build-identity.json',prior/'build-identity.json'}
    return required
