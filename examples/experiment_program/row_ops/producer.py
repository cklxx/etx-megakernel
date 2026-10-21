"""Original-checker provenance; a literal baseline label is never admission."""
import json
from pathlib import Path
from examples.experiment_program.protocol import require,sha,verify_files,resolve_path,selected_artifacts
BASE='b573664faa4e7988dff353bc03aca4ce8561a729'

def verify(meta,pack):
    path=resolve_path(meta['path']);require(sha(path)==meta['sha256'],'original producer certificate changed')
    c=json.loads(path.read_text());require(c.get('schema')=='etx.row.original-producer.v1' and c.get('baseline_revision')==BASE,'wrong original producer baseline')
    require(c.get('producer_kind')=='unchanged-original-checker' and c.get('exit')==0,'candidate/failed original producer')
    verify_files(c['files']);verify_files(c['inputs']);verify_files(c['exports'])
    require(c['files'].get(str(resolve_path(c['exit_file'])))==sha(c['exit_file']) and resolve_path(c['exit_file']).read_text().strip()=='0','raw original checker exit failed/incomplete')
    log=resolve_path(c['check_log']);require(c['files'].get(str(log))==sha(log),'original checker log missing/unbound')
    import re
    text=log.read_text();require('CHECK OK: 0 launch(es) differ' in text,'complete original checker correctness missing')
    if c.get('capture_format','row-v1')=='row-v1':
        require(f"ROW-CAPTURE original folded native snapshots={c['capture_directory']}" in text,'references not exported by this checking execution')
    else:require(c['capture_format']=='bf16-native-v1','unknown original capture format')
    build=resolve_path(c['build']);identity=json.loads((build/'build-identity.json').read_text())
    require(c['files'].get(str(build/'build-identity.json'))==sha(build/'build-identity.json'),'original build identity unbound')
    require(identity.get('host_link_exit')==0 and identity.get('check_enabled') is True,'original checker not genuinely linked')
    verify_files(identity['original_reference_objects']);verify_files(identity['dependency_files'])
    selected_artifacts(build,identity['hashes'],['run','model.hsaco','dsl_tiles.hip','plan.json'])
    from examples.vllm_llm import import_certificate as cert
    fingerprint=build/'build_identity.json';require(c['files'].get(str(fingerprint))==sha(fingerprint),'original checker fingerprint absent/unbound')
    checked=cert.verified_build(build,resolve_path(c['model_dir']))
    expected=cert.expected_instances((build/'vl_plan.txt').read_text(),json.loads((build/'imp/imports.json').read_text()))
    require(len(expected)==97,'original checked launch population differs')
    cert.check_output(text,expected,pack['position'],checked)
    require(identity['flags']==c['math_flags'] and identity.get('host_source_sha256')==c['capture_host_sha256'],'original compiler/math/host identity differs')
    # Independently derive the allowed checking-host hook from the frozen source.
    import subprocess,tempfile
    from .capture import checking_host
    root=Path(__file__).resolve().parents[3]
    original=subprocess.check_output(['git','show',BASE+':examples/vllm_llm/host.hip'],cwd=root)
    with tempfile.TemporaryDirectory() as d:
        inp=Path(d)/'original.hip';out=Path(d)/'capture.hip';inp.write_bytes(original);checking_host(inp,out)
        expected=sha(out) if c.get('capture_format','row-v1')=='row-v1' else sha(inp)
        require(expected==c['capture_host_sha256'],'candidate or changed original checking host')
    authority=json.loads((Path(__file__).with_name('original-contract.json')).read_text())
    verify_files(authority['compiler_files']);verify_files(identity['runtime_files'])
    require(sha(build/'model.hsaco')==authority['model.hsaco'],'candidate-produced or changed original device module')
    verify_files(identity['reused_imports'])
    require(set(identity['reused_imports'].values())=={authority['imports.bc']},'original imports differ')
    for name in ('dsl_tiles.hip','plan.json'):
        require(sha(build/name)==authority[name],'original module arithmetic/plan/import contract differs: '+name)
    require(c.get('position')==pack['position'] and c.get('layer')==pack['layer'] and c.get('role')==pack['role'],'original capture population differs')
    if c.get('capture_format')=='bf16-native-v1':
        original=resolve_path(c['original_pack']);require(c['files'].get(str(original))==sha(original),'historical original pack unbound')
        old=json.loads(original.read_text());reference=old['reference']
        require(reference['original_host_sha256']==sha(build/'run') and reference['plan_sha256']==sha(build/'plan.json') and reference['check_log_sha256']==sha(log),'historical producer identity differs before adaptation')
        from examples.vllm_llm.bf16_300.launch import validate_pack
        validate_pack(old,resolve_path(reference['file']),resolve_path(old['arguments']),old['layer'],old['first'],0,0)
    if c.get('capture_format','row-v1')=='row-v1':
        raw_path=resolve_path(c['capture_directory'])/'capture.json'
        require(c['exports'].get(str(raw_path))==sha(raw_path),'raw native capture metadata not bound to execution')
        original=json.loads(raw_path.read_text())
        def shape(value):
            if isinstance(value,dict):return {k:shape(v) for k,v in value.items() if k!='sha256'}
            if isinstance(value,list):return [shape(v) for v in value]
            return value
        for key in ('role','position','layer','first','count','arguments','buffers','pointers','references'):
            require(shape(original[key])==shape(pack[key]),'capture metadata differs from original producer: '+key)
    argument=resolve_path(pack['arguments']['file']) if isinstance(pack.get('arguments'),dict) else None
    if argument is not None:require(c['inputs'].get(str(argument),c['exports'].get(str(argument)))==sha(argument),'original argument block not bound to execution')
    for ref in pack['references']:
        require(c['exports'].get(str(resolve_path(ref['file'])))==sha(ref['file']),'reference not bound to original execution')
    for b in pack['buffers']:
        require(str(resolve_path(b['file'])) in c['inputs'] or str(resolve_path(b['file'])) in c['exports'],'native operand not bound to original execution')
    return c

def seal(recipe,out):
    c=json.loads(resolve_path(recipe).read_text())
    c['schema']='etx.row.original-producer.v1'
    for key in ('files','inputs','exports'):
        require(isinstance(c[key],list) and c[key],'explicit producer '+key+' population required')
        c[key]={str(resolve_path(p)):sha(p) for p in c[key]}
    require(c['files'].get(str(resolve_path(c['exit_file'])))==sha(c['exit_file']),'original exit receipt not sealed')
    require(resolve_path(c['exit_file']).read_text().strip()=='0','original checker exit failed/incomplete')
    c['exit']=0
    identity=json.loads((resolve_path(c['build'])/'build-identity.json').read_text())
    c['math_flags']=identity['flags'];c['capture_host_sha256']=identity['host_source_sha256']
    with resolve_path(out).open('x') as f:f.write(json.dumps(c,indent=2)+'\n')
    meta=dict(path=str(resolve_path(out)),sha256=sha(out))
    if c.get('capture_format','row-v1')=='row-v1':raw=json.loads((resolve_path(c['capture_directory'])/'capture.json').read_text())
    else:
        original=json.loads(resolve_path(c['original_pack']).read_text())
        raw=dict(position=original['position'],layer=original['layer'],role='dsl_'+original['role'],references=[original['reference']],buffers=original['buffers'])
    verify(meta,raw)
    return meta
