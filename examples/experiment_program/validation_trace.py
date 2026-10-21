"""Separate exact-step runtime traces; no subtraction from timed pairs.

The ordinary runtime stamps body-done before event publication. These records
establish the current graph/queue context, not exact publication timestamps or
removable/model time. G11 sparse replay remains a separate clock population.
"""
from __future__ import annotations
import json,shutil,struct
from pathlib import Path
from .protocol import require,sha,verify_files,resolve_path


def options(c):
    return dict(fold_decision_active=c.get('fold_decision_active',False),trace_positions=tuple(c.get('fold_trace_positions',[1040])))


def validate_raw(plan_file,raw_file):
    p=json.loads(resolve_path(plan_file).read_text());raw=resolve_path(raw_file).read_bytes()
    require(len(raw)==len(p['descs'])*72,'trace must contain every planned task exactly once')
    rows=list(struct.iter_unpack('<9q',raw));previous={}
    for queue in p['static_queues'].values():
        for i,tid in enumerate(queue):previous[tid]=queue[i-1] if i else None
    require(len(p['task_in'])==len(rows),'exact emitted dependency population required')
    for i,(d,row) in enumerate(zip(p['descs'],rows)):
        coords=tuple(d['coord'])+(0,)*(4-len(d['coord']))
        require(d['id']==i and row[:5]==(d['type'],*coords),'trace type/coordinate differs from exact plan')
        require(0<row[5]<=row[6]<=row[7] and 0<=row[8]<p['domains']*p['workers_per_domain'],'missing/invalid completed task trace')
        if d.get('mode','static')=='static':require(row[8]==d['worker'] and i in previous,'static worker/queue differs')
        for event in p['task_in'][i]:
            producers=p['ev_inst'][event][2];require(producers and all(0<=t<len(rows) for t in producers),'missing actual event producers')
            require(all(rows[t][7]<=row[6] for t in producers),'producer body-done exceeds consumer deps-ready')
        prior=previous.get(i)
        if prior is not None:require(rows[prior][7]<=row[5],'static queue predecessor timing differs')
    return dict(tasks=len(rows),complete=True,publication_timestamp_available=False,
                boundary='taken / deps-ready / all-wave body-drain-done BEFORE event releases-arrives',
                interpretation='current dependency/queue context only; no exact publication or removable-work price')


def compiled_contract(directory):
    d=resolve_path(directory);native=json.loads((d/'build-identity.json').read_text());verify_files(native['hashes'])
    def file(name):
        matches=[value for path,value in native['hashes'].items() if Path(path).name==name]
        require(len(matches)==1,'missing/ambiguous native '+name);return matches[0]
    return dict(device_module_sha256=file('model.hsaco'),device_assembly_sha256=file('mk.s'),device_source_sha256=file('megakernel_d0.hip'),dsl_source_sha256=file('dsl_tiles.hip'),plan_sha256=file('plan.json'),imports_sha256=sorted(native.get('reused_imports',{}).values()),flags=native['flags'])


def reserve(models,out,positions):
    size=16<<20
    for arm in ('control','candidate'):
        d=resolve_path(models['arms'][arm]['directory']);p=json.loads((d/'plan.json').read_text())
        size+=len(positions)*(len(p['descs'])*72+(d/'plan.json').stat().st_size+2*(d/'build-identity.json').stat().st_size)
    require(shutil.disk_usage(resolve_path(out)).free>=size+(2<<30),'trace export storage plus protected2GiB floor unavailable')
    return size


def create(c,models,out,row):
    from .validation import qwen_correct
    from examples.vllm_llm import import_certificate as cert
    out=resolve_path(out);arm=row['arm'];position=row['ordinal'];native=resolve_path(models['arms'][arm]['directory'])
    directory=out/'trace'/f'{arm}_p{position}';directory.mkdir(exist_ok=False)
    plan=directory/'plan.json';shutil.copyfile(native/'plan.json',plan)
    for source,name in [(native/'build-identity.json','native-build-identity.json'),(out/arm/'build_identity.json','build_identity.json')]:shutil.copyfile(source,directory/name)
    raw=out/'trace'/f'{arm}_p{position}.bin';log=out/'trace'/f'{arm}_p{position}.log'
    identity=json.loads((directory/'build_identity.json').read_text());qwen_correct(log.read_text(),identity)
    marker=f'G12TRACE position={position} run=1 tasks={len(json.loads(plan.read_text())["descs"])} completed=1'
    require(log.read_text().splitlines().count(marker)==1,'missing exact post-step trace identity marker')
    checked=cert.validate(out/arm,timing=False);require(checked['checked']==checked['planned']==97,'trace requires original97 same-build import gates')
    population=validate_raw(plan,raw);preparation=json.loads(resolve_path(c['preparation']).read_text())
    host=resolve_path(preparation[arm]['host']);native_identity=json.loads((native/'build-identity.json').read_text())
    require(sha(host)==native_identity['host_source_sha256'] and 'ETX_PROGRAM_TRACE_POS' in host.read_text(),'compiled exact-step host trace guard unavailable')
    model=resolve_path(c['model_dir']);inputs={str(model/name):sha(model/name) for name in ('vl.bin','vl.manifest','golden_c1024.txt')}
    files={str(p.resolve()):sha(p) for p in (raw,log,plan,directory/'native-build-identity.json',directory/'build_identity.json',host,out/'hardware.json')}
    value=dict(schema='etx.g12.trace.v1',session_directory=str(out),session_id=c['session_id'],arm=arm,position=position,model='qwen',batch=1,context=1024,greedy_tokens=32,trace_file=str(raw),trace_log=str(log),plan_file=str(plan),certificate_file=str(directory/'build_identity.json'),native_build_directory=str(native),native_identity_sha256=sha(native/'build-identity.json'),compiled_contract=compiled_contract(native),model_inputs=inputs,files=files,population=population,source_guard_revision={str(p.resolve()):sha(p) for p in (Path(__file__),Path(__file__).with_name('validation_run.py'),Path(__file__).with_name('validation_prepare.py'))},performance_claim=False,exact_publication_interval_available=False,uninstrumented_pairs_are_separate=True)
    proof=directory/'proof.json';proof.write_text(json.dumps(value,indent=2)+'\n')
    return dict(proof=str(proof),proof_files={str(proof):sha(proof)})


def verify(meta,*,validate_session=True):
    from .validation import qwen_correct
    verify_files(meta['proof_files']);path=resolve_path(meta['proof']);require(any(resolve_path(k)==path and v==sha(path) for k,v in meta['proof_files'].items()),'trace proof itself must be sealed')
    p=json.loads(path.read_text());require(p['schema']=='etx.g12.trace.v1' and p['model']=='qwen' and p['batch']==1 and p['position'] in (0,1040),'wrong current trace context')
    verify_files(p['files']);verify_files(p['model_inputs']);verify_files(p['source_guard_revision'])
    require(validate_raw(p['plan_file'],p['trace_file'])==p['population'],'trace raw population differs')
    identity=json.loads(resolve_path(p['certificate_file']).read_text());qwen_correct(resolve_path(p['trace_log']).read_text(),identity)
    marker=f'G12TRACE position={p["position"]} run=1 tasks={p["population"]["tasks"]} completed=1'
    require(resolve_path(p['trace_log']).read_text().splitlines().count(marker)==1,'trace selected-step marker differs')
    native=resolve_path(p['native_build_directory']);require(sha(native/'build-identity.json')==p['native_identity_sha256'] and compiled_contract(native)==p['compiled_contract'],'trace native source/module/flags changed')
    if validate_session:
        from .validation_run import transcript
        root=resolve_path(p['session_directory']);c=json.loads((root/'session-config.json').read_text());transcript(c,root)
        require(c['session_id']==p['session_id'] and c.get('fold_decision_active') is True,'trace is not the active G12 fold session')
        require(any(x.get('proof')==meta['proof'] for x in c.get('fold_traces',[])),'trace not retained in complete session proof population')
    return p


def bind_g11(pack,meta):
    p=verify(meta);require(pack['position']==p['position'],'G11/trace position mismatch')
    require(pack['dependency_edges']['plan_sha256']==p['compiled_contract']['plan_sha256'],'G11/trace emitted plan mismatch')
    require(pack.get('g12_session_id')==p['session_id'],'G11 trace session mismatch')
    from .row_ops.capture import plan_edges
    actual=plan_edges(p['plan_file'],pack['role'],pack['layer'])
    require(all(pack['dependency_edges'][k]==actual[k] for k in ('plan_sha256','grid','tasks','events')),'G11 actual event/queue edges differ from trace')
    capture=resolve_path(pack['capture_native_build']);require(compiled_contract(capture)==p['compiled_contract'],'capture-host device module/source/flags differ from G12 trace')
    require(sha(capture/'build-identity.json')==pack['capture_native_identity_sha256'],'native capture identity differs')
    weight=next((pack['buffers'][ptr['buffer']]['file'] for ptr in pack['pointers'] if ptr['offset']==0),None)
    require(weight is not None and any(resolve_path(k)==resolve_path(weight) and digest==sha(weight) for k,digest in p['model_inputs'].items()),'native captured weights differ from trace model input')
    return dict(matching_current_chain_context=True,proof_sha256=sha(meta['proof']),position=p['position'],session_id=p['session_id'],exact_publication_interval_available=False,model_or_removable_work_price=None,scope='same module/source/flags/input/plan/position; separate sparse replay clocks cannot be subtracted')
