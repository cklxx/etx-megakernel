"""Winner dispatch over precompiled registered G12 model pairs.

CPU preflight can inspect dormant source/native/input closures before any
micro result exists. Only a completed, sealed prior record in this same owned
session can activate the first declared admitted mechanism. No command arrays
or new compilation are accepted by this dispatcher.
"""
from __future__ import annotations
import json
from .protocol import require,resolve_path,sha,verify_files


def candidates(c):
    verify_files(c['input_files']);verify_files(c['source_snapshot']);verify_files(c['readiness_files']);rows=c['conditional_candidates']
    sealed={resolve_path(k):v for k,v in c['input_files'].items()}
    require(all(sealed.get(resolve_path(k))==v for group in (c['source_snapshot'],c['readiness_files']) for k,v in group.items()),'conditional source snapshot/readiness not sealed by root inputs')
    require(isinstance(rows,list) and bool(rows),'registered conditional model pairs required')
    seen=set();result=[]
    for row in rows:
        require(set(row)=={'preparation','models','template','admission'},'exact conditional pair schema required')
        prep=resolve_path(row['preparation']);models=resolve_path(row['models']);template=resolve_path(row['template'])
        sealed={resolve_path(k):v for k,v in c['input_files'].items()}
        require(all(p in sealed and sealed[p]==sha(p) for p in (prep,models,template)),'conditional source/models/template not sealed')
        value=json.loads(template.read_text());require('stages' not in value and 'winner' not in value,'dormant template must not invent a winner or arbitrary stages')
        require(all(sealed.get(resolve_path(k))==v for k,v in value['input_files'].items()),'dormant exact input artifact missing from conditional root seal')
        require(resolve_path(value['preparation'])==prep and resolve_path(value['models'])==models,'template differs from registered pair')
        record=row['admission']['record'];key=(record,row['admission'].get('variant'),value['batch'])
        require(key not in seen,'duplicate conditional mechanism/variant/batch');seen.add(key)
        original=json.loads(prep.read_text());require(original['record']==record and original['local_validation'] and not original['execution_enabled'],'conditional pair is not this dormant mechanism')
        identity=resolve_path(row['admission']['build'])/'identity.json'
        require(identity in sealed and sealed[identity]==sha(identity),'micro build identity not sealed before selection')
        result.append((row,value))
    return result


def preflight(c):
    from .validation_run import configuration
    rows=[]
    for entry,value in candidates(c):
        try:
            configuration(value,admission=False,dormant=True)
            rows.append(dict(record=entry['admission']['record'],variant=entry['admission'].get('variant'),status='DORMANT_NATIVE_AND_INPUT_CLOSURE_PASS',winner_required_after_micros=True))
        except (OSError,ValueError,KeyError) as error:rows.append(dict(record=entry['admission']['record'],status='BLOCKED',reason=str(error)))
    return dict(record='G12',status='PASS' if any(x['status']=='DORMANT_NATIVE_AND_INPUT_CLOSURE_PASS' for x in rows) else 'BLOCKED',candidates=rows,enabled=False,winner_invented=False,gpu_outcomes='UNMEASURED')


def select(c,root):
    from .validation import require_admission
    from etx.experiments.sealed import bind_output
    root=resolve_path(root);bound=bind_output(c,root);state=json.loads((root/'session.json').read_text())
    require(state['source_revision']==c['source_revision'],'micro session source revision differs')
    rejected=[]
    for entry,template in candidates(bound):
        admission=entry['admission'];record=admission['record'];path=resolve_path(admission['decision'])
        if state['stages'].get(record,{}).get('status')!='COMPLETE':continue
        require(path.is_relative_to(root) and state['sealed'].get(str(path))==sha(path),'micro winner is not a sealed prior result in this session')
        raw=json.loads(path.read_text());status=raw.get('status',raw.get('verdict',{}).get('status'))
        claimed=raw.get('admitted') is True or status in ('MICRO_PASS','MICRO_WINNER','MICRO_PASS_BOTH_REGIMES','MICRO_PASS_BOTH_POSITIONS')
        if not claimed:continue
        proof=require_admission(admission)
        from .validation_run import configuration
        try:configuration(template,admission=False,dormant=True)
        except (OSError,ValueError,KeyError) as error:
            rejected.append(dict(record=record,reason=str(error)));continue
        return entry,template,proof,dict(directory=str(root),source_revision=state['source_revision'],micro_decision=str(path),micro_decision_sha256=sha(path),earlier_model_prerequisite_rejections=rejected)
    raise ValueError('no preregistered admitted winner in the completed same-session micro populations')


def run(c,out,root):
    from .validation_prepare import activate
    from .validation_run import run as native_run
    entry,template,proof,session=select(c,root)
    out=resolve_path(out);out.mkdir(parents=True,exist_ok=False)
    activated=out/'activated-preparation.json';activate(entry['preparation'],entry['admission'],activated)
    value=dict(template,preparation=str(activated),winner=entry['admission'])
    value['input_files']=dict(value['input_files'],**{str(activated):sha(activated),session['micro_decision']:session['micro_decision_sha256']})
    config=out/'activated-run.json';config.write_text(json.dumps(value,indent=2)+'\n')
    receipt=dict(schema='etx.g12.dispatch.v1',selected=entry,admission=proof,micro_session=session,activated_config=str(config),activated_config_sha256=sha(config),conditional_config=c)
    (out/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    result=native_run(value,out/'native-session');result['conditional_dispatch']=dict(record=proof['record'],receipt_sha256=sha(out/'dispatch.json'))
    (out/'decision.json').write_text(json.dumps(result,indent=2)+'\n');return result


def replay(c,out):
    from .validation import replay as native_replay
    out=resolve_path(out);receipt=json.loads((out/'dispatch.json').read_text())
    require(receipt['schema']=='etx.g12.dispatch.v1' and receipt['conditional_config']==c,'conditional registration changed')
    entry,template,proof,session=select(c,receipt['micro_session']['directory'])
    require((entry,proof,session)==(receipt['selected'],receipt['admission'],receipt['micro_session']),'conditional winner selection changed')
    path=resolve_path(receipt['activated_config']);require(sha(path)==receipt['activated_config_sha256'],'activated conditional execution config changed')
    result=native_replay(json.loads(path.read_text()),out/'native-session',write=False)
    result['conditional_dispatch']=dict(record=proof['record'],receipt_sha256=sha(out/'dispatch.json'));return result
