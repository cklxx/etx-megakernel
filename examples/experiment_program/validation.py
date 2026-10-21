"""G12 winner-only model, same-kernel graph and real-system gates.

Local source preparation is explicit and timing disabled. Native execution is
never triggered by finding a candidate or by finishing a local build.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import statistics
import sys

from .protocol import require,sha,verify_files,paired_decision,resolve_path


def model_rule(control,candidate):
    require(len(control)==len(candidate)==5,'five complete model pairs')
    require(all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in control+candidate),'invalid model timing')
    gains=[c-x for c,x in zip(control,candidate)];m=statistics.median(gains);spread=max(gains)-min(gains)
    return dict(keep=min(gains)>0 and m>=30000 and m>2*spread,median_ns=m,spread_ns=spread,gains_ns=gains,basis='host wall decode step')


def system_rule(candidate,reference,latency,reference_latency,*,batch):
    from etx.session.batch import verdict
    require(batch in (1,2,4),'registered batch only')
    require(all(len(v)==5 for v in (candidate,reference,latency,reference_latency)),'five complete system pairs')
    require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for v in (candidate,reference,latency,reference_latency) for x in v),'invalid system endpoints')
    throughput=verdict([Fraction(str(r))/Fraction(str(c))-1 for c,r in zip(candidate,reference)])
    request=verdict([Fraction(str(r))/Fraction(str(c))-1 for c,r in zip(latency,reference_latency)])
    return dict(throughput_verdict=throughput['outcome'],request_latency_verdict=request['outcome'],
                candidate_tokens_per_second=batch*1e9/statistics.median(candidate),
                candidate_worst_request_interval_ns=statistics.median(latency),
                pursuit_target_pass=statistics.median(candidate)<=.9*statistics.median(reference) if batch==4 else None,
                reference_scope='strongest among same-session tested lever-proven legal configurations')


def require_admission(c):
    """Reread raw native evidence. A self-authored `admitted:true` is insufficient."""
    record=c['record'];p=resolve_path(c['decision']);r=json.loads(p.read_text())
    build=resolve_path(c['build']);identity=json.loads((build/'identity.json').read_text())
    if record=='G01':
        if r.get('schema')=='etx.router.program.v1':
            raw=resolve_path(r['raw_control']);require(sha(raw)==r['raw_control_sha256'],'router program raw control changed')
            from .router.diagnostics import qualify
            diagnostic=qualify(build,p.parent/'control',p.parent/'probe',p.parent/'empty')
            require(isinstance(diagnostic,dict) and diagnostic.get('qualified') is True,'router diagnostic unqualified/malformed')
            require(diagnostic==r['diagnostic'],'router qualification changed')
            require(r.get('admitted') is True,'router program did not admit validation')
            return require_admission(dict(c,decision=str(raw)))
        from .router.receipts import parse_native
        require(r.get('record')=='G01' and r['build_identity_sha256']==sha(build/'identity.json'),'G01 winner build mismatch')
        verify_files(r['inputs']);verify_files(identity['files'])
        result=paired_decision(r['samples'],seed=c.get('seed',19),qwen_layer=True)
        require(result['admitted'] and r.get('admitted') is True,'G01 did not win')
        text=(p.parent/'native.log').read_text()
        seal=json.loads((p.parent/'seal.json').read_text())
        actual=parse_native(text,population=seal['population'],seed=seal['seed'])
        require(actual['admitted'] and actual['samples']==r['samples'],'raw G01 population not admitted')
        require(len(re.findall(r'^CHECK G01 case=\d+ exact=1$',text,re.M))==96 and 'G01 COMPLETE all96 exact' in text,'missing96-case native correctness')
        # Pair means must agree with the retained raw complete invocation order.
        rows=re.findall(r'^PAIR G01 ordinal=(\d) arm=(\d) device_ns=([\d.]+) wall_ns=([\d.]+) population=(\d+)$',text,re.M)
        require(len(rows)==10,'missing native raw pairs')
        for i,s in enumerate(r['samples']):
            by={int(x[1]):float(x[2]) for x in rows[2*i:2*i+2]}
            require(by=={0:s['control_ns'],1:s['candidate_ns']},'edited G01 decision')
    elif record in ('G06','G07','G08'):
        from .issue_order.integration import admit_winner
        from .issue_order.__main__ import verify_build
        identity=verify_build(build);identity['_identity_sha256']=sha(build/'identity.json')
        r['_evidence_directory']=str(p.parent.resolve())
        from .telemetry import verify as telemetry_verify
        for entry in json.loads((p.parent/'commands.json').read_text()):
            telemetry_verify(p.parent/(entry['frame'].replace('/','_')+'.telemetry'),dict(command=entry['argv'],build_sha256=r['build_sha256'],capture_sha256=r['capture_seal_sha256']))
        admit_winner(r,record,identity)
    elif record in ('G09','G10'):
        from .row_ops.replay import validate_build,validate_pack
        from .row_ops.results import summarize
        variant=c['variant']
        if r.get('schema')=='etx.row.program-report.v1':
            from .row_ops.program import admit
            return admit(p,build,variant)
        pack_path=resolve_path(c['pack']);pack=json.loads(pack_path.read_text())
        validate_pack(pack);validate_build(build,variant)
        if variant=='g10_down':
            from .row_ops.dependencies import up_evidence
            require(up_evidence(r['up_evidence']['path'],build,pack['layer'])==r['up_evidence'],'down raw up dependency changed')
        require(r.get('gpu_executed') is True and r.get('build_identity_sha256')==sha(build/'identity.json') and r.get('input_pack_sha256')==sha(pack_path),'row winner identity')
        diagnostic=c.get('diagnostic');require(isinstance(diagnostic,dict),'raw matched sparse diagnostic required')
        from .row_ops.diagnostics import summarize as diagnostic_summary
        db=resolve_path(diagnostic['build']);dr=json.loads(resolve_path(diagnostic['decision']).read_text())
        di=json.loads((db/'identity.json').read_text());di['_directory']=str(db)
        from .protocol import selected_artifacts
        selected_artifacts(db,di['files'],['replay']+[variant+'_'+m+'.hsaco' for m in ('control','probe','empty')])
        actual_diag=diagnostic_summary(diagnostic['output_prefix'],pack,di,variant)
        require(dr['build_identity_sha256']==sha(db/'identity.json') and dr['input_pack_sha256']==sha(pack_path) and actual_diag==dr['verdict'] and actual_diag['qualified'],'sparse raw diagnostic unqualified/changed')
        require(resolve_path(str(diagnostic['output_prefix'])+'.exit').read_text().strip()=='0','sparse attempt incomplete')
        prefix=resolve_path(c['output_prefix'])
        require(resolve_path(str(prefix)+'.exit').read_text().strip()=='0','row attempt incomplete/failed')
        from .telemetry import verify
        verify(str(prefix)+'.telemetry',dict(command=r['command'],pack_sha256=sha(pack_path),build_sha256=sha(build/'identity.json')))
        actual=summarize(prefix,pack,identity,variant)
        require(actual==r['verdict'] and actual.get('exact') and actual['measurements']['device_us']['admitted'],'row micro did not win')
    elif record in ('G02','G05'):
        from .columns.runner import admit_model_winner
        actual=admit_model_winner(p,build)
        require(actual['record']==record,'wrong aggregate column winner record')
    else:raise ValueError('only an optimization winner can enter G12')
    return dict(record=record,decision_sha256=sha(p),micro_build_sha256=sha(build/'identity.json'))


def qwen_correct(text,identity):
    from examples.vllm_llm import import_certificate as cert
    from examples.vllm_llm.nvref_report import correctness
    cert.stamped(text,identity);correctness(text,'G12')
    rows=re.findall(r'^RESULT .*\bmatched=(\d+) gen=(\d+) median_ms=([\d.]+) wall_ms=([\d.]+)',text,re.M)
    require(len(rows)==1 and rows[0][:2]==('32','32'),'exact32-token full-model invocation')
    repeats=re.findall(r'^run (\d+): (\d+)/(\d+) generated tokens match HF greedy[^\n]*$',text,re.M)
    require(repeats==[('0','32','32'),('1','32','32')],'exact one warmup and one full greedy invocation')
    population=native_population(text,'qwen',1)
    values=sorted(r['wall_ms'] for r in population if r['run']==1);wall=values[len(values)//2]
    require(abs(wall-float(rows[0][3]))<=.000501,'native wall samples disagree with RESULT')
    return wall*1e6


def native_population(text,model,batch):
    from etx.session.batch import kv,etx_result,ns
    rows=[kv(x) for x in re.findall(r'^G12TOKEN (.*)$',text,re.M)]
    expected={(run,r,t) for run in ((0,1) if model=='qwen' else (1,)) for r in range(batch) for t in range(32)}
    result=[]
    for row in rows:
        values={key:int(row[key]) for key in ('run','request','token','start_ticks','end_ticks')};values['wall_ms']=float(row['wall_ms'])
        require(0<values['start_ticks']<values['end_ticks'] and math.isfinite(values['wall_ms']) and values['wall_ms']>0,'invalid native token endpoint')
        result.append(values)
    actual=[(x['run'],x['request'],x['token']) for x in result]
    require(len(actual)==len(set(actual)) and set(actual)==expected,'complete raw native token/timestamp population required')
    if model=='deepseek':
        parsed=etx_result(text,batch)
        for r in range(batch):
            mine=sorted((x for x in result if x['request']==r),key=lambda x:x['token'])
            require(all(b['end_ticks']>a['end_ticks'] for a,b in zip(mine,mine[1:])),'nonmonotonic request timestamps')
            interval=statistics.median([(b['end_ticks']-a['end_ticks'])*10 for a,b in zip(mine,mine[1:])])
            require(interval==ns(parsed['requests'][r]['interval_ms']),'native timestamp interval differs from BATCHREQ')
    return result


def system_population(text,label,batch,expected_version='0.27.1'):
    from etx.session.batch import kv,vllm_results
    rows=[kv(line) for line in re.findall(r'^VLLMSEQ (.*)$',text,re.M)]
    phases={phase:(1 if phase.endswith('_short') else 32) for phase in ['warmup','warmup_short',*[p for k in range(5) for p in (f'sample{k}',f'sample{k}_short')]]}
    expected={(phase,r) for phase in phases for r in range(batch)}
    actual=[(r.get('phase'),int(r.get('seq',-1))) for r in rows]
    require(len(rows)==len(expected) and set(actual)==expected and len(set(actual))==len(actual),'exact full/short system population required')
    require(all(r.get('label')==label and r.get('batch')==str(batch) and r.get('tokens')==str(phases[r['phase']]) and r.get('first_divergence')==f"{phases[r['phase']]}/{phases[r['phase']]}" for r in rows),'system full/short correctness or identity failed')
    timing=re.findall(r'^VLLMB (.*)$',text,re.M)
    require(len(timing)==1 and kv(timing[0]).get('label')==label and kv(timing[0]).get('batch')==str(batch),'exact one registered system timing population')
    heads=re.findall(r'^VLLM-BATCH-MODE (.*)$',text,re.M)
    require(len(heads)==1 and kv(heads[0]).get('label')==label and kv(heads[0]).get('context')=='1024' and kv(heads[0]).get('batches')==str(batch) and kv(heads[0]).get('version')==expected_version,'system context/batch/version changed')
    parsed=vllm_results(text,label);require(set(parsed)=={batch} and parsed[batch]['legal'],'system correctness/population invalid')
    return parsed[batch]


def applied_system(meta,label,batch,*,session_id=None,log_path=None):
    verify_files(meta['proof_files']);path=resolve_path(meta['applied_proof']).resolve()
    bound={resolve_path(p).resolve():digest for p,digest in meta['proof_files'].items()}
    require(path in bound and bound[path]==sha(path),'applied proof itself must be sealed')
    proof=json.loads(path.read_text())
    require(proof.get('label')==label and proof.get('batch')==batch and proof.get('applied_counters') and all(type(v) is int and v>0 for v in proof['applied_counters'].values()),'unproven system lever')
    if session_id is not None:require(proof.get('session_id')==session_id,'historical system proof cannot transfer sessions')
    if log_path is not None:require(proof.get('benchmark_log_sha256')==sha(log_path),'applied proof benchmark changed')
    return proof


def schedule(model,batch,systems,selected,seed,*,fold_decision_active=False,trace_positions=(1040,)):
    from .protocol import pair_order
    require(model in ('qwen','deepseek') and batch in ((1,) if model=='qwen' else (2,4)),'registered model/batch')
    require(selected in systems and len(set(systems))==len(systems),'selected legal system missing/duplicate')
    rows=[]
    if model=='qwen':
        for arm in ('control','candidate'):
            for position in (0,1040):rows.append(dict(phase='imports',ordinal=position,arm=arm,order='CHECK'))
            rows.append(dict(phase='dsl',ordinal=0,arm=arm,order='CHECK'))
        rows.append(dict(phase='graph-gate',ordinal=0,arm='candidate',order='CHECK'))
    for label in systems:rows.append(dict(phase='selection',ordinal=0,arm=label,order='SELECT'))
    comparisons=[('model',seed,('control','candidate'))]+([('fusion',seed+1,('fused','graph'))] if model=='qwen' else [])+[('system',seed+2,('candidate',selected))]
    for phase,seed,arms in comparisons:
        for ordinal,order in enumerate(pair_order(seed)):
            for index in ((0,1) if order=='AB' else (1,0)):rows.append(dict(phase=phase,ordinal=ordinal,arm=arms[index],order=order))
    if fold_decision_active:
        require(model=='qwen' and bool(trace_positions) and len(set(trace_positions))==len(trace_positions) and set(trace_positions)<={0,1040},'G11 trace requires Qwen and exact registered positions0/1040')
        for position in trace_positions:
            for arm in ('control','candidate'):rows.append(dict(phase='trace',ordinal=position,arm=arm,order='DIAGNOSTIC_AFTER_PAIRS'))
    return rows


def replay(config,out,*,write=True):
    """Consume only this session's completed native logs and original certificates."""
    from etx.session.batch import etx_result,vllm_results,ns
    from examples.vllm_llm import import_certificate as cert,pairs
    if 'conditional_candidates' in config:
        from .validation_dispatch import replay as conditional_replay
        return conditional_replay(config,out)
    from .validation_run import transcript
    c=config;out=resolve_path(out)
    if 'output_files' not in c:
        require(c==json.loads((out/'registration.json').read_text()),'original G12 run configuration differs')
        c=json.loads((out/'session-config.json').read_text())
    session=transcript(c,out);winner=require_admission(c['winner'])
    verify_files(c['input_files']);require(c['model'] in ('qwen','deepseek'),'registered model')
    batch=c['batch'];controls=[];candidate=[];latency=[]
    if c['model']=='qwen':
        require(batch==1,'QwenB1 only')
        for arm in ('control','candidate'):
            directory=out/arm
            checked=cert.validate(directory,timing=False)
            require(checked['checked']==checked['planned']==97,'Qwen full97 same-build imports')
            identity=json.loads((directory/'build_identity.json').read_text())
            require(re.search(r'^CHECK-DSL OK\b',(directory/'gate_chkx.log').read_text(),re.M) is not None,'missing CHECK-DSL')
            qwen_correct((directory/'gate_chkx.log').read_text(),identity)
        for i in range(5):
            controls.append(qwen_correct((out/f'control/model_r{i}.log').read_text(),json.loads((out/'control/build_identity.json').read_text())))
            candidate.append(qwen_correct((out/f'candidate/model_r{i}.log').read_text(),json.loads((out/'candidate/build_identity.json').read_text())))
        latency=list(candidate)
        # Existing validator enforces same binary, all checks and exact graph
        # modes at both endpoints; it does not accept a fabricated certificate.
        fusion=cert.validate(out/'candidate',timing=True)
        fusion_scope='ESTABLISHED' if fusion['eligible'] else 'NOT_ESTABLISHED'
    else:
        require(batch in (2,4),'DeepSeekB2/B4 only')
        for i in range(5):
            control_text=(out/f'control/model_r{i}.log').read_text();actual_text=(out/f'candidate/model_r{i}.log').read_text()
            native_population(control_text,'deepseek',batch);native_population(actual_text,'deepseek',batch)
            control=etx_result(control_text,batch);actual=etx_result(actual_text,batch)
            controls.append(ns(control['step_ms']));candidate.append(ns(actual['step_ms']));latency.append(ns(actual['latency_ms']))
        fusion_scope='UNAVAILABLE_LDS_LIFETIME';fusion=None
    # Select the strongest legal system on a separate selection population,
    # then use that same selected lever-proven config in all five final pairs.
    selection={}
    for label,meta in c['systems'].items():
        path=out/f'select_{label}.log'
        applied_system(meta,label,batch,session_id=c['session_id'],log_path=path)
        from examples.vllm_llm.vlbest_report import proof,config as system_config,expected_replacements
        expected=expected_replacements((out/'candidate/vl_plan.txt').read_text()) if c['model']=='qwen' else None
        require(proof(path.read_text(),system_config(label),expected) is None,'selection system lever not actually applied')
        selection[label]=system_population(path.read_text(),label,batch)
    require(bool(selection),'no legal lever-proven reference')
    selected=min(selection,key=lambda k:ns(selection[k]['step_ms']))
    require(selected==c['selected_system'],'selected system differs from separate raw selection population')
    reference=[];system_candidate=[];system_latency=[]
    for i in range(5):
        path=out/f'system_{selected}_r{i}.log'
        proof_path=resolve_path(str(path)+'.applied.json')
        applied_system(dict(applied_proof=str(proof_path),proof_files={str(proof_path):c['output_files'][str(proof_path.resolve())]}),selected,batch,session_id=c['session_id'],log_path=path)
        parsed=system_population(path.read_text(),selected,batch)
        from examples.vllm_llm.vlbest_report import proof,config as system_config,expected_replacements
        expected=expected_replacements((out/'candidate/vl_plan.txt').read_text()) if c['model']=='qwen' else None
        require(proof(path.read_text(),system_config(selected),expected) is None,'final pair system lever not applied')
        reference.append(ns(parsed['step_ms']))
        text=(out/f'system_candidate_r{i}.log').read_text()
        if c['model']=='qwen':
            value=qwen_correct(text,json.loads((out/'candidate/build_identity.json').read_text()));system_candidate.append(value);system_latency.append(value)
        else:
            native_population(text,'deepseek',batch);value=etx_result(text,batch);system_candidate.append(ns(value['step_ms']));system_latency.append(ns(value['latency_ms']))
    from .validation_trace import verify as verify_trace
    traces=[verify_trace(meta,validate_session=False) for meta in c.get('fold_traces',[])]
    require(len(traces)==(2*len(c.get('fold_trace_positions',[1040])) if c.get('fold_decision_active') else 0),'missing/extra optional trace proof')
    if traces:
        require({(p['arm'],p['position']) for p in traces}=={(arm,pos) for arm in ('control','candidate') for pos in c.get('fold_trace_positions',[1040])},'trace arm/position population differs')
    result=dict(record='G12',complete=True,winner=winner,model=model_rule(controls,candidate),fold_traces=c.get('fold_traces',[]),
                fusion=fusion_scope,graph_certificate=fusion,system=system_rule(system_candidate,reference,system_latency,reference,batch=batch),selected_system=selected,session=session,
                paired_system_wall_ns=dict(candidate=system_candidate,reference=reference,candidate_worst_request=system_latency),
                paired_model_wall_ns=dict(control=controls,candidate=candidate),scope='one session, matched inputs/batch/context; no cross-session subtraction')
    if write:(out/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','build-models','activate','compile-import','preflight','dry-run','run','replay'])
    p.add_argument('--config',type=Path);p.add_argument('--preparation',type=Path);p.add_argument('--winner',type=Path);p.add_argument('--reuse-control',type=Path);p.add_argument('--micro-session-directory',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--local-validation',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    if a.config:a.config=resolve_path(a.config)
    if a.preparation:a.preparation=resolve_path(a.preparation)
    if a.winner:a.winner=resolve_path(a.winner)
    a.out=resolve_path(a.out)
    from .validation_prepare import prepare_sources,build_models,activate
    if a.action=='build-models':require(a.preparation is not None,'--preparation required');r=build_models(a.preparation,a.out,reuse_control=a.reuse_control)
    elif a.action=='activate':require(a.preparation is not None and a.winner is not None,'--preparation and --winner required');r=activate(a.preparation,json.loads(a.winner.read_text()),a.out)
    elif a.action=='compile-import':
        from .issue_order.integration import compile_import_overlay
        require(a.preparation is not None,'--preparation overlay required');r=compile_import_overlay(a.preparation,resolve_path('/code/rocm-headers'),resolve_path('/code/vllm'))
    else:
        require(a.config is not None,'--config required');c=json.loads(a.config.read_text())
        if a.action=='prepare':r=prepare_sources(c,a.out,local_validation=a.local_validation)
        elif a.action=='dry-run':
            r=dict(record='G12',enabled=False,registered_phases=['imports at0/1040 and DSL (Qwen only)','lever-proven system selection','five interleaved native model pairs','five same-kernel graph pairs (Qwen only)','five independent native/system pairs'],preparation=c.get('preparation'),models=c.get('models'),deepseek_graph='UNAVAILABLE_LDS_LIFETIME',arbitrary_commands_accepted=False)
            if 'conditional_candidates' in c:r.update(conditional_dispatch=True,preregistered_candidates=[x['admission']['record'] for x in c['conditional_candidates']],selection='first declared raw-admitted completed and sealed same-session micro winner')
        elif a.action=='preflight':
            from .validation_run import configuration
            try:
                if 'conditional_candidates' in c:
                    from .validation_dispatch import preflight
                    r=preflight(c)
                else:configuration(c);r=dict(record='G12',preflight='PASS',enabled=False,gpu_outcomes='UNMEASURED')
            except (ValueError,OSError,KeyError) as e:r=dict(record='G12',status='BLOCKED',enabled=False,reason=str(e),gpu_outcomes='UNMEASURED')
        elif a.action=='run':
            from .validation_run import run
            require(a.execute,'native model execution disabled by default')
            if 'conditional_candidates' in c:
                from .validation_dispatch import run as conditional_run
                root=a.micro_session_directory or c.get('micro_session_directory');require(root is not None,'conditional dispatch requires this owned micro session directory')
                r=conditional_run(c,a.out,root)
            else:r=run(c,a.out)
        else:r=replay(c,a.out)
    print(json.dumps(r,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
