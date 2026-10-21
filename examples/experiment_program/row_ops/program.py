"""Finite row coverage and same-session derived-pack handoff, disabled by default."""
import json,contextlib,io
from pathlib import Path
from examples.experiment_program.protocol import require,resolve_path,sha,verify_files

def cases(record,layer=24):
    require(record in ('G09','G10','G11'),'row program record')
    require(type(layer) is int and 0<=layer<48,'representative layer')
    layers=(1,24,47) if record=='G11' else (layer,)
    return [dict(id=f'L{l}-p{p}',layer=l,position=p) for l in layers for p in (0,1040)]

def configuration(path,record):
    p=resolve_path(path);c=json.loads(p.read_text())
    require(c.get('schema')=='etx.row.program.v1' and c.get('record')==record,'row program schema/record')
    verify_files(c['inputs']);expected=cases(record,c.get('layer',24))
    for key in (('build',) if record=='G11' else ('build','diagnostic_build')):
        identity=resolve_path(c[key])/'identity.json'
        require(c['inputs'].get(str(identity))==sha(identity),'selected row build must be sealed by the immutable recipe: '+key)
    require(set(c['packs'])=={r['id'] for r in expected},'complete registered both-position row coverage required')
    for row in expected:
        for variant in (('g10_up','g10_down') if record=='G10' else ('g09',) if record=='G09' else ('g11_probe',)):
            path=resolve_path(c['packs'][row['id']][variant]);require(c['inputs'].get(str(path))==sha(path),'row recipe pack must be presealed')
            pack=json.loads(path.read_text());require((pack['position'],pack['layer'])==(row['position'],row['layer']),'recipe pack layer/position differs')
    return c,expected

def derive(pack_path,trace_meta,out,*,expected_session):
    """Seal only after a successful producer; immutable recipe/base bytes survive."""
    from examples.experiment_program.validation_trace import verify,bind_g11
    p=json.loads(resolve_path(pack_path).read_text());t=verify(trace_meta)
    require(t['session_id']==expected_session and t['position']==p['position'],'derived trace must be same active session/position')
    p.update(matching_trace=trace_meta,g12_session_id=t['session_id'])
    p['trace_binding']=bind_g11(p,trace_meta)
    p['derived_from']={str(resolve_path(pack_path)):sha(pack_path),**trace_meta['proof_files']}
    out=Path(out)
    with out.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return out

def readmit(descriptor,*,position=None,layer=None):
    from examples.experiment_program.validation import require_admission
    require(descriptor.get('record') in ('G09','G10'),'row dependency descriptor record')
    require_admission(descriptor)
    pack=json.loads(resolve_path(descriptor['pack']).read_text())
    if position is not None:require(pack['position']==position,'up dependency position differs')
    if layer is not None:require(pack['layer']==layer,'up dependency layer differs')
    return descriptor

def _run(path,record,out,*,execute,case_main,trace_session=None):
    c,expected=configuration(path,record);out=resolve_path(out)
    report=dict(schema='etx.row.program-report.v1',record=record,cases=[],admitted=False,gpu_executed=False,configuration_sha256=sha(path),status='PREPARED_CONDITIONAL')
    if execute:out.mkdir(parents=True,exist_ok=False)
    if record=='G11':
        from examples.experiment_program.publication import admission
        pub=resolve_path(c['publication_preparation']);require(c['inputs'].get(str(pub))==sha(pub),'G11 publication build must be presealed')
        gate=admission(pub);report['publication_admission']=gate
        if not gate['admitted']:
            report.update(status=gate['status'],gpu_executed=False)
            if execute:(out/'program-report.json').write_text(json.dumps(report,indent=2)+'\n')
            return report
    traces={};session_id=None
    if record=='G11' and execute:
        require(trace_session is not None,'G11 requires completed same-session G12 producer')
        root=resolve_path(trace_session);dispatch=root/'dispatch.json'
        if dispatch.is_file():root=root/'native-session'
        active=json.loads((root/'session-config.json').read_text());session_id=active['session_id']
        from examples.experiment_program.validation_run import transcript
        transcript(active,root)
        traces={json.loads(resolve_path(m['proof']).read_text())['position']:m for m in active.get('fold_traces',[])}
        require(set(traces)=={0,1040},'G11 both-position trace producers required')
    up=[]
    for row in expected:
        variant='g09' if record=='G09' else 'g10_up' if record=='G10' else 'g11_probe'
        pack=resolve_path(c['packs'][row['id']][variant]);prefix=out/(row['id']+'-'+variant)
        if record=='G11' and execute:pack=derive(pack,traces[row['position']],out/(row['id']+'.derived-pack.json'),expected_session=session_id)
        # G11 dry-run validates the immutable recipe and native build, with future
        # trace dependency explicit; it never invents a completed G12 trace.
        if record=='G11' and not execute:
            from .replay import validate_pack,validate_build
            validate_pack(json.loads(pack.read_text()));validate_build(c['build'],variant)
            report['cases'].append(dict(**row,status='PENDING_SAME_SESSION_TRACE',variant=variant));continue
        diagnostic=None
        if record!='G11':
            from examples.experiment_program.protocol import selected_artifacts
            diag_build=resolve_path(c['diagnostic_build']);di=json.loads((diag_build/'identity.json').read_text())
            selected_artifacts(diag_build,di['files'],['replay']+[variant+'_'+m+'.hsaco' for m in ('control','probe','empty')])
            if execute:
                diagprefix=out/(row['id']+'-'+variant+'-diagnostic')
                diagargv=['execute','--build',str(diag_build),'--pack',str(pack),'--variant',variant,'--output',str(diagprefix),'--diagnostic']
                with contextlib.redirect_stdout(io.StringIO()):rc=case_main(diagargv)
                require(rc in (None,0),'diagnostic correctness/liveness failure')
                diagnostic=dict(build=str(diag_build),output_prefix=str(diagprefix),decision=str(diagprefix)+'.result.json')
                dr=json.loads(resolve_path(diagnostic['decision']).read_text())
                if not dr['verdict']['qualified']:
                    report['cases'].append(dict(**row,variant=variant,status='CLOSED_DIAGNOSTIC_QUALIFICATION',diagnostic=diagnostic,admitted=False));continue
        argv=['execute','--build',c['build'],'--pack',str(pack),'--variant',variant,'--output',str(prefix)]+([] if execute else ['--dry-run'])
        stdout=io.StringIO()
        with contextlib.redirect_stdout(stdout):code=case_main(argv)
        require(code in (None,0),'row program correctness/liveness failure')
        entry=dict(**row,variant=variant,command=argv,status='PREFLIGHT_PASS' if not execute else 'COMPLETE')
        if execute:
            (out/(row['id']+'.stdout.json')).write_text(stdout.getvalue())
            descriptor=dict(record=record,build=c['build'],variant=variant,pack=str(pack),output_prefix=str(prefix),decision=str(prefix)+'.result.json')
            if diagnostic:descriptor['diagnostic']=diagnostic
            result=json.loads(resolve_path(descriptor['decision']).read_text());entry['evidence']=descriptor
            winner=result['verdict'].get('status')=='MICRO_PASS'
            entry['admitted']=winner
            if record!='G11' and winner:readmit(descriptor);up.append(descriptor)
        report['cases'].append(entry)
    # Down only follows the exact complete both-position up winner population.
    if record=='G10':
        if execute and len(up)==len(expected):
            evidence=out/'up-evidence.json';evidence.write_text(json.dumps(dict(schema='etx.row.up-evidence.v1',positions=up),indent=2)+'\n')
            down=[]
            for row in expected:
                pack=resolve_path(c['packs'][row['id']]['g10_down']);prefix=out/(row['id']+'-g10_down')
                diagprefix=out/(row['id']+'-g10_down-diagnostic')
                diagargv=['execute','--build',c['diagnostic_build'],'--pack',str(pack),'--variant','g10_down','--output',str(diagprefix),'--up-evidence',str(evidence),'--up-build',c['build'],'--diagnostic']
                with contextlib.redirect_stdout(io.StringIO()):rc=case_main(diagargv)
                require(rc in (None,0),'down diagnostic correctness/liveness failure')
                diagnostic=dict(build=c['diagnostic_build'],output_prefix=str(diagprefix),decision=str(diagprefix)+'.result.json')
                if not json.loads(resolve_path(diagnostic['decision']).read_text())['verdict']['qualified']:
                    report['cases'].append(dict(**row,variant='g10_down',status='CLOSED_DIAGNOSTIC_QUALIFICATION',admitted=False,diagnostic=diagnostic));continue
                argv=['execute','--build',c['build'],'--pack',str(pack),'--variant','g10_down','--output',str(prefix),'--up-evidence',str(evidence)]
                with contextlib.redirect_stdout(io.StringIO()):code=case_main(argv)
                require(code in (None,0),'conditional down native failure')
                descriptor=dict(record='G10',build=c['build'],variant='g10_down',pack=str(pack),output_prefix=str(prefix),decision=str(prefix)+'.result.json',diagnostic=diagnostic)
                result=json.loads(resolve_path(descriptor['decision']).read_text());won=result['verdict']['status']=='MICRO_PASS'
                if won:readmit(descriptor);down.append(descriptor)
                report['cases'].append(dict(**row,variant='g10_down',status='COMPLETE',admitted=won,evidence=descriptor))
            report['down_admitted']=len(down)==len(expected)
        else:report['down_status']='PENDING_UP_BOTH_POSITIONS' if not execute else 'CLOSED_UP_MISS'
    report['admitted']=execute and record!='G11' and len(up)==len(expected)
    report['gpu_executed']=bool(execute);report['status']='MICRO_PASS_BOTH_POSITIONS' if report['admitted'] else 'PROGRAM_COMPLETE' if execute else 'PREPARED_CONDITIONAL'
    if execute:(out/'program-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def admit(path,build,variant):
    r=json.loads(resolve_path(path).read_text());require(r.get('schema')=='etx.row.program-report.v1' and r.get('gpu_executed'),'complete native row program required')
    target=[x for x in r['cases'] if x['variant']==variant]
    require(len(target)==2 and {x['position'] for x in target}=={0,1040} and len({x['layer'] for x in target})==1,'both-position row admission required')
    for x in target:
        require(x.get('admitted') and resolve_path(x['evidence']['build'])==resolve_path(build),'row program variant/build not admitted')
        readmit(x['evidence'],position=x['position'],layer=x['layer'])
    return dict(record=r['record'],decision_sha256=sha(path),micro_build_sha256=sha(resolve_path(build)/'identity.json'),variant=variant,both_positions=True)


def run(path,record,out,*,execute,case_main,trace_session=None):
    output=resolve_path(out);unoccupied=not output.exists()
    try:return _run(path,record,out,execute=execute,case_main=case_main,trace_session=trace_session)
    except BaseException as error:
        if execute and unoccupied and output.is_dir():
            (output/'program-error.json').write_text(json.dumps(dict(record=record,status='NOT_COMPLETE',configuration_sha256=sha(path),error=str(error),performance_claim=False),indent=2)+'\n')
        raise


def verify_diagnostic(path,stage):
    """Read all six raw G11 cases again; an edited summary cannot survive resume."""
    from .replay import validate_pack,validate_build
    from .results import summarize
    path=resolve_path(path);r=json.loads(path.read_text())
    require(r.get('record')=='G11' and r.get('gpu_executed') is True and r.get('status')=='PROGRAM_COMPLETE','complete native G11 program required')
    cmd=stage['command'];recipe_path=resolve_path(cmd[cmd.index('--config')+1]);recipe,_=configuration(recipe_path,'G11')
    require(r['configuration_sha256']==sha(recipe_path),'G11 derivation recipe changed')
    trace_root=resolve_path(cmd[cmd.index('--trace-session')+1]).resolve()
    require([{k:x[k] for k in ('id','layer','position')} for x in r['cases']]==cases('G11'),'all early/middle/late and both positions required')
    for row in r['cases']:
        d=row['evidence'];build=resolve_path(d['build']);pack_path=resolve_path(d['pack']);pack=json.loads(pack_path.read_text());validate_pack(pack);identity=validate_build(build,'g11_probe');identity['_directory']=str(build)
        result=json.loads(resolve_path(d['decision']).read_text());prefix=d['output_prefix']
        require(resolve_path(prefix)==path.parent/(row['id']+'-g11_probe') and resolve_path(d['decision'])==resolve_path(str(prefix)+'.result.json') and build==resolve_path(recipe['build']),'G11 selected raw paths/build differ')
        base=resolve_path(recipe['packs'][row['id']]['g11_probe']);require(pack['derived_from'].get(str(base))==sha(base),'G11 derived pack base changed')
        from examples.experiment_program.validation_trace import verify as trace_verify
        trace=trace_verify(pack['matching_trace']);require(resolve_path(trace['session_directory']).resolve().is_relative_to(trace_root),'G11 trace belongs to another session')
        require(resolve_path(str(prefix)+'.exit').read_text().strip()=='0','G11 native child incomplete')
        require(result['gpu_executed'] and result['input_pack_sha256']==sha(pack_path) and result['build_identity_sha256']==sha(build/'identity.json'),'G11 native identity differs')
        require(summarize(prefix,pack,identity,'g11_probe')==result['verdict'],'G11 raw diagnostic summary differs')
    return dict(record='G11',complete=True,admitted=False,decision_sha256=sha(path),publication_status='UNAVAILABLE_PUBLICATION_TIMESTAMP')
