"""Serial registered column program; native cases and dependencies are explicit."""
from __future__ import annotations
import contextlib,io,json
from pathlib import Path
from examples.experiment_program.protocol import resolve_path,relocation_seal
from .protocol import require,sha256,ProtocolError

def registered(record):
    if record=='G02':
        cases=[dict(id=f'qkv-B{b}',batch=b,role='qkv',depends=[] if b==2 else ['qkv-B2']) for b in (2,4)]
        for b in (2,4):
            for x in (6,7):
                for role in ('gate_up','down'):cases.append(dict(id=f'shared-B{b}-x{x}-{role}',batch=b,role=role,native_xcd=x,depends=[f'qkv-B{b}']))
        return cases
    if record=='G05':return [dict(id=f'routed-B4-{role}',batch=4,role=role,depends=[]) for role in ('gate_up','down')]
    if record=='G13':return [dict(id='G13-B2-'+order,batch=2,role='expert_up_down',depends=[],capture_id='G13-B2',g13_order=order) for order in ('request','both-up-first')]
    return [dict(id=record+'-B2',batch=2,role='qkv' if record=='G03' else 'expert_up_down',depends=[])]

def run_program(config_path,record,out,*,execute,case_main):
    path=resolve_path(config_path);c=json.loads(path.read_text());require(c.get('schema')=='etx.columns.program.v1' and c.get('record')==record,'registered program record/schema mismatch')
    require(set(c)-{'schema','record','build','capture_root','dependencies','warmup','seed','timeout','model_batch'}==set(),'unknown program options/cases are not admitted')
    root=resolve_path(c['capture_root']);build=resolve_path(c['build']);cases=registered(record);positions=(1024,1040) if record in ('G02','G05') else (0,1040)
    report=dict(schema='etx.columns.program-report.v1',record=record,enabled=bool(execute),cases=[],configuration_sha256=sha256(path),relocation_seal=relocation_seal(),device_outcome='UNMEASURED')
    if execute:
        require(out is not None,'program --out is required');out=Path(out);require(not out.exists(),'program refuses occupied output');out.mkdir(parents=True)
    winners={}
    if record=='G02':
        report['model_batch']=c.get('model_batch',4);require(report['model_batch'] in (2,4),'G02 model batch must be explicitly2or4')
    for case in cases:
        capture_paths=[root/case.get('capture_id',case['id'])/f'p{p}'/'capture.json' for p in positions]
        entry=dict(**case,captures=[str(p) for p in capture_paths],status='PREPARED',command=['python','-m','examples.experiment_program.columns','--record',record])
        missing=[str(p) for p in capture_paths if not p.is_file()]
        if missing:entry.update(status='BLOCKED_NATIVE_CAPTURE',missing=missing)
        blocked=[key for key in case['depends'] if key not in winners]
        if execute and blocked:entry.update(status='SKIPPED_DEPENDENCY',missing_winners=blocked)
        elif not missing:
            deps=list(c.get('dependencies',[]))+[str(winners[key]) for key in case['depends'] if key in winners]
            if not execute and blocked:entry['pending_winners']=blocked
            # Confirm each declared case consumes its exact native role/batch.
            for capture in capture_paths:
                data=json.loads(capture.read_text())
                require(data['record']==record and data['batch']==case['batch'] and data['role']==case['role'],'program native case identity differs')
                if 'native_xcd' in case:require(data.get('native_xcd')==case['native_xcd'],'program shared expert XCD differs')
            target=out/case['id'] if execute else Path('/unused-dry-run')/case['id'];argv=['--record',record,'--build',str(build),'--out',str(target),'--regime','both','--warmup',str(c.get('warmup',3)),'--seed',str(c.get('seed',20261016)),'--timeout',str(c.get('timeout',120)),'--execute' if execute else '--dry-run']
            if not execute:
                for dep in blocked:argv+=['--pending-winner',dep]
            if record=='G13':argv+=['--g13-order',case['g13_order']]
            for capture in capture_paths:argv+=['--capture',str(capture)]
            for dep in deps:argv+=['--dependency',str(resolve_path(dep))]
            stdout=io.StringIO();stderr=io.StringIO()
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):code=case_main(argv)
            if execute:
                (out/(case['id']+'.cli.stdout.json')).write_text(stdout.getvalue());(out/(case['id']+'.cli.stderr.log')).write_text(stderr.getvalue())
            entry.update(command=argv,exit=code,status='CASE_FAILED')
            if not execute:entry['status']='LOCAL_PREFLIGHT_PASS' if code==0 else 'REFUSED'
            if execute and code!=0:
                report['cases'].append(entry);report['status']='ABORTED_NATIVE_FAILURE';report['admitted']=False
                (out/'program-report.json').write_text(json.dumps(report,indent=2)+'\n')
                raise ProtocolError('native correctness/liveness/source failure stops the column program; full evidence retained: '+str(out))
            result=target/'report.json'
            if code==0 and result.is_file():
                raw=json.loads(result.read_text());entry.update(status=raw['status'],report=str(result.resolve()),report_sha256=sha256(result))
                if raw.get('admitted') is True:winners[case['id']]=result.resolve()
        report['cases'].append(entry)
        if execute:(out/'program-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if execute and record=='G05' and len(winners)==2:
        from .bundle import bundle
        bundle_path=out/'gate-up-down.json';bundle(winners['routed-B4-gate_up'],winners['routed-B4-down'],build,bundle_path)
        report['matched_roles']=dict(path=str(bundle_path.resolve()),sha256=sha256(bundle_path))
    if execute and record=='G02':
        report['qkv_winners']={case:dict(path=str(p),sha256=sha256(p)) for case,p in winners.items() if case.startswith('qkv')}
    blocked=any(case['status'] in ('BLOCKED_NATIVE_CAPTURE','REFUSED','CASE_FAILED') for case in report['cases'])
    report['status']='BLOCKED' if blocked else 'PROGRAM_COMPLETE' if execute else 'PREPARED_DISABLED'
    report['admitted']=False
    if execute and not blocked:
        from .runner import admit_model_winner
        selected=report.get('matched_roles') if record=='G05' else report.get('qkv_winners',{}).get('qkv-B'+str(report.get('model_batch',4))) if record=='G02' else None
        if selected:
            admitted=admit_model_winner(selected['path'],build);report['winner']=selected;report['admitted']=admitted['admitted'];report['winner_scope']={key:admitted[key] for key in ('record','batch','role','status')}
    if execute and record=='G13' and not blocked:
        from .runner import diagnose_g13
        captures=[json.loads(p.read_text()) for p in [root/'G13-B2'/f'p{p}'/'capture.json' for p in (0,1040)]]
        report['diagnostic']=diagnose_g13(out,captures)
    if any(case.get('report') for case in report['cases']):report['device_outcome']='MEASURED_ISOLATED_SUBGRAPHS'
    if execute:(out/'program-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
