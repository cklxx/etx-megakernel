"""Local staged experiment session; no provisioning, SSH, cloud or publication.

All device entry points default off. Future execution is separately authorized;
this module's own tests use explicitly labeled synthetic CPU child processes.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tarfile
import time
import uuid
from contextvars import ContextVar

from etx.artifacts.sealed import source_provenance
from etx.contracts.execution import run_logged

_stage_runner = ContextVar('sealed_stage_runner', default=None)

from etx.experiments.sealed_protocol import require,sha,verify_files,relocation_seal

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'examples/experiment_program'
SPECS={x['id']:x for x in json.loads(Path(__file__).with_name('specs.json').read_text())['experiments']}
MODULES={'G01':'router','G02':'columns','G03':'columns','G04':'routing','G05':'columns',
         'G06':'issue_order','G07':'issue_order','G08':'issue_order',
         'G09':'row_ops','G10':'row_ops','G11':'row_ops','G12':'validation','G13':'columns'}



def execution_order(selected):
    # G11 needs the same session's completed G12 trace. Numeric record order
    # would silently skip it before G12 has run. Selection never adds records.
    order=('G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G13','G12','G11')
    require(len(selected)==len(set(selected)) and set(selected)<=set(order),'duplicate or unknown session selection')
    return [record for record in order if record in selected]


def eligible(record,decisions,options):
    if record=='G05':return decisions.get('G04',{}).get('complete') is True
    if record in ('G03','G13'):
        return decisions.get('G02',{}).get('admitted') is True and (record!='G13' or options.get('coalescing_decision_active') is True)
    if record=='G12':
        return any(decisions.get(k,{}).get('admitted') is True for k in ('G01','G02','G05','G06','G07','G08','G09','G10'))
    if record=='G11':return decisions.get('G12',{}).get('complete') is True and options.get('fold_decision_active') is True
    return True



def decision_for_dependencies(record,stage,path):
    """Canonical dependency evidence is recomputed from raw native receipts."""
    path=Path(path);value=json.loads(path.read_text())
    if record=='G04':
        from etx.experiments.sealed_routing import session_decision
        return session_decision(stage,path,value=value)
    if record in ('G01','G02','G05','G06','G07','G08','G09','G10'):
        status=value.get('status',value.get('verdict',{}).get('status'))
        claimed=value.get('admitted') is True or status in ('MICRO_WINNER','MICRO_PASS','MICRO_PASS_BOTH_REGIMES','MICRO_PASS_BOTH_POSITIONS')
        if not claimed:return dict(record=record,admitted=False,decision_sha256=sha(path))
        require(isinstance(stage.get('admission'),dict),'raw admission configuration required')
        c=dict(stage['admission'],record=record,decision=str(path.resolve()))
        identity=Path(c['build'])/'identity.json'
        require(stage['input_files'].get(str(identity.resolve()))==sha(identity),'admission build not sealed by session')
        from examples.experiment_program.validation import require_admission
        proof=require_admission(c)
        return dict(record=record,admitted=True,complete=True,proof=proof,decision_sha256=sha(path))
    if record=='G11':
        if value.get('schema')=='etx.g11.report.v1':
            from examples.experiment_program.publication_run import verify_report
            return verify_report(path,stage)
        from examples.experiment_program.row_ops.program import verify_diagnostic
        return verify_diagnostic(path,stage)
    if record=='G13':
        from examples.experiment_program.columns.runner import diagnose_g13
        cmd=stage['command'];cfg=Path(cmd[cmd.index('--program')+1]);recipe=json.loads(cfg.read_text())
        root=Path(recipe['capture_root']);captures=[json.loads((root/'G13-B2'/f'p{p}/capture.json').read_text()) for p in (0,1040)]
        require(value.get('diagnostic')==diagnose_g13(path.parent,captures),'G13 summary differs from raw diagnostic controls')
        return dict(record=record,complete=True,admitted=False,decision_sha256=sha(path))
    if record=='G12':
        from examples.experiment_program.validation import replay
        cmd=stage['command'];c=json.loads(Path(cmd[cmd.index('--config')+1]).read_text())
        actual=replay(c,path.parent,write=False)
        require(actual==value,'validation summary differs from raw logs')
        return dict(record=record,complete=True,admitted=False,decision_sha256=sha(path))
    return dict(record=record,complete=True,admitted=False,decision_sha256=sha(path))


def verify_resume(state,config_sha,revision):
    require(state.get('config_sha256')==config_sha and state.get('source_revision')==revision,'resume identity mismatch')
    if state.get('sealed'):verify_files(state['sealed'])
    require(state.get('status') not in ('RUNNING','FAILED','NOT_COMPLETE'),'refuse ambiguous/incomplete attempt; preserve it and choose a new output directory')


def export_record(directory,destination):
    directory=Path(directory).resolve();destination=Path(destination).resolve()
    require(not destination.is_relative_to(directory),'export must be outside retained record')
    owner=destination.with_name(destination.name+'.owner')
    if destination.exists():require(owner.is_file() and owner.read_text()==str(directory)+'\n','refuse overwriting unowned archive')
    if not owner.exists():
        with owner.open('x') as f:f.write(str(directory)+'\n')
    require(owner.read_text()==str(directory)+'\n','archive ownership mismatch')
    partial=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    with tarfile.open(partial,'x') as archive:
        for p in sorted(directory.rglob('*')):
            require(not p.is_symlink(),'export refuses symlink or escaped shared data')
            if p.is_file():archive.add(p,arcname=str(p.relative_to(directory)),recursive=False)
    # Verify every exported byte before an atomic publication to the owned local
    # archive. An interrupted export cannot relabel an incomplete stage complete.
    with tarfile.open(partial) as archive:
        import hashlib
        for member in archive:
            stream=archive.extractfile(member)
            require(stream is not None and hashlib.sha256(stream.read()).hexdigest()==sha(directory/member.name),'export verification failed')
    partial.replace(destination)
    return sha(destination)


def configuration(path):
    c=json.loads(Path(path).read_text())
    require(c.get('base')=='b573664faa4e7988dff353bc03aca4ce8561a729','wrong published preparation base')
    require(isinstance(c.get('source_snapshot'),dict) and bool(c['source_snapshot']),'complete staged source snapshot required')
    require(isinstance(c.get('records'),dict) and set(c['records'])<=set(SPECS),'unknown records')
    require(0<c.get('budget_seconds',0)<=14400,'bounded declared session budget required')
    for record,stage in c['records'].items():
        if stage.get('local_terminal'):
            require(record=='G11' and stage['local_terminal'].get('kind')=='G11_PUBLICATION','only the registered publication resource terminal is supported')
        require(isinstance(stage.get('input_files'),dict) and bool(stage['input_files']),'stage input seal required')
        require(all(stage['input_files'].get(path)==digest for path,digest in relocation_seal().items()),'explicit staging map must be sealed by every session stage')
        for key in ('preflight','command'):
            cmd=stage.get(key)
            expected='examples.experiment_program.'+MODULES[record]
            if record=='G11' and isinstance(cmd,list) and len(cmd)>2 and cmd[2]=='examples.experiment_program.publication_run':
                require('--config' in cmd and cmd.index('--config')+1<len(cmd),'G11 integrated recipe required')
                recipe=Path(cmd[cmd.index('--config')+1]);r=json.loads(recipe.read_text())
                require(r.get('schema')=='etx.g11.program.v1' and stage.get('local_terminal',{}).get('identity')==r.get('publication_preparation'),'G11 integrated module requires sealed canonical publication recipe')
                expected='examples.experiment_program.publication_run'

            require(isinstance(cmd,list) and len(cmd)>=3 and cmd[1:3]==['-m',expected],f'{record}: exact prepared module entry required')
            require(Path(cmd[0]).resolve()==Path(sys.executable).resolve(),f'{record}: current staged project Python required')
        require('--execute' not in stage['preflight'],f'{record}: preflight may not execute a device')
        cmd=stage['preflight'];module=MODULES[record]
        require('--device-execution-authorized' not in cmd,'device authorization forbidden in preflight')
        safe=(cmd[3:6]==['--record',record,'preflight'] if module=='issue_order' else
              cmd[3:5]==['--record',record] and '--dry-run' in cmd if module=='columns' else
              len(cmd)>3 and cmd[3] in ('execute','program') and '--dry-run' in cmd if module=='row_ops' else
              len(cmd)>3 and cmd[3]=='preflight')
        require(safe,f'{record}: unsafe preflight action')
        limit=13500 if record=='G12' else 900
        require(0<stage.get('timeout_seconds',0)<=limit,f'{record}: bounded timeout')
        rel=Path(stage['decision_file']);require(not rel.is_absolute() and '..' not in rel.parts,'decision file must be relative to session output')
    return c



def bind_output(config,output):
    """Bind only declared output paths to a runner's newly owned record."""
    original=Path(config['output_directory']).resolve() if 'output_directory' in config else None
    if original is None:return config
    output=Path(output).resolve()
    def bind(value):
        if isinstance(value,dict):return {bind(k):bind(v) for k,v in value.items()}
        if isinstance(value,list):return [bind(v) for v in value]
        if isinstance(value,str) and value.startswith(str(original)):
            p=Path(value)
            if p.is_relative_to(original):return str(output/p.relative_to(original))
        return value
    return bind(config)


def run(request):
    from types import SimpleNamespace
    a=SimpleNamespace(action=request.action, config=Path(request.config) if request.config else None,
                      out=Path(request.output) if request.output else None, record=request.records,
                      resume=request.resume, execute=request.execute)
    if a.action=='list':
        return dict(records=[dict(id=k,title=v['title'],kind=v['kind'],dependencies=v['dependencies'],module=MODULES[k],enabled=False) for k,v in SPECS.items()])
    require(a.out is not None,'retained output/staging path required')
    if a.action=='export':
        return dict(sha256=export_record(a.out,a.out.with_suffix('.tar')))
    require(a.config is not None,'session config required')
    c=bind_output(configuration(a.config),a.out);selected=execution_order(a.record or list(c['records']))
    require(set(selected)<=set(c['records']),'selected record missing stage configuration')
    if a.action=='dry-run':
        return dict(selected=selected,stages={k:c['records'][k] for k in selected},budget_seconds=c['budget_seconds'],enabled=False,deepseek_graph='UNAVAILABLE_LDS_LIFETIME')
    # Preparation never executes a GPU. The public run entry requires a separate
    # explicit invocation after the owner/user GPU authorization, not a timer.
    if a.action=='run':require(a.execute,'device execution disabled by default')
    verify_files(c['source_snapshot'])
    provenance=source_provenance(c['source_snapshot'], c.get('source_revision'))
    revision=provenance['commit']
    stamp=sha(a.config)
    state_path=a.out/'session.json'
    if a.resume:
        state=json.loads(state_path.read_text());verify_resume(state,stamp,revision)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        state=dict(status='PREPARED',config_sha256=stamp,source_revision=revision,source_provenance=provenance,stages={},sealed={})
    start=time.monotonic();decisions={}
    try:
        for record in selected:
            stage=c['records'][record];status=state['stages'].get(record,{})
            if stage.get('local_terminal'):
                from examples.experiment_program.publication import admission
                t=stage['local_terminal'];require(sha(t['identity'])==t['sha256'],'publication terminal identity changed')
                gate=admission(t['identity'])
                if not gate['admitted']:
                    state['stages'][record]=dict(status='CLOSED_LOCAL_DIAGNOSTIC',evidence=gate);continue
            try:verify_files(stage['input_files'])
            except (ValueError,OSError) as e:
                state['stages'][record]=dict(status='MISSING_PREREQUISITE',reason=str(e));continue
            if a.action=='preflight' and status.get('status')=='PREFLIGHT_PASS':continue
            if status.get('status')=='COMPLETE':
                decisions[record]=decision_for_dependencies(record,stage,a.out/stage['decision_file']);continue
            if a.action=='run' and not eligible(record,decisions,c.get('options',{})):
                state['stages'][record]=dict(status='NOT_ADMITTED');continue
            remaining=c['budget_seconds']-(time.monotonic()-start)
            require(remaining>0,'session budget exhausted')
            cmd=stage['preflight'] if a.action=='preflight' else stage['command']
            state['status']='RUNNING';state['stages'][record]=dict(status='RUNNING',argv=cmd)
            state_path.write_text(json.dumps(state,indent=2)+'\n')
            log=a.out/(record+'-'+a.action+'.log')
            rc=(_stage_runner.get() or run_logged)(cmd,log,seconds=min(remaining,stage['timeout_seconds']))
            state['sealed'][str(log.resolve())]=sha(log)
            if rc:
                if a.action=='preflight':state['stages'][record]=dict(status='MISSING_PREREQUISITE',exit=rc);continue
                raise ValueError(f'{record}: native stage exit{rc}; stop correctness/liveness failure')
            if a.action=='preflight':
                value=json.loads(log.read_text())
                require(isinstance(value,dict),'preflight must emit structured evidence')
                marker=str(value.get('status',''))+' '+str(value.get('native_input_status',''))
                blocked=any(word in marker.upper() for word in ('BLOCKED','REFUSED','MISSING','REJECT','UNAVAILABLE'))
                blocked |= value.get('native_admission',{}).get('timing_eligible') is False
                state['stages'][record]=dict(status='MISSING_PREREQUISITE' if blocked else 'PREFLIGHT_PASS',evidence=value)
                continue
            decision=a.out/stage['decision_file'];require(decision.is_file(),'native stage missing complete result')
            decisions[record]=decision_for_dependencies(record,stage,decision)
            state['sealed'][str(decision.resolve())]=sha(decision)
            state['stages'][record]=dict(status='COMPLETE',exit=rc)
        state['status']='LOCAL_PREFLIGHT_COMPLETE' if a.action=='preflight' else 'COMPLETE'
    except BaseException as e:
        state['status']='NOT_COMPLETE';state['error']=str(e);raise
    finally:
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        export_record(a.out,a.out.with_suffix('.tar'))
    return state

