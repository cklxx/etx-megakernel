"""Run optional qualified exposure capture only after a future GPU authorization."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
from .__main__ import verify_build,DEFAULT_CAPTURE
from .receipts import validate_native_log
from .evidence import require,sha,validate_captures
from .diagnostic import check_stamps,qualify
from .build import ROOT
from examples.experiment_program.protocol import resolve_path

def diagnostic_identity(path,base):
    path=resolve_path(path);record=json.loads((path/'identity.json').read_text())
    require(record['schema']=='etx.issue-order.diagnostic.v1','diagnostic schema')
    require(record['base_build_sha256']==sha(Path(base)/'identity.json'),'diagnostic original build identity')
    for name,digest in record['hashes'].items():require(sha(path/name)==digest,'stale diagnostic '+name)
    for name,digest in record['source_hashes'].items():require(sha(ROOT/name)==digest,'stale diagnostic implementation '+name)
    return record

def parse_trace(text,record,position,enabled,phases=None):
    validate_native_log(text,record=record,expected_warmup=1,expected_samples=1) # exposure qualification only
    matches=re.findall(r'^IO_STAMP arm=(control|candidate) epoch=(\d+) index=(\d+) tick=(\d+)$',text,re.M)+re.findall(r'^IO_PLACEMENT arm=(control|candidate) epoch=(\d+) index=(\d+) raw=(\d+)$',text,re.M)
    if not enabled:require(not matches,'empty trace emitted payload');return {}
    epochs={}
    for arm,epoch,index,tick in matches:
        value=epochs.setdefault(int(epoch),{'arm':arm,'rows':[]});require(value['arm']==arm,'mixed epoch arm')
        value['rows'].append({'index':int(index),'tick':int(tick)})
    # Two declared warmup + ten measured launches, original driver one sample.
    require(set(epochs)==set(range(1,13)),'incomplete probe epoch population')
    return {str(epoch):dict(arm=value['arm'],**check_stamps(record,position,value['rows'],phases)) for epoch,value in epochs.items()}

def times(text,arm):
    rows=re.findall(r'^IO_PAIR ordinal=\d+ control_ns=(\S+) candidate_ns=(\S+) ',text,re.M)
    require(len(rows)==5,'probe pair count');return statistics.median(float(row[arm=='candidate']) for row in rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--probe-build',type=Path,required=True);p.add_argument('--capture',type=Path,default=DEFAULT_CAPTURE);p.add_argument('--out',type=Path);p.add_argument('--matched-build',type=Path);p.add_argument('--layers',default='0,24,47');p.add_argument('--dry-run',action='store_true');p.add_argument('--device-execution-authorized',action='store_true');a=p.parse_args()
    for key in ('base','probe_build','capture','matched_build'):
        if getattr(a,key,None) is not None:setattr(a,key,resolve_path(getattr(a,key)))
    base=verify_build(a.base);probe=diagnostic_identity(a.probe_build,a.base);frames=validate_captures(a.capture,base['original_imports'],tuple(map(int,a.layers.split(','))))
    matched=None
    if a.matched_build:
        matched=json.loads((a.matched_build/'identity.json').read_text())
        require(matched.get('schema')=='etx.issue-order.matched-diagnostic.v1' and matched['base_sha256']==sha(a.base/'identity.json') and matched['probe_sha256']==sha(a.probe_build/'identity.json'),'matched module/input identity')
        for name,digest in matched['hashes'].items():require(sha(a.matched_build/name)==digest,'stale matched native module '+name)
        for name,digest in matched['source_hashes'].items():require(sha(ROOT/name)==digest,'stale matching helper '+name)
    print(json.dumps({'matched_status':matched.get('status') if matched else None,'record':base['record'],'declared_frames':list(frames),'allocation_equal':probe['allocation_equal'],'status':probe['status'],'runs':['original uninstrumented','same diagnostic kernel disabled stamps','same diagnostic kernel enabled stamps'],'perturbation_limit_ns':500,'causal_claim_before_qualification':False},indent=2),flush=True)
    if a.dry_run:return
    require(a.device_execution_authorized,'future GPU authorization required')
    require(all(matched['physical_equal'].values()) if matched else all(probe['allocation_equal'].values()),'probe allocation differs from uninstrumented; instrumentation attribution closed before GPU')
    require(a.out is not None and not a.out.exists(),'fresh --out required');a.out.mkdir(parents=True)
    result={}
    for name in frames:
        logs={};coverage={}
        modes=[('uninstrumented',a.base,None),('empty',a.probe_build,False),('probe',a.probe_build,True)]
        if matched:modes.insert(0,('raw',a.base,None))
        for mode,build,enabled in modes:
            print('STAGE diagnostic',name,mode,flush=True)
            module=build/'kernel.hsaco'
            if matched and mode!='raw':module=a.matched_build/('uninstrumented.hsaco' if mode=='uninstrumented' else 'diagnostic.hsaco')
            cmd=[str(build/'replay'),str(module),str(a.capture/name),'--warmup','1','--samples','1','--seed','617']
            if enabled is not None:cmd+=['--probe-mode','1' if enabled else '0']
            log=a.out/(name.replace('/','_')+'_'+mode+'.log')
            with log.open('w') as f:run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=120)
            require(run.returncode==0,'native probe failure '+str(log));logs[mode]=log.read_text();validate_native_log(logs[mode],record=base['record'],expected_warmup=1,expected_samples=1)
            if enabled is not None:coverage[mode]=parse_trace(logs[mode],base['record'],frames[name]['position'],enabled,probe.get('probe_phases'))
        observed={}
        for arm in ('control','candidate'):
            r=next(r for r in (matched['roles']['diagnostic']['resources'] if matched else probe['resources']) if r['kernel']=='io_'+arm)
            b=next(r for r in matched['roles']['uninstrumented']['resources'] if r['kernel']=='io_'+arm) if matched else probe['base_resources'][arm]
            reference=times(logs['raw'] if matched else logs['uninstrumented'],arm)
            observed[arm]=qualify(b,r,r,reference,times(logs['empty'],arm),times(logs['probe'],arm),bool(coverage['probe']))
            if matched:
                observed[arm]['reserve_perturbation_ns']=abs(times(logs['uninstrumented'],arm)-reference)
                if observed[arm]['reserve_perturbation_ns']>500:observed[arm]['status']='UNQUALIFIED'
        result[name]={'qualification':observed,'landmarks':coverage['probe'],'output_identity':frames[name]}
    (a.out/'exposure.json').write_text(json.dumps({'schema':'etx.issue-order.exposure.v1','record':base['record'],'results':result,'optimization_claim':False,'probe_sha256':sha(a.probe_build/'identity.json'),'base_sha256':sha(a.base/'identity.json'),'matched_sha256':sha(a.matched_build/'identity.json') if matched else None,'declared_probe_phases':probe.get('probe_phases'),'full_phase_interval_claim':False},indent=2)+'\n')
if __name__=='__main__':main()
