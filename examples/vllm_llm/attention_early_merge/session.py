"""Bounded C6 preparation/measurement pipeline. No provisioning or remote calls.

The caller must select exactly one GPU. A work directory has one fixed deadline.
Each build retry has a new attempt. A run never trusts an earlier correctness gate.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys
import time
from .build import ROOT, clean_env, sha
from . import evidence as ev
from examples.vllm_llm import import_certificate as cert
from etx.session import proc

CAP_SECONDS=2400


def save(path,data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(data,indent=2)+'\n'); tmp.replace(path)


@contextmanager
def locked(work,model,now=time.time):
    work=Path(work); work.mkdir(parents=True,exist_ok=True)
    with (work/'runner.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('duplicate C6 runner; inspect runner.lock and stage.json') from None
        lease=work/'lease.json'
        if lease.exists():
            record=json.loads(lease.read_text()); ev.require(record['model']==str(Path(model).resolve()),'lease model changed')
        else:
            record=dict(model=str(Path(model).resolve()),started=now(),deadline=now()+CAP_SECONDS)
            save(lease,record)
        ev.require(now()<record['deadline'],'fixed C6 deadline expired; do not silently extend budget')
        yield record['deadline']


class Runner:
    def __init__(self,work,model,deadline):
        self.work=Path(work); self.model=Path(model).resolve(); self.deadline=deadline
        self.attempt=None; self.action="run"

    def stage(self,name,blocker=None):
        record=dict(stage=name,blocker=blocker,evidence=str(self.work),deadline=self.deadline,
                    safe_resume=f'python -m examples.vllm_llm.attention_early_merge.session --work {shlex.quote(str(self.work))} --model {shlex.quote(str(self.model))} {self.action}')
        save(self.work/'stage.json',record); print(json.dumps(record),flush=True)

    def command(self,name,args,seconds=120,extra=None):
        self.stage(name)
        log=self.attempt/(name+'.log'); ev.require(not log.exists(),'refuse raw-log overwrite '+str(log))
        env=clean_env(); env.update(extra or {})
        remaining=min(seconds,self.deadline-time.time()); ev.require(remaining>0,'deadline before '+name)
        def progress(elapsed): print(f'PROGRESS stage={name} elapsed={elapsed:.0f}s evidence={log}',flush=True)
        with log.open('x') as sink:
            rc=proc.run(list(map(str,args)),sink=sink,env=env,cwd=ROOT,deadline_s=remaining,on_poll=progress,poll_s=20)
        (self.attempt/(name+'.rc')).write_text(str(rc)+'\n')
        ev.require(rc==0,f'{name} exit={rc}; see {log}')
        return log.read_text()

    def inputs(self):
        names=('vl.bin','vl.manifest','golden_c1024.txt')
        return {n:sha(self.model/n) for n in names}

    def build(self):
        self.action="build"
        index=len(list(self.work.glob('attempt-*')))
        self.attempt=self.work/f'attempt-{index:03d}'; self.attempt.mkdir(exist_ok=False)
        save(self.attempt/'inputs.json',self.inputs())
        self.command('build_envelope',[sys.executable,'-m','examples.vllm_llm.attention_early_merge.offline','--compiler','rocm','--out',self.attempt/'envelope'],600)
        for label,args in [('probe',[]),('diagnostic',['--diagnostic'])]:
            self.command('build_'+label,[sys.executable,'-m','examples.vllm_llm.attention_early_merge.build','--out',self.attempt/label,'--footprint',self.attempt/'envelope',*args],600)
        self.command('build_model',[sys.executable,'-m','examples.vllm_llm.attention_early_merge.build_model','--out',self.attempt/'model','--model',self.model],900)
        # Native full-dispatch resource envelope uses the SAME toolchain as the device link.
        self.verify()
        save(self.work/'current.json',dict(attempt=self.attempt.name))
        self.stage('BUILT; mandatory device checks pending')

    def verify(self):
        ev.require(json.loads((self.attempt/'inputs.json').read_text())==self.inputs(),'stale model/golden identity')
        p=ev.identity(self.attempt/'probe'); d=ev.identity(self.attempt/'diagnostic')
        ev.require(p.get('retained_footprint') and d.get('retained_footprint'),'timing requires retained whole dispatcher')
        ev.require(p['retained_footprint']==d['retained_footprint'] and p['retained_footprint']['directory']==str((self.attempt/'envelope').resolve()),'wrong retained dispatcher')
        ev.require(p['diagnostic'] is False and d['diagnostic'] is True,'diagnostic/performance identity')
        ev.require(p['imports']==d['imports'],'diagnostic imports differ')
        ev.identity(self.attempt/'model','c6_model_identity.json')
        m=cert.verified_build(self.attempt/'model',self.model)
        full=ev.identity(self.attempt/'envelope','offline.json')
        ev.resource_gate(p['resources']); ev.resource_gate(d['resources'])
        ev.resource_gate(full['resources']['envelope'])
        # Raw hashes remain bound above. Independent build paths differ only in
        # LLVM path headers and HIP's unused unit-ID symbol. No code is stripped.
        ev.same_imports(self.attempt/'probe/imp',self.attempt/'model/imp',p['imports'])
        return m

    def run(self):
        self.action="run"
        self.attempt=self.work/json.loads((self.work/'current.json').read_text())['attempt']
        # New run directory retains all failed and prior logs. Deadline is shared.
        checks=self.attempt/f'checks-{len(list(self.attempt.glob("checks-*"))):03d}'; checks.mkdir()
        build=self.attempt; self.verify(); self.attempt=checks
        capture=checks/'capture'; model=build/'model'
        imports=json.loads((build/'probe/imp/imports.json').read_text())
        expected=cert.expected_instances((model/'vl_plan.txt').read_text(),json.loads((model/'imp/imports.json').read_text()))
        # self.host expects build-relative paths. Keep build checks explicit here.
        def host(name,args,extra=None):
            ident=cert.verified_build(model,self.model)
            text=self.command(name,[sys.executable,'-m','examples.vllm_llm.import_certificate','--run-build',model,'--model',self.model,'--',*args],300,extra)
            ev.require(ident==cert.verified_build(model,self.model),'stale model check')
            return text,ident
        for pos in ev.POSITIONS:
            text,ident=host('imports_p'+str(pos),['--check','48','--check-pos',str(pos)],{'C6_CAPTURE_DIR':str(capture)})
            cert.check_output(text,expected,pos,ident)
        ev.seal_capture(capture,imports)
        for label,extra in [('control',{}),('candidate',{'C6_HYBRID':'1'})]:
            text,ident=host('tokens_'+label,['--repeat','1','--check-dsl'],extra)
            cert.stamped(text,ident); ev.tokens(text,hybrid=label=='candidate')
        # Fault and skew checks use real cached payloads, with repeated epoch reset.
        first=capture/'p1040/L0'
        for label in ('probe','diagnostic'):
            for fault in ('duplicate','stale','future','timeout'):
                text=self.command(label+'_fault_'+fault,[build/label/'run',first,'--fault',fault,'--repeats','2'])
                ev.require('C6-FAULT rejected='+fault in text,'missing fault rejection')
            self.command(label+'_skew',[build/label/'run',first,'--skew','10000','--repeats','8'])
            for pos in ev.POSITIONS:
                for layer in range(48):
                    text=self.command(f'{label}_check_p{pos}_L{layer}',[build/label/'run',capture/f'p{pos}/L{layer}','--repeats','4'],30)
                    ev.require(f'C6-CHECK OK layer={layer} position={pos} ' in text,'missing payload check')
        # No timing call precedes these mandatory gates. Verify identities again.
        self.attempt=build; self.verify(); self.attempt=checks
        ev.verify_capture(capture,imports)
        save(checks/'correctness.json',dict(status='PASS',inputs=self.inputs(),capture_sha=sha(capture/'capture.json'),
             binaries={n:sha(build/n/'run') for n in ('probe','diagnostic','model')}))
        decisions={}
        for pos in ev.POSITIONS:
            layers=[]
            for layer in range(48):
                text=self.command(f'measure_p{pos}_L{layer}',[build/'probe/run',capture/f'p{pos}/L{layer}','--measure','--repeats','1'],30)
                layers.append(ev.parse_measurement(text,layer,pos))
            decisions[str(pos)]=ev.decision(layers)
        # Diagnostic builds are correctness-only. There is no attributed timing claim.
        result=dict(schema='etx.c6.measurement.v1',positions=decisions,correctness_sha=sha(checks/'correctness.json'),
                    probe_identity_sha=sha(build/'probe/identity.json'),model_identity_sha=sha(build/'model/c6_model_identity.json'),
                    wall_gain='UNPROVEN',reset='same reset in both arms; reset/copies outside local window',
                    control='resident separate-task protocol; not ETX scheduler or graph wall baseline',
                    diagnostic_timing='NOT_USED',target_pass=all(x['combined_50us_target']=='PASS' for x in decisions.values()))
        save(checks/'measurement.json',result); self.stage('COMPLETE local probe; full-megakernel/wall claim remains unproved')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--work',type=Path,required=True); p.add_argument('--model',type=Path,required=True)
    p.add_argument('action',choices=['build','run','status']); a=p.parse_args()
    if a.action=='status':
        print((a.work/'stage.json').read_text() if (a.work/'stage.json').exists() else 'NOT_STARTED'); return
    try:
        with locked(a.work,a.model) as deadline:
            runner=Runner(a.work,a.model,deadline)
            try: getattr(runner,a.action)()
            except Exception as exc: runner.stage('BLOCKED',str(exc)); raise
    except Exception as exc:
        print('BLOCKER',exc,flush=True); raise SystemExit(2)

if __name__=='__main__': main()
