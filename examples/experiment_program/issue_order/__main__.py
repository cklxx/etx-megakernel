"""Prepare/replay G06-G08. All ordinary commands are CPU-only and offline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .build import ROOT, HERE, prepare
from .evidence import require, sha, validate_captures, decide
from .source import RECORDS
from examples.experiment_program.protocol import resolve_path
from .paths import dependency_path

DEFAULT_CAPTURE=Path.home()/'.hermes/cache/scratch/etx-gpu-live-20261015-evidence/c6-work/attempt-001/checks-000/capture'
DEFAULT_ATTENTION=Path.home()/'code/etx/build/qwen300-rocm-on/imp/attention.hip'

def verify_build(build):
    build=resolve_path(build); value=json.loads((build/'identity.json').read_text())
    for name,digest in value['hashes'].items(): require(sha(build/name)==digest,'stale final build '+name)
    for name,digest in value['source_hashes'].items(): require(sha(ROOT/name)==digest,'stale implementation source '+name)
    for name,digest in value.get('dependency_files',{}).items():
        local=dependency_path(name)
        require(sha(local)==digest,'changed mounted compiler/header dependency '+name)
    return value

def parse_pair_means(text):
    import re
    rows=re.findall(r'^IO_PAIR ordinal=(\d+) control_ns=(\S+) candidate_ns=(\S+) control_host_ns=(\S+) candidate_host_ns=(\S+) samples=(\d+) layer=(\d+) position=(\d+)$',text,re.M)
    require(len(rows)==5 and {int(x[0]) for x in rows}==set(range(5)),'incomplete/duplicate native pair population')
    require(len({x[5] for x in rows})==len({x[6] for x in rows})==len({x[7] for x in rows})==1,'pair endpoint/sample identity changed')
    checks=re.findall(r'^IO_CHECK OK record=(G0[678]) layer=(\d+) position=(\d+) endpoints=(\d+) epochs=(\d+)$',text,re.M)
    require(len(checks)==1,'missing/duplicate native byte gate')
    require(checks[0][1:3]==rows[0][6:8] and int(checks[0][3])==({'G06':3,'G07':1,'G08':3}[checks[0][0]]),'unbound correctness endpoint')
    return decide([(float(x[1]),float(x[2])) for x in rows])

def parse_pairs(text):
    from .receipts import validate_native_log
    return validate_native_log(text)['decision']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',choices=RECORDS,required=True)
    sub=p.add_subparsers(dest='action',required=True)
    for name in ('dry-run','build','preflight','execute','report'):
        s=sub.add_parser(name)
        s.add_argument('--out',type=Path,required=name in ('build','execute'))
        s.add_argument('--build',type=Path)
        s.add_argument('--capture',type=Path,default=DEFAULT_CAPTURE)
        s.add_argument('--layers',default='0,24,47')
        if name=='build':
            s.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
            s.add_argument('--attention-source',type=Path,default=DEFAULT_ATTENTION)
            s.add_argument('--reference-imports',type=Path,default=DEFAULT_CAPTURE/'capture.json')
            s.add_argument('--headers',type=Path,default=Path.home()/'code/rocm-headers')
            s.add_argument('--vllm',type=Path,default=Path.home()/'code/vllm')
        if name=='execute':
            s.add_argument('--device-execution-authorized',action='store_true')
            s.add_argument('--warmup',type=int,default=8);s.add_argument('--samples',type=int,default=50);s.add_argument('--seed',type=int,default=617)
    a=p.parse_args()
    for key in ('build','capture','attention_source','reference_imports','headers','vllm'):
        if getattr(a,key,None) is not None:setattr(a,key,resolve_path(getattr(a,key)))
    if a.action=='dry-run':
        print(json.dumps({'record':a.record,'enabled_by_default':False,'gpu_executed':False,'stages':['offline build + final native mapping/resources','four-boundary retained-frame provenance preflight','resident exact-byte original/candidate checks','five seeded AB/BA pairs; no model arm on miss'],'commands':[f'{sys.executable} -m examples.experiment_program.issue_order --record {a.record} build --out BUILD',f'{sys.executable} -m examples.experiment_program.issue_order --record {a.record} preflight --build BUILD --capture CAPTURE',f'{sys.executable} -m examples.experiment_program.issue_order --record {a.record} execute --build BUILD --capture CAPTURE --out NEW_LOGDIR --device-execution-authorized'],'native_admission':'REQUIRED_BEFORE_GPU'},indent=2));return
    if a.action=='build':
        require(not a.out.exists(),'refuse existing build output')
        if a.inside:
            prepare(a.record,a.out,a.attention_source,a.reference_imports,a.vllm,a.headers);return
        require(str(a.out.resolve()).startswith(str(Path.home()/'code')+'/'),'build must be in container code mount')
        # Read-only captured import descriptors are small; no tensor data copied.
        original=json.loads(a.reference_imports.read_text());original=original.get('imports',original)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        receipt=a.out.parent/(a.out.name+'.input-imports.json'); require(not receipt.exists(),'refuse overwrite input receipt')
        receipt.write_text(json.dumps(original,indent=2)+'\n')
        def mapped(path): return '/code/'+str(Path(path).resolve().relative_to(Path.home()/'code'))
        command=['docker','run','--rm','--pull=never','--network=none','--platform','linux/amd64','-v',str(Path.home()/'code')+':/code','-v',str(Path.home()/'.hermes/cache/scratch/expert-early-merge-prep-evidence/resume-final/link-deps/root')+':/link-deps:ro','-e','LD_LIBRARY_PATH=/link-deps/usr/lib/x86_64-linux-gnu','-e','TMPDIR='+mapped(a.out.parent),'--env','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH='+mapped(ROOT),'-w',mapped(ROOT),'etx-rocm22','python3','-m','examples.experiment_program.issue_order','--record',a.record,'build','--inside','--out',mapped(a.out),'--attention-source',mapped(a.attention_source),'--reference-imports',mapped(receipt),'--headers',mapped(a.headers),'--vllm',mapped(a.vllm)]
        print('CPU_OFFLINE',*command,flush=True);subprocess.run(command,check=True,timeout=900);return
    require(a.build is not None,'--build required')
    identity=verify_build(a.build);require(identity['record']==a.record,'record/build mismatch')
    layers=tuple(map(int,a.layers.split(',')))
    if a.action in ('preflight','execute'):
        frames=validate_captures(a.capture,identity['original_imports'],layers)
        value={'schema':'etx.issue-order.preflight.v1','record':a.record,'build_sha256':sha(a.build/'identity.json'),'capture_seal_sha256':sha(a.capture/'capture.json'),'frames':frames,'native_admission':identity['admission'],'gpu_execution':'NOT_RUN'}
        if a.action=='preflight': print(json.dumps(value,indent=2));return
        require(1<=a.warmup<=100 and 1<=a.samples<=1000 and 0<=a.seed<2**32,'invalid declared warmup/sample/seed budget')
        value['population']={'warmup_rounds':a.warmup,'samples_per_arm_per_pair':a.samples,'seed':a.seed,'pairs':5}
        require(a.device_execution_authorized,'GPU execution requires explicit future authorization')
        require(identity['admission']['timing_eligible'],'native/resources gate closed; do not run GPU')
        require(not a.out.exists(),'refuse overwrite/resume without complete identity revalidation')
        a.out.mkdir(parents=True);(a.out/'preflight.json').write_text(json.dumps(value,indent=2)+'\n')
        results={};commands=[]
        for name in frames:
            print('STAGE resident_replay',name,flush=True)
            cmd=[str(a.build/'replay'),str(a.build/'kernel.hsaco'),str(a.capture/name),'--warmup',str(a.warmup),'--samples',str(a.samples),'--seed',str(a.seed)]
            log=a.out/(name.replace('/','_')+'.log')
            commands.append({'frame':name,'argv':cmd,'timeout_seconds':120,'status':'STARTED'})
            (a.out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            from examples.experiment_program.telemetry import capture
            with capture(a.out/(name.replace('/','_')+'.telemetry'),dict(command=cmd,build_sha256=value['build_sha256'],capture_sha256=value['capture_seal_sha256'])):
                with log.open('w') as stream: run=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            commands[-1].update(status='COMPLETED',exit=run.returncode)
            (a.out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            require(run.returncode==0,'native correctness/liveness failure; see '+str(log))
            from .receipts import validate_native_log
            results[name]=validate_native_log(log.read_text(),record=a.record,expected_warmup=a.warmup,expected_samples=a.samples)['decision']
        # All adjacent contexts remain correctness gates; the actual context1040
        # alone supplies the target operator performance decision. Never sum layers.
        eligible=all(r['status']=='MICRO_WINNER' for n,r in results.items() if n.startswith('p1040/'))
        final={'status':'MICRO_WINNER' if eligible else 'MICRO_MISS','record':a.record,'results':results,'build_sha256':value['build_sha256'],'capture_seal_sha256':value['capture_seal_sha256'],'model_admission':'G12_REQUIRED' if eligible else 'CLOSED','gpu_execution':'COMPLETED'}
        (a.out/'result.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2));return
    print(json.dumps({'record':a.record,'build_sha256':sha(a.build/'identity.json'),'admission':identity['admission'],'resources':identity['resources'],'gpu_execution':'NOT_RUN','status':'LOCAL_BUILD_COMPLETE_AWAITING_REVIEW' if identity['admission']['timing_eligible'] else 'CLOSED_OR_BLOCKED_BEFORE_GPU'},indent=2))

if __name__=='__main__':
    try: main()
    except (ValueError,OSError,subprocess.SubprocessError) as e: print('BLOCKER',e,file=sys.stderr);raise SystemExit(2)
