"""G01 preparation/build and explicitly requested future resident replay."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .prepare import capture_manifest,source,HERE
from ..protocol import require,sha,verify_files,pair_order,paired_decision,relocation_seal,selected_artifacts,resolve_path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','preflight','replay','dry-run','qualify','program'])
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--capture',type=Path,action='append',default=[])
    p.add_argument('--out',type=Path)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--population',type=int,default=64)
    p.add_argument('--timing-layer',type=int,default=0)
    p.add_argument('--seed',type=int,default=19)
    p.add_argument('--deadline',type=float,default=120)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--control-result',type=Path);p.add_argument('--probe-result',type=Path);p.add_argument('--empty-result',type=Path)
    p.add_argument('--stress',action='store_true')
    p.add_argument('--diagnostic-mode',choices=('probe','empty'),default='probe')
    p.add_argument('--diagnostic',action='store_true',help='separate raw landmarks only; never an optimization verdict')
    a=p.parse_args(argv)
    a.build=resolve_path(a.build).resolve()
    if a.action=='qualify':
        from .diagnostics import qualify
        print(json.dumps(qualify(a.build,a.control_result,a.probe_result,a.empty_result),indent=2));return 0
    require(a.warmup>0 and a.population>0 and 0<=a.timing_layer<48 and 0<a.deadline<=600,'bounded protocol settings')
    if a.action=='build':
        from ..build_support import build
        identity=build(a.build,dict(router=source(),router_probe=source(probe=True),router_empty='#define G01_EMPTY 1\n'+source(probe=True)),HERE/'replay.cpp')
        r=next(x for x in identity['resources']['router'] if x['kernel']=='g01_run')
        identity['g01_resource_admitted']=r['vgpr']<=128 and r['lds_bytes']<=65536 and r['static_blocks_per_cu']==1 and not any(r[k] for k in ('vgpr_spills','sgpr_spills','scratch_bytes_lane'))
        identity['g01_resource_scope']='one common control/candidate kernel,304blocks,one perCU; physical38workers/XCD must be checked natively'
        (a.build/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        print(json.dumps(identity,indent=2));return 0
    command=[str((a.build/'replay').resolve()),str((a.out or Path('RUN_DIRECTORY'))/'config.txt')]
    if a.action=='dry-run':
        print(json.dumps(dict(record='G01',command=command,enabled=False,requires='both retained original captures, sealed target build, future GPU authorization',order=pair_order(a.seed))));return 0
    require(len(a.capture)==2,'positions0 and1040 original captures required')
    captures={capture_manifest(d)['positions'][0]:d.resolve() for d in a.capture}
    require(set(captures)=={0,1040},'missing/duplicate decode position')
    identity=json.loads((a.build/'identity.json').read_text());selected_artifacts(a.build,identity['files'],['replay','router.hsaco','router_probe.hsaco','router_empty.hsaco']);verify_files(identity['files']);verify_files(identity['dependency_files'])
    require(identity['dynamic_header_sha256']==sha(HERE.parent/'hip_dynamic.h'),'HIP ABI loader changed; rebuild')
    require(identity.get('g01_resource_admitted') is True,'G01 resource gate not established for this build')
    require(identity['host_source_sha256']==sha(HERE/'replay.cpp'),'host source identity changed; rebuild')
    require((a.build/'router.hip').read_text()==source() and (a.build/'router_probe.hip').read_text()==source(probe=True),'candidate source/complete switches changed; rebuild')
    if a.action=='preflight':
        print(json.dumps(dict(record='G01',local_preflight='PASS',native_cases=96,gpu='NOT_RUN')));return 0
    if a.action=='program':
        require(a.execute,'device program disabled by default')
        from .program import run
        print(json.dumps(run(a,main),indent=2));return 0
    require(a.execute,'device execution disabled; explicit --execute required only after separate authorization')
    require(a.out is not None,'retained output directory required')
    a.out.mkdir(parents=True,exist_ok=False)
    prefixes=[str(captures[pos]/f'L{layer}_') for pos in (0,1040) for layer in range(48)]
    line=' '.join(json.dumps(s) for s in [str((a.build/(('router_empty.hsaco' if a.diagnostic_mode=='empty' else 'router_probe.hsaco') if a.diagnostic else 'router.hsaco')).resolve()),str(a.out.resolve())])
    line+=f' {a.warmup} {a.population} {48+a.timing_layer} 96 {a.deadline} {int(a.diagnostic)} {int(a.stress)}\n'+' '.join(pair_order(a.seed))+'\n'
    (a.out/'config.txt').write_text(line+'\n'.join(map(json.dumps,prefixes))+'\n')
    seal={str(a.build/'identity.json'):sha(a.build/'identity.json'),**relocation_seal()}
    for d in a.capture:
        m=capture_manifest(d);seal[str(d/'capture.json')]=sha(d/'capture.json')
        seal.update({str(d/n):h for n,h in m['files'].items()})
    (a.out/'seal.json').write_text(json.dumps(dict(inputs=seal,command=command,warmup=a.warmup,population=a.population,seed=a.seed,stress=a.stress),indent=2))
    # Clocks/power are retained provenance; they cannot create a counter claim.
    with (a.out/'telemetry-before.log').open('w') as f:
        subprocess.run(['rocm-smi','--showclocks','--showpower'],stdout=f,stderr=subprocess.STDOUT,timeout=10,check=True)
    with (a.out/'native.log').open('w') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=a.deadline+10,check=True)
    with (a.out/'telemetry-after.log').open('w') as f:
        subprocess.run(['rocm-smi','--showclocks','--showpower'],stdout=f,stderr=subprocess.STDOUT,timeout=10,check=True)
    verify_files(seal)
    log=(a.out/'native.log').read_text()
    from .receipts import parse_native
    result=parse_native(log,population=a.population,seed=a.seed,diagnostic=a.diagnostic)
    samples=result.pop('samples')
    result.update(record='G01',instrumented=a.diagnostic,build_identity_sha256=sha(a.build/'identity.json'),inputs=seal,samples=samples)
    (a.out/'decision.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
