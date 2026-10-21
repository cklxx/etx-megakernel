"""Recompute matched diagnostic qualification; no optimization verdict from stamps."""
import json,re,struct
from pathlib import Path
from ..protocol import require,sha,verify_files,selected_artifacts

def qualify(build,control,probe,empty):
    from .receipts import parse_native
    build=Path(build);identity=json.loads((build/'identity.json').read_text());results=[];resources=[];reasons=[]
    selected_artifacts(build,identity['files'],['replay','router.hsaco','router_probe.hsaco','router_empty.hsaco'])
    from ..row_ops.elf import descriptors,allocation_signature
    for name,directory in zip(('router','router_probe','router_empty'),(control,probe,empty)):
        d=Path(directory);r=json.loads((d/'decision.json').read_text());seal=json.loads((d/'seal.json').read_text());verify_files(seal['inputs'])
        require(r['build_identity_sha256']==sha(build/'identity.json'),'diagnostic selected build differs')
        raw=parse_native((d/'native.log').read_text(),population=seal['population'],seed=seal['seed'],diagnostic=name!='router')
        require(raw['samples']==r['samples'],'diagnostic raw pair means changed')
        results.append(raw['samples']);rr=next(x for x in identity['resources'][name] if x['kernel']=='g01_run')
        resources.append(allocation_signature(descriptors(build/(name+'.hsaco'))['g01_run'],rr['agpr']))
        if name!='router':
            placements=re.findall(r'^PLACEMENT G01 epoch=(\d+) arm=(\d) reverse=(\d) winner=(\d+) reset_count=(\d+) workers_per_xcd=38 readers=8$',(d/'native.log').read_text(),re.M)
            require(len(placements)==384 and len({p[0] for p in placements})==384,'complete original/reverse control/candidate placement population required')
            expected={f'landmarks_{p[0]}.bin':p for p in placements}
            files=list(d.glob('landmarks_*.bin'));require({p.name for p in files}==set(expected),'native diagnostic record population missing/extra')
            for p in files:
                epoch,arm,reverse,winner,resets=map(int,expected[p.name])
                validate_landmarks(p.read_bytes(),epoch=epoch,arm=arm,winner=winner)
    if not resources[0]==resources[1]==resources[2]:reasons.append('native allocations differ')
    delta=max(abs(x[k]-b[k]) for arm in results[1:] for b,x in zip(results[0],arm) for k in ('control_ns','candidate_ns'))
    if delta>500:reasons.append('diagnostic perturbation exceeds500ns')
    stress=(Path(probe)/'native.log').read_text()
    if not re.search(r'^RESET G01 previous_epoch=4294967295 quiescent=1 fresh_allocation=1 reset_count=1$',stress,re.M):reasons.append('quiescent rollover boundary not observed')
    return dict(status='QUALIFIED_DIAGNOSTIC' if not reasons else 'UNQUALIFIED_DIAGNOSTIC',qualified=not reasons,reasons=reasons,resources=resources,max_perturbation_ns=delta,performance_claim=False,scope='native issue landmarks; no removable-work price')


def validate_landmarks(raw,*,epoch,arm,winner):
    require(len(raw)==264*16*13*8,'G01 task/wave placement extent')
    rows=list(struct.iter_unpack('<13Q',raw));physical={}
    for task in range(256):
        for wave in range(16):
            row=rows[task*16+wave]
            require(0<row[0]<=row[1]<=row[2] and row[8]<16 and row[9]==task//32 and row[10]<38 and row[11]==epoch and row[12]<(1<<32),'G01 incomplete native task/wave/worker/XCD identity')
            physical.setdefault(row[9],set()).add(row[8])
    require(set(physical)==set(range(8)) and all(len(v)==1 for v in physical.values()) and len(set(next(iter(v)) for v in physical.values()))==8,'G01 physical XCD population differs')
    owners=[task for task in range(128) if any(rows[task*16+w][3] for w in range(16))]
    require(len(owners)==(1 if arm==1 else 0),'actual elected owner coverage differs')
    for task in owners:
        for wave in range(16):
            row=rows[task*16+wave];require(0<row[2]<=row[3]<=row[4]<=row[5] and row[9]==winner,'selected winner task/wave/physical XCD missing')
    for domain in range(8):
        for wave in range(16):
            row=rows[(256+domain)*16+wave];require(0<row[6]<=row[7] and row[9]==domain and row[10]==0 and row[11]==epoch and row[8] in physical[domain] and row[12]<(1<<32),'G01 reader coverage')
    return dict(tasks=256,waves=16,readers=8,elected_task=owners,physical_xcds={str(d):next(iter(v)) for d,v in physical.items()})
