"""Fail-closed evidence rules shared by all thirteen conditional records.

These CPU protocol checks do not certify GPU visibility or device output bits.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
from pathlib import Path
import random
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_path(path):
    """Resolve an explicitly registered staging prefix without editing seals.

    The original metadata and hashes remain authoritative. A mapping changes
    only where identical bytes are opened. No environment mapping is implicit.
    """
    path=Path(path).expanduser().absolute()
    mapping=os.environ.get('ETX_PROGRAM_RELOCATIONS')
    if not mapping:return path
    value=json.loads(Path(mapping).read_text())
    require(value.get('schema')=='etx.path-map.v1' and isinstance(value.get('paths'),dict),'unknown explicit path-map schema')
    prefixes=[]
    for origin,target in value['paths'].items():
        require(isinstance(origin,str) and isinstance(target,str) and Path(origin).is_absolute() and Path(target).is_absolute(),'staging prefixes must be absolute')
        require('..' not in Path(origin).parts and '..' not in Path(target).parts,'staging prefix traversal')
        prefixes.append((Path(origin),Path(target)))
    require(all(not target.is_relative_to(other) for origin,target in prefixes for other,_ in prefixes if origin!=target),'staging prefixes must not chain or cycle')
    for origin,target in sorted(prefixes,key=lambda pair:len(pair[0].parts),reverse=True):
        if path.is_relative_to(origin):return target/path.relative_to(origin)
    return path


def sha(path):
    h=hashlib.sha256()
    with resolve_path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def relocation_seal():
    mapping=os.environ.get('ETX_PROGRAM_RELOCATIONS')
    if not mapping:return {}
    p=Path(mapping).resolve()
    # Validate the mapping before recording its own unchanged bytes.
    resolve_path(p)
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()}


def verify_files(files):
    require(isinstance(files,dict) and bool(files),'missing file identity')
    for path,digest in files.items():
        p=resolve_path(path)
        require(p.is_file() and len(digest)==64 and sha(p)==digest,'missing/mismatched sealed file: '+str(p))


def pair_order(seed):
    order=['AB','BA','AB','BA','AB']
    random.Random(seed).shuffle(order)
    return order


def paired_decision(samples, *, seed, qwen_layer=False, model=False):
    require(len(samples)==5,'exactly five complete pairs required')
    order=pair_order(seed)
    gains=[];controls=[]
    for i,s in enumerate(samples):
        require(s.get('ordinal')==i and s.get('order')==order[i],'pair identity/order')
        require(s.get('correct') is True and s.get('instrumented') is False,'correct uninstrumented pairs required')
        for k in ('control_ns','candidate_ns'):
            require(type(s.get(k)) in (int,float) and math.isfinite(s[k]) and s[k]>0,'invalid timing')
        controls.append(s['control_ns']);gains.append(s['control_ns']-s['candidate_ns'])
    median=statistics.median(gains);spread=max(gains)-min(gains)
    floor=30000 if model else max(625 if qwen_layer else 0,.02*statistics.median(controls))
    admitted=all(g>0 for g in gains) and median>=floor and median>2*spread
    return dict(admitted=admitted,gains_ns=gains,median_ns=median,spread_ns=spread,floor_ns=floor,
                status='ADMIT_VALIDATION' if admitted else 'CLOSE',performance_claim=False)


def capture_coverage(indices, *, batch):
    require(batch in (2,4),'native capture B2/B4 only')
    expected={(r,t,l,s) for r in range(batch) for t in range(32) for l in range(1,27) for s in range(6)}
    actual=[tuple(x) for x in indices]
    require(len(actual)==len(set(actual)) and set(actual)==expected,'incomplete/duplicate route capture')
    return dict(slots=len(actual),dense_first_layer='separately labeled; not routed',performance_claim=False)


def qualify_exposure(*, expected, observed, allocations, perturbation_ns, exact):
    observed=[tuple(x) for x in observed]
    require(len(observed)==len(set(observed)) and set(observed)==set(expected),'incomplete/duplicate probe coverage')
    require(len(allocations)==3 and allocations[0]==allocations[1]==allocations[2],'probe resource mismatch')
    require(type(perturbation_ns) in (int,float) and math.isfinite(perturbation_ns) and 0<=perturbation_ns<=500,'probe perturbation')
    require(exact is True,'probe changed outputs')
    return dict(qualified=True,performance_claim=False,scope='exposure only; not removable work')


class Election:
    """Happens-before model of G01. Never a hardware visibility certificate."""
    def __init__(self):
        self.epoch=0;self.complete=True

    def begin(self,epoch,*,quiescent):
        require(quiescent and self.complete,'reset requires all readers and launch quiescence')
        require(type(epoch) is int and self.epoch<epoch<2**32,'stale/wrapped epoch')
        self.epoch=epoch;self.drains=set();self.owners=set();self.readers=set();self.published=False;self.complete=False

    def check(self,epoch):require(epoch==self.epoch and not self.complete,'stale epoch or completed launch')

    def drain(self,owner,wave,*,epoch):
        self.check(epoch)
        require(owner in range(128) and wave in range(16) and (owner,wave) not in self.drains,'invalid/duplicate producing wave')
        self.drains.add((owner,wave))

    def arrive(self,owner,*,epoch,norm_ready=False):
        self.check(epoch)
        require(owner in range(128) and owner not in self.owners,'invalid/duplicate useful owner')
        require(all((owner,w) in self.drains for w in range(16)),'all writing waves must drain')
        require(owner!=0 or norm_ready,'owner zero canonical norm not published')
        self.owners.add(owner)

    def publish(self,*,epoch,replicas,acquire):
        self.check(epoch)
        require(acquire and self.owners==set(range(128)) and not self.published,'early finalization or missing acquire')
        replicas=list(replicas)
        require(len(replicas)==8 and set(replicas)==set(range(8)),'all eight routing payloads required')
        self.published=True

    def read(self,reader,*,epoch,acquire):
        self.check(epoch)
        require(acquire and self.published and reader in range(8) and reader not in self.readers,'reader before publication or duplicate')
        self.readers.add(reader);self.complete=len(self.readers)==8


def selected_artifacts(directory,files,names):
    """Bind the actual launch directory, including copied/explicitly relocated bytes."""
    directory=resolve_path(directory).resolve()
    require(isinstance(files,dict) and bool(files),'missing artifact identity')
    selected={}
    for name in names:
        require(Path(name).name==name,'artifact name must be local')
        matches=[digest for path,digest in files.items() if Path(path).name==name]
        require(len(matches)==1,'missing/ambiguous selected artifact identity: '+name)
        path=directory/name
        require(path.is_file() and sha(path)==matches[0],'selected artifact differs: '+str(path))
        selected[str(path)]=matches[0]
    return selected
