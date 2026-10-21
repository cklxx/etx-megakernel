"""Validate sealed native snapshots and generate a single-process resident replay."""
import hashlib
import json
from pathlib import Path
from examples.experiment_program.protocol import resolve_path
import random
import struct
import shutil
from examples.experiment_program.protocol import sha
from examples.vllm_llm.bf16_300.pack import sha_slice as _sha_slice

def sha_slice(path,offset,size):return _sha_slice(resolve_path(path),offset,size)

from .export_contract import intermediate_extent

POINTERS=(0,8,16,24,32,64,72,80,88,112)


def output_reservation(pack,arms,samples,output,args=None):
    """Bound every retained native export before any execution or copy."""
    # replay.cpp exports every correctness reference and all 72B task/wave
    # records for every arm/sample, plus one 16384B/task intermediate per arm.
    payload=samples*arms*(sum(r['bytes'] for r in pack['references'])+pack['count']*16*72)+arms*intermediate_extent(pack,args)
    overhead=max(16<<20,samples*arms*2048)
    reserve=payload+overhead;parent=resolve_path(output).resolve().parent
    while not parent.exists():parent=parent.parent
    free=shutil.disk_usage(parent).free
    if free<reserve+(2<<30):raise ValueError(f'insufficient output storage: need {reserve} export bytes plus protected2GiB floor; free {free}')
    return dict(estimated_export_bytes=reserve,free_bytes=free,protected_free_floor_bytes=2<<30)


def validate_pack(p):
    if p.get('schema')!='etx-row-replay-v1' or p.get('position') not in (0,1040) or p.get('baseline_revision')!='b573664faa4e7988dff353bc03aca4ce8561a729':
        raise ValueError('requires exact frozen Qwen native snapshot at position 0/1040')
    if p['role'] not in ('dsl_moe_up','dsl_moe_down','dsl_gemv_fold'): raise ValueError('role')
    if p['count']<=0 or p['first']<0: raise ValueError('task range')
    args=resolve_path(p['arguments']['file']).read_bytes()
    if len(args)!=160 or sha(p['arguments']['file'])!=p['arguments']['sha256']: raise ValueError('arguments identity')
    if struct.unpack_from('<i',args,120)[0]!=1: raise ValueError('B1 only')
    if p['first']+p['count']>struct.unpack_from('<i',args,156)[0]: raise ValueError('task range exceeds native grid')
    offsets=[v['offset'] for v in p['pointers']]
    if len(set(offsets))!=len(offsets) or any(v not in POINTERS for v in offsets): raise ValueError('pointer map')
    mode=struct.unpack_from('<i',args,104)[0]
    used=POINTERS if mode>=2 else POINTERS[:-1]
    required={o for o in used if struct.unpack_from('<Q',args,o)[0]}
    if not required.issubset(set(offsets)) or any(not struct.unpack_from('<Q',args,o)[0] for o in offsets): raise ValueError('all live pointers must be relocated')
    for b in p['buffers']:
        if sha_slice(b['file'],b['offset'],b['bytes'])!=b['sha256']: raise ValueError('native buffer identity')
    # Every writable region/intermediate must be independently captured from the
    # original host (not computed by a CPU approximation or candidate).
    if not p['references'] or p.get('reference_provenance')!='unchanged-original-host': raise ValueError('independent native references required')
    for r in p['references']:
        if not 0<=r['buffer']<len(p['buffers']) or r['offset']<0 or r['offset']+r['bytes']>p['buffers'][r['buffer']]['bytes']:raise ValueError('reference range')
        if sha(r['file'])!=r['sha256'] or resolve_path(r['file']).stat().st_size!=r['bytes']: raise ValueError('reference identity')
    for ptr in p['pointers']:
        if not 0<=ptr['buffer']<len(p['buffers']) or not 0<=ptr['add']<p['buffers'][ptr['buffer']]['bytes']: raise ValueError('pointer range')
    from .producer import verify
    verify(p.get('producer_certificate',{}),p)
    validate_alias_map(p,args)
    M,K=struct.unpack_from('<2i',args,48);R=struct.unpack_from('<i',args,96)[0];E=struct.unpack_from('<i',args,108)[0]
    if min(M,K,R)<=0 or max(M,K)>5120 or R!=8:raise ValueError('unsupported native model shape')
    if p['role']=='dsl_gemv_fold':needs={0:M*K*2,8:K*2,16:M*2,64:R*K*2,72:K*2,80:K*2,88:K*2}
    else:
        if E!=128:raise ValueError('native expert identity')
        stride=struct.unpack_from('<q',args,40)[0]
        expected_stride=(2 if p['role']=='dsl_moe_up' else 1)*M*K
        if stride!=expected_stride:raise ValueError('native expert weight stride')
        needs={0:E*stride*2,8:(R if p['role']=='dsl_moe_down' else 1)*K*2,16:R*M*2,24:R*4,32:R*4,112:E*2}
    for ptr in p['pointers']:
        if p['buffers'][ptr['buffer']]['bytes']-ptr['add']<needs.get(ptr['offset'],0):raise ValueError('native buffer smaller than actual role access')
    mode=struct.unpack_from('<i',args,104)[0]
    if p['role']=='dsl_gemv_fold' and (mode!=1 or p.get('layer') not in (1,24,47) or not p.get('dependency_edges',{}).get('tasks')):
        raise ValueError('G11 requires current fold chain, representative layer, actual predecessor edges')
    return args


def config(pack,build,variant,output,seconds=120,warmup=16,samples=5,seed=91011,diagnostic=False):
    args=validate_pack(pack)
    M,K,rpt,tpe=struct.unpack_from('<4i',args,48)
    expected=(768,2048,32,24,192) if variant in ('g09','g10_up') else (2048,768,64,32,256) if variant=='g10_down' else (5120,2048,20,0,256)
    if (M,K,rpt,pack['count'])!=(expected[0],expected[1],expected[2],expected[4]) or pack['first']!=0:raise ValueError('requires complete fixed native row-owner window')
    if variant=='g11_probe' and pack['role']!='dsl_gemv_fold':raise ValueError('G11 fold role')
    if variant!='g11_probe' and pack['role']!=('dsl_moe_down' if variant=='g10_down' else 'dsl_moe_up'):raise ValueError('candidate role mismatch')
    if diagnostic:
        from .diagnostics import PHASES
        if variant not in ('g09','g10_up','g10_down'):raise ValueError('sparse diagnostic variant')
        names=tuple(variant+'_'+mode for mode in ('control','probe','empty'));samples=pack['count']*16*len(PHASES)
    elif variant=='g11_probe':
        names=('g11_control','g11_probe','g11_empty'); samples=pack['count']*16
        if not pack.get('matching_trace'):raise ValueError('G11 execution requires an active matching G12 trace proof; no isolated full-chain price')
        from examples.experiment_program.validation_trace import bind_g11
        trace_binding=bind_g11(pack,pack['matching_trace'])
    elif variant in ('g09','g10_up','g10_down'): names=('control',variant)
    else: raise ValueError('candidate')
    if samples<1 or warmup<0 or seconds<=0: raise ValueError('population')
    if len(names)==2 and samples!=5: raise ValueError('exactly five complete pairs required')
    storage=output_reservation(pack,len(names),samples,output,args)
    q=lambda v:json.dumps(str(v)); build=resolve_path(build)
    fields=[q(resolve_path(output)),q(resolve_path(pack['arguments']['file'])),q('row_'+pack['role']),str(len(names)),str(pack['first']),str(pack['count']),str(warmup),str(samples),str(seconds),str(len(pack['buffers'])),str(len(pack['pointers'])),str(len(pack['references']))]
    lines=[' '.join(fields)]+[q(build/(n+'.hsaco')) for n in names]
    lines += [f'{q(resolve_path(b["file"]))} {b["offset"]} {b["bytes"]}' for b in pack['buffers']]
    lines += [f'{p["offset"]} {p["buffer"]} {p["add"]}' for p in pack['pointers']]
    lines += [f'{r["buffer"]} {r["offset"]} {q(resolve_path(r["file"]))} {r["bytes"]} {int(reference_output_only(pack,r,args))}' for r in pack['references']]
    from .protocol import invocation_orders
    orders=invocation_orders(len(names),samples,seed)
    lines+=[' '.join(str(a) for order in orders for a in order)]
    return '\n'.join(lines)+'\n',dict(arms=names,orders=orders,seed=seed,samples=samples,warmup=warmup,position=pack['position'],layer=pack.get('layer'),gpu_executed=False,storage=storage,trace_binding=trace_binding if variant=='g11_probe' else None)


def validate_build(build,variant):
    b=resolve_path(build);identity=json.loads((b/'identity.json').read_text())
    names=('g11_control',variant,'g11_empty') if variant=='g11_probe' else ('control',variant)
    for name in [n+'.hsaco' for n in names]+['replay']:
        if sha(b/name)!=identity['hashes'][name]:raise ValueError('sealed native executable changed')
    if variant=='g09' and not identity.get('g09_resource_admitted'):raise ValueError('G09 static resource rejection')
    return identity


def reference_output_only(pack,reference,args=None):
    """Poison precisely when the reference slice overlaps no original live read."""
    if args is None:
        reads={0,64,72,88} if pack['role']=='dsl_gemv_fold' else {0,8,24,32,112}
        return not any(p['buffer']==reference['buffer'] and p['offset'] in reads for p in pack['pointers'])
    M,K=struct.unpack_from('<2i',args,48);R=struct.unpack_from('<i',args,96)[0];mode=struct.unpack_from('<i',args,104)[0];E=struct.unpack_from('<i',args,108)[0]
    if pack['role']=='dsl_gemv_fold':reads={0:M*K*2,64:R*K*2,72:K*2,88:K*2}
    else:
        reads={0:E*struct.unpack_from('<q',args,40)[0]*2,8:(R if pack['role']=='dsl_moe_down' else 1)*K*2}
        if mode>=2:reads[112]=E*2
        else:
            reads[24]=R*4
            if pack['role']=='dsl_moe_down':reads[32]=R*4
    begin=reference['offset'];end=begin+reference['bytes']
    return not any(p['buffer']==reference['buffer'] and p['offset'] in reads and max(begin,p['add'])<min(end,p['add']+reads[p['offset']]) for p in pack['pointers'])


def validate_alias_map(pack,args):
    """Reject snapshot relocation that splits one original native allocation."""
    bases={}
    for pointer in pack['pointers']:
        original=struct.unpack_from('<Q',args,pointer['offset'])[0]
        base=original-pointer['add'];index=pointer['buffer']
        if index in bases and bases[index]!=base:raise ValueError('noncanonical native alias mapping')
        bases[index]=base
    indices=sorted(bases)
    for i,index in enumerate(indices):
        for other in indices[i+1:]:
            if max(bases[index],bases[other])<min(bases[index]+pack['buffers'][index]['bytes'],bases[other]+pack['buffers'][other]['bytes']):raise ValueError('overlapping native buffers must retain one canonical allocation')
