"""Fail-closed local admission and retained native-input validation."""
import json
from pathlib import Path
import re
import statistics
from etx.models.vllm_llm.attention_early_merge.evidence import frame, sha

from examples.experiment_program.protocol import resolve_path

EXPORTS={'G06':'dsl_attnprep','G07':'attn_reduce','G08':'attn_mfma16_g8'}
POSITIONS=(0,1023,1024,1040)

def require(ok,why):
    if not ok: raise ValueError(why)

def address_contract(path,sizes,position):
    import struct
    path=resolve_path(path); args=(path/'dsl_attnprep.args').read_bytes()
    require(len(args)==112,'prep argument bytes')
    nh,nkv,hd=struct.unpack_from('3i',args,64);bs,x=struct.unpack_from('2i',args,80)
    require((nh,nkv,hd,bs,x)==(32,4,128,16,8),'unsupported prep/cache shape')
    pos=int.from_bytes((path/'positions.before').read_bytes(),'little',signed=True)
    slot=int.from_bytes((path/'slot_mapping.before').read_bytes(),'little',signed=True)
    row_bytes=2*hd
    block_bytes=2*nkv*hd*bs
    require(sizes['kc']==sizes['vc'] and sizes['kc']%block_bytes==0,'cache layout bounds')
    blocks=sizes['kc']//block_bytes
    require(pos==position and 0<=pos<sizes['cos_sin']//row_bytes,'position outside immutable RoPE cache')
    require(0<=slot<blocks*bs,'slot outside admitted cache')
    return {'position':pos,'slot':slot,'cache_blocks':blocks}

def validate_frame(path,imports,sealed):
    path=resolve_path(path)
    require(sealed is not None,'missing original capture seal')
    observed=frame(path,imports)
    require(observed==sealed,'stale/truncated capture')
    rows=(path/'frame.txt').read_text().splitlines()
    spans=[]
    for row in rows[1:16]:
        name,address,size=row.split(); address=int(address); size=int(size)
        require(all(address+size<=a or a+s<=address for a,s in spans),'overlapping capture buffers')
        spans.append((address,size))
    # These are immutable metadata in this intervention, not arbitrary aliased
    # user pointers. The capture's bounded pointer resolver validates all args.
    for n in ('positions','slot_mapping','block_tables','seq_lens','qw','kw','cos_sin'):
        require((path/(n+'.before')).read_bytes()==(path/(n+'.after')).read_bytes(),'immutable input changed '+n)
    sizes={row.split()[0]:int(row.split()[2]) for row in rows[1:16]}
    address_contract(path,sizes,observed['position'])
    return observed

def validate_captures(root,imports,layers=(0,)):
    root=resolve_path(root); seal=json.loads((root/'capture.json').read_text())
    require(seal.get('schema')=='etx.c6.capture.v1' and seal.get('synthetic') is False,'unsealed/synthetic captures')
    require(seal['imports']==imports,'original capture import identity')
    require(len(set(layers))==len(layers) and all(0<=l<48 for l in layers),'duplicate/invalid layer')
    return {f'p{p}/L{l}':validate_frame(root/f'p{p}/L{l}',imports,seal['frames'].get(f'p{p}/L{l}')) for p in POSITIONS for l in layers}

def native_events(assembly, source, markers, source_file=None):
    """Source-mapped native READ issue order, never source-order inference.

    .loc inline annotations include the call-site line for coherent accessors.
    An unidentifiable marker yields no admission. Only actual load instructions
    are recorded, and the floating-point anchor for G07 is separately mapped.
    """
    lines=source.splitlines(); targets=[]
    for marker in markers:
        found=[i+1 for i,line in enumerate(lines) if marker in line]
        require(bool(found),'missing native marker '+marker)
        targets.append(found[0])
    events={'target':[],'anchor':[]}; locations=set()
    for ordinal,line in enumerate(assembly.splitlines()):
        if re.match(r'\s*\.loc\s',line):
            match=re.match(r'\s*\.loc\s+\d+\s+(\d+)',line)
            if source_file:
                locations=set(map(int,re.findall(re.escape(source_file)+r':(\d+):',line)))
            else:
                locations={int(match[1])} if match else set()
                locations.update(map(int,re.findall(r'kernel\.hip:(\d+):',line)))
        read=bool(re.match(r'\s*(?:global|flat|buffer|s)_load_',line))
        for name,target in zip(('target','anchor'),targets):
            if target in locations and (read or (name=='anchor' and re.search(r'\bv_max(?:3)?_f32(?:_e(?:32|64))?\b',line))):
                events[name].append(ordinal)
    return events

def native_admission(control,candidate,control_resources,candidate_resources):
    require(all(k in c and c[k] for c in (control,candidate) for k in ('target','anchor')),'missing source-mapped native order')
    require(len(control_resources)==len(candidate_resources)==1,'resource entry coverage')
    def ok(r): return r['vgpr']<=128 and r['agpr']==0 and r['lds_bytes']<=32768 and not any(r[k] for k in ('scratch_bytes_lane','vgpr_spills','sgpr_spills'))
    if not all(ok(r) for r in control_resources+candidate_resources):
        return {'status':'CLOSED_RESOURCE','timing_eligible':False}
    # Must cross the declared actual issue anchor; changing instruction offsets,
    # source line numbers or register names alone can never qualify.
    before=lambda x:min(x['target'])<min(x['anchor'])
    if before(control) or not before(candidate):
        return {'status':'CLOSED_NO_NATIVE_REORDER','timing_eligible':False}
    return {'status':'ADMITTED_OPERATOR_ONLY','timing_eligible':True,'model_resource_identity':'PENDING_G12'}

def decide(pairs):
    if len(pairs)!=5 or any(not isinstance(x,(int,float)) or not 0<x<1e12 for pair in pairs for x in pair):
        return {'status':'INVALID','reason':'five complete finite positive pairs required'}
    gains=[a-b for a,b in pairs]; median=statistics.median(gains); spread=max(gains)-min(gains)
    passed=all(x>0 for x in gains) and median>2*spread and median>=625 and median>=.02*statistics.median(a for a,b in pairs)
    return {'status':'MICRO_WINNER' if passed else 'MICRO_MISS','median_ns':median,'spread_ns':spread,'gains_ns':gains}
