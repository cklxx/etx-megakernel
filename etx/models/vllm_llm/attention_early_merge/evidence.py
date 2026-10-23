"""Fail-closed C6 identities, capture coverage and exact integer decisions."""
from __future__ import annotations
import json
from pathlib import Path
import re
import statistics
from . import paths
from .build import sha
from etx.session.experiment import ns

POSITIONS=(0,1023,1024,1040)
BUFFERS=('qkv','qw','kw','cos_sin','positions','slot_mapping','kc','vc','block_tables',
         'seq_lens','kv_scale','exp_sums','max_logits','tmp_out','attn_out')
EXPORTS=('dsl_attnprep','attn_mfma16_g8','attn_reduce')


def require(ok,why):
    if not ok: raise ValueError(why)


def import_ir(path):
    """Ignore path headers and alpha-rename HIP's unused unit-ID marker only."""
    lines=Path(path).read_bytes().splitlines(keepends=True)
    module=re.compile(rb"; ModuleID = '[^\r\n]*'")
    source=re.compile(rb'source_filename = "(?:[^"\\\r\n]|\\[0-9A-Fa-f]{2})*"')
    counts=[0,0]; kept=[]
    for line in lines:
        body=line.rstrip(b'\r\n')
        match=next((i for i,p in enumerate((module,source)) if p.fullmatch(body)),None)
        if match is None: kept.append(line)
        else: counts[match]+=1
    require(counts==[1,1],'unexpected LLVM provenance '+str(path))
    data=b''.join(kept)
    if b'@__hip_cuid_' in data:
        ids=re.findall(rb'@__hip_cuid_[0-9a-f]+',data)
        require(len(ids)==2 and ids[0]==ids[1],'unexpected HIP unit-ID uses '+str(path))
        symbol=ids[0]
        definition=symbol+b' = addrspace(1) global i8 0'
        used=(b'@llvm.compiler.used = appending addrspace(1) global [1 x ptr] '
              b'[ptr addrspacecast (ptr addrspace(1) '+symbol+b' to ptr)], section "llvm.metadata"')
        bodies=[line.rstrip(b'\r\n') for line in kept]
        require(bodies.count(definition)==1 and bodies.count(used)==1,
                'unexpected HIP unit-ID marker '+str(path))
        data=data.replace(symbol,b'@__hip_cuid_C6_UNIT')
    return data


def same_imports(left,right,names):
    require(bool(names),'missing import IR coverage')
    for name in names:
        require(import_ir(Path(left)/name)==import_ir(Path(right)/name),
                'probe/model import IR mismatch '+name)


def identity(build,filename='identity.json',runnable=True):
    build=Path(build); record=json.loads((build/filename).read_text())
    require(record.get('hashes'), 'missing build hashes')
    for name,value in record['hashes'].items(): require(sha(build/name)==value,'stale build '+name)
    require(record.get('sources'),'missing source hashes')
    for name,value in record['sources'].items(): require(sha(paths.ROOT/name)==value,'stale source '+name)
    for name,value in record.get('imports',{}).items():
        require(sha(build/'imp'/name)==value,'stale import IR '+name)
    if filename=='identity.json':
        require(record['runnable'] or not runnable,'device binary is missing: local compilation only')
    if record.get('retained_footprint'):
        fp=Path(record['retained_footprint']['directory'])
        require(sha(fp/'offline.json')==record['retained_footprint']['identity_sha'],'stale retained footprint')
        require(record['retained_footprint'].get('headers'),'missing retained header hashes')
        for name,value in record['retained_footprint']['headers'].items():
            require(sha(fp/name)==value,'stale retained header '+name)
        identity(fp,'offline.json',runnable=False)
    return record


def frame(path,imports):
    path=Path(path); rows=(path/'frame.txt').read_text().splitlines(); hdr=rows[0].split()
    require(len(hdr)==4 and hdr[0]=='C6CAP1','capture schema')
    layer,position,count=map(int,hdr[1:]); require(0<=layer<48 and position in POSITIONS and count==15,'capture shape')
    require(len(rows)==19,'capture record count')
    hashes={'frame.txt':sha(path/'frame.txt')}; names=[]; spans=[]
    for row in rows[1:16]:
        name,address,size=row.split(); address=int(address); size=int(size)
        require(name in BUFFERS and name not in names and address>0 and 0<size<2**27,'capture buffer')
        names.append(name); spans.append((address,size))
        for suffix in ('before','after'):
            f=path/(name+'.'+suffix); require(f.stat().st_size==size,'capture byte count'); hashes[f.name]=sha(f)
    require(set(names)==set(BUFFERS),'capture buffers missing')
    for row,export in zip(rows[16:],EXPORTS):
        name,size=row.split(); size=int(size); require(name==export and size==imports[name]['arg_bytes'],'capture argument identity')
        f=path/(name+'.args'); require(f.stat().st_size==size,'capture argument bytes'); hashes[f.name]=sha(f)
        data=f.read_bytes()
        for param in imports[name]['params']:
            if param['ty'].startswith('ptr'):
                address=int.from_bytes(data[param['offset']:param['offset']+8],'little')
                require(address==0 or sum(a<=address<a+n for a,n in spans)==1,'unresolved capture pointer')
    require(int.from_bytes((path/'seq_lens.before').read_bytes(),'little')==position+1,'capture sequence length')
    return dict(layer=layer,position=position,hashes=hashes)


def seal_capture(directory,imports):
    directory=Path(directory)
    frames={f'p{p}/L{l}':frame(directory/f'p{p}/L{l}',imports) for p in POSITIONS for l in range(48)}
    record=dict(schema='etx.c6.capture.v1',frames=frames,synthetic=False,imports=imports)
    (directory/'capture.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def verify_capture(directory,imports):
    directory=Path(directory); record=json.loads((directory/'capture.json').read_text())
    expected={f'p{p}/L{l}' for p in POSITIONS for l in range(48)}
    require(set(record['frames'])==expected and record['imports']==imports,'capture coverage/import identity')
    for name,value in record['frames'].items(): require(frame(directory/name,imports)==value,'stale capture '+name)
    return record


def tokens(text,hybrid=False):
    rows=re.findall(r'^run \d+(?: request \d+)?: (\d+)/(\d+) generated tokens match HF greedy(.*)$',text,re.M)
    require(bool(rows) and all(a=='32' and b=='32' and not tail for a,b,tail in rows),'missing/failed 32/32 greedy token gate')
    require('CHECK-DSL OK:' in text and 'CHECK-DSL FAILED' not in text,'missing/failed CHECK-DSL')
    if hybrid: require('C6-HYBRID enabled=1 mode=combined purpose=correctness-only' in text,'candidate label lacks implementation proof')


def resource_gate(records):
    require(records,'missing native resources')
    for r in records:
        require(r['vgpr']<=128 and r['lds_bytes']<=29752 and r['static_blocks_per_cu']>=1,'C6 resource limit')


def parse_measurement(text,layer,position):
    rows=re.findall(r'^C6-RESULT arm=(control|candidate) layer=(\d+) position=(\d+) device_ms=(\d+\.\d+)$',text,re.M)
    require(f'C6-CHECK OK layer={layer} position={position} ' in text,'timing lacks its payload check')
    require(len(rows)==10 and all(int(l)==layer and int(p)==position for _,l,p,_ in rows),'incomplete paired probe')
    rounds=[]
    for r in range(5):
        pair=rows[r*2:r*2+2]; require([x[0] for x in pair]==(['control','candidate'] if r%2==0 else ['candidate','control']),'wrong interleaving')
        rounds.append({arm:ns(value) for arm,_,_,value in pair})
    return rounds


def decision(layers):
    require(len(layers)==48,'need all 48 layers, identical endpoints')
    gains=[sum(rows[r]['control']-rows[r]['candidate'] for rows in layers) for r in range(5)]
    median=statistics.median(gains); spread=max(gains)-min(gains)
    noise=min(gains)>0 and median>2*spread
    target=noise and median>=50000
    keep=noise and median>=30000
    return dict(basis='sum of 48 captured local segment device spans; NOT host-wall step speedup',
        gains_ns=gains,median_ns=median,spread_ns=spread,every_positive=min(gains)>0,
        combined_50us_target='PASS' if target else 'CLOSE',
        general_30us_rule='KEEP' if keep else 'DROP',
        target_next='full-megakernel integration and wall pairs permitted' if target else 'close registered target lever',
        net_full_step_target='PENDING runtime integration, reset and queue costs',wall_gain='UNPROVEN')
