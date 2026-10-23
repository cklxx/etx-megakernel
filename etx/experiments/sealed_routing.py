"""G04 capture evidence shared by the sealed lifecycle and legacy providers.

No execution or optimization admission: complete routing observations remain
capture-only evidence. Consumer-specific dependency predicates stay separate.
"""
from collections import Counter
import json
import math
from pathlib import Path
import re
import struct

from .sealed_protocol import (require, capture_coverage, verify_files, sha,
                              resolve_path, relocation_seal)

ROLES={'source','plan','executable','compiler','math_flags','resources','weights','inputs','prefixes','goldens','golden_producers'}


def config(path):
    c=json.loads(resolve_path(path).read_text())
    require(c.get('batch') in (2,4),'capture B2/B4')
    require(set(c.get('roles',{}))==ROLES,'complete capture identity roles required')
    for files in c['roles'].values():verify_files(files)
    require(isinstance(c.get('command'),list) and all(isinstance(x,str) for x in c['command']) and c['command'],'native capture argv')
    require(resolve_path(c['command'][0]) in [resolve_path(p) for p in c['roles']['executable']],'command executable not bound')
    c['command']=[str(resolve_path(s)) if Path(s).is_absolute() else s for s in c['command']]
    all_seals={p:h for files in c['roles'].values() for p,h in files.items()}
    require(all(all_seals.get(p)==h for p,h in relocation_seal().items()),'explicit staging map must be in capture source seal')
    flags='\n'.join(resolve_path(p).read_text() for p in c['roles']['math_flags'])
    require('ETX_PROGRAM_ROUTE_CAPTURE=1' in flags,'capture flag must belong to actual build')
    require(0<c.get('deadline_seconds',0)<=900,'bounded capture deadline')
    return c


def gates(text,batch):
    require(f'G04 CAPTURE COMPLETE requests={batch} steps=32 routed_layers=26 replicas=8 slots=6 dense_layer=0' in text,'incomplete native capture')
    require(not re.search(r'MISMATCH|\[FAIL\]|REFUSED|no reference token',text),'capture correctness failure')
    rows=re.findall(r'^BATCHREQ r=(\d+) tokens=(\d+/\d+) layers=(\d+) warmup=(\d+/\d+)',text,re.M)
    require(len(rows)==batch and sorted(int(x[0]) for x in rows)==list(range(batch)),'missing/duplicate requests')
    require(all(x[1]=='32/32' and x[2]=='27' and x[3]=='32/32' for x in rows),'per-request warmup/greedy/hidden gate')


def decode_records(raw, *, batch):
    require(batch in (2,4) and len(raw)==batch,'B2/B4 request coverage')
    expected_size=32*26*8*64
    ids={};mix={};logical=[];replica_equal=True
    for r,data in enumerate(raw):
        require(len(data)==expected_size,'truncated/extra capture bytes')
        for t in range(32):
            for l in range(1,27):
                for x in range(8):
                    words=struct.unpack_from('<16I',data,((t*26+l-1)*8+x)*64)
                    require(words[0]==33+t and words[1]==1,'stale epoch or missing/duplicate native writer')
                    picks=words[2:8];weights=struct.unpack('<6f',struct.pack('<6I',*words[8:14]))
                    require(len(set(picks))==6 and all(e<64 for e in picks),'invalid native expert/slot identity')
                    require(all(math.isfinite(w) and w>=0 for w in weights),'invalid native mixing bytes')
                    ids[r,t,l,x]=picks;mix[r,t,l,x]=words[8:14]
                replica_equal &= all(ids[r,t,l,0]==ids[r,t,l,x] and mix[r,t,l,0]==mix[r,t,l,x] for x in range(1,8))
                # The actual routed slot s is consumed on XCD s. Never substitute
                # replica0 when another XCD's independent selector differs.
                logical.extend((r,t,l,s) for s in range(6))
    capture_coverage(logical,batch=batch)
    hist=Counter();shared=0;groups=0
    for t in range(32):
        for l in range(1,27):
            counts=Counter(ids[r,t,l,s][s] for r in range(batch) for s in range(6))
            hist.update(counts.values());shared+=sum(n-1 for n in counts.values());groups+=len(counts)
    return dict(requests=batch,steps=32,routed_layers=26,dense_layer=0,replicas=8,slots=6,
                replica_bytes_equal=replica_equal,effective_batch_histogram={str(k):v for k,v in sorted(hist.items())},
                duplicate_weight_request_potential=shared,expert_groups=groups,
                slot_ids=[dict(request=r,step=t,layer=l,slot=s,xcd=s,expert=ids[r,t,l,s][s],mix_bits=mix[r,t,l,s][s])
                          for r,t,l,s in logical],performance_claim=False,
                interpretation='native selected columns, not measured physical traffic or a grouped-kernel speedup')


def session_decision(stage,path,*,value):
    """Recompute the session's G04 receipt without changing its predicates."""
    path=Path(path)
    require(value.get('record')=='G04' and value.get('complete') is True,'capture incomplete')
    cmd=stage['command'];cfg=Path(cmd[cmd.index('--config')+1]);c=config(cfg)
    require(value['config_sha256']==sha(cfg) and value['roles']==c['roles'],'capture configuration identity')
    verify_files(value['raw_files'])
    log=path.parent/'capture.log';require(sha(log)==value['log_sha256'],'capture log changed')
    gates(log.read_text(),value['requests'])
    actual=decode_records([(path.parent/f'routes_r{i}.bin').read_bytes() for i in range(value['requests'])],batch=value['requests'])
    require(all(value[k]==v for k,v in actual.items()),'capture summary differs from full native records')
    return dict(record='G04',complete=True,admitted=False,decision_sha256=sha(path))
