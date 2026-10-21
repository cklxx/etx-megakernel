"""G04 native routing observation. No timing trial or performance verdict."""
from collections import Counter
import math
import struct

from ..protocol import require,capture_coverage


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
