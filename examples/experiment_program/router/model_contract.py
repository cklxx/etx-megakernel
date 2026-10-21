"""G01 emitted-plan reset/publication coverage and static-queue liveness checks."""
from collections import Counter
from ..protocol import require


def validate(plan):
    require((plan['devices'],plan['domains'],plan['workers_per_domain'],plan['threads'])==(1,8,38,1024),'exact retained physical worker geometry')
    types=plan['types'];by_id={v['id']:k for k,v in types.items()}
    descs=plan['descs'];tasks={name:[d['id'] for d in descs if by_id[d['type']]==name] for name in types}
    require('g01_state' in plan['args'],'raw lifecycle state absent')
    events=plan['ev_inst'];incoming=plan['task_in']
    event_for_producers={frozenset(producers):i for i,(_,_,producers) in enumerate(events)}
    for layer in range(48):
        reset=tasks[f'g01_reset_{layer}'];router=tasks[f'router_{layer}']
        require(len(reset)==1 and len(router)==256,'reset/unchanged router population')
        require(not any(name in types for name in (f'topk_{layer}',f'topk_{layer}_x')),'standalone selector remains')
        ready=event_for_producers[frozenset(reset)]
        require(all(ready in incoming[t] for t in router),'router may precede reset')
        complete=event_for_producers[frozenset(router)]
        require(all(complete in incoming[t] for t in tasks[f'moe_up_{layer}']),'expert may observe incomplete eight-slab publication')
        placement=Counter(d['domain'] for d in descs if d['id'] in router)
        require(placement=={x:32 for x in range(8)},'router topology changed')
    # A persistent worker cannot skip a blocked queue head. Simulate exactly
    # those queue heads and actual event producer populations, not an abstract
    # grid topological sort that misses a queue-induced deadlock.
    queues=[list(q) for q in plan['static_queues'].values()]
    require(sum(map(len,queues))==len(descs),'static task coverage')
    done=set();ready=set();cursor=[0]*len(queues)
    while len(done)<len(descs):
        advanced=False
        for i,q in enumerate(queues):
            if cursor[i]==len(q):continue
            t=q[cursor[i]]
            if set(incoming[t])<=ready:
                require(t not in done,'duplicate queued task')
                done.add(t);cursor[i]+=1;advanced=True
        ready={i for i,(_,_,producers) in enumerate(events) if set(producers)<=done}
        require(advanced,'static queue deadlock before a required producer')
    return dict(record='G01',layers=48,reset_tasks=48,router_tasks=12288,
                all_router_completion_before_expert=True,static_tasks=len(done),
                queue_liveness='PASS',scope='local emitted schedule and native protocol proof; not device correctness or performance')
