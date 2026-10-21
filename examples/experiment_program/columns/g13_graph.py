"""Common G13 ETX executable: fixed native task bodies, explicit legal queues.

Both request states and the runtime combined-event switch are device argument
buffers. Host order changes rewrite only static task order, never worker pins.
"""
from etx.frontends import hip_link
from etx.ir import Graph, Resource
from examples.experiment_program.columns.protocol import queue_reachable
SOURCE='examples/experiment_program/columns/g13_tiles.hip'

def build():
    g=Graph('g13_unshared_expert_layer')
    resource=Resource(threads=1024,vgpr=128,agpr=0,lds_bytes=50280)
    for r in range(2):
        g.tensor(f'r{r}',(512,),role='runtime',bytes_per_elem=1)
        g.etensor(f'ready{r}',(1,),wait_count=1)
    for r in range(2): g.etensor(f'up{r}',(8,),wait_count=24)
    for r in range(2): g.etensor(f'down{r}',(1,),wait_count=256)
    for r in range(2):
        g.call_device(f'ready_r{r}',(1,1),hip_link(SOURCE,'g13_ready'),resource=resource,args=[f'r{r}'],
                      consts=(r,),domain_map=f'xy->({r})',worker_map='xy->(37)',
                      out_edges={f'ready{r}':'xy->(0)'},duration_us=0.01)
    for phase,count,symbol in [('up',24,'g13_up'),('down',32,'g13_down')]:
        for r in range(2):
            g.call_device(f'{phase}_r{r}',(8,count),hip_link(SOURCE,symbol),resource=resource,args=[f'r{r}'],
                          consts=(r,3-r),domain_map='xw->x',worker_map='xw->(w)',
                          in_edges={f'ready{r}':'xw->(0)'} if phase=='up' else {f'up{r}':'xw->x'},
                          out_edges={f'up{r}':'xw->x'} if phase=='up' else {f'down{r}':'xw->(0)'},duration_us=20.0)
    return g

def bindings(**overrides):
    if overrides:raise ValueError('G13 native task dimensions are frozen')
    return {}

def runtime(bindings,rng=None): return {}

def queue_variants():
    tasks={};queues={(x,w):[] for x in range(8) for w in range(38)}
    for r in range(2):
        name=f'ready{r}';tasks[name]={'request':r,'phase':'ready','xcd':r,'worker':37};queues[r,37].append(name)
    for phase,count in [('up',24),('down',32)]:
        for r in range(2):
            for x in range(8):
                for w in range(count):
                    name=f'{phase}{r}:{x}:{w}';tasks[name]={'request':r,'phase':phase,'xcd':x,'worker':w};queues[x,w].append(name)
    own={}
    combined={}
    for name,t in tasks.items():
        r,x=t['request'],t['xcd']
        own[name]=[] if t['phase']=='ready' else ([f'ready{r}'] if t['phase']=='up' else [f'up{r}:{x}:{w}' for w in range(24)])
        combined[name]=own[name]+([f'up{1-r}:{x}:{w}' for w in range(24)] if t['phase']=='down' else [])
    independent={key:sorted(q,key=lambda name:(tasks[name]['request']*3+{'ready':0,'up':1,'down':2}[tasks[name]['phase']])) for key,q in queues.items()}
    # Each exogenous producer has its own unused physical worker; arbitrary
    # arrival order therefore adds no producer-to-producer dependencies.
    for r in range(2):independent[r,37]=queues[r,37]
    result={'same_task_bodies':True,'same_worker_assignment':True,'tasks':len(tasks),'worker_geometry':[8,38],'independent_ready_producers':True,'nonmonotonic_arrivals_replayable':True}
    for name,q,deps in [('combined_both_up_first',queues,combined),('independent_request_order',independent,own),('combined_request_order',independent,combined),('independent_both_up_first',queues,own)]:
        reachable=queue_reachable(list(q.values()),deps)
        result[name]={'reachable':reachable,'admitted':reachable,'queues':{f'{x}:{w}':rows for (x,w),rows in q.items()},'dependencies':deps}
    return result

def verify_compiled_queues(plan):
    """Compare the exact emitted task/worker queues with the finite proof model."""
    from examples.experiment_program.columns.protocol import require
    proof=queue_variants();types={v['id']:name for name,v in plan['types'].items()}
    names={}
    for d in plan['descs']:
        name=types[d['type']];phase,request=name.split('_r');r=int(request)
        require(d['mode']=='static' and d['worker']//38==d['domain'],'unmatched ETX physical worker-domain assignment')
        if phase=='ready':
            n=f'ready{r}';require(d['domain']==r and d['worker']==r*38+37,'unmatched readiness producer')
        else:
            x,w=d['coord'];n=f'{phase}{r}:{x}:{w}'
            require(d['domain']==x and d['worker']%38==w,'native G13 task row/worker pin changed')
        require(n not in names.values(),'duplicate generated native task');names[d['id']]=n
    require(set(names.values())==set(proof['combined_both_up_first']['dependencies']),'missing/extra generated native tasks')
    queues={f'{int(key.split(":")[1])//38}:{int(key.split(":")[1])%38}':[names[t] for t in ids]
            for key,ids in plan['static_queues'].items()}
    expected=proof['combined_both_up_first']['queues']
    require(all(queues.get(key,[])==value for key,value in expected.items()),'emitted queues differ from reachability proof')
    proof['generated_tasks']=len(names);proof['compiled_queue_match']=True
    return proof
