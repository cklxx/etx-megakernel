"""G03 common native-placement ETX NB1/NB2 readiness experiment."""
import json,os
from pathlib import Path
from etx.ir import Graph,Resource
from etx.frontends import hip_link
from examples.experiment_program.columns.protocol import require,sha256,queue_reachable
from examples.experiment_program.protocol import resolve_path
REFERENCE=Path(os.environ.get('ETX_G03_NATIVE_PLAN',str(Path.home()/'code/etx/results_2026-10-01/box_final/b14_b2/plan.json')))

def native_pins():
    p=json.loads(resolve_path(REFERENCE).read_text());t=p['types']['qkv_11']['id'];rows=[d for d in p['descs'] if d['type']==t]
    require(len(rows)==256 and p['workers_per_domain']==38,'native historical QKV plan differs')
    pins={d['coord'][0]:(d['domain'],d['worker']%38) for d in rows}
    for x in range(8):
        require(all(pins[x*32+w][0]==x for w in range(32)),'native QKV row/domain assignment changed')
        require(all(pins[x*32+w][1]==(pins[x*32][1]+w)%38 for w in range(32)),'native QKV worker pattern changed')
    return pins

def build():
    pins=native_pins();g=Graph('g03_native_dense_readiness');resource=Resource(threads=1024,vgpr=128,agpr=0,lds_bytes=50280)
    for r in range(2):g.tensor(f'r{r}',(512,),role='runtime',bytes_per_elem=1);g.etensor(f'ready{r}',(1,),wait_count=1)
    free={r:next(w for w in range(38) if w not in {pins[r*32+j][1] for j in range(32)}) for r in range(2)}
    for r in range(2):g.call_device(f'ready_r{r}',(1,),hip_link('examples/experiment_program/columns/g03_tiles.hip','g03_ready'),resource=resource,args=[f'r{r}'],consts=(r,),domain_map=f'x->({r})',worker_map=f'x->({free[r]})',out_edges={f'ready{r}':'x->(0)'},duration_us=.01)
    for r in range(2):
        for x in range(8):
            g.call_device(f'qkv_r{r}_x{x}',(32,),hip_link('examples/experiment_program/columns/g03_tiles.hip','g03_service'),resource=resource,args=[f'r{r}'],consts=(r,1-r,x),domain_map=f'w->({x})',worker_map=f'w->((w+{pins[x*32][1]})%38)',in_edges={f'ready{r}':'w->(0)'},duration_us=20.)
    return g

def bindings(**kw):
    if kw:raise ValueError('historical placement frozen')
    return {}
def runtime(bindings,rng=None):return {}

def verify(plan):
    pins=native_pins();types={v['id']:name for name,v in plan['types'].items()};deps={};services={};ready={}
    require(plan['workers_per_domain']==38,'G03 native resource/worker budget differs')
    for d in plan['descs']:
        name=types[d['type']]
        if name.startswith('ready'):
            r=int(name[-1]);ready[r]=d['id']
            require(d['domain']==r and d['worker']%38 not in {pins[r*32+j][1] for j in range(32)},'G03 independent producer uses a service worker')
        else:
            r=int(name.split('_r')[1].split('_')[0]);x=int(name.split('_x')[1]);block=x*32+d['coord'][0]
            require((d['domain'],d['worker']%38)==pins[block],'G03 row/native worker placement differs');services[r,block]=d['id']
    require(len(services)==512 and len(ready)==2,'G03 full native tasks absent')
    for tid in ready.values():deps[tid]=[]
    for (r,b),tid in services.items():deps[tid]=list(ready.values())+([services[0,b]] if r else [])
    require(queue_reachable(list(plan['static_queues'].values()),deps),'G03 combined ready/owner waits can producer-block')
    return dict(actual_tasks=514,native_plan_sha256=sha256(REFERENCE),native_worker_pins=True,all_layouts_reachable=True,independent_ready_producers=True,nonmonotonic_arrivals_replayable=True)
