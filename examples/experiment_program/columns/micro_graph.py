"""Native Fleet resident micrograph: original request slots and fixed queues."""
import os
from etx.ir import Graph,Resource
from etx.frontends import hip_link

def parameters():
    B=int(os.environ.get('ETX_COLUMN_BATCH','2'));kind=os.environ.get('ETX_COLUMN_KIND','qkv');xcd=int(os.environ.get('ETX_COLUMN_NATIVE_XCD','6'))
    if B not in (2,4) or kind not in ('qkv','shared','group') or xcd not in (6,7):raise ValueError('unregistered native Fleet micro geometry')
    return B,kind,xcd

def build():
    B,kind,xcd=parameters();g=Graph(f'columns_native_{kind}_B{B}')
    g.tensor('state',(16384,),role='runtime',bytes_per_elem=1)
    resource=Resource(threads=256,vgpr=256,agpr=0,lds_bytes=B*2048*4)
    for request in range(B):
        g.call_device(f'ready_r{request}',(1,),hip_link('examples/experiment_program/columns/micro_tiles.hip','column_micro_ready'),resource=resource,args=['state'],consts=(request,B,{'qkv':0,'shared':1,'group':2}[kind]),domain_map=f'x->({request})',worker_map='x->(37)',duration_us=.01)
    for request in range(B):
        g.call_device(f'service_r{request}',(8 if kind=='qkv' else 6 if kind=='group' else 1,37),hip_link('examples/experiment_program/columns/micro_tiles.hip','column_micro_service'),resource=resource,args=['state'],consts=(request,B,{'qkv':0,'shared':1,'group':2}[kind]),domain_map='xw->x' if kind!='shared' else f'xw->({xcd})',worker_map=f'xw->({38*(request%2)}+w)',duration_us=20.)
    return g

def bindings(**overrides):
    if overrides:raise ValueError('native geometry is frozen')
    return {}
def runtime(bindings,rng=None):return {}

def verify_queues(plan):
    from .protocol import require,queue_reachable
    B,kind,xcd=parameters();require(plan['workers_per_domain']==75,'native persistent worker budget changed');types={value['id']:name for name,value in plan['types'].items()}
    rows={};deps={};queues={}
    for d in plan['descs']:
        name=types[d['type']];request=int(name.split('_r')[1]);task=d['id']
        require(d['mode']=='static','native columns must use fixed queues')
        if name.startswith('ready'):
            require(d['worker']%plan['workers_per_domain']==37 and d['domain']==request,'ready producer pin changed')
            deps[task]=[]
        else:
            x,w=d['coord'][:2];domain=x if kind!='shared' else xcd
            require(d['domain']==domain and d['worker']%plan['workers_per_domain']==38*(request%2)+w,'native request worker/XCD pin changed')
            # Candidate waits only on ready producers and a same-worker owner
            # from a strictly earlier request. This conservative superset covers
            # every possible stable ID group without assuming any routing IDs.
            deps[task]=[r['id'] for r in plan['descs'] if types[r['type']].startswith('ready')]
            deps[task]+=[r['id'] for r in plan['descs'] if types[r['type']].startswith('service') and int(types[r['type']].split('_r')[1])<request and r['coord'][1]==w]
        rows[task]=d
    for worker,q in plan['static_queues'].items():queues[worker]=[n for n in q if n>=0]
    require(queue_reachable(list(queues.values()),deps),'candidate stable-owner/readiness queue can producer-block')
    return dict(native_worker_pins=True,stable_owner_all_possible_groups_reachable=True,independent_ready_producers=True,nonmonotonic_arrivals_replayable=True,actual_tasks=len(rows),workers_per_domain=plan['workers_per_domain'],kind=kind,batch=B)


def freeze_native_budget(plan):
    """Emit exact native slots for compiler checks, with no residency waiver."""
    from etx.passes import p5_queues,p6_memory,p7_prefetch
    B,_,_=parameters();estimated=plan.wg_per_cu
    plan.workers_per_domain=75;plan.wg_per_cu=2
    p5_queues.run(plan);p6_memory.run(plan);p7_prefetch.run(plan)
    plan.say(f'NATIVE_FIXED_BUDGET: required=2 blocks/CU, declared-estimate={estimated}; no execute without exact target capacity gate')
    return estimated
