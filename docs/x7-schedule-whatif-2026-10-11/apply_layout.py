"""Future build hook: call apply(plan) after compile_graph, before any code emission.

Off by default; this file does not change the production compiler or launch a GPU.
"""
import gzip,json
from pathlib import Path
from etx.passes.plan import verify_plan

HERE=Path(__file__).resolve().parent

def apply(plan):
    base=json.loads(gzip.decompress((HERE/'original-plan.json.gz').read_bytes()))
    layout=json.loads((HERE/'oracle-best-layout.json').read_text())
    names={v['id']:k for k,v in base['types'].items()}
    assert plan.n_domains==8 and plan.workers_per_domain==38
    assert len(plan.tasks)==len(base['descs'])==len(layout['workers'])
    for task,desc,worker in zip(plan.tasks,base['descs'],layout['workers']):
        assert (task.id,task.grid,list(task.coord),task.device,task.domain)==(desc['id'],names[desc['type']],desc['coord'],desc['device'],desc['domain'])
        assert task.mode=='static' and worker//38==task.domain
    # Every original scope, event coordinate, producer set and consumer set stays intact.
    for ev in base['events']:
        assert plan.events[ev['name']].scope.name==ev['scope']
    for task,worker in zip(plan.tasks,layout['workers']):task.worker=worker
    plan.static_queues={tuple(map(int,k.split(':'))):ids.copy() for k,ids in layout['static_queues'].items()}
    errors=verify_plan(plan)
    if errors:raise ValueError(errors)
    plan.say('X7 frozen single-trace oracle worker layout; no GPU performance certificate')
