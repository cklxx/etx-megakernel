#!/usr/bin/env python3
import copy,csv,gzip,json
from collections import defaultdict
from pathlib import Path
import search as s
from etx.sim.xcd import Cost,Hw,ReplayTiming,simulate
O=Path(__file__).resolve().parent;p=json.loads(gzip.decompress((O/'original-plan.json.gz').read_bytes()));raw=s.load_raw()
tr,basecp=s.cp_account(p,raw);i=json.loads((O/'inputs.json').read_text());space=json.loads((O/'finalist-space.json').read_text())
names={v['id']:k for k,v in p['types'].items()};groups=defaultdict(list)
for d in p['descs']:groups[d['type']].append(d['id'])
origin=min(r[5] for r in raw);oldlatest=[max(raw[z][7] for z in ev[2]) for ev in p['ev_inst']]
baseperiod=(max(raw[t][7] for t,d in enumerate(p['descs']) if s.stem(names[d['type']])=='moe_down')-max(raw[t][7] for t,d in enumerate(p['descs']) if s.stem(names[d['type']])=='in_norm'))*.01/48
profiles=defaultdict(lambda: [[],[]])
for t,d in enumerate(p['descs']):
 key=(s.stem(names[d['type']]),tuple(d['coord']))
 profiles[key][0].append(i['body_us'][t]);profiles[key][1].append(i['ready_after_us'][t])
import statistics
profiles={k:[statistics.median(x) for x in v] for k,v in profiles.items()}
frozen=json.loads((O/'oracle-best-layout.json').read_text())
assert frozen['model']=='additive'
results=[];best=None
for model in space['models']:
 local=i['ready_after_us'].copy();notif=None
 if model!='additive':
  floor=.52 if model=='branch052' else 0.;notif=[]
  for t,r in enumerate(raw):
   dep=max((oldlatest[e] for e in p['task_in'][t]),default=origin)
   if not p['task_in'][t]:notif.append(0.)
   elif r[5]<=dep:
    local[t]=min(floor,local[t]);notif.append(i['ready_after_us'][t]-local[t])
   else:notif.append(min(1.,(r[5]-dep)*.01))
 timing=ReplayTiming(i['queue_after_us'],local,{int(k):v for k,v in i['initial_taken_us'].items()},notif)
 for label in space['layouts']:
  c=copy.deepcopy(p);qs={f'0:{w}':[] for w in range(304)};avail=timing.initial_taken_us.copy();done=[0.]*len(raw)
  for ty,ids in groups.items():
   by=defaultdict(list)
   for t in ids:by[c['descs'][t]['domain']].append(t)
   for d,ts in by.items():
    ts=sorted(ts,key=lambda t:(-sum(profiles[(s.stem(names[ty]),tuple(c['descs'][t]['coord']))]),t))
    for t in ts:
     dep=max((done[z] for e in c['task_in'][t] for z in c['ev_inst'][e][2]),default=0.)+(notif[t] if notif else 0.)
     w=min(range(d*38,(d+1)*38),key=lambda w:(avail.get(w,0),w))
     done[t]=max(dep,avail.get(w,0))+local[t]+i['body_us'][t];avail[w]=done[t]+i['queue_after_us'][t]
     c['descs'][t]['worker']=w;qs[f'0:{w}'].append(t)
  c['static_queues']=qs
  if label=='frozen_additive_oracle':
   for t,d in enumerate(c['descs']):d['worker']=frozen['workers'][t]
   c['static_queues']=frozen['static_queues']
  s.validate(c,p)
  rows,stats=simulate(c,lambda _,d:Cost(chain_us=i['body_us'][d['id']]),Hw(handoff_us={'DOMAIN':0.,'DEVICE':0.}),replay=timing)
  assert stats['unfinished']==0
  ctr,cp=s.cp_account(c,rows,tr.deps);latest=[max(rows[z][7] for z in ev[2]) for ev in c['ev_inst']]
  assert all((r[7]-r[6])==raw[t][7]-raw[t][6] for t,r in enumerate(rows))
  assert all(rows[t][6]>=max((latest[e] for e in ins),default=0) for t,ins in enumerate(c['task_in']))
  period=(max(rows[t][7] for t,d in enumerate(c['descs']) if s.stem(names[d['type']])=='moe_down')-max(rows[t][7] for t,d in enumerate(c['descs']) if s.stem(names[d['type']])=='in_norm'))*.01/48
  r=dict(id=label,model=model,legal=True,clairvoyant=label=='frozen_additive_oracle',period_us=period,period_delta_us=period-baseperiod,span_us=ctr.span,step_delta_us=ctr.span-tr.span,wall_ms_est=4.180+(ctr.span-tr.span)/1000,cp=cp,queue_cost_sum_us=sum(timing.queue_after_us),local_cost_sum_us=sum(local),notification_cost_sum_us=sum(notif or []))
  results.append(r);s.dump(O/'finalist-results.json',results)
  with gzip.open(O/f'finalist-edges-{label}-{model}.csv.gz','wt',newline='') as f:
   w=csv.writer(f,lineterminator='\n');w.writerow(['task','event','gap_before_us','gap_after_us','delta_us'])
   for t in sorted({a['task'].id for a in s.tl.critical_path(tr)}|{a['task'].id for a in s.tl.critical_path(ctr)}):
    for e in p['task_in'][t]:
     b=(raw[t][6]-oldlatest[e])*.01;a=(rows[t][6]-latest[e])*.01
     w.writerow([t,p['ev_inst'][e][0],round(b,3),round(a,3),round(a-b,3)])
  if best is None or period<best[0]['period_us']:best=(r,c,rows)
  print(f'E {model} {label}: period_delta={r["period_delta_us"]:+.6f} step_delta={r["step_delta_us"]:+.3f}',flush=True)
r,c,rows=best
s.dump(O/'finalist-best-layout.json',dict(id=r['id'],model=r['model'],workers=[d['worker'] for d in c['descs']],domains=[d['domain'] for d in c['descs']],static_queues=c['static_queues']))
with gzip.GzipFile(filename=str(O/'finalist-best-plan.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(c,separators=(',',':')).encode())
s.write_rows(O/'finalist-best.bin.gz',rows)
print('BEST',json.dumps(r),flush=True)
