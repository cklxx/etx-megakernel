#!/usr/bin/env python3
import copy,csv,gzip,json,statistics
from collections import defaultdict
from pathlib import Path
import search as s
from etx.sim.xcd import Cost,Hw,ReplayTiming,simulate
O=Path(__file__).resolve().parent
p=json.loads(gzip.decompress((O/'original-plan.json.gz').read_bytes()));raw=s.load_raw()
tr,basecp=s.cp_account(p,raw);inputs=json.loads((O/'inputs.json').read_text());space=json.loads((O/'branch-space.json').read_text())
names={v['id']:k for k,v in p['types'].items()};groups=defaultdict(list); profiles=defaultdict(lambda:[[],[],[]])
for t,d in enumerate(p['descs']):
 groups[d['type']].append(t);key=(s.stem(names[d['type']]),tuple(d['coord']))
 for k,field in enumerate(('body_us','ready_after_us','queue_after_us')):profiles[key][k].append(inputs[field][t])
profiles={k:[statistics.median(v) for v in a] for k,a in profiles.items()}
baseperiod=(max(raw[t][7] for t,d in enumerate(p['descs']) if s.stem(names[d['type']])=='moe_down')-max(raw[t][7] for t,d in enumerate(p['descs']) if s.stem(names[d['type']])=='in_norm'))*.01/48
origin=min(r[5] for r in raw);depdone=[max((raw[z][7] for e in ins for z in p['ev_inst'][e][2]),default=origin) for ins in p['task_in']]
records=[];best=None
for local_floor in space['common_local_setup_floor_us']:
 local=[];notification=[]
 for t,r in enumerate(raw):
  lag=inputs['ready_after_us'][t]
  if not p['task_in'][t]:local.append(lag);notification.append(0)
  elif r[5]<=depdone[t]:
   f=min(local_floor,lag);local.append(f);notification.append(lag-f)
  else:local.append(lag);notification.append(min(1.,(r[5]-depdone[t])*.01))
 timing=ReplayTiming(inputs['queue_after_us'],local,{int(k):v for k,v in inputs['initial_taken_us'].items()},notification)
 for label in space['layouts']:
  c=copy.deepcopy(p)
  if label!='baseline':
   qs={f'0:{w}':[] for w in range(304)};avail=timing.initial_taken_us.copy();pseudo=[0.]*len(raw)
   for ty,ids in groups.items():
    role=s.stem(names[ty]);phase=s.ROLES.index(role) if role in s.ROLES else 0;by=defaultdict(list)
    for t in ids:by[c['descs'][t]['domain']].append(t)
    for d,ts in by.items():
     if label.startswith('preserve'):
      workers=[p['descs'][t]['worker']%38 for t in ts];ts=ts[::-1];stride=int(label.rsplit('s',1)[1])
      for t,w in zip(ts,workers):c['descs'][t]['worker']=d*38+(w+stride*phase)%38
     else:
      if label!='greedy_coord':
       sign=1 if label=='greedy_short_first' else -1
       ts=sorted(ts,key=lambda t:(sign*profiles[(role,tuple(c['descs'][t]['coord']))][0],t))
      for t in ts:
       w=min(range(d*38,(d+1)*38),key=lambda w:(avail.get(w,0),w))
       dep=max((pseudo[z] for e in c['task_in'][t] for z in c['ev_inst'][e][2]),default=0)
       b,r,q=profiles[(role,tuple(c['descs'][t]['coord']))]
       pseudo[t]=max(dep,avail.get(w,0))+r+b;avail[w]=pseudo[t]+q;c['descs'][t]['worker']=w
   for t,d in enumerate(c['descs']):qs[f'0:{d["worker"]}'].append(t)
   c['static_queues']=qs
  s.validate(c,p)
  rows,stats=simulate(c,lambda _,d:Cost(chain_us=inputs['body_us'][d['id']]),Hw(handoff_us={'DOMAIN':0.,'DEVICE':0.}),replay=timing)
  assert stats['unfinished']==0
  ctr,cp=s.cp_account(c,rows,tr.deps)
  assert all((r[7]-r[6])==raw[t][7]-raw[t][6] for t,r in enumerate(rows))
  latest=[max(rows[z][7] for z in ev[2]) for ev in c['ev_inst']]
  assert all(rows[t][6]>=max((latest[e] for e in ins),default=0) for t,ins in enumerate(c['task_in']))
  if label=='baseline':assert all(rows[t][k]==raw[t][k]-origin+1 for t in range(len(raw)) for k in (5,6,7))
  period=(max(rows[t][7] for t,d in enumerate(c['descs']) if s.stem(names[d['type']])=='moe_down')-max(rows[t][7] for t,d in enumerate(c['descs']) if s.stem(names[d['type']])=='in_norm'))*.01/48
  r=dict(id=label,local_floor_us=local_floor,period_us=period,period_delta_us=period-baseperiod,span_us=ctr.span,step_delta_us=ctr.span-tr.span,wall_ms_est=4.180+(ctr.span-tr.span)/1000,cp=cp,queue_cost_sum_us=sum(timing.queue_after_us),local_cost_sum_us=sum(local),notification_cost_sum_us=sum(notification))
  records.append(r);s.dump(O/'branch-results.json',records)
  with gzip.open(O/f'branch-edges-{label}-f{local_floor}.csv.gz','wt',newline='') as f:
   w=csv.writer(f,lineterminator='\n');w.writerow(['task','event','gap_before_us','gap_after_us','delta_us'])
   oldlatest=[max(raw[z][7] for z in ev[2]) for ev in p['ev_inst']]
   for t in sorted({a['task'].id for a in s.tl.critical_path(tr)}|{a['task'].id for a in s.tl.critical_path(ctr)}):
    for e in p['task_in'][t]:
     b=(raw[t][6]-oldlatest[e])*.01;a=(rows[t][6]-latest[e])*.01
     w.writerow([t,p['ev_inst'][e][0],round(b,3),round(a,3),round(a-b,3)])
  if best is None or period<best[0]['period_us']:best=(r,c,rows)
  print(f'E floor={local_floor} {label}: period_delta={r["period_delta_us"]:+.6f} us/layer step_delta={r["step_delta_us"]:+.3f}',flush=True)
r,c,rows=best
s.dump(O/'branch-best-layout.json',dict(id=r['id'],local_floor_us=r['local_floor_us'],workers=[d['worker'] for d in c['descs']],domains=[d['domain'] for d in c['descs']],static_queues=c['static_queues']))
with gzip.GzipFile(filename=str(O/'branch-best-plan.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(c,separators=(',',':')).encode())
s.write_rows(O/'branch-best.bin.gz',rows)
print('BEST',json.dumps(r),flush=True)
