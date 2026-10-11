#!/usr/bin/env python3
"""Additional preregistered layout policies and boundary feasibility bounds."""
import copy,csv,gzip,json,statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import search as s
from etx.sim.xcd import Cost,Hw,ReplayTiming,simulate

O=Path(__file__).resolve().parent
p=json.loads(gzip.decompress((O/'original-plan.json.gz').read_bytes()))
raw=s.tl.read_raw(s.RAW); tr,basecp=s.cp_account(p,raw)
i=json.loads((O/'inputs.json').read_text())
names={v['id']:k for k,v in p['types'].items()}; groups=defaultdict(list); values=defaultdict(lambda:[[],[],[]])
for d in p['descs']:
 t=d['id'];groups[d['type']].append(t);key=(s.stem(names[d['type']]),tuple(d['coord']))
 for k,field in enumerate(('body_us','ready_after_us','queue_after_us')):values[key][k].append(i[field][t])
profile={k:[statistics.median(x) for x in v] for k,v in values.items()}
baseperiod=(max(raw[t][7] for t,d in enumerate(p['descs']) if s.stem(names[d['type']])=='moe_down')-max(raw[t][7] for t,d in enumerate(p['descs']) if s.stem(names[d['type']])=='in_norm'))*.01/48
space=json.loads((O/'secondary-space.json').read_text()); records=[]; best=None

def run(label,c,timing,legal=True,reason='same tasks and dependencies',nominal=True):
 global best
 rows,stats=simulate(c,lambda n,d:Cost(chain_us=i['body_us'][d['id']]),Hw(handoff_us={'DOMAIN':0.,'DEVICE':0.}),replay=timing)
 assert stats['unfinished']==0
 ctr,cp=s.cp_account(c,rows,tr.deps if c['ev_inst']==p['ev_inst'] and c['task_in']==p['task_in'] else None)
 assert all((r[7]-r[6])==raw[t][7]-raw[t][6] for t,r in enumerate(rows))
 latest=[max(rows[z][7] for z in e[2]) for e in c['ev_inst']]
 assert all(rows[t][6]>=max((latest[e] for e in ins),default=0) for t,ins in enumerate(c['task_in']))
 period=(max(rows[t][7] for t,d in enumerate(c['descs']) if s.stem(names[d['type']])=='moe_down')-max(rows[t][7] for t,d in enumerate(c['descs']) if s.stem(names[d['type']])=='in_norm'))*.01/48
 r=dict(id=label,legal=legal,reason=reason,period_us=period,period_delta_us=period-baseperiod,span_us=ctr.span,step_delta_us=ctr.span-tr.span,wall_ms_est=4.180+(ctr.span-tr.span)/1000,cp=cp,nominal=nominal,
        queue_cost_sum_us=sum(timing.queue_after_us),ready_cost_sum_us=sum(timing.ready_after_us))
 records.append(r);s.dump(O/'secondary-results.json',records)
 with gzip.open(O/f'edges-{label}.csv.gz','wt',newline='') as f:
  w=csv.writer(f);w.writerow(['task','event','gap_before_us','gap_after_us','delta_us'])
  oldlatest=[max(raw[z][7] for z in e[2]) for e in p['ev_inst']]
  for t in sorted({a['task'].id for a in s.tl.critical_path(tr)}|{a['task'].id for a in s.tl.critical_path(ctr)}):
   for e in p['task_in'][t]:
    before=(raw[t][6]-oldlatest[e])*.01;after=(rows[t][6]-latest[e])*.01
    w.writerow([t,p['ev_inst'][e][0],round(before,3),round(after,3),round(after-before,3)])
 if legal and nominal and (best is None or period<best[0]['period_us']):best=(r,c,rows)
 print(f'E {label} period_delta={r["period_delta_us"]:+.6f} us/layer step_delta={r["step_delta_us"]:+.3f} us legal={legal}',flush=True)
 return r

base=ReplayTiming(i['queue_after_us'],i['ready_after_us'],{int(k):v for k,v in i['initial_taken_us'].items()})
for order in space['offset_preserving_original']['orders']:
 for stride in space['offset_preserving_original']['strides']:
  c=copy.deepcopy(p);queues={f'0:{w}':[] for w in range(304)}
  for ty,ids in groups.items():
   role=s.stem(names[ty]);phase=s.ROLES.index(role) if role in s.ROLES else 0; by=defaultdict(list)
   for t in ids:by[c['descs'][t]['domain']].append(t)
   for d,ts in by.items():
    workers=[p['descs'][t]['worker']%38 for t in ts]
    if order=='reverse':ts=ts[::-1]
    elif order in ('long_first','short_first'):
     ts=sorted(ts,key=lambda t:((-1 if order=='long_first' else 1)*profile[(role,tuple(c['descs'][t]['coord']))][0],t))
    for t,w in zip(ts,workers):c['descs'][t]['worker']=d*38+(w+stride*phase)%38
  for t in range(len(c['descs'])):queues[f'0:{c["descs"][t]["worker"]}'].append(t)
  c['static_queues']=queues;s.validate(c,p);run(f'preserve-{order}-s{stride}',c,base)
for policy in space['profile_guided']+['narrow6','narrow14','one_xcd_domain']:
 c=copy.deepcopy(p);queues={f'0:{w}':[] for w in range(304)};avail=base.initial_taken_us.copy();done=[0.]*len(raw)
 if policy=='one_xcd_domain':
  for d in c['descs']:d['domain']=0
  for e in c['events']:e['scope']='DOMAIN'
 for ty,ids in groups.items():
  role=s.stem(names[ty]);by=defaultdict(list)
  for t in ids:by[c['descs'][t]['domain']].append(t)
  for d,ts in by.items():
   sign=1 if policy=='greedy_short_first' else -1
   if policy!='greedy_coord':ts=sorted(ts,key=lambda t:(sign*profile[(role,tuple(c['descs'][t]['coord']))][0],t))
   width=int(policy.removeprefix('narrow')) if policy.startswith('narrow') else 38
   for t in ts:
    w=min(range(d*38,d*38+width),key=lambda w:(avail.get(w,0),w))
    dep=max((done[z] for e in c['task_in'][t] for z in c['ev_inst'][e][2]),default=0)
    b,r,q=profile[(role,tuple(c['descs'][t]['coord']))]
    done[t]=max(dep,avail.get(w,0))+r+b;avail[w]=done[t]+q
    c['descs'][t]['worker']=w;queues[f'0:{w}'].append(t)
 c['static_queues']=queues;s.validate(c,p)
 timing=base
 if policy=='one_xcd_domain':
  # Optimistic scope-price sensitivity; only documented .3-us difference, never a fitted measurement.
  oldscope={e['name']:e['scope'] for e in p['events']}
  timing=replace(base,ready_after_us=[max(0,r-(.3 if any(oldscope[p['ev_inst'][e][0]]=='DEVICE' for e in ins) else 0)) for r,ins in zip(base.ready_after_us,p['task_in'])])
 run(policy,c,timing,nominal=policy!='one_xcd_domain',reason='all DOMAIN parties co-located; fixed bodies optimistic for reduced streaming width' if policy=='one_xcd_domain' or policy.startswith('narrow') else 'profile medians choose earliest free worker, longest task first')
# Whole-head composite bound: serialize the unchanged prep, five attn and reduce calls.
# Give all internal publication/fetch/ready cost away: if it loses, any real wrapper loses in this frozen-body model.
for kind in ('serial_head_optimistic','owner_prep_optimistic'):
 c=copy.deepcopy(p);queues={f'0:{w}':[] for w in range(304)};q=base.queue_after_us.copy();r=base.ready_after_us.copy()
 for L in range(48):
  roleids={role:groups[p['types'][f'{role}_{L}']['id']] for role in ('prep','attn','reduce')}
  for h in range(4):
   prep=roleids['prep'][h];attn=roleids['attn'][h*5:(h+1)*5];reduce=roleids['reduce'][h]
   slot=p['descs'][prep]['worker'];chain=[prep]+(attn+[reduce] if kind.startswith('serial') else attn[:1])
   for t in chain:c['descs'][t]['worker']=slot
   for a,b in zip(chain,chain[1:]):q[a]=0;r[b]=0
   if kind.startswith('owner'):
    # Extra serialization models the prep signal emitted after the owner's whole body.
    c['ev_inst'].append([f'owner_{L}_{h}',[0],[attn[0]]]);c['events'].append(dict(name=f'owner_{L}_{h}',scope='DOMAIN'))
    for t in attn[1:]:c['task_in'][t].append(len(c['ev_inst'])-1)
 for t,d in enumerate(c['descs']):queues[f'0:{d["worker"]}'].append(t)
 c['static_queues']=queues
 run(kind,c,ReplayTiming(q,r,base.initial_taken_us),legal=False,nominal=False,
     reason='same-call composite optimistic bound, internal residual deletion not certified; whole-body plan can express serial composite but no mid-body publication')
required=s.shares(p);bad=[n for n,v in required.items() if v>1]
records.append(dict(id='domain_all_on_original',legal=False,reason=f'{len(bad)} event names contain cross-XCD producer-consumer pairs',examples=bad[:8]))
for key,role in [('delete_prep_dependency','attn'),('delete_reduce_dependency','o_proj'),('delete_moe_fold_dependency','moe_down'),('delete_layer_dependency','qkv')]:
 records.append(dict(id=key,legal=False,reason=f'consumer {role} still needs existing producer data; deleting the event without an ordering substitute deletes real dependencies'))
s.dump(O/'secondary-results.json',records)
r,c,rows=best;s.dump(O/'secondary-best-layout.json',dict(id=r['id'],workers=[d['worker'] for d in c['descs']],domains=[d['domain'] for d in c['descs']],static_queues=c['static_queues']))
with gzip.GzipFile(filename=str(O/'secondary-best-plan.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(c,separators=(',',':')).encode())
s.write_rows(O/'secondary-best.bin.gz',rows)
print('BEST',json.dumps(r),flush=True)
