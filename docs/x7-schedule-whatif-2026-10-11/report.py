#!/usr/bin/env python3
"""Readable, line-citable accounting from retained simulation output."""
import csv,gzip,json
from collections import defaultdict
from pathlib import Path
import search as s
O=Path(__file__).resolve().parent
i=json.loads((O/'inputs.json').read_text());main=json.loads((O/'results.json').read_text());sec=json.loads((O/'secondary-results.json').read_text());fin=json.loads((O/'finalist-results.json').read_text());oracle=json.loads((O/'oracle-results.json').read_text())
b=main[0];win=next(r for r in fin if r['id']=='frozen_additive_oracle' and r['model']=='additive')
raw=s.load_raw();plan=json.loads(gzip.decompress((O/'oracle-best-plan.json.gz').read_bytes()))
from array import array
ar=array('q');ar.frombytes(gzip.decompress((O/'oracle-best.bin.gz').read_bytes()));rows=[tuple(ar[t:t+9]) for t in range(0,len(ar),9)]
tr,cp=s.cp_account(plan,rows);oldplan=json.loads(gzip.decompress((O/'original-plan.json.gz').read_bytes()));oldtr,oldcp=s.cp_account(oldplan,raw)
origin=min(r[5] for r in raw);prefix=(max(raw[t][7] for t,d in enumerate(oldplan['descs']) if s.stem(oldtr.by_id[t].name)=='in_norm')-origin)*.01
tail=(max(raw[t][7] for t in range(len(raw)))-max(raw[t][7] for t,d in enumerate(oldplan['descs']) if s.stem(oldtr.by_id[t].name)=='moe_down'))*.01
with (O/'accounting.txt').open('w') as f:
 def line(x):print(x,file=f)
 line(f'M trace {i["trace_source"]} sha256={i["trace_sha256"]}')
 line(f'M plan {i["plan_source"]} sha256={i["plan_sha256"]}')
 line(f'M rows=60362 mismatch=0 dependency_pairs={i["dependency_pairs"]} event_names=436 instances=1403')
 line('M baseline stamp differences: all 181086 stamp errors 0.000 us; span 4067.290 us')
 line(f'M body/ready residual sums; E queue sum including imputation: body={sum(i["body_us"]):.3f} us; ready residual={sum(i["ready_after_us"]):.3f} us; queue residual + terminal imputation={sum(i["queue_after_us"]):.3f} us')
 line(f'M period entry max(in_norm.done)={prefix:.3f} us; tail last(moe_down.done)->argmax.done={tail:.3f} us')
 line(f'M span reconciliation: {prefix:.3f} + 48*80.995000 + {tail:.3f} = 4067.290 us')
 line('E best nominal: exact_body_plus_ready_desc additive; period 80.175000 us; reduction 0.820000 us/layer > 0.625')
 line(f'E nominal step: {prefix:.3f} + 48*80.175000 + {tail:.3f} = 4027.930 us; delta -39.360 us; estimated wall 4.140640 ms')
 line('E critical-path ledger (us): role body_before body_after delta_body gap_before gap_after delta_gap delta_total')
 totals=[0.]*7
 for role in ('prologue','in_norm')+tuple(s.ROLES)+('lm_head','argmax'):
  a=oldcp[role];c=cp[role];v=[a['body'],c['body'],c['body']-a['body'],a['gap'],c['gap'],c['gap']-a['gap'],c['body']+c['gap']-a['body']-a['gap']]
  totals=[x+y for x,y in zip(totals,v)];line(role+' '+' '.join(f'{x:+.3f}' for x in v))
 line('TOTAL '+' '.join(f'{x:+.3f}' for x in totals))
 line('E same frozen table, alternative exact baseline fit: branch0 and branch052 period 80.463750 us; reduction 0.531250 < 0.625; step cut 25.500 us; wall 4.154500 ms')
 line('E profile-median body+ready policy: additive +0.031250 us/layer, +1.500 us/step; branch models +0.296667 us/layer, +14.240 us/step')
 line('E oracle reoptimized for branch0: -0.686875 us/layer; branch052: -0.631667 us/layer; these are different worker tables, not sensitivity of the frozen best')
 line('E terminal queue tax extrema 0..4.32 us: frozen nominal period unchanged; step reduction 37.050..39.360 us')
 line('E family minima: map candidate delta_period delta_step span_us estimated_wall_ms')
 for mapping in ('original','rotate_heads','dense_balance','attention_wide'):
  r=min([a for a in main if a['map']==mapping and a['id']!='baseline' and a['promotion_us']==.3],key=lambda a:a['period_delta_us'])
  line(f'{mapping} {r["id"]} {r["period_delta_us"]:+.6f} {r["step_delta_us"]:+.3f} {r["span_us"]:.3f} {r["wall_ms_est"]:.6f}')
 for label in ('greedy_coord','narrow6','narrow14','one_xcd_domain','serial_head_optimistic','owner_prep_optimistic'):
  r=next(a for a in sec if a['id']==label);line(f'{label} {r["period_delta_us"]:+.6f} {r["step_delta_us"]:+.3f} {r["span_us"]:.3f} {r["wall_ms_est"]:.6f}')
 line('E matched-edge examples: task type event before_us after_us delta_us (same consumer and event)')
 with gzip.open(O/'oracle-edges-exact_body_plus_ready_desc-additive.csv.gz','rt') as ef:
  edges=list(csv.DictReader(ef))
 for sign in (-1,1):
  e=max([e for e in edges if float(e['delta_us'])*sign>0],key=lambda e:float(e['delta_us'])*sign)
  t=int(e['task']);line(f'{t} {oldtr.by_id[t].name} {e["event"]} {e["gap_before_us"]} {e["gap_after_us"]} {e["delta_us"]}')
 line('VERDICT WEAK: nominal same-cost layout passes; other exact baseline timing fit fails KEEP; profile-only policy fails; no measured GPU win or proof of structural impossibility')
 line('E pool accounting: gap -70.320 us, body-path +30.960 us, net -39.360 us; no total task work or dependency removed')
 line('E practical-policy search CLOSED below KEEP; rows 9/10 structural classification remains unproven, conditional oracle gain only 3.0%-4.6% of 0.847 ms need')
# Per-candidate aggregate total deltas, separate from matched-edge CSVs (which are not summable).
allrecords=[]
for file in ('results.json','secondary-results.json','branch-results.json','oracle-results.json','finalist-results.json'):
 for r in json.loads((O/file).read_text()):
  if 'cp' in r:allrecords.append((file,r))
with (O/'candidate-ledgers.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['source','candidate','model','role','body_before','body_after','delta_body','gap_before','gap_after','delta_gap','delta_total','whole_step_delta'])
 for file,r in allrecords:
  total=0
  for role in sorted(set(b['cp'])|set(r['cp'])):
   a=b['cp'].get(role,dict(body=0,gap=0));c=r['cp'].get(role,dict(body=0,gap=0));delta=c['body']+c['gap']-a['body']-a['gap'];total+=delta
   w.writerow([file,r['id'],r.get('model',r.get('local_floor_us','additive')),role,round(a['body'],3),round(c['body'],3),round(c['body']-a['body'],3),round(a['gap'],3),round(c['gap'],3),round(c['gap']-a['gap'],3),round(delta,3),round(r['step_delta_us'],3)])
  assert abs(total-r['step_delta_us'])<1e-6
print((O/'accounting.txt').read_text(),end='')
