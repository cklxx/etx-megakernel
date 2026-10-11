#!/usr/bin/env python3
import gzip,json
from dataclasses import replace
from pathlib import Path
import search as s
from etx.sim.xcd import Cost,Hw,ReplayTiming,simulate
O=Path(__file__).resolve().parent;i=json.loads((O/'inputs.json').read_text());p=json.loads(gzip.decompress((O/'oracle-best-plan.json.gz').read_bytes()));raw=s.load_raw()
# Preregistered extremes: imputed final-worker epilogues 0 or max observed queue tax, not arbitrary deletion of measured taxes.
s.dump(O/'tail-space.json',dict(registered_before_tail_results=True,tail_values_us=[0,4.32],model='additive',layout='frozen additive oracle'))
results=[]
for value in (0.,4.32):
 q=i['queue_after_us'].copy()
 for t in i['queue_tail_imputed_task_ids']:q[t]=value
 rows,stats=simulate(p,lambda _,d:Cost(chain_us=i['body_us'][d['id']]),Hw(handoff_us={'DOMAIN':0.,'DEVICE':0.}),replay=ReplayTiming(q,i['ready_after_us'],{int(k):v for k,v in i['initial_taken_us'].items()}))
 tr=s.tl.build_trace(p,rows);names={v['id']:k for k,v in p['types'].items()}
 layer=(max(rows[t][7] for t,d in enumerate(p['descs']) if s.stem(names[d['type']])=='moe_down')-max(rows[t][7] for t,d in enumerate(p['descs']) if s.stem(names[d['type']])=='in_norm'))*.01/48
 r=dict(tail_us=value,span_us=tr.span,period_us=layer,period_delta_us=layer-80.995,step_delta_us=tr.span-i['measured_span_us'])
 results.append(r);print(json.dumps(r),flush=True)
s.dump(O/'tail-results.json',results)
