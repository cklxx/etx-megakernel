#!/usr/bin/env python3
"""Validate retained input identity, output bodies/dependencies and whole-step ledgers."""
import csv,gzip,hashlib,json
from array import array
from pathlib import Path
import search as s
O=Path(__file__).resolve().parent
inputs=json.loads((O/'inputs.json').read_text());raw=s.load_raw()
assert hashlib.sha256(gzip.decompress((O/'measured.bin.gz').read_bytes())).hexdigest()==inputs['trace_sha256']
assert hashlib.sha256(gzip.decompress((O/'original-plan.json.gz').read_bytes())).hexdigest()==inputs['plan_sha256']
base=json.loads(gzip.decompress((O/'original-plan.json.gz').read_bytes()))
for prefix,plan,trace in [('measured','original-plan','measured'),('main','best-plan','best'),('secondary','secondary-best-plan','secondary-best'),('branch','branch-best-plan','branch-best'),('oracle','oracle-best-plan','oracle-best'),('finalist','finalist-best-plan','finalist-best')]:
 p=json.loads(gzip.decompress((O/f'{plan}.json.gz').read_bytes()));a=array('q');a.frombytes(gzip.decompress((O/f'{trace}.bin.gz').read_bytes()));rows=[tuple(a[x:x+9]) for x in range(0,len(a),9)]
 s.validate(p,base)
 assert len(rows)==len(raw)==60362
 latest=[max(rows[z][7] for z in e[2]) for e in p['ev_inst']]
 for t,r in enumerate(rows):
  d=p['descs'][t];assert r[0]==d['type'] and list(r[1:len(d['coord'])+1])==d['coord'] and r[8]==d['worker']
  assert r[7]-r[6]==raw[t][7]-raw[t][6]
  assert r[6]>=max((latest[e] for e in p['task_in'][t]),default=0)
 print(f'{prefix}: all bodies and dependency gates preserved for 60362 tasks')
main=json.loads((O/'results.json').read_text());assert len(main)==len(json.loads((O/'registered-candidates.json').read_text()))==145
assert sum('span_us' in r for r in json.loads((O/'secondary-results.json').read_text()))==32
sums={};targets={}
with (O/'candidate-ledgers.csv').open(newline='') as f:
 for r in csv.DictReader(f):
  key=(r['source'],r['candidate'],r['model']);sums[key]=sums.get(key,0)+float(r['delta_total']);targets[key]=float(r['whole_step_delta'])
assert len(sums)==201 and all(abs(sums[k]-targets[k])<.001 for k in sums)
print('201 candidate critical-path ledgers reconcile to their whole-step deltas')
assert len((O.parent/'X7-SCHEDULE-WHATIF-2026-10-11.md').read_text().splitlines())<=200
print('Report <=200 lines; compressed input hashes match; validation complete')
