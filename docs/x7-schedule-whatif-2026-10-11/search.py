#!/usr/bin/env python3
"""CPU-only frozen-body schedule search. Run from any cwd; outputs live beside this script."""
from __future__ import annotations
import copy, csv, gzip, hashlib, json, re, statistics, sys
from array import array
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from etx.sim.xcd import Cost, Hw, ReplayTiming, simulate
from etx.tools import timeline as tl

OUT = Path(__file__).resolve().parent
ARCH = Path.home() / 'code/etx'
PLAN = ARCH / 'results_vlbest2_2026-10-10/b_all/plan.json'
RAW = ARCH / 'results_qwen300_2026-10-11/tr_on.bin'
ROLES = ['qkv', 'prep', 'attn', 'reduce', 'o_proj', 'router', 'topk', 'moe_up', 'moe_down']

def stem(name):
    return re.sub(r'_\d+(?:_x)?$', '', name)

def dump(path, value):
    path.write_text(json.dumps(value, separators=(',', ':'), sort_keys=True) + '\n')

def write_rows(path, rows):
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as f:
        f.write(array('q', [x for r in rows for x in r]).tobytes())

def shares(p):
    consumers = defaultdict(list)
    for t, ins in enumerate(p['task_in']):
        for e in ins:
            consumers[e].append(t)
    result = {}
    for i, (name, coord, producers) in enumerate(p['ev_inst']):
        parties = {p['descs'][t]['domain'] for t in producers + consumers[i]}
        result[name] = max(result.get(name, 0), 2 if len(parties) > 1 else 1)
    return result

def validate(p, base):
    assert p['ev_inst'] == base['ev_inst'] and p['task_in'] == base['task_in']
    seen = set()
    for k, ids in p['static_queues'].items():
        d, w = map(int, k.split(':'))
        assert d == 0 and 0 <= w < 304
        assert ids == sorted(ids) or all(p['descs'][a]['type'] <= p['descs'][b]['type'] for a,b in zip(ids, ids[1:]))
        for t in ids:
            assert t not in seen
            seen.add(t)
            desc = p['descs'][t]
            assert desc['domain'] == w // 38 and desc['worker'] == w
            assert all(desc[k] == base['descs'][t][k] for k in ('id','type','coord','device','mode'))
    assert len(seen) == len(base['descs'])
    required = shares(p)
    scopes = {e['name']: e['scope'] for e in p['events']}
    assert all(scopes[n] == 'DEVICE' or required[n] == 1 for n in scopes)

def cp_account(p, rows, deps=None):
    view = {k:v for k,v in p.items() if k not in ('ev_inst','task_in')} if deps is not None else p
    tr = tl.build_trace(view, rows)
    if deps is not None: tr.deps = deps
    a = defaultdict(lambda: {'body': 0., 'gap': 0., 'tasks': 0})
    for s in tl.critical_path(tr):
        r = stem(s['task'].name)
        for k in ('body','gap'): a[r][k] += s[k]
        a[r]['tasks'] += 1
    assert abs(sum(v['body'] + v['gap'] for v in a.values()) - tr.span) < 1e-7
    return tr, dict(a)

def main():
    space = json.loads((OUT / 'search-space.json').read_text())
    p = json.loads(PLAN.read_text()); raw = tl.read_raw(RAW)
    tr, measured_cp = cp_account(p, raw)
    assert len(raw) == len(p['descs']) == 60362
    names = {t['id']: n for n,t in p['types'].items()}
    groups = defaultdict(list)
    for d,r in zip(p['descs'],raw):
        assert r[0] == d['type'] and list(r[1:1+len(d['coord'])]) == d['coord']
        assert (r[8] & 0xffffffff) == d['worker']
        groups[d['type']].append(d['id'])
    # Use integer tick differences to avoid subtracting large absolute GPU timestamps.
    origin = min(r[5] for r in raw)
    body = [(r[7]-r[6])*.01 for r in raw]
    ready_after = []
    for t,r in enumerate(raw):
        depdone = max((raw[z][7] for e in p['task_in'][t] for z in p['ev_inst'][e][2]), default=origin)
        ready_after.append((r[6]-max(r[5], depdone))*.01)
    assert min(ready_after) >= 0
    queue_after = [None]*len(raw); initial = {}
    for k, ids in p['static_queues'].items():
        if not ids: continue
        initial[p['descs'][ids[0]]['worker']] = (raw[ids[0]][5]-origin)*.01
        for a,b in zip(ids,ids[1:]): queue_after[a] = (raw[b][5]-raw[a][7])*.01
    medq = {ty: statistics.median(queue_after[t] for t in ids if queue_after[t] is not None)
            for ty,ids in groups.items() if any(queue_after[t] is not None for t in ids)}
    missing = [t for t,v in enumerate(queue_after) if v is None]
    for t in missing: queue_after[t] = medq.get(p['descs'][t]['type'], statistics.median(v for v in queue_after if v is not None))
    assert min(queue_after) >= 0
    timing = ReplayTiming(queue_after, ready_after, initial)
    # Frozen role-coordinate median avoids per-layer assignment to a lucky noisy task.
    profile = defaultdict(list)
    for t,d in enumerate(p['descs']): profile[(stem(names[d['type']]),tuple(d['coord']))].append(body[t])
    profile = {k:statistics.median(v) for k,v in profile.items()}
    inputs = dict(base_commit='e31bf04', plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                  trace_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(), plan_source=str(PLAN), trace_source=str(RAW),
                  measured_span_us=tr.span, task_count=len(raw), mismatches=0, dependency_pairs=sum(len(x) for x in tr.deps),
                  boundary_events=len(p['events']), event_instances=len(p['ev_inst']),
                  queue_tail_imputed_task_ids=missing, queue_tail_rule='same type median, unused by baseline successors',
                  body_us=body, queue_after_us=queue_after, ready_after_us=ready_after, initial_taken_us=initial)
    dump(OUT/'inputs.json', inputs)
    with gzip.GzipFile(filename=str(OUT/'original-plan.json.gz'),mode='wb',mtime=0) as f: f.write(PLAN.read_bytes())
    write_rows(OUT/'measured.bin.gz',raw)
    results = []; best = None
    edgefile = gzip.open(OUT/'candidate-edge-deltas.csv.gz','wt',newline='')
    edgewriter = csv.writer(edgefile)
    edgewriter.writerow(['candidate','task','event','baseline_path','candidate_path','gap_before_us','gap_after_us','delta_us'])
    basepath={x['task'].id for x in tl.critical_path(tr)}
    base_latest=[max(raw[z][7] for z in ev[2]) for ev in p['ev_inst']]
    baseline_rows = None
    layouts = [('baseline', 'original', 'original', 0, .3)]
    for mapping in space['maps']:
        for order in space['orders']:
            for stride in space['worker_strides']:
                # Promotion has no effect unless DOMAIN heads are split.
                premiums = space['scope_promotion_us'] if mapping == 'attention_wide' else [.3]
                for premium in premiums:
                    layouts.append((f'{mapping}-{order}-s{stride}-p{premium}',mapping,order,stride,premium))
    dump(OUT/'registered-candidates.json', layouts)
    print(f'INPUT span={tr.span:.3f} us tasks={len(raw)} mismatches=0 layouts={len(layouts)}',flush=True)
    for index,(label,mapping,order,stride,premium) in enumerate(layouts):
        c = copy.deepcopy(p)
        for ty,ids in groups.items():
            role = stem(names[ty]); nm = names[ty]
            if mapping == 'rotate_heads' and role in ('prep','attn','reduce'):
                layer=int(re.search(r'_(\d+)',nm)[1])
                for t in ids: c['descs'][t]['domain'] = (p['descs'][t]['domain']+layer%8)%8
            elif mapping == 'attention_wide' and role == 'attn':
                for t in ids: c['descs'][t]['domain']=c['descs'][t]['coord'][0]%8
            elif mapping == 'dense_balance' and role in ('qkv','o_proj','router','lm_head'):
                loads=[0.]*8; counts=[0]*8
                for t in sorted(ids,key=lambda t:(-profile[(role,tuple(c['descs'][t]['coord']))],t)):
                    d=min((x for x in range(8) if counts[x]<len(ids)//8),key=lambda x:(loads[x],x))
                    c['descs'][t]['domain']=d;counts[d]+=1
                    loads[d]+=profile[(role,tuple(c['descs'][t]['coord']))]
        required=shares(c); promoted=set()
        for e in c['events']:
            if required[e['name']]>1 and e['scope']=='DOMAIN': e['scope']='DEVICE'; promoted.add(e['name'])
        if order!='original':
            queues={f'0:{w}':[] for w in range(304)}
            for ty,ids in groups.items():
                role=stem(names[ty]); bydom=defaultdict(list)
                for t in ids: bydom[c['descs'][t]['domain']].append(t)
                phase=ROLES.index(role) if role in ROLES else 0
                for d,ts in bydom.items():
                    if order=='reverse': ts=ts[::-1]
                    if order in ('long_first','short_first'):
                        sign=-1 if order=='long_first' else 1
                        ts=sorted(ts,key=lambda t:(sign*profile[(role,tuple(c['descs'][t]['coord']))],t))
                    for j,t in enumerate(ts):
                        slot=(j+stride*phase)%38;c['descs'][t]['worker']=d*38+slot
                        queues[f'0:{d*38+slot}'].append(t)
            c['static_queues']=queues
        validate(c,p)
        extra=[premium if any(c['ev_inst'][e][0] in promoted for e in ins) else 0. for ins in c['task_in']]
        tt=replace(timing,ready_after_us=[a+b for a,b in zip(ready_after,extra)])
        rows,stats=simulate(c,lambda name,d:Cost(chain_us=body[d['id']]),
                            Hw(handoff_us={'DOMAIN':0.,'DEVICE':0.}),replay=tt)
        assert stats['unfinished']==0
        ctr,cp=cp_account(c,rows,tr.deps)
        # Individual bodies and every event's producer completion gate are preserved.
        assert all((r[7]-r[6])==(raw[t][7]-raw[t][6]) for t,r in enumerate(rows))
        latest=[max(rows[z][7] for z in ev[2]) for ev in c['ev_inst']]
        assert all(rows[t][6]>=max((latest[e] for e in ins),default=0) for t,ins in enumerate(c['task_in']))
        cpath={x['task'].id for x in tl.critical_path(ctr)}
        for t in sorted(basepath|cpath):
            for e in c['task_in'][t]:
                before=(raw[t][6]-base_latest[e])*.01; after=(rows[t][6]-latest[e])*.01
                edgewriter.writerow([label,t,c['ev_inst'][e][0],int(t in basepath),int(t in cpath),round(before,3),round(after,3),round(after-before,3)])
        if index==0:
            baseline_rows=rows
            stamp_error=max(abs(rows[t][k]-(raw[t][k]-origin+1))*.01 for t in range(len(rows)) for k in (5,6,7))
            assert stamp_error==0
            print(f'BASELINE all-stamp error={stamp_error:.3f} us span={ctr.span:.3f}',flush=True)
        end={role:max(r[7]*.01 for t,r in enumerate(rows) if stem(names[c['descs'][t]['type']])==role) for role in ('moe_down','in_norm')}
        layer_period=(end['moe_down']-end['in_norm'])/48
        step_delta=ctr.span-tr.span
        entry=dict(id=label,map=mapping,order=order,stride=stride,promotion_us=premium,promoted_events=len(promoted),
                   span_us=ctr.span,step_delta_us=step_delta,wall_ms_est=4.180+step_delta/1000,
                   layer_period_us=layer_period,cp=cp,queue_cost_sum_us=sum(queue_after),
                   ready_cost_sum_us=sum(tt.ready_after_us),legal=True)
        results.append(entry)
        dump(OUT/'partial-results.json',results)
        if index==0: baseperiod=layer_period
        entry['period_delta_us']=layer_period-baseperiod
        # Rank nominal .3-us promotion estimate; optimistic premiums remain sensitivities.
        if index and premium==.3 and (best is None or layer_period<best[0]['layer_period_us']):
            best=(entry,c,rows)
        if index%12==0 or index+1==len(layouts):
            print(f'PROGRESS {index+1}/{len(layouts)} best={best[0]["id"] if best else "none"}',flush=True)
    edgefile.close()
    dump(OUT/'results.json',results)
    (OUT/'partial-results.json').unlink()
    entry,c,rows=best
    dump(OUT/'best-layout.json',dict(id=entry['id'],domains=[d['domain'] for d in c['descs']],
                                   workers=[d['worker'] for d in c['descs']],events=c['events'],static_queues=c['static_queues']))
    with gzip.GzipFile(filename=str(OUT/'best-plan.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(c,separators=(',',':')).encode())
    write_rows(OUT/'best.bin.gz',rows)
    # Same consumer+event identity before/after, including edges not on either critical path.
    with gzip.open(OUT/'edge-deltas.csv.gz','wt',newline='') as f:
        writer=csv.writer(f);writer.writerow(['task','type','event','latest_producer_before','latest_producer_after','gap_before_us','gap_after_us','delta_us','queue_delay_before_us','queue_delay_after_us'])
        for t,ins in enumerate(c['task_in']):
            for e in ins:
                producers=c['ev_inst'][e][2];b=max(producers,key=lambda z:raw[z][7]);a=max(producers,key=lambda z:rows[z][7])
                before=(raw[t][6]-raw[b][7])*.01;after=(rows[t][6]-rows[a][7])*.01
                writer.writerow([t,names[c['descs'][t]['type']],c['ev_inst'][e][0],b,a,round(before,3),round(after,3),round(after-before,3),
                                 round(max(0,(raw[t][5]-raw[b][7])*.01),3),round(max(0,(rows[t][5]-rows[a][7])*.01),3)])
    with (OUT/'candidates.csv').open('w',newline='') as f:
        fields=[k for k in results[0] if k!='cp'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:v for k,v in r.items() if k!='cp'} for r in results)
    with (OUT/'summary.txt').open('w') as f:
        print(f'M input: span {tr.span:.3f} us; 60362 tasks; 436 events; {len(p["ev_inst"])} instances; zero mismatches',file=f)
        print('M calibrated baseline: all 181086 stamps identical; no residual dropped',file=f)
        print(f'E layer period baseline {baseperiod:.6f} us; definition: (last moe_down - last in_norm) / 48',file=f)
        print(f'E best nominal: {entry["id"]}; period {entry["layer_period_us"]:.6f}; delta {entry["period_delta_us"]:.6f} us/layer',file=f)
        print(f'E best span {entry["span_us"]:.3f} us; delta {entry["step_delta_us"]:.3f}; wall {entry["wall_ms_est"]:.6f} ms',file=f)
        print('E critical-path reconciliation: role body_before body_after dbody gap_before gap_after dgap total_delta_us',file=f)
        for role in sorted(set(measured_cp)|set(entry['cp'])):
            b=measured_cp.get(role,dict(body=0,gap=0));a=entry['cp'].get(role,dict(body=0,gap=0))
            print(f'{role:10s} {b["body"]:.3f} {a["body"]:.3f} {a["body"]-b["body"]:+.3f} {b["gap"]:.3f} {a["gap"]:.3f} {a["gap"]-b["gap"]:+.3f} {a["body"]+a["gap"]-b["body"]-b["gap"]:+.3f}',file=f)
        print(f'E residual conservation: queue {sum(queue_after):.3f} us (worker-summed) unchanged; ready {sum(ready_after):.3f} -> {entry["ready_cost_sum_us"]:.3f}',file=f)
        for r in sorted(results,key=lambda r:r['period_delta_us'])[:12]:
            print(f'E rank {r["id"]}: period_delta={r["period_delta_us"]:+.6f}; step_delta={r["step_delta_us"]:+.3f}',file=f)
    print((OUT/'summary.txt').read_text(),flush=True)

if __name__=='__main__':main()
