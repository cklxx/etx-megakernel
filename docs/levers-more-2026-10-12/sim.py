#!/usr/bin/env python3
"""Run: .venv/bin/python docs/levers-more-2026-10-12/sim.py > docs/levers-more-2026-10-12/sim.log  (11 s, CPU)

CPU what-ifs for docs/LEVERS-MORE-2026-10-12.md on the X7 calibrated replay (tr_on.bin, 4067.29 us).

Each lever edits only the bodies (done-ready) or the replay residuals of the tasks it touches; every other task keeps
its recorded body, queue residual q and ready residual r. Prints span deltas in us and ms per step [E]."""
import gzip, json, statistics, sys
from collections import defaultdict
from pathlib import Path

X7 = Path(__file__).resolve().parents[1] / 'x7-schedule-whatif-2026-10-11'
sys.path.insert(0, str(X7)); sys.path.insert(0, str(X7.parents[1]))
import search as s
from etx.sim.xcd import Cost, Hw, ReplayTiming, simulate

p = json.loads(gzip.decompress((X7 / 'original-plan.json.gz').read_bytes()))
raw = s.load_raw()
inp = json.loads((X7 / 'inputs.json').read_text())
body0, q0, r0 = inp['body_us'], inp['queue_after_us'], inp['ready_after_us']
init = {int(k): v for k, v in inp['initial_taken_us'].items()}
names = {t['id']: n for n, t in p['types'].items()}
role = [s.stem(names[d['type']]) for d in p['descs']]
scope = {e['name']: e['scope'] for e in p['events']}
origin = min(r[5] for r in raw)
N = len(raw)
cons = defaultdict(list)
for t, ins in enumerate(p['task_in']):
    for e in ins: cons[e].append(t)
latest = [max(raw[z][7] for z in ev[2]) for ev in p['ev_inst']]
free = [bool(p['task_in'][t]) and raw[t][5] <= max(latest[e] for e in p['task_in'][t]) for t in range(N)]


def run(body=None, q=None, r=None):
    body = body or body0; q = q or q0; r = r or r0
    rows, st = simulate(p, lambda n, d: Cost(chain_us=body[d['id']]), Hw(handoff_us={'DOMAIN': 0., 'DEVICE': 0.}),
                        replay=ReplayTiming(q, r, init))
    assert st['unfinished'] == 0
    return (max(x[7] for x in rows) - min(x[5] for x in rows)) * .01


base = run()
print(f'BASE span {base:.3f} us (measured 4067.290)')


def rep(label, span):
    print(f'{label:58s} span {span:9.3f}  delta {span - base:+8.3f} us = {(span - base) / 1000:+.4f} ms')


# L1: coherent-load batching. Removable serial coherent round trips per task: topk 1 (two logit loads -> one),
# moe_up 1 (expert id || activation stage), moe_down 2 (id || stage || routing weight). Sweep the round trip; also each site alone.
for rtt in (0.30, 0.50, 0.69):
    for sites in (('topk',), ('moe_up', 'moe_down'), ('topk', 'moe_up', 'moe_down')):
        k = {'topk': 1, 'moe_up': 1, 'moe_down': 2}                     # serial coherent trips removable per task
        b = [max(0., x - k[role[t]] * rtt) if role[t] in sites else x for t, x in enumerate(body0)]
        rep(f'L1 rtt={rtt:.2f} sites={"+".join(sites)}', run(body=b))

# L2: top-k folded into moe_up only (moe_down keeps reading ids). topk bodies -> 0; moe_up's notification residual
# after topk -> local floor; moe_up body += fold chain F + hot-spot h - expert-id hop 0.6.
for F in (2.1, 2.4, 2.7):
    for h in (0.0, 0.5, 1.0):
        b = list(body0); r = list(r0)
        for t in range(N):
            if role[t] == 'topk': b[t] = 0.
            elif role[t] == 'moe_up':
                b[t] = body0[t] + F + h - 0.6
                if free[t]: r[t] = min(r0[t], 0.52)
        rep(f'L2 fold-up F={F} h={h}', run(body=b, r=r))

# L3: flat arrive (ETX_NO_LASTFLUSH=1) on the four coherent DEVICE events. Fan-in queue on the measured done stamps:
# hierarchical = per-XCD L2 atomic with return (rl) then one device atomic; flat = one FIFO device atomic per
# producer with service sv. Delta per event instance = hier - flat completion; consumers' r shrink by it, producers'
# q shrink by rl (no local return to wait for).
dev_roles = {'qkv', 'o_proj', 'router', 'moe_down'}
dom = [d['domain'] for d in p['descs']]
inst = [i for i, (n, c, pr) in enumerate(p['ev_inst']) if scope[n] == 'DEVICE' and pr and role[pr[0]] in dev_roles]
print(f'L3 instances {len(inst)}; producers per instance {statistics.median(len(p["ev_inst"][i][2]) for i in inst)}')
for rl in (0.2, 0.4):
    for sv in (0.010, 0.028, 0.050):
        deltas = {}
        for i in inst:
            pr = p['ev_inst'][i][2]
            at = sorted((raw[z][7] - origin) * .01 for z in pr)
            c = 0.
            for a in at: c = max(c, a) + sv
            flat = c
            byx = defaultdict(list)
            for z in pr: byx[dom[z]].append((raw[z][7] - origin) * .01)
            hier = max(max(v) + rl for v in byx.values()) + sv
            deltas[i] = hier - flat
        r = list(r0); q = list(q0)
        for i, dl in deltas.items():
            for t in cons[i]:
                if free[t]: r[t] = max(0., r[t] - dl)
            for z in p['ev_inst'][i][2]: q[z] = max(0., q0[z] - rl)
        ds = sorted(deltas.values())
        rep(f'L3 rl={rl} sv={sv*1000:.0f}ns med_d={statistics.median(ds):+.2f} min={ds[0]:+.2f}', run(q=q, r=r))

# L4: prep run by the last qkv arriver of each kv head. qkv tasks wait for a device atomic return (+A in q);
# prep's notification residual -> A; attn waits on a DEVICE event (+X on its r).
for A, X in ((0.7, 0.3), (0.9, 0.3), (1.2, 0.4)):
    r = list(r0); q = list(q0)
    for t in range(N):
        if role[t] == 'qkv': q[t] = q0[t] + A
        elif role[t] == 'prep' and free[t]: r[t] = A
        elif role[t] == 'attn' and free[t]: r[t] = r0[t] + X
    rep(f'L4 prep-in-qkv A={A} X={X}', run(q=q, r=r))

# L3 break-even and the arrival spread; L1 (moe_down site) + L2 combined.
dev_roles={'qkv','o_proj','router','moe_down'}
dom=[d['domain'] for d in p['descs']]
inst=[i for i,(n,c,pr) in enumerate(p['ev_inst']) if scope[n]=='DEVICE' and pr and role[pr[0]] in dev_roles]
# arrival spread of the last 32 producers per instance
spreads=[]
for i in inst:
    at=sorted((raw[z][7]-origin)*.01 for z in p['ev_inst'][i][2]); spreads.append((at[-1]-at[-32], at[-1]-at[0]))
print('L3 spread last32 median %.2f us, all-256 median %.2f us'%(statistics.median(a for a,b in spreads), statistics.median(b for a,b in spreads)))
for rl in (0.2,0.4):
  for sv in (0.002,0.004,0.006):
    deltas={}
    for i in inst:
        pr=p['ev_inst'][i][2]; at=sorted((raw[z][7]-origin)*.01 for z in pr); c=0.
        for a in at: c=max(c,a)+sv
        byx=defaultdict(list)
        for z in pr: byx[dom[z]].append((raw[z][7]-origin)*.01)
        deltas[i]=max(max(v)+rl for v in byx.values())+sv-c
    r=list(r0); q=list(q0)
    for i,dl in deltas.items():
        for t in cons[i]:
            if free[t]: r[t]=max(0.,r[t]-dl)
        for z in p['ev_inst'][i][2]: q[z]=max(0.,q0[z]-rl)
    rep(f'L3 rl={rl} sv={sv*1000:.0f}ns med_d={statistics.median(deltas.values()):+.2f}',run(q=q,r=r))
# L1+L2 combined: fold-up (F=2.4, h=0.5) plus the moe_down id||stage batching at rtt 0.5
for F,h,rtt in ((2.4,0.5,0.5),(2.1,0.0,0.69),(2.7,1.0,0.3)):
    b=list(body0); r=list(r0)
    for t in range(N):
        if role[t]=='topk': b[t]=0.
        elif role[t]=='moe_up':
            b[t]=body0[t]+F+h-0.6
            if free[t]: r[t]=min(r0[t],0.52)
        elif role[t]=='moe_down': b[t]=max(0.,body0[t]-2*rtt)
    rep(f'L1(down)+L2 F={F} h={h} rtt={rtt}',run(body=b,r=r))
