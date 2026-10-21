"""Parse complete raw native populations, never a selected mean-only summary."""
import math
import re
from ..protocol import require,pair_order,paired_decision


def parse_native(log,*,population,seed,diagnostic=False):
    checks=re.findall(r'^CHECK G01 case=(\d+) exact=1$',log,re.M)
    require(list(map(int,checks))==list(range(96)) and ('G01 DIAGNOSTIC COMPLETE all96 exact' if diagnostic else 'G01 COMPLETE all96 exact') in log,'incomplete96-case correctness')
    raw=re.findall(r'^SAMPLE G01 ordinal=(\d) arm=(\d) sample=(\d+) device_ns=([\d.]+) wall_ns=([\d.]+)$',log,re.M)
    expected=[(i,0 if ch=='A' else 1,j) for i,order in enumerate(pair_order(seed)) for ch in order for j in range(population)]
    require(len(re.findall(r'^CHECK G01 ',log,re.M))==len(checks),'malformed correctness row')
    require(len(re.findall(r'^SAMPLE G01 ',log,re.M))==len(raw),'malformed raw sample')
    require(population>0 and [(int(i),int(a),int(j)) for i,a,j,_,_ in raw]==expected,'incomplete/duplicate/reordered raw samples')
    require(all(math.isfinite(float(v)) and float(v)>0 for x in raw for v in x[3:]),'invalid raw timing')
    means=re.findall(r'^PAIR G01 ordinal=(\d) arm=(\d) device_ns=([\d.]+) wall_ns=([\d.]+) population=(\d+)$',log,re.M)
    require(len(means)==len(re.findall(r'^PAIR G01 ',log,re.M))==10,'incomplete/malformed paired means')
    samples=[]
    for i,order in enumerate(pair_order(seed)):
        pair=means[2*i:2*i+2]
        require([int(x[1]) for x in pair]==[0 if c=='A' else 1 for c in order] and all(int(x[0])==i and int(x[4])==population for x in pair),'wrong pair mean population/order')
        by={int(x[1]):x for x in pair}
        for arm in (0,1):
            own=[x for x in raw if int(x[0])==i and int(x[1])==arm]
            require(all(math.isclose(sum(float(x[k]) for x in own)/population,float(by[arm][k-1]),rel_tol=1e-9,abs_tol=1e-5) for k in (3,4)),'edited pair mean')
        samples.append(dict(ordinal=i,order=order,control_ns=float(by[0][2]),candidate_ns=float(by[1][2]),correct=True,instrumented=False))
    if diagnostic:
        return dict(samples=[dict(s,instrumented=True) for s in samples],admitted=False,status='UNQUALIFIED_DIAGNOSTIC',reason='matched native allocation and <=500ns perturbation not established; no exposure/optimization attribution')
    return dict(samples=samples,**paired_decision(samples,seed=seed,qwen_layer=True))
