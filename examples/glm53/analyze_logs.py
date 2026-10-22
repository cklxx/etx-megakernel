"""Analyze archived GLM logs only: no model imports, tests, compilation or GPU access.

Host step timing excludes reset/upload/completion checks/D2H. Trace tables contain
per-grid spans, not additive per-task waits; never sum wait columns as overhead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

STEP=re.compile(r'^GLM-MODEL-STEP pos=(\d+) batch=(\d+) ms=([0-9.]+)$',re.M)
TRACE=re.compile(r'^\s+(l\d+_\w+|embedding|head|argmax|feedback)\s+(\d+)\s+'
                 r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')


def stats(values):
    if not values:return None
    a=sorted(values)
    def percentile(q):
        x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
    return dict(n=len(a),median_ms=statistics.median(a),mean_ms=statistics.mean(a),
                p90_ms=percentile(.9),p99_ms=percentile(.99),min_ms=a[0],max_ms=a[-1])


def analyze(path,context=1024,batch=1,layers=45):
    raw=path.read_bytes();text=raw.decode();steps=[(int(p),int(b),float(t)) for p,b,t in STEP.findall(text)]
    expected=context+31
    completion=f'GLM-MODEL-EXECUTED batch={batch} layers={layers} generated_each=32; reference comparison required'
    complete=(text.count(completion)==1 and [(p,b) for p,b,t in steps]==[(p,batch) for p in range(expected)])
    result=dict(log=str(path),log_sha256=hashlib.sha256(raw).hexdigest(),complete=complete,
                observed_steps=len(steps),expected_steps=expected,
                qualification='UNVERIFIED: no independent full-model HF certificate',
                timing='host launch-to-all-streams-retired; reset/upload/checks/D2H excluded',
                resource_leak_warnings=re.findall(r'^.*(?:SharedSignalPool|Signals leaked).*$' ,text,re.M))
    if not complete:
        result['status']='NOT-COMPLETE';return result
    result.update(status='COMPLETED-CANDIDATE',cold_step_ms=steps[0][2],
                  all=stats([t for p,b,t in steps]),steady=stats([t for p,b,t in steps if p>0]),
                  early100=stats([t for p,b,t in steps if 1<=p<=100]),
                  last100=stats([t for p,b,t in steps if p>=expected-100]),
                  decode32=stats([t for p,b,t in steps if context-1<=p<context+31]))
    allocation=[]
    for rank,used,free in re.findall(r'^GLM-ALLOCATED rank=(\d+) used=(\d+) free=(\d+)$',text,re.M):
        allocation.append(dict(rank=int(rank),used_bytes=int(used),free_bytes=int(free)))
    result['allocations']=allocation
    trace=[];rank=None;position=None;device_spans={}
    for line in text.splitlines():
        marker=re.match(r'GLM-TRACE rank=(\d+) pos=(\d+) ',line)
        if marker:rank,position=map(int,marker.groups());continue
        span=re.match(r'phase trace \(us\): total span ([\d.]+);',line)
        if span and rank is not None:device_spans[(position,rank)]=float(span[1])
        match=TRACE.match(line)
        if rank is None or not match:continue
        name,n,wait,body,ready,last_ready,span=match.groups()
        trace.append(dict(rank=rank,position=position,grid=name,tasks=int(n),mean_wait_us=float(wait),
                          mean_body_us=float(body),first_ready_us=float(ready),span_us=float(span)))
    if trace:
        positions=sorted({r['position'] for r in trace});summaries=[]
        for pos in positions:
            rows=[r for r in trace if r['position']==pos]
            names=[r['grid'] for r in rows]
            if len(set(names))!=len(names):raise ValueError(f'duplicate named trace grids in {path} at {pos}')
            kinds={};sparse=0;dense=0
            # First three full-model FFNs are dense. This script's default is the
            # archived pinned full model, not a general stage-schema replacement.
            for r in rows:
                kind=re.sub(r'^l\d+_','',r['grid']);kinds[kind]=kinds.get(kind,0)+r['span_us']
                m=re.fullmatch(r'l(\d+)_(up|down)',r['grid'])
                if m:
                    if int(m[1])>=3:sparse+=r['span_us']
                    else:dense+=r['span_us']
            # Each rank prints first-ready relative to its OWN earliest taken
            # task. Without raw timestamps these origins cannot be aligned.
            # Use the corresponding measured host wall, not a merged timeline.
            total=next(t*1000 for p,b,t in steps if p==pos);ffn=dense+sparse
            ceilings=[]
            for degree in (2,4,8):
                ideal=total-ffn+ffn/degree
                ep_ideal=total-sparse+sparse/degree
                ceilings.append(dict(degree=degree,ideal_ffn_tp_us=ideal,ideal_tp_speedup=total/ideal,
                                     ideal_routed_ep_us=ep_ideal,ideal_ep_speedup=total/ep_ideal))
            summaries.append(dict(position=pos,table_precision_us=.1,host_wall_us=total,
                                  device_trace_spans_us={str(r):v for (p,r),v in device_spans.items() if p==pos},
                                  summed_grid_spans_us=sum(kinds.values()),
                                  host_minus_grid_spans_us=total-sum(kinds.values()),phase_spans_us=kinds,
                                  residual_caveat='Not pure kernel gaps: includes host launch/retirement; grid spans can include scheduling tails. Rank table time origins are not aligned.',
                                  sparse_up_down_us=sparse,dense_up_down_us=dense,
                                  ideal_ffn_only_scaling=ceilings,
                                  ceiling_caveat='Estimate from one early traced step; zero communication, balanced work, fixed non-FFN cost. EP scales routed sparse FFN only; shared fraction is not isolated and makes EP estimate optimistic. Not a decode prediction.'))
        result['traces']=summaries
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('logs',nargs='+',type=Path)
    ap.add_argument('--context',type=int,default=1024);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();result=dict(schema=1,runs=[analyze(p,a.context) for p in a.logs],
        repeat_claim='Independent single runs, not interleaved pairs; no significance inferred across widths.')
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['runs']:
        if not r['complete']:print(r['log'],'NOT-COMPLETE');continue
        print(f'{r["log"]}: all median={r["all"]["median_ms"]:.3f} ms; decode32 median={r["decode32"]["median_ms"]:.3f}, mean={r["decode32"]["mean_ms"]:.3f}; UNVERIFIED')
        for t in r.get('traces',[]):
            print(f'  pos={t["position"]}: host={t["host_wall_us"]:.1f} us, FFN={t["dense_up_down_us"]+t["sparse_up_down_us"]:.1f}, sparse={t["sparse_up_down_us"]:.1f}, unattributed={t["host_minus_grid_spans_us"]:.1f}')


if __name__=='__main__':main()
