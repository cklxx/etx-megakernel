"""Name event/producer observations from an incomplete diagnostic snapshot.

No cause is inferred. No missing stamp proves why a task stopped, and no
snapshot is evidence that a synchronization mechanism can be removed.
Version-1 final-arriver records cannot identify all partially arrived tasks.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .inbody import read_raw, event_words


def dump(plan,raw):
    nt=len(plan['descs'])
    if len(raw['legacy'])!=nt:raise ValueError('plan/task count mismatch')
    words=event_words(plan);ledger=raw.get('producer_ledger')
    names={v['id']:name for name,v in plan['types'].items()}
    expected=defaultdict(Counter)
    for event,(_,coord,producers) in enumerate(plan['ev_inst']):
        for task in producers:expected[task][words[event]]+=1
    observed={}
    if ledger:
        offsets=ledger['offsets']
        if len(offsets)!=nt+1 or len(ledger['remaining'])!=len(raw['events']):raise ValueError('ledger shape mismatch')
        for task in range(nt):
            positions=range(offsets[task],offsets[task+1])
            if Counter(ledger['words'][i] for i in positions)!=expected[task]:raise ValueError('producer ledger differs from exact plan')
            for i in positions:
                before,after=ledger['stamps'][i]
                if (after and (not before or after<before)):raise ValueError('invalid arrival bracket')
                observed.setdefault((task,ledger['words'][i]),[]).append((before,after))
    event_meta={e['name']:e for e in plan['events']}
    events=[]
    for event,(name,coord,producers) in enumerate(plan['ev_inst']):
        word=words[event];remaining=ledger['remaining'][word] if ledger else None
        if remaining==0:continue
        records=[];seen=Counter()
        for task in producers:
            d=plan['descs'][task];stamps=raw['legacy'][task];taken,body,done,worker=stamps
            ty=names[d['type']]; nw=raw['waves_per_task']
            wave=raw['waves'][task*nw:(task+1)*nw]
            history=[('taken',taken),('poll_success',raw['poll'][task][0]),('body_entry',body),('post_drain',done)]
            history.extend((f'wave_{w}_{kind}',tick) for w,row in enumerate(wave) for kind,tick in zip(('return','drain'),row))
            pair=observed.get((task,word),[])
            occurrence=seen[task];seen[task]+=1
            before,after=pair[occurrence] if occurrence<len(pair) else (None,None)
            if before is not None:history.extend([('arrive_before',before),('arrive_after',after)])
            last=max(((label,tick) for label,tick in history if tick),key=lambda p:p[1],default=(None,None))
            status='arrival_complete_observed' if after else 'arrival_in_flight_observed' if before else 'arrival_not_observed' if ledger else 'unknown_no_producer_ledger'
            records.append(dict(task=task,occurrence=occurrence,type=ty,phase=re.sub(r'_\d+(?:_x)?$','',ty),coord=d['coord'],
                observed_worker=worker if taken else None,planned_worker=d.get('worker'),domain=d.get('domain'),
                arrival_status=status,arrival_bracket=[before,after],last_stamp=dict(kind=last[0],ticks=last[1]),cause=None))
        completed=sum(r['arrival_status']=='arrival_complete_observed' for r in records)
        meta=event_meta[name]
        runtime=meta.get('runtime_init',False)
        if runtime: initial=None
        elif 'counts' in meta: initial=meta['counts'][word-meta['offset']]
        else: initial=len(producers)
        if not runtime and initial!=len(producers):raise ValueError('initial count differs from static producer incidence count')
        events.append(dict(event=name,coord=coord,word=word,expected=initial,producer_incidence_count=len(producers),runtime_obligation_unknown=runtime,remaining=remaining,
            globally_accounted_arrivals=initial-remaining if initial is not None and remaining is not None and 0<=remaining<=initial else None,
            completed_calls_observed=completed if ledger else None,
            subcounters=[r[word] for r in ledger['sub']] if ledger else None,
            producers=records,unobserved_producer_tasks=sorted({r['task'] for r in records if r['arrival_status']=='arrival_not_observed'}) if not runtime else []))
    return dict(schema=1,events=events,raw=raw,cause=None,removability_evidence=False,
        limitation='non-atomic snapshot; diagnostic writes can lag arrival. Global count can batch domain arrivals; subcounters and per-call observations are separate. Missing stamp means not observed, never a cause.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);p.add_argument('raw',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=dump(json.loads(a.plan.read_text()),read_raw(a.raw))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for event in result['events']:
        print('WATCHDOG '+json.dumps({k:event[k] for k in ('event','coord','expected','remaining','completed_calls_observed','unobserved_producer_tasks')},sort_keys=True))


if __name__=='__main__':main()
