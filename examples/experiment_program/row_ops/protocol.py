"""Fail-closed CPU admission/qualification, separate from GPU observations."""
import statistics
import math


def admit_down(savings,controls):
    return (len(savings)==len(controls)==5 and all(math.isfinite(v) for v in savings+controls) and min(controls)>0 and min(savings)>0 and
            statistics.median(savings)>2*(max(savings)-min(savings)) and
            statistics.median(savings)>=max(.625,.02*statistics.median(controls)))


def qualify(records,tasks,original,probe,empty,base_us,probe_us,empty_us,exact):
    reasons=[]
    expected={(t,w) for t in range(tasks) for w in range(16)}
    got=[(r['task'],r['wave']) for r in records]
    if len(got)!=len(expected) or set(got)!=expected: reasons.append('incomplete/duplicate task-wave coverage')
    if any(r['begin']<=0 or r['end']<r['begin'] for r in records): reasons.append('invalid landmarks')
    if not original==probe==empty: reasons.append('native allocation mismatch')
    if not exact: reasons.append('changed output/intermediate bytes')
    if not base_us or len(base_us)!=len(probe_us) or len(base_us)!=len(empty_us): reasons.append('incomplete qualification population')
    elif not all(math.isfinite(v) and v>0 for v in base_us+probe_us+empty_us): reasons.append('invalid qualification time')
    elif max(abs(p-b) for b,p in zip(base_us,probe_us))>.5 or max(abs(e-b) for b,e in zip(base_us,empty_us))>.5: reasons.append('perturbation above 0.5us')
    last=max(records,key=lambda r:r['end']) if records else None
    return dict(qualified=not reasons,reasons=reasons,last_task_wave=[last['task'],last['wave']] if last else None,
                interpretation='exposure only; no optimization saving or removable-work claim')


def invocation_orders(arms,samples,seed=91011):
    import random
    rng=random.Random(seed);result=[]
    for sample in range(samples):
        order=list(range(arms));rng.shuffle(order);result.append(order)
    return result
