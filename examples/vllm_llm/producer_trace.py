"""Attribute a same-build task trace. No cross-build wall subtraction."""
import hashlib
import numpy as np


def summarize(plan, raw, candidate):
    if len(raw)%72:raise ValueError('truncated trace')
    data=np.frombuffer(raw,dtype='<i8')
    if not data.size or data.size%9:raise ValueError('truncated or empty trace')
    data=data.reshape(-1,9)
    if np.any(data[:,5]>data[:,6]) or np.any(data[:,6]>data[:,7]):raise ValueError('trace timestamps')
    def tasks(name):
        rows=data[data[:,0]==plan['types'][name]['id']]
        expected=int(np.prod(plan['types'][name]['grid']))
        if len(rows)!=expected:raise ValueError('trace coverage '+name)
        return rows
    layers=[]
    for layer in range(48):
        o=tasks(f'o_proj_{layer}');moe=tasks(f'moe_up_{layer}')
        start=int(o[:,6].min())
        if candidate:publish=int(o[:,7].max())
        else:publish=int(tasks(f'topk_{layer}_x')[:,7].max());tasks(f'router_{layer}')
        ready=int(moe[:,5].max())
        if ready<publish:raise ValueError('consumer ready precedes all required publication')
        layers.append(dict(layer=layer,o_start_to_all_outputs_published_us=(publish-start)*.01,
                           o_start_to_all_expert_tasks_ready_us=(ready-start)*.01,
                           o_body_mean_us=float(np.mean(o[:,7]-o[:,6])*.01),
                           publication_to_all_expert_ready_us=(ready-publish)*.01))
    return dict(trace_sha256=hashlib.sha256(raw).hexdigest(),candidate=candidate,layers=layers,
                basis='instrumented GPU task trace; endpoints share one clock',
                substage_costs='UNKNOWN: scoring/interference, counter/readback, scale, norm, top-k, fanout not separately stamped',
                use='attribution only; no gain prediction or wall-time transfer; trace overhead not calibrated')
