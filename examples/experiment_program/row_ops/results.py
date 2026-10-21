"""Read every raw invocation and reject incomplete/changed qualification evidence."""
import hashlib,json,math,statistics,struct
from pathlib import Path
from examples.experiment_program.protocol import resolve_path
from .protocol import qualify,admit_down
from .export_contract import intermediate_extent
RECORD=struct.Struct('<7Q4I')


def summarize(prefix,pack,identity,variant):
    prefix=str(prefix);items=[json.loads(l) for l in resolve_path(prefix+'.timing.jsonl').read_text().splitlines()]
    arms=3 if variant=='g11_probe' else 2;population=pack['count']*16 if arms==3 else 5
    if len(items)!=arms*population or {(x['sample'],x['arm']) for x in items}!={(s,a) for s in range(population) for a in range(arms)}:raise ValueError('incomplete declared invocation population')
    from .protocol import invocation_orders
    if [(x['sample'],x['arm']) for x in items]!=[(s,a) for s,order in enumerate(invocation_orders(arms,population)) for a in order]:raise ValueError('native invocation order differs from declared seeded resident replay')
    if not all(x['exact'] for x in items):raise ValueError('changed output bytes')
    if any(type(x.get(key)) not in (int,float) or not math.isfinite(x[key]) or x[key]<=0 for x in items for key in ('device_us','host_wall_us')):raise ValueError('invalid native timing endpoint')
    raw_files={}
    def read(path):
        path=resolve_path(path);data=path.read_bytes();raw_files[str(path)]=hashlib.sha256(data).hexdigest();return data
    goldens=[resolve_path(reference['file']).read_bytes() for reference in pack['references']]
    for sample in range(population):
        for arm in range(arms):
            for r,golden in enumerate(goldens):
                if read(prefix+f'.s{sample}.a{arm}.r{r}.bin')!=golden:raise ValueError('retained raw output export differs from sealed original')
    original=None
    for arm in range(arms):
        data=read(prefix+f'.a{arm}.intermediates')
        if len(data)!=intermediate_extent(pack):raise ValueError('incomplete native intermediate export')
        if original is None:original=data
        elif data!=original:raise ValueError('retained candidate intermediates differ from original control')
    indexed={(x['sample'],x['arm']):x for x in items}
    if arms==2:
        result={}
        for basis in ('device_us','host_wall_us'):
            controls=[indexed[s,0][basis] for s in range(5)];savings=[indexed[s,0][basis]-indexed[s,1][basis] for s in range(5)]
            result[basis]=dict(savings_us=savings,control_us=controls,admitted=admit_down(savings,controls),median_us=statistics.median(savings),range_us=max(savings)-min(savings))
        return dict(status='MICRO_PASS' if result['device_us']['admitted'] else 'MICRO_MISS',exact=True,measurements=result,raw_evidence=raw_files,full_model='NOT_RUN')
    records=[];selected=[]
    for sample in range(population):
        raw=read(prefix+f'.s{sample}.a1.records')
        if len(raw)!=pack['count']*16*RECORD.size:raise ValueError('incomplete native task/wave records')
        rows=list(RECORD.iter_unpack(raw));observed=[]
        for i,r in enumerate(rows):
            if (r[7],r[8])!=(i//16,i%16) or not 0<r[0]<=r[5]<=r[6] or not 0<=r[9]<8:raise ValueError('task/wave completion coverage')
            observed.append(dict(task=r[7],wave=r[8],begin=r[0],end=r[6]))
        r=rows[sample]
        if not 0<r[1]<=r[3]<=r[4]<=r[5]:raise ValueError('missing selected fold/norm/publication landmarks')
        if sample%16<4 and not r[1]<=r[2]<=r[3]:raise ValueError('active wave residual-read marker missing')
        last=max(observed,key=lambda q:q['end'])
        selected.append(dict(sample=sample,task=r[7],wave=r[8],stamps=list(r[:7]),load_wave=sample%16<4,last_task_wave=[last['task'],last['wave']],selected_is_last=(r[7],r[8])==(last['task'],last['wave'])))
        records=observed
    def resource(name):
        r=next(r for r in identity['resources'][name] if r['kernel']=='row_dsl_gemv_fold')
        if identity.get('_directory'):
            from .elf import descriptors,allocation_signature
            return allocation_signature(descriptors(resolve_path(identity['_directory'])/(name+'.hsaco'))['row_dsl_gemv_fold'],r['agpr'])
        return r.get('physical_signature',r)
    q=qualify(records,pack['count'],resource('g11_control'),resource('g11_probe'),resource('g11_empty'),[indexed[s,0]['device_us'] for s in range(population)],[indexed[s,1]['device_us'] for s in range(population)],[indexed[s,2]['device_us'] for s in range(population)],True)
    from examples.experiment_program.validation_trace import bind_g11
    binding=bind_g11(pack,pack['matching_trace']) if pack.get('matching_trace') else None
    return dict(status='QUALIFIED_CONSUMER_EXPOSURE' if q['qualified'] and binding else 'UNQUALIFIED',qualification=q,observations=selected,raw_evidence=raw_files,trace_binding=binding,full_chain_status='UNAVAILABLE_PUBLICATION_TIMESTAMP',full_chain_interval_ns=None,optimization_saving=None)
