"""Runnable G11 subgraph protocol, typed raw reader and six-case qualification.

CPU fixtures exercise these boundaries without becoming GPU observations.
Publication is an issue/forced-retirement bracket, never an exact atomic point.
"""
import argparse,json,math,struct,subprocess,shutil
from pathlib import Path
from .protocol import require,resolve_path,sha,verify_files
from .publication_subgraph import LAYERS,MODES,PHASES
MAGIC=0x314C425550313147
TASKS=512;WAVES=16;ROW_WORDS=25;SLOT_WORDS=8*(2+2*256)
SOURCE_NAMES=('publication_subgraph.py','publication_run.py','publication_capture.py','publication_host.cpp','publication_build.py')

def interface_sources():
    here=Path(__file__).parent
    return {str(here/n):sha(here/n) for n in SOURCE_NAMES}

def verify_interface(identity):
    i=identity['interface'];verify_files(i['sources']);require(i['sources']==interface_sources(),'integrated G11 implementation changed')
    verify_files(i['files']);require(i['host_layers']==list(LAYERS),'all actual representative-layer hosts required')
    require(all(x['exit']==0 for x in i['commands']) and set(i['completed'])=={'host-L1','host-L24','host-L47','capture-host-object','capture-host-link','capture-loader','host-L1-help','host-L24-help','host-L47-help'},'complete real interface build/link/help receipts required')
    for l in LAYERS:require(str(resolve_path(i['directory'])/f'host-L{l}') in i['files'],'selected host absent from seal')
    return i

def bind_trace(pack,meta,session_id):
    from .validation_trace import verify,compiled_contract
    t=verify(meta);require(t['session_id']==session_id and t['position']==pack['position'],'G11 requires same active G12 position/session')
    require(compiled_contract(pack['capture_build'])==t['compiled_contract'],'original capture is a different module/source/plan/flags')
    from .publication_subgraph import extract
    p=json.loads(resolve_path(t['plan_file']).read_text());context=extract(p,pack['layer'])
    require(context['consumers'] and context['producers'],'actual matching current chain')
    c=json.loads(resolve_path(pack['producer_certificate']['path']).read_text());model=resolve_path(c['model_dir'])
    require(all(t['model_inputs'].get(str(model/n))==sha(model/n) for n in ('vl.bin','vl.manifest','golden_c1024.txt')),'capture/G12 model bytes differ')
    return dict(proof=meta,session_id=session_id,position=pack['position'],current_context=context,separate_clock_populations=True)

def config(pack,identity,output,deadline):
    from .publication_capture import validate
    from .publication_subgraph import admission
    validate(pack);gate=admission(identity);require(gate['admitted'],'integrated local publication preparation not admitted')
    c=json.loads(resolve_path(identity).read_text());directory=resolve_path(identity).parent;hostdir=resolve_path(c['interface']['directory']);require(deadline>0 and math.isfinite(deadline),'finite native deadline')
    q=lambda v:json.dumps(str(v));buffers=pack['buffers'];pointers=pack['pointers'];refs=pack['references'];layer=pack['layer']
    # Initial live operands exclude down-produced slots and fold-written norm/output.
    # Every other range is preserved, even if it is also an output reference.
    external=[]
    for x in pointers:
        if x['offset'] in ((0,72,88) if x['block']==8 else (0,8,24,32,112)):
            b=buffers[x['buffer']];external.append((x['buffer'],x['add'],b['bytes']))
    lines=[f'{q(output)} {layer} 4096 {deadline} {len(buffers)} {len(pointers)} {len(refs)}']
    lines += [q(directory/(m+'.hsaco')) for m in MODES]
    lines += [q(a['file']) for a in pack['arguments']]
    lines += [f'{q(b["file"])} {b["offset"]} {b["bytes"]}' for b in buffers]
    lines += [f'{x["block"]} {x["offset"]} {x["buffer"]} {x["add"]}' for x in pointers]
    for r in refs:
        poison=not any(b==r['buffer'] and off<r['offset']+r['bytes'] and end>r['offset'] for b,off,end in external)
        lines.append(f'{r["buffer"]} {r["offset"]} {q(r["file"])} {r["bytes"]} {int(poison)}')
    return '\n'.join(lines)+'\n',[str(hostdir/f'host-L{layer}'),str(output)+'.config']

def read_raw(path,plan,selector,arm,*,coverage=False):
    data=resolve_path(path).read_bytes();nwords=7+512*4+512+3+512*16*3+16*25
    expected=nwords*8+SLOT_WORDS*4+512*4;require(len(data)==expected,'truncated/extra typed G11 raw')
    words=struct.unpack_from('<'+str(nwords)+'Q',data);header=words[:7]
    require(header[:2]==(MAGIC,1) and header[2] in LAYERS and header[3:5]==(selector,arm) and header[5]>0 and header[6]==coverage,'G11 raw context/clock/arm differs')
    at=7
    def take(n,w):
        nonlocal at
        rows=[words[j:j+w] for j in range(at,at+n*w,w)];at+=n*w;return rows
    legacy=take(512,4);poll=take(512,1);event=take(1,3)[0];waves=take(512*16,3);phase=take(16,25)
    slots=struct.unpack_from('<'+str(SLOT_WORDS)+'i',data,nwords*8);executions=struct.unpack_from('<512i',data,nwords*8+SLOT_WORDS*4)
    require(list(slots[:8])==[38]*8,'native XCD/worker population differs')
    if coverage or arm:
        worker_locations={}
        for t,d in enumerate(plan['descs']):
            keys=set();locations=[]
            for w in range(16):
                location=waves[t*16+w][2];require(location>>32==d['domain'],'task/wave physical XCD differs')
                keys.add((location>>8)&255);locations.append(location)
            require(len(keys)==1 and len(set(locations))==16,'one workgroup must retain16 distinct hardware waves on one CU')
            key=next(iter(keys));worker=d['worker'];require(worker not in worker_locations or worker_locations[worker]==key,'worker migrated between tasks');worker_locations[worker]=key
        for domain in range(8):
            keys=[key for worker,key in worker_locations.items() if worker//38==domain];require(len(keys)==len(set(keys)),'resident workers share physical CU')
    if coverage or arm:
        require(executions==(1,)*512,'all512 tasks must execute exactly once')
        for d,row in zip(plan['descs'],legacy):require(0<row[0]<=row[1]<=row[2] and row[3]==d['worker'],'incomplete actual task/worker trace')
        for queue in plan['static_queues'].values():
            for a,b in zip(queue,queue[1:]):require(legacy[a][2]<=legacy[b][0],'actual restricted queue order failed')
    if arm==0:return dict(layer=header[2],selector=selector,coverage=coverage,native_placement=True)
    begin,end,owner=event;require(0<begin<=end and owner in range(256),'missing original final-arrival owner/bounds')
    for t in range(512):
        for w in range(16):
            a,b,_=waves[t*16+w];require(legacy[t][1]<=a<=b<=legacy[t][2] and a>0,'all512/all16-wave drain coverage incomplete')
        if t>=256:require(legacy[t][0]<=poll[t][0]<=legacy[t][1] and begin<=poll[t][0] and max(legacy[k][2] for k in range(256))<=legacy[t][1],'consumer poll/publication dependency coverage failed')
    selected=selector%16;row=phase[selected];require(row[0]==5 and all(r[0]==0 for w,r in enumerate(phase) if w!=selected),'sparse selector population differs')
    target=256+selector//16;intervals={};last=0
    for i,name in enumerate(PHASES):
        a,b,location=row[1+3*i:4+3*i]
        if selected>=4 and i<4:intervals[name]=None;continue
        require(legacy[target][1]<=a<=b<=legacy[target][2] and a>0 and last<=a,'disjoint selected read/fold/norm landmarks missing/reordered')
        require(location==waves[target*16+selected][2] and location>>32==plan['descs'][target]['domain'],'selected physical XCD differs from actual worker domain');last=b
        intervals[name]=(b-a)*1000/header[5]
    final=max(range(256,512),key=lambda t:legacy[t][2])
    ready=row[14] # phase4 end = norm-ready
    require(ready>0,'norm-ready missing')
    return dict(layer=header[2],selector=selector,publication_bounds=[begin,end],owner=owner,consumer=target,wave=selected,phase_us=intervals,norm_ready_after_publication_us_bounds=[(ready-end)*1000/header[5],(ready-begin)*1000/header[5]],final_consumer=final,final_output_after_publication_us_bounds=[(legacy[final][2]-end)*1000/header[5],(legacy[final][2]-begin)*1000/header[5]],native_placement=True,clock_khz=header[5])

def summarize(prefix,pack,identity):
    prefix=str(prefix);c=json.loads(resolve_path(identity).read_text());plan=c['plans'][str(pack['layer'])]
    rows=[json.loads(line) for line in resolve_path(prefix+'.timing.jsonl').read_text().splitlines()]
    require([(r['selector'],r['arm']) for r in rows]==[(s,(s+j)%3) for s in range(4096) for j in range(3)],'complete4096-selector matched interleaving required')
    indexed={}
    for r in rows:
        require(r['exact'] is True and all(type(r[k]) in (float,int) and math.isfinite(r[k]) and r[k]>0 for k in ('device_us','host_wall_us')),'invalid native timing/exactness');indexed[r['selector'],r['arm']]=r
    expected=b''.join(resolve_path(r['file']).read_bytes() for r in pack['references']);selected=[]
    def item(s,a,coverage=False):
        base=prefix+f'.s{s}.a{a}'+('.coverage' if coverage else '')
        require(resolve_path(base+'.references').read_bytes()==expected,'raw outputs/intermediates differ from original')
        value=read_raw(base+'.raw',plan,s,a,coverage=coverage);require(value['layer']==pack['layer'],'raw layer differs');return value
    for arm in range(3):item(0,arm,True)
    for s in range(4096):
        for a in range(3):
            v=item(s,a)
            if a==1:selected.append(v)
    delta=max(abs(indexed[s,a]['device_us']-indexed[s,0]['device_us']) for s in range(4096) for a in (1,2))
    return dict(status='QUALIFIED_PUBLICATION_DIAGNOSTIC' if delta<=.5 else 'UNQUALIFIED_PERTURBATION',qualified=delta<=.5,perturbation_us=delta,threshold_us=.5,selectors=4096,selected=selected,performance_claim=False,exposure_is_removable_work=False,exact_atomic_point=False)

def configuration(path):
    c=json.loads(resolve_path(path).read_text());require(c['schema']=='etx.g11.program.v1','integrated G11 recipe required');verify_files(c['inputs']);identity=resolve_path(c['publication_preparation'])
    require(c['inputs'].get(str(identity))==sha(identity),'publication identity must be sealed')
    require(set(c['packs'])=={f'L{l}-p{p}' for l in LAYERS for p in (0,1040)},'six actual G11 cases required')
    for name,p in c['packs'].items():
        path=resolve_path(p);require(c['inputs'].get(str(path))==sha(path),'operand pack must be presealed')
        pack=json.loads(path.read_text());require(name==f'L{pack["layer"]}-p{pack["position"]}','G11 pack case differs')
    return c

def program(path,out,execute=False,trace_session=None):
    c=configuration(path);out=resolve_path(out);identity=c['publication_preparation'];report=dict(schema='etx.g11.report.v1',record='G11',configuration_sha256=sha(path),gpu_executed=False,admitted=False,cases=[])
    if execute:out.mkdir(exist_ok=False)
    traces={};session_id=None
    if execute:
        require(trace_session is not None,'G11 requires completed active G12 session')
        root=resolve_path(trace_session)
        if (root/'dispatch.json').exists():root=root/'native-session'
        active=json.loads((root/'session-config.json').read_text());from .validation_run import transcript
        transcript(active,root);require(active.get('fold_decision_active') is True,'G11 is opportunistic only with active fold decision');session_id=active['session_id'];traces={json.loads(resolve_path(m['proof']).read_text())['position']:m for m in active['fold_traces']};require(set(traces)=={0,1040},'both-position current G12 traces required')
    if execute:
        reserve=16<<20
        for pp in c['packs'].values():
            pack=json.loads(resolve_path(pp).read_text())
            per=(7+512*4+512+3+512*16*3+400)*8+SLOT_WORDS*4+512*4+sum(r['bytes'] for r in pack['references'])+(16<<10)
            reserve+=per*(4096*3+3)
        require(shutil.disk_usage(out).free>(2<<30)+reserve,'G11 complete six-case export budget plus2GiB floor unavailable')
    for name,pp in c['packs'].items():
        p=json.loads(resolve_path(pp).read_text());prefix=out/name;text,argv=config(p,identity,prefix,c.get('deadline_seconds',600));row=dict(id=name,pack=str(resolve_path(pp)),pack_sha256=sha(pp),command=argv)
        if execute:
            require(resolve_path(json.loads(resolve_path(traces[p['position']]['proof']).read_text())['session_directory'])==root,'G11 trace belongs to a different active session');row['trace_binding']=bind_trace(p,traces[p['position']],session_id)
            # Retain all raw evidence and reserve the complete declared write population.
            per=(7+512*4+512+3+512*16*3+400)*8+SLOT_WORDS*4+512*4+sum(r['bytes'] for r in p['references'])+(16<<10)
            require(shutil.disk_usage(out).free>(2<<30)+per*(4096*3+3),'G11 complete raw export budget plus2GiB floor unavailable')
            resolve_path(str(prefix)+'.config').write_text(text)
            try:
                with resolve_path(str(prefix)+'.log').open('x') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=c.get('deadline_seconds',600)+5)
                resolve_path(str(prefix)+'.exit').write_text(str(r.returncode)+'\n');require(r.returncode==0,'G11 native child failed; evidence retained')
            except subprocess.TimeoutExpired:resolve_path(str(prefix)+'.exit').write_text('TIMEOUT\n');raise
            row['verdict']=summarize(prefix,p,identity)
        else:row['status']='PREPARED_PENDING_NATIVE_G12_CAPTURE_AND_QUALIFICATION'
        report['cases'].append(row)
    report.update(status='PROGRAM_COMPLETE' if execute else 'PREPARED_CONDITIONAL',gpu_executed=execute,qualified=execute and all(r['verdict']['qualified'] for r in report['cases']))
    if execute:(out/'program-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def verify_report(path,stage):
    r=json.loads(resolve_path(path).read_text());cmd=stage['command'];recipe=resolve_path(cmd[cmd.index('--config')+1]);c=configuration(recipe)
    require(r['schema']=='etx.g11.report.v1' and r['gpu_executed'] and r['status']=='PROGRAM_COMPLETE' and r['configuration_sha256']==sha(recipe),'complete integrated G11 native report required')
    require([x['id'] for x in r['cases']]==list(c['packs']),'complete six-case diagnostic population')
    trace_root=resolve_path(cmd[cmd.index('--trace-session')+1])
    if (trace_root/'dispatch.json').exists():trace_root=trace_root/'native-session'
    active=json.loads((trace_root/'session-config.json').read_text())
    for row in r['cases']:
        p=json.loads(resolve_path(row['pack']).read_text());require(row['pack']==str(resolve_path(c['packs'][row['id']])) and row['pack_sha256']==sha(row['pack']),'selected operand pack differs');text,argv=config(p,c['publication_preparation'],resolve_path(path).parent/row['id'],c.get('deadline_seconds',600));require(row['command']==argv,'selected native command differs')
        require(resolve_path(json.loads(resolve_path(row['trace_binding']['proof']['proof']).read_text())['session_directory'])==trace_root,'G11 proof belongs to another trace session');require(row['trace_binding']==bind_trace(p,row['trace_binding']['proof'],active['session_id']),'active trace binding differs')
        prefix=resolve_path(path).parent/row['id'];require(resolve_path(str(prefix)+'.config').read_text()==text,'native configuration differs');require(resolve_path(str(prefix)+'.exit').read_text().strip()=='0','native G11 incomplete');require(summarize(prefix,p,c['publication_preparation'])==row['verdict'],'G11 raw qualification changed')
    require(r['qualified']==all(row['verdict']['qualified'] for row in r['cases']),'aggregate G11 qualification differs')
    return dict(record='G11',complete=True,admitted=False,qualified=r['qualified'],decision_sha256=sha(path),performance_claim=False)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sp=parser.add_subparsers(dest='action',required=True)
    p=sp.add_parser('capture-source');p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sp.add_parser('capture');p.add_argument('--identity',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--layer',type=int,required=True);p.add_argument('--position',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p=sp.add_parser('seal-capture');p.add_argument('--certificate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sp.add_parser('program');p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--trace-session',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=parser.parse_args(argv)
    if a.action=='capture-source':
        from .publication_capture import checking_source
        require(not a.out.exists(),'preserve source output');a.out.write_text(checking_source(a.original.read_text()));result=dict(source=str(a.out),gpu_executed=False)
    elif a.action=='capture':
        from .publication_capture import job
        result=job(a.identity,a.model,a.layer,a.position,a.output,execute=a.execute)
    elif a.action=='seal-capture':
        from .publication_capture import assemble
        require(not a.out.exists(),'preserve capture pack');result=assemble(a.certificate,a.out)
    else:result=program(a.config,a.output,a.execute,a.trace_session)
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
