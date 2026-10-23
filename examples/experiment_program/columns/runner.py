"""Resident native run preparation and strict result qualification."""
from __future__ import annotations
import hashlib, json, math, os, struct, subprocess, time
from pathlib import Path
from statistics import median
from examples.experiment_program.protocol import resolve_path,verify_files
from .protocol import ProtocolError, require, sha256, validate_capture, stable_groups

def require_finite_golden(data,element):
    require(element in (2,4) and len(data)%element==0,'invalid golden byte extent')
    bits=struct.iter_unpack('<I' if element==4 else '<H',data)
    require(all((v[0] & (0x7f800000 if element==4 else 0x7f80)) != (0x7f800000 if element==4 else 0x7f80) for v in bits),'nonfinite golden invalidates NaN poison/correctness gate')

def admit_native(c, build):
    validate_capture(c)
    require(c.get('decode_position') in ((1024,1040) if c['family']=='fleet' else (0,1040)), 'decode position must be explicitly source-bound')
    build=resolve_path(build)
    require((build/'SHA256SUMS').is_file(), 'compiled build identity missing')
    identity=json.loads((build/'identity.json').read_text())
    verify_files(identity['files'])
    require(identity.get('local_device_compile_complete') is True,'candidate device compilation incomplete')
    sources=json.loads((build/'sources.json').read_text())
    require({'fleet','b5eaa67','ff9ef92'} <= sources.keys(), 'incomplete native arithmetic source identity')
    source=resolve_path(c.get('source_path','/nonexistent'))
    require(source.is_file() and sha256(source)==c['source_sha256'], 'sealed native source missing/mismatch')
    expected=sources['fleet']['gemv_sha256'] if c['family']=='fleet' else sources['ff9ef92' if c['record']=='G13' else 'b5eaa67']['sha256']
    require(c['source_sha256']==expected,'native arithmetic identity differs from the source actually compiled')
    cert_path=resolve_path(c.get('native_capture_certificate','/nonexistent'))
    require(cert_path.is_file(), 'producer-bound native capture certificate missing')
    cert=json.loads(cert_path.read_text())
    for key in ('source_revision','source_sha256','decode_position','per_column_dag'):
        require(cert.get(key)==c[key], 'capture certificate mismatch: '+key)
    require(cert.get('payload_sha256')==c['payload']['sha256'], 'capture producer/payload mismatch')
    require(cert.get('all_task_ids_once') is True and cert.get('all_writing_waves_drained') is True,
            'missing native complete task/wave coverage')
    require(cert.get('placement_qualified') is True, 'actual worker/XCD placement unqualified')
    from .capture_pack import verify_seal,decode_rows,replay_rows,task_identity,request_ready
    producer=resolve_path(c.get('native_producer_seal','/nonexistent'))
    require(producer.is_file() and sha256(producer)==c.get('native_producer_sha256')==cert.get('producer_seal_sha256'),'native producer seal absent/changed')
    seal=verify_seal(json.loads(producer.read_text()))
    require(all(seal.get(k)==c.get(k) for k in ('record','batch','role','decode_position')),'native producer target identity mismatch')
    raw=[row for name in seal['raw_files'] for row in decode_rows(resolve_path(name).read_bytes(),family=c['family'])]
    require(len(raw)==seal['native_task_count']==cert.get('native_task_count'),'native producer task coverage changed')
    positions={str(x):sorted({r['h'][40] for r in raw if r['h'][7]==x}) for x in range(8)}
    require(all(len(v)==1 for v in positions.values()) and len({v[0] for v in positions.values()})==8 and cert.get('physical_xcds')==positions,'complete native eight-XCD provenance changed')
    if c['record']=='G02' and c['role']!='qkv':require(c.get('native_xcd')==seal.get('shared_xcd'),'selected native role/XCD differs from producer')
    selected=replay_rows(raw,seal)
    require(cert.get('selected_task_count')==len(selected) and cert.get('selected_task_ids')==[task_identity(r,c['family']) for r in selected],'selected task identities/coverage changed')
    require(cert.get('selected_role')==c['role'] and cert.get('selected_native_xcd')==c.get('native_xcd'),'selected role/XCD differs')
    ready=request_ready(raw,seal)
    require(c['arrivals_ticks']==[t-min(ready) for t in ready],'original request readiness changed')

    require(c['K'] <= 2048 and c['K'] > 256, 'shared service supports native 512..2048 K only')
    require(c['role']!='gate_up' or c['K']==2048, 'native gate/up requires K2048')
    require(c['record'] not in ('G03','G13') or c['batch']==2, 'historical subgraphs are B2')
    if c['record']=='G03':
        require((c['N'],c['K'])==(5120,2048), 'historical QKV shape mismatch')
    if c['record']=='G13':
        require((c['N'],c['K'],c.get('slots'))==(2048,2048,8), 'historical expert body/task shape mismatch')
        require(c.get('historical_identity')=='NEW_CAPTURE', 'new inputs cannot recreate missing historical native identity')
    require(c['ld']>=c['K'] and c['ld']%8==0, 'invalid native leading dimension')
    if c['record']=='G02' and c['role']=='qkv':
        require((c['N'],c['K'],c['ld'])==(3648,2048,2048) and c.get('kva_layout')=='K1_SHARED', 'frozen QKV Q3072/KVA576 K1 layout required')
    segments={s['name']:s for s in c['segments']}
    require({'weights','inputs','scales','golden'} <= segments.keys(), 'missing captured native scales/golden')
    with resolve_path(c['payload']['path']).open('rb') as stream:
        stream.seek(segments['golden']['offset']);golden=stream.read(segments['golden']['bytes'])
    require(hashlib.sha256(golden).hexdigest()==c['golden_sha256']==cert.get('golden_sha256'), 'producer golden bytes mismatch')
    columns=c['batch']*(c.get('slots',1) if c['record'] in ('G05','G13') else 1)
    element=2 if c['family'].startswith('dsl') or c['role']=='down' else 4
    require(len(golden)==columns*c['N']*element, 'golden native output shape mismatch')
    require_finite_golden(golden,element)
    if c['record']=='G13':
        with resolve_path(c['payload']['path']).open('rb') as stream:
            stream.seek(segments['golden_up']['offset']);up_golden=stream.read(segments['golden_up']['bytes'])
        require(len(up_golden)==c['batch']*8*768*2,'native expert-up golden extent differs')
        require_finite_golden(up_golden,2)
    if c['record']=='G05':
        require(c.get('input_layout')=='request-slot-K', 'group gather requires real per-slot operands')
        require(segments['inputs']['bytes']==columns*c['K']*4, 'group native input shape mismatch')
        require(segments['weights']['bytes'] >= ((c['expert_count']-1)*c['expert_stride']+c['N']*c['ld']*(2 if c['role']=='gate_up' else 1))*2, 'expert weight extent mismatch')
        require(max(v for row in c['picks'] for v in row)<c['expert_count'], 'expert ID outside weight capture')
    if c['record']=='G13':
        require({'ids','down_weights','golden_up'}<=segments.keys(), 'missing exact expert IDs/down weights/up output')
        require(c.get('expert_stride')==2*768*2048 and c.get('down_stride')==2048*768, 'historical expert weight stride mismatch')
    # Binary/source identity belongs to this build, never a relabeled old object.
    for line in (build/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        relative=name.removeprefix('/out/')
        path=build/relative
        require(path.resolve().is_relative_to(build.resolve()),'build seal path traversal')
        require(path.is_file() and sha256(path)==digest, 'build artifact changed: '+str(path))
    return segments,golden

def write_launch(c,build,out,regime,*,warmup=3,seed=20261016,g13_order="request"):
    segments,golden=admit_native(c,build)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    family=c['family'];module=('native-ready-final/'+native_variant(c)+'/micro') if family=='fleet' else ('g13-ready-final/g13' if family=='dsl-expert' else 'g03-ready-final/g03')
    require(family!='fleet' or identity_resource_admitted(build,[c]),'CLOSED_RESOURCE: native fixed grid cannot reside or new spills exceed the gate')
    role={'qkv':0,'gate_up':1,'down':2,'expert_up_down':3}[c['role']]
    slots=c.get('slots',1);columns=c['batch']*(slots if c['record'] in ('G05','G13') else 1)
    values=[str(resolve_path(build)/(module+'.hsaco')),str(resolve_path(c['payload']['path'])),str(out.resolve()),c['record'],c['N'],c['K'],c['ld'],c['batch'],role,columns,slots,c.get('expert_count',1),c.get('expert_stride',0)]
    values += [segments.get(n,{'offset':0})['offset'] for n in ('weights','inputs','scales','ids','down_weights','golden','golden_up')]
    values += [c.get('down_stride',0),warmup,5,seed,0]
    values += [0]*c['batch'] if regime=='simultaneous' else c['arrivals_ticks']
    if c['record']=='G05': values += [v for row in c['picks'] for v in row]
    if True:
        cert=json.loads(resolve_path(c['native_capture_certificate']).read_text())
        values += [c.get('native_xcd',6)]+[cert['physical_xcds'][str(x)][0] for x in range(8)]+[int(g13_order=='both-up-first')]
    require(all(not any(ch.isspace() for ch in str(v)) for v in values), 'native launch paths cannot contain whitespace')
    (out/'launch.txt').write_text('\n'.join(map(str,values))+'\n')
    (out/'golden.bin').write_bytes(golden)
    (out/'capture.json').write_text(json.dumps(c,indent=2)+'\n')
    return [str(resolve_path(build)/('native-ready-final/'+native_variant(c)+'/columns-run' if family=='fleet' else 'g13-ready-run' if c['record']=='G13' else 'g03-ready-run')),'--execute',str((out/'launch.txt').resolve())]

def decide_pairs(rows):
    require(len(rows)==10, 'five complete native pairs required')
    pairs={}
    for index,row in enumerate(rows):
        require(row.get('ordinal')==index//2, 'native pair population reordered/truncated')
        expected=('control','candidate') if (index//2)%2==0 else ('candidate','control')
        require(row.get('arm')==expected[index%2], 'AB/BA order differs from declared protocol')
        require(row.get('pair') in range(5) and row.get('ordinal') in range(5), 'unknown pair ordinal')
        arm=row.get('arm');require(arm in ('control','candidate'), 'unknown arm')
        group=pairs.setdefault(row['pair'],{});require(arm not in group, 'duplicate paired sample');group[arm]=row
        for key in ('host_ms','device_ms'):
            require(isinstance(row.get(key),(int,float)) and math.isfinite(row[key]) and row[key]>0, 'invalid timing')
        endpoints=row.get('request_ms',[]);require(endpoints, 'missing request endpoints')
        require(len({p['request'] for p in endpoints})==len(endpoints), 'duplicate request endpoint')
        for p in endpoints:
            require(all(math.isfinite(p.get(k,float('nan'))) for k in ('ready','done','latency'))
                    and 0<=p['ready']<=p['done'] and p['latency']>=0
                    and abs((p['done']-p['ready'])-p['latency'])<0.001, 'invalid request timing/endpoint')
    require(len(pairs)==5 and all(set(p)=={'control','candidate'} for p in pairs.values()), 'incomplete native pair')
    require(len({p['control']['ordinal'] for p in pairs.values()})==5
            and all(p['control']['ordinal']==p['candidate']['ordinal'] for p in pairs.values()), 'inconsistent pair ordinal')
    savings=[p['control']['host_ms']-p['candidate']['host_ms'] for p in pairs.values()]
    control=median(p['control']['host_ms'] for p in pairs.values());spread=max(savings)-min(savings)
    passed=all(v>0 for v in savings) and median(savings)>2*spread and median(savings)>=.02*control
    return {'verdict':'MICRO_PASS' if passed else 'MICRO_MISS','host_savings_ms':savings,
            'median_ms':median(savings),'spread_ms':spread,'model_admitted':False,
            'traffic_cause':'UNKNOWN_NO_QUALIFIED_COUNTERS'}

def run_native(command,out,*,timeout=120):
    out=Path(out);begin=time.monotonic()
    (out/'command.json').write_text(json.dumps(command)+'\n')
    with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        try:
            from examples.experiment_program.telemetry import capture
            with capture(out/'telemetry',dict(command=command,capture_sha256=sha256(out/'capture.json'))):
                result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=timeout)
        except subprocess.TimeoutExpired:
            (out/'exit.json').write_text(json.dumps({'exit':'TIMEOUT','seconds':time.monotonic()-begin}));raise ProtocolError('native deadline exhausted; no result admitted')
    (out/'exit.json').write_text(json.dumps({'exit':result.returncode,'seconds':time.monotonic()-begin})+'\n')
    require(result.returncode==0,'native correctness/liveness failure; no performance admission')
    for name in ('control.bin','candidate.bin'):
        require((out/name).read_bytes()==(out/'golden.bin').read_bytes(),'native output differs from sealed producer golden')
    rows=[json.loads(line) for line in (out/'stdout.jsonl').read_text().splitlines()]
    captured=json.loads((out/'capture.json').read_text());build=Path(command[0]).parent.parent.parent if captured['family']=='fleet' else Path(command[0]).parent
    verify_native_probe(out,captured,build)
    decision=qualify_rows(rows,captured)
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    return decision

def verify_dependency(path,*,needed=()):
    """Verify actual capture-only G04 or all four raw G02 readiness populations."""
    path=resolve_path(path);r=json.loads(path.read_text())
    if r.get('schema')=='etx.columns.program-report.v1':
        require(r.get('record')=='G02','only actual QKV winner program can be a diagnostic dependency')
        child=r.get('qkv_winners',{}).get('qkv-B2',{});p=resolve_path(child.get('path','/nonexistent'))
        require(p.is_file() and sha256(p)==child.get('sha256'),'program QKV raw report absent/changed')
        return verify_dependency(p,needed=needed)
    if r.get('record')=='G04':
        require(r.get('complete') is True,'G04 capture is incomplete')
        from etx.experiments.sealed_routing import decode_records,gates,ROLES
        from examples.experiment_program.protocol import verify_files
        require(set(r.get('roles',{}))==ROLES,'G04 complete source/input roles absent')
        for files in r['roles'].values():verify_files(files)
        files=r.get('raw_files',{});require(files,'G04 raw capture seal absent')
        for name,digest in files.items():require(resolve_path(name).is_file() and sha256(name)==digest,'G04 raw capture changed')
        log=path.parent/'capture.log';require(log.is_file() and sha256(log)==r.get('log_sha256'),'G04 capture log changed')
        batch=r.get('requests');gates(log.read_text(),batch)
        ordered=[path.parent/f'routes_r{request}.bin' for request in range(batch)]
        require({str(p) for p in ordered}==set(files),'G04 unexpected/missing request export')
        actual=json.loads(json.dumps(decode_records([p.read_bytes() for p in ordered],batch=batch)))
        require(all(r.get(key)==value for key,value in actual.items()),'G04 decoded capture differs from observation seal')
        return {'record':'G04','complete':True,'batch':batch,'decision_sha256':sha256(path)}
    require(r.get('record')=='G02','only a qualified native G02/G04 dependency is accepted')
    return admit_model_winner(path,Path(r.get('build','/nonexistent')))

def admit_model_winner(report_path,build):
    """Re-read both positions x both readiness regimes before any model expansion."""
    report_path=resolve_path(report_path);build=resolve_path(build);r=json.loads(report_path.read_text())
    if r.get('schema')=='etx.columns.program-report.v1':
        return admit_program(report_path,build)
    require(r.get('record') in ('G02','G05'),'not a column optimization record')
    if r.get('record')=='G05' and r.get('role')=='gate_up_down':
        require(set(r.get('role_reports',{}))=={'gate_up','down'},'both routed raw role reports required')
        roles={}
        for role,entry in r['role_reports'].items():
            path=resolve_path(entry['path'])
            require(path.is_file() and sha256(path)==entry['sha256'],'paired role report changed')
            roles[role]=admit_model_winner(path,build)
            require(roles[role]['role']==role and roles[role]['batch']==r.get('batch'),'paired role/batch mismatch')
        require(r.get('matched_up_down') is True and r.get('admitted') is True,'paired role bundle not admitted')
        require(r.get('build_identity_sha256')==sha256(build/'identity.json'),'paired build identity mismatch')
        return dict(record='G05',batch=r['batch'],role='gate_up_down',status='MICRO_PASS_BOTH_REGIMES',
                    admitted=True,native_exact=True,resource_admitted=True,native_pick_coverage=True,
                    singleton_fallback_passed=True,matched_up_down=True,decision_sha256=sha256(report_path),
                    micro_build_sha256=sha256(build/'identity.json'))

    require(r.get('status')=='MICRO_PASS_BOTH_REGIMES','both readiness regimes did not pass')
    require((build/'identity.json').is_file(),'frozen candidate build identity absent')
    require(r.get('build_identity_sha256')==sha256(build/'identity.json'),'aggregate build identity mismatch')
    decisions=[];captures=[]
    for position in (1024,1040):
        for regime in ('simultaneous','archived'):
            out=report_path.parent/f'p{position}-{regime}'
            c=json.loads((out/'capture.json').read_text());admit_native(c,build);captures.append(c)
            require(c['record']==r['record'] and c['decode_position']==position and c['batch']==r.get('batch') and c['role']==r.get('role'),'aggregate capture identity mismatch')
            require(json.loads((out/'exit.json').read_text()).get('exit')==0,'incomplete native child process')
            from examples.experiment_program.telemetry import verify as telemetry_verify
            telemetry_verify(out/'telemetry',dict(command=json.loads((out/'command.json').read_text()),capture_sha256=sha256(out/'capture.json')))
            actual=qualify_rows([json.loads(line) for line in (out/'stdout.jsonl').read_text().splitlines()],c)
            verify_native_probe(out,c,build)
            require(actual==json.loads((out/'decision.json').read_text()) and actual['verdict']=='MICRO_PASS','edited/incomplete/nonwinning native pairs')
            for name in ('control.bin','candidate.bin'):
                require((out/name).read_bytes()==(out/'golden.bin').read_bytes(),'native original/candidate golden mismatch')
            decisions.append({'position':position,'regime':regime,**actual})
    # The CLI order is capture order; compare by explicit identity, never silently
    # trim a miss or choose one decode position/readiness population.
    require(sorted(decisions,key=lambda d:(d['position'],d['regime']))==sorted(r.get('decisions',[]),key=lambda d:(d['position'],d['regime'])),'aggregate raw decision population mismatch')
    require(identity_resource_admitted(build,captures),'candidate target resource identity not admitted')
    require(r.get('native_exact') is True and r.get('resource_admitted') is True,'native exact/resource gate absent')
    if r['record']=='G05':
        require(r.get('native_pick_coverage') is True and r.get('singleton_fallback_passed') is True,'real-mix/native picks/singleton fallback gate absent')
    require(r.get('admitted') is True,'column aggregate not admitted')
    return dict(record=r['record'],batch=r['batch'],role=r['role'],status=r['status'],admitted=True,
                native_exact=True,resource_admitted=True,native_pick_coverage=r.get('native_pick_coverage',False),
                singleton_fallback_passed=r.get('singleton_fallback_passed',False),decision_sha256=sha256(report_path),
                micro_build_sha256=sha256(build/'identity.json'))


def native_variant(c):
    kind='group' if c['record']=='G05' else 'qkv' if c['role']=='qkv' else 'shared'
    return f"native-B{c['batch']}-{kind}"+(f"-x{c.get('native_xcd',6)}" if kind=='shared' else '')


def identity_resource_admitted(build,captures):
    value=json.loads((resolve_path(build)/'identity.json').read_text())
    for capture in captures:
        if capture['family']=='fleet':
            entry=value.get('native_resources',{}).get(native_variant(capture),{})
            if entry.get('admitted') is not True:return False
        else:
            module='g13' if capture['record']=='G13' else 'b5eaa67'
            if not value.get('resources',{}).get(module):return False
    return True


def qualify_rows(rows,c):
    decision=decide_pairs(rows)
    if c['record']=='G05':
        early=[r for r,t in enumerate(c['arrivals_ticks']) if t<max(c['arrivals_ticks'])]
        pairs={}
        for row in rows:pairs.setdefault(row['pair'],{})[row['arm']]=row
        held=[]
        for pair,arms in pairs.items():
            for request in early:
                control=next(p for p in arms['control']['request_ms'] if p['request']==request)
                candidate=next(p for p in arms['candidate']['request_ms'] if p['request']==request)
                if candidate['latency']>control['latency']+1e-6:held.append(dict(pair=pair,request=request))
        decision.update(early_request_not_held=not held,early_requests_held=held)
        if held:decision['verdict']='MICRO_MISS'
    return decision


def verify_native_probe(out,c,build):
    out=Path(out);fleet=c['family']=='fleet'
    directory=resolve_path(build)/('native-ready-final/'+native_variant(c) if fleet else 'g03-ready-final' if c['record']=='G03' else 'g13-ready-final')
    plan=json.loads((directory/'plan.json').read_text());cert=json.loads(resolve_path(c['native_capture_certificate']).read_text())
    types={value['id']:name for name,value in plan['types'].items()}
    for arm in ('control','candidate'):
        counts=(out/(arm+'.counts.bin')).read_bytes()
        require(len(counts)==len(plan['descs'])*4 and all(v[0]==1 for v in struct.iter_unpack('<i',counts)),'native ETX actual task coverage incomplete')
        trace=(out/(arm+'.trace.bin')).read_bytes();require(len(trace)==len(plan['descs'])*32,'native ETX complete task trace absent')
        placement=(out/(arm+'.placement.bin')).read_bytes();require(len(placement)==(c['batch']*297 if fleet else 898)*8,'native ETX placement extent differs')
        physical=struct.unpack('<'+'Q'*(len(placement)//8),placement)
        for d in plan['descs']:
            name=types[d['type']];r=int(name.split('_r')[1].split('_')[0])
            if name.startswith('ready'):index=(c['batch']*296 if fleet else 896)+r
            elif fleet:
                x,w=d['coord'];index=r*296+x*37+w if c['role']=='qkv' else (r*(6 if c['record']=='G05' else 1)+x)*37+w
            elif c['record']=='G03':index=r*256+int(name.split('_x')[1])*32+d['coord'][0]
            else:
                x,w=d['coord'];up=name.startswith('up');index=r*448+(0 if up else 192)+x*(24 if up else 32)+w
            word=physical[index];domain=d['domain']
            require(word>>32==cert['physical_xcds'][str(domain)][0] and (word&0xffffffff)==d['worker'],'actual trial worker/XCC differs from native fixed queue')


def admit_program(path,build):
    path=resolve_path(path);r=json.loads(path.read_text())
    from .program import registered
    require(r.get('schema')=='etx.columns.program-report.v1' and r.get('record') in ('G02','G05'),'not a registered optimization program')
    require(r.get('status')=='PROGRAM_COMPLETE','column program incomplete/failed/blocked')
    expected=registered(r['record']);require([c['id'] for c in r.get('cases',[])]==[c['id'] for c in expected],'program case population truncated/changed')
    for case in r['cases']:
        require(case.get('exit',0)==0 and case.get('status') not in ('CASE_FAILED','REFUSED','BLOCKED_NATIVE_CAPTURE'),'native failure cannot become a nonwinning case')
        if 'report' in case:
            child=resolve_path(case['report']);require(child.is_file() and sha256(child)==case['report_sha256'],'program raw child changed')
            for position in (1024,1040):
                for regime in ('simultaneous','archived'):
                    loc=child.parent/f'p{position}-{regime}'
                    require(json.loads((loc/'exit.json').read_text()).get('exit')==0,'program child contains native correctness/liveness failure')
    winner=r.get('winner',{});p=resolve_path(winner.get('path','/nonexistent'))
    require(p.is_file() and sha256(p)==winner.get('sha256'),'program qualified winner absent/changed')
    raw=admit_model_winner(p,build)
    require(raw['record']==r['record'],'program winner record differs')
    if r['record']=='G02':
        require(raw['role']=='qkv' and raw['batch']==r.get('model_batch',4),'program model selection is not its actual batch QKV winner')
        b2=r.get('qkv_winners',{}).get('qkv-B2',{});p2=resolve_path(b2.get('path','/nonexistent'))
        require(p2.is_file() and sha256(p2)==b2.get('sha256'),'B4 lost its required B2 native precursor')
        earlier=admit_model_winner(p2,build);require(earlier['batch']==2 and earlier['role']=='qkv','program B2 precursor differs')
    else:require(raw.get('matched_up_down') is True,'program G05 lacks both exact matched roles')
    return {**raw,'program_sha256':sha256(path)}


def diagnose_readiness(out,captures):
    out=Path(out);results=[]
    for c in captures:
        populations={}
        for regime in ('simultaneous','archived'):
            loc=out/f"p{c['decode_position']}-{regime}"
            if not loc.exists():continue
            rows=[json.loads(line) for line in (loc/'stdout.jsonl').read_text().splitlines()];decide_pairs(rows)
            pairs={}
            for row in rows:pairs.setdefault(row['pair'],{})[row['arm']]=row
            populations[regime]={pair:arms['candidate']['host_ms']-arms['control']['host_ms'] for pair,arms in pairs.items()}
        if set(populations)!={'simultaneous','archived'}:continue
        deficit=[populations['archived'][p] for p in range(5)];recovery=[populations['archived'][p]-populations['simultaneous'][p] for p in range(5)]
        dominant=c['record']=='G03' and all(d>0 and r>d/2 for d,r in zip(deficit,recovery)) and median(recovery)>2*(max(recovery)-min(recovery))
        results.append(dict(position=c['decode_position'],archived_layout_deficit_ms=deficit,readiness_difference_in_differences_ms=recovery,verdict='DOMINANT_READINESS_NEW_ISOLATED' if dominant else 'PARTIAL_MIXED_OR_UNKNOWN',traffic_cause='UNKNOWN_NO_QUALIFIED_COUNTERS'))
    return results


def diagnose_g13(out,captures):
    """Raw four-cell completion/order contrast; never the G03 readiness DiD."""
    root=Path(out);results=[]
    for c in captures:
        cells={};edges={};durations={};position=c['decode_position']
        for order in ('request','both-up-first'):
            for regime in ('simultaneous','archived'):
                loc=root/('G13-B2-'+order)/f'p{position}-{regime}'
                require(json.loads((loc/'exit.json').read_text()).get('exit')==0,'G13 incomplete native population')
                saved=json.loads((loc/'capture.json').read_text());require(saved==c,'G13 orthogonal capture differs')
                build=json.loads((loc.parent/'report.json').read_text())['build'];_,golden=admit_native(c,build);verify_native_probe(loc,c,build)
                require((loc/'golden.bin').read_bytes()==golden and all((loc/(arm+'.bin')).read_bytes()==golden for arm in ('control','candidate')),'G13 raw original/candidate outputs changed')
                rows=[json.loads(l) for l in (loc/'stdout.jsonl').read_text().splitlines()];decide_pairs(rows)
                for arm in ('control','candidate'):
                    mine={r['pair']:r for r in rows if r['arm']==arm};cells[order,regime,arm]=mine
                    raw=(loc/(arm+'.trace.bin')).read_bytes();tr=list(struct.iter_unpack('<4Q',raw))
                    require(all(0<a<=b<=d and w<304 for a,b,d,w in tr),'G13 incomplete body/worker records')
                    durations[order,regime,arm]=[d-b for a,b,d,w in tr]
                plan_path=resolve_path(build)/'g13-ready-final/plan.json';plan=json.loads(plan_path.read_text());types={x['id']:n for n,x in plan['types'].items()};actual_queues={}
                for arm in ('control','candidate'):
                    trace=list(struct.iter_unpack('<4Q',(loc/(arm+'.trace.bin')).read_bytes()))
                    require(len(trace)==len(plan['descs']) and all(t[3]==d['worker'] for d,t in zip(plan['descs'],trace)),'G13 actual task/worker identity differs')
                    queues={}
                    for key,ids in plan['static_queues'].items():
                        worker=int(key.split(':')[1]);ids=[i for i in ids if i>=0]
                        observed=sorted(ids,key=lambda i:trace[i][0]);require(len({trace[i][0] for i in ids})==len(ids),'ambiguous native queue order')
                        expected=ids
                        if arm=='candidate' and order=='request':
                            def priority(i):
                                n=types[plan['descs'][i]['type']];r=int(n.split('_r')[1]);return r*3+(0 if n.startswith('ready') else 1 if n.startswith('up') else 2)
                            expected=sorted(ids,key=priority)
                        require(observed==expected and all(trace[a][2]<=trace[b][0] for a,b in zip(observed,observed[1:])),'actual G13 queue/predecessor edges differ')
                        queues[str(worker)]=observed
                    actual_queues[arm]=queues
                edges[order,regime]=dict(plan_sha256=sha256(plan_path),task_in=plan['task_in'],events=plan['ev_inst'],actual_queues=actual_queues,native_endpoints=rows,control_queue='both-up-first (combined/request order is not reachable)',candidate_queue=order)
        # Three reachable states, observed in two actual paired groups: combined/
        # both-up-first vs independent/both-up-first, and a repeated combined/
        # both-up-first vs independent/request-order. Never invent the deadlocked
        # combined/request-order cell or cross-subtract unmatched controls.
        key=lambda order,arm:[cells[order,'simultaneous',arm][p]['host_ms'] for p in range(5)]
        cb=key('both-up-first','control');ib=key('both-up-first','candidate');cr=key('request','control');ir=key('request','candidate')
        deficit=[a-b for a,b in zip(cb,ib)];recovery=[a-b for a,b in zip(cr,ir)]
        dominant=all(d>0 and r>d/2 for d,r in zip(deficit,recovery)) and median(recovery)>2*(max(recovery)-min(recovery))
        decision=dict(verdict='DOMINANT_NEW_ISOLATED' if dominant else 'PARTIAL_OR_UNKNOWN',deficit=deficit,recovery=recovery)
        body_changes={}
        for order in ('both-up-first','request'):
            original=durations[order,'simultaneous','control'];changed=durations[order,'simultaneous','candidate']
            body_changes[order]=[abs(b-a)/max(1,a) for a,b in zip(original,changed)]
        # A predeclared qualification flag, not a causal price; keep every body.
        contention=any(change>.10 for group in body_changes.values() for change in group)
        if contention:decision['verdict']='MIXED_CONTENTION_REQUIRES_ATTRIBUTION'
        decision['historical_attribution']='UNKNOWN_MISSING_ORIGINAL_RUN_IDENTITY'
        results.append(dict(position=position,**decision,denominator='simultaneous paired combined minus independent completion, both-up-first queue fixed',restoration='separate simultaneous paired repeated combined/both-up-first minus independent/request-order',legal_unique_states=3,paired_control_groups=2,orthogonal_controls={str(k):v for k,v in cells.items()},task_worker_edges={str(k):v for k,v in edges.items()},body_duration_ticks={str(k):v for k,v in durations.items()},body_change_ratios=body_changes,contention_qualification=contention,traffic_cause='UNKNOWN_NO_QUALIFIED_COUNTERS'))
    return results
