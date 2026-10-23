"""Bounded producer experiment operations. No provisioning or remote action.

Run only through examples/sessions/qwen_producer.py after explicit GPU approval.
The report can run offline. It refuses partial correctness or timing evidence.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
from pathlib import Path
import statistics
import sys
import time
from examples.vllm_llm import g6_session as g6, import_certificate as cert, producer_replay as replay
from examples.vllm_llm.kept_build import KEPT_ENV, BASE_FLAGS, source_hashes
from examples.vllm_llm.qwen300_config import IMAGE
from examples.vllm_llm.pairs import parse, verdict, kv
from etx.session import proc

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'examples/vllm_llm/results/2026-10-15-producer-prep'
RULE='five AB/BA interleaved host RESULT wall pairs; every gain >0; median >=0.030ms and >2*(max-min)'
RESERVE=300
ARMS=('baseline','producer')


def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2);f.write('\n')


def clean_env():
    # Drop all inherited experiments, diagnostic capture and debug controls.
    return {k:v for k,v in os.environ.items() if (k != 'CFLAGS' and not k.startswith(('ETX_','DSL_ABLATE_')))
            or k in ('ETX_SESS_LOGDIR','ETX_SESS_REMAINING_SECONDS','ETX_QWEN_G6_CONFIG',
                     'ETX_ROCM_HEADERS','ETX_TORCH_HEADERS','ETX_DEVLIBS','ETX_LLVM')}


def build_env(arm):
    if arm not in ARMS: raise ValueError('unknown arm')
    return dict(clean_env(),**KEPT_ENV, ETX_VL_PRODUCER_SCORE=str(int(arm=='producer')),
                ETX_HIPCC_FLAGS=BASE_FLAGS,ETX_OUT=str(g6.build_dir(arm)))


def remaining():
    # Runner supplies remaining time once per job. Use a fixed process deadline.
    return DEADLINE-time.monotonic()


def run(name,argv,*,env=None,seconds=240):
    if remaining() < seconds+RESERVE:
        raise ValueError(f'evidence reserve: need {seconds+RESERVE}s, have {remaining():.1f}s')
    path=g6.record()/f'{name}.raw.log'
    print(f'PHASE {name} deadline_s={seconds} evidence={path}',flush=True)
    last=[time.monotonic()]
    with path.open('x') as out:
        out.write('# command '+json.dumps(list(map(str,argv)))+'\n');out.flush()
        def poll(elapsed):
            now=time.monotonic()
            if now-last[0]>=30:
                out.flush();print(f'PHASE {name} elapsed_s={elapsed:.1f} log_bytes={path.stat().st_size} evidence={path}',flush=True)
                last[0]=now
            if elapsed >= 180 and path.stat().st_mtime < time.time()-180:
                # A bounded read-only probe. Never kill by command-string match.
                import subprocess
                probe=subprocess.run(['ps','-eo','pid,ppid,etime,stat,comm'],capture_output=True,text=True,timeout=5)
                out.write('\nSILENCE-PROBE\n'+probe.stdout);out.flush()
                raise ValueError(f'{name}: silent >180s; inspect retained process state')
        rc=proc.run(list(map(str,argv)),sink=out,deadline_s=seconds,poll_s=5,on_poll=poll,
                    env=env or clean_env(),cwd=ROOT,log=out)
    if rc: raise ValueError(f'{name}: exit={rc}; evidence={path}')
    return path.read_text(errors='replace')


def audit_plan(build, candidate):
    plan=json.loads((Path(build)/'plan.json').read_text())
    types=plan['types']
    for layer in range(48):
        o=types[f'o_proj_{layer}']
        expected='etx_tile_dsl_gemv_producer' if candidate else 'etx_tile_dsl_gemv'
        if o['symbol']!=expected or o['grid']!=[256]:
            raise ValueError('emitted o_proj does not reach the intended code')
        if candidate:
            if any(n.startswith((f'router_{layer}_',f'topk_{layer}_')) or n in (f'router_{layer}',f'topk_{layer}') for n in types):
                raise ValueError('candidate retains router/topk scheduling')
        elif types[f'router_{layer}']['symbol']!='etx_tile_dsl_gemv_router_fold' or f'topk_{layer}_x' not in types:
            raise ValueError('baseline recipe changed')
    src=(Path(build)/'dsl_tiles.hip').read_text()
    if candidate and ("etx_producer_dot8" not in src or "dsl_topk_lds(d.logits,128,8,1" not in src or 'producer_scratch' not in plan['args']):
        raise ValueError('missing producer arithmetic, scratch or kept top-k')
    return dict(candidate=candidate, owners=256,layers=48,plan_sha256=cert.sha(Path(build)/'plan.json'))


def local_sources():
    files=source_hashes()
    files.update({str(p.relative_to(ROOT)):cert.sha(p) for p in (ROOT/'tests').rglob('*.py')})
    for name in ('examples/sessions/qwen_producer.py','examples/sessions/qwen300.py',
                 'tools/producer_start.py','tools/session.sh','etx/session/transport.sh','pyproject.toml'):
        files[name]=cert.sha(ROOT/name)
    return files



def input_paths(config):
    model=Path(config['model']).expanduser().resolve(strict=True)
    hf=Path(config['hf']).expanduser().resolve(strict=True)
    if hf != (model/'hf').resolve(strict=True):
        raise ValueError('HF path differs from the model/hf launch path')
    golden=(model/'golden_c1024.txt').resolve(strict=True)
    return dict(model=str(model),hf=str(hf),golden=str(golden),prompt=str(golden))


def input_binding(config, hashes):
    return dict(**input_paths(config),input_sha256_digest=cert.digest(hashes),
                golden_sha256=hashes['packed/golden_c1024.txt'],
                prompt_sha256=hashes['packed/golden_c1024.txt'])


def validate_reference(ref, experiment):
    expected=experiment['input_binding']
    if expected['input_sha256_digest'] != cert.digest(experiment['input_sha256']):
        raise ValueError('S3 input digest mismatch')
    if (expected['golden'] != expected['prompt'] or
        expected['golden_sha256'] != experiment['input_sha256']['packed/golden_c1024.txt'] or
        expected['prompt_sha256'] != expected['golden_sha256'] or
        ref.get('input_binding') != expected or
        ref.get('launch',{}).get('model') != expected['hf']):
        raise ValueError('S3 checkpoint/golden/prompt identity mismatch')


def input_identity(config):
    paths=input_paths(config)
    model=Path(paths['model']);hf=Path(paths['hf'])
    files={f'packed/{n}':model/n for n in ('vl.bin','vl.manifest','golden_c1024.txt')}
    checkpoint_files=[p for p in hf.rglob('*') if p.is_file() and '.cache' not in p.relative_to(hf).parts]
    if not any(p.suffix=='.safetensors' for p in checkpoint_files):
        raise ValueError('HF checkpoint tensors missing')
    files.update({f'hf/{p.relative_to(hf)}':p for p in checkpoint_files})
    if 'hf/config.json' not in files or 'hf/REVISION' not in files:
        raise ValueError('HF config or pinned revision missing')
    return {n:cert.sha(p) for n,p in sorted(files.items())}


def preflight():
    input_paths(g6.config())  # Reject a path mismatch before any probe/build/compute.
    ready=json.loads((ART/'local-ready.json').read_text())
    if ready['sources']!=local_sources() or ready['ready'] is not True:
        raise ValueError('local acceptance missing or source hashes changed; rerun local preflight')
    if (ROOT/'.git').exists():
        if run('source_clean',['git','status','--porcelain']).strip().splitlines()[1:]:
            raise ValueError('dirty source; commit before GPU run')
        source_identity=run('source_commit',['git','rev-parse','HEAD']).splitlines()[-1]
    else:
        source_identity=(ROOT/'SOURCE_IDENTITY').read_text()
        if not re.search(r'^commit [0-9a-f]{40}$',source_identity,re.M):
            raise ValueError('missing stage identity: use tools/session.sh stage/push')
    g6.preflight()
    c=g6.config();model=Path(c['model'])
    # Full packed input identity. Manifest-only identity is insufficient here.
    data=json.loads((g6.record()/'inputs.json').read_text())
    hashes=input_identity(c)
    data.update(input_sha256=hashes,input_binding=input_binding(c,hashes),
                source_hashes=local_sources(),source_identity=source_identity,
                rule=RULE,condition='one MI300X, bf16, B1, context1024, 32 HF greedy',
                budget_seconds=3600,evidence_reserve_seconds=RESERVE)
    # g6 preflight already created inputs.json. Append the immutable identity once.
    save(g6.record()/'experiment.json',data)
    run('hardware',['rocm-smi','--showproductname','--showuniqueid','--showclocks'])
    run('image',['sudo','docker','image','inspect',IMAGE],seconds=30)
    print('PHASE preflight OK; no image pull or model download',flush=True)


def build(arm):
    b=g6.build_dir(arm);b.mkdir()
    run(f'build_{arm}',['bash','examples/vllm_llm/build.sh',g6.config()['model']],env=build_env(arm),seconds=840)
    cert.verified_build(b,Path(g6.config()['model']))
    proof=audit_plan(b,arm=='producer');save(g6.record()/f'plan_{arm}.json',proof)
    print('PLAN '+json.dumps(proof),flush=True)


def host(arm,name,args,extra=None):
    b=g6.build_dir(arm);model=Path(g6.config()['model'])
    text=run(name,[sys.executable,'-u','-m','examples.vllm_llm.import_certificate','--run-build',b,
                   '--model',model,'--',*args],env=dict(clean_env(),ETX_VL_PROGRESS='64',**(extra or {})),seconds=180)
    cert.stamped(text,cert.verified_build(b,model));return text


def token_gate(text, mode='fused',dsl=False):
    from examples.vllm_llm.nvref_report import correctness
    correctness(text,'producer')
    runs=parse(text.splitlines()).runs
    if len(runs)!=1 or runs[0].arm!=mode: raise ValueError('wrong or missing executable mode')
    if not all(math.isfinite(x) and x>0 for x in (runs[0].wall_ms,runs[0].device_ms)):
        raise ValueError('nonfinite or nonpositive timing')
    if dsl and 'CHECK-DSL OK' not in text: raise ValueError('CHECK-DSL missing')
    return runs[0]


def capture_gate(record):
    hashes={}
    for pos in (0,1040):
        a,b=[Path(record)/f'capture_{arm}_p{pos}' for arm in ARMS]
        am,bm=[replay.read_capture(p,actual=True) for p in (a,b)]
        if am['layers']!=48 or bm['layers']!=48 or am['positions']!=[pos] or bm['positions']!=[pos]:
            raise ValueError('actual capture coverage')
        # These are production gates. A diagnostic CPU replay mismatch alone
        # is reported separately. Routing ID/mix and expert data must match.
        for name in am['files']:
            # Raw router logits retain the existing CHECK-DSL numerical gate.
            # The registered exact checks cover IDs/mix and expert inputs.
            if name.endswith('_logits.bin'):
                continue
            if (a/name).read_bytes()!=(b/name).read_bytes():
                raise ValueError(f'actual-model acceptance mismatch at position={pos} file={name}; performance blocked')
        hashes[str(pos)]={arm:cert.sha(p/'capture.json') for arm,p in zip(ARMS,(a,b))}
    return hashes


def qualify():
    for arm in ARMS:
        archive=g6.record()/f'certificate_{arm}';archive.mkdir()
        identity=cert.verified_build(g6.build_dir(arm),Path(g6.config()['model']))
        cert.export_build(g6.build_dir(arm),Path(g6.config()['model']),archive)
        for pos in (0,1040):
            cap=g6.record()/f'capture_{arm}_p{pos}';cap.mkdir()
            text=host(arm,f'{arm}_check_p{pos}',['--check','48','--check-pos',str(pos)],{'ETX_PRODUCER_CAPTURE':str(cap)})
            (archive/f'check_p{pos}.txt').write_text(text)
            replay.seal(cap,identity=cert.digest(identity),provenance='actual-model-check',positions=[pos])
        cert.validate(archive,timing=False)
    # Keep diagnostics even if the production byte gate fails.
    for pos in (0,1040):
        result=replay.replay(g6.record()/f'capture_baseline_p{pos}',replay.Arithmetic(),actual=True)
        save(g6.record()/f'replay_p{pos}.json',result)
    # Preserve raw logit changes as diagnostics. They are not a new exact gate.
    import numpy as np
    diagnostics=[]
    for pos in (0,1040):
        for layer in range(48):
            a,b=[np.fromfile(g6.record()/f'capture_{arm}_p{pos}/L{layer}_logits.bin',dtype='<u2') for arm in ARMS]
            diagnostics.append(dict(position=pos,layer=layer,bf16_logit_bits_different=int(np.count_nonzero(a!=b))))
    save(g6.record()/'router_logit_diagnostics.json',dict(kind='diagnostic; CHECK-DSL owns numerical acceptance',cases=diagnostics))
    captures=capture_gate(g6.record())
    gates={}
    for arm in ARMS:
        for mode in ('fused','graph'):
            args=['--repeat','1']+(['--check-dsl'] if mode=='fused' else ['--unfused'])
            text=host(arm,f'{arm}_gate_{mode}',args)
            token_gate(text,mode,dsl=mode=='fused')
            gates[f'{arm}_{mode}']=cert.sha(g6.record()/f'{arm}_gate_{mode}.raw.log')
    save(g6.record()/'correctness.json',dict(pass_all=True,captures=captures,gates=gates,
         builds={a:cert.digest(cert.verified_build(g6.build_dir(a),Path(g6.config()['model']))) for a in ARMS}))
    print('CERTIFIED both builds: imports 0/1040; exact captured router/expert data; 32/32; CHECK-DSL',flush=True)


def require_correctness(record, *, live=True):
    record=Path(record)
    gate=json.loads((record/'correctness.json').read_text())
    if gate.get('pass_all') is not True or set(gate['gates'])!={f'{a}_{m}' for a in ARMS for m in ('fused','graph')}:
        raise ValueError('failed or missing correctness gate')
    if capture_gate(record)!=gate['captures']: raise ValueError('capture evidence changed')
    for arm in ARMS:
        cert.validate(record/f'certificate_{arm}',timing=False)
        archived=json.loads((record/f'certificate_{arm}'/'build_identity.json').read_text())
        if gate['builds'][arm]!=cert.digest(archived): raise ValueError('correctness build mismatch')
        if live and cert.verified_build(record/f'b_{arm}',Path(g6.config()['model']))!=archived:
            raise ValueError('timed executable differs from correctness executable')
        for mode in ('fused','graph'):
            path=record/f'{arm}_gate_{mode}.raw.log'
            if cert.sha(path)!=gate['gates'][f'{arm}_{mode}']: raise ValueError('gate log changed')
            text=path.read_text();cert.stamped(text,archived);token_gate(text,mode,dsl=mode=='fused')
    return gate


def s3_command(name,seconds=180):
    model=Path(g6.config()['model']);out=g6.record()/name
    out.mkdir()
    home=str(Path.home())
    # Image is content-addressed. timeout is INSIDE Docker as well as outside.
    return ['sudo','docker','run','--rm','--init','--pull=never','--network=none','--ipc=host','--shm-size=16g',
            '--security-opt','seccomp=unconfined','--device=/dev/kfd','--device=/dev/dri','--group-add=video',
            '-v',f'{home}:{home}','-w',str(ROOT),'-e',f'HOME={home}',
            '-e',f'HIP_VISIBLE_DEVICES={os.environ.get("GPU","0")}', '-e','VLLM_ROCM_USE_AITER=1',
            '-e','VLLM_ENABLE_V1_MULTIPROCESSING=0','-e',f'ETX_SESS_LOGDIR={out}',
            '-e',f'ETX_QWEN_G6_CONFIG={os.environ["ETX_QWEN_G6_CONFIG"]}', IMAGE,
            'timeout','--signal=KILL',str(seconds-10),'python3','-u','-m',__name__.replace('__main__','examples.vllm_llm.producer_session'),'s3-worker']


def s3_worker():
    from examples.sessions import qwen300
    # Reuse the established estimator and S3 options with the staged model.
    c=g6.config();binding=input_binding(c,input_identity(c))
    experiment=json.loads((g6.record().parent/'experiment.json').read_text())
    if binding != experiment['input_binding']:
        raise ValueError('S3 input changed before launch')
    qwen300.vllm_run('bf16','bench',input_binding=binding)
    ref=json.loads((g6.record()/'bf16_bar.json').read_text())
    validate_reference(ref,experiment)
    if binding != input_binding(c,input_identity(c)):
        raise ValueError('S3 input changed during launch')


def measure():
    inputs=json.loads((g6.record()/'experiment.json').read_text())
    if input_binding(g6.config(),input_identity(g6.config())) != inputs['input_binding']:
        raise ValueError('S3 input changed before compute')
    require_correctness(g6.record())
    for rnd in range(5):
        order=[('baseline','fused'),('producer','fused'),('producer','graph')]
        if rnd%2: order.reverse()
        for arm,mode in order:
            args=['--repeat','1']+(['--unfused'] if mode=='graph' else [])
            token_gate(host(arm,f'r{rnd}_{arm}_{mode}',args),mode)
        run(f'r{rnd}_s3',s3_command(f's3_r{rnd}'),seconds=180)
    require_correctness(g6.record())
    inputs=json.loads((g6.record()/'experiment.json').read_text())
    if inputs['input_sha256']!=input_identity(g6.config()) or inputs['source_hashes']!=local_sources():
        raise ValueError('source or packed input changed; timing refused')
    save(g6.record()/'measurement.complete.json',dict(rounds=5,basis='host RESULT wall_ms',complete=True))



def diagnose():
    from examples.vllm_llm.producer_trace import summarize
    require_correctness(g6.record())
    if not (g6.record()/'measurement.complete.json').exists():raise ValueError('measurement not complete')
    for arm in ARMS:
        raw=g6.record()/f'trace_{arm}.bin'
        text=host(arm,f'trace_{arm}',['--repeat','1'],{'ETX_TRACE':'1','ETX_TRACE_RAW':str(raw)})
        token_gate(text)
        plan=json.loads((g6.build_dir(arm)/'plan.json').read_text())
        save(g6.record()/f'trace_{arm}.json',summarize(plan,raw.read_bytes(),arm=='producer'))


def result(record):
    record=Path(record)
    require_correctness(record,live=False)
    complete=json.loads((record/'measurement.complete.json').read_text())
    if complete!={'rounds':5,'basis':'host RESULT wall_ms','complete':True}:
        raise ValueError('incomplete measurement population')
    experiment=json.loads((record/'experiment.json').read_text())
    rows=[]
    for rnd in range(5):
        row={}
        for arm,mode in (('baseline','fused'),('producer','fused'),('producer','graph')):
            text=(record/f'r{rnd}_{arm}_{mode}.raw.log').read_text()
            identity=json.loads((record/f'certificate_{arm}/build_identity.json').read_text())
            cert.stamped(text,identity);r=token_gate(text,mode)
            fields=kv(next(line for line in text.splitlines() if line.startswith('RESULT ')))
            row[f'{arm}_{mode}']=dict(wall_ms=r.wall_ms,wall_decimal=fields['wall_ms'],device_ms=r.device_ms)
        raw=(record/f'r{rnd}_s3.raw.log').read_text()
        if raw.count('CORRECTNESS OK 32/32 precision=bf16')!=6 or 'BENCH-PROGRESS sample=5/5 correctness=32/32' not in raw:
            raise ValueError('S3 raw correctness/timing evidence missing')
        ref=json.loads((record/f's3_r{rnd}/bf16_bar.json').read_text())
        validate_reference(ref,experiment)
        launches=[json.loads(line.removeprefix('EXACT-LLM-ARGS ')) for line in raw.splitlines() if line.startswith('EXACT-LLM-ARGS ')]
        bindings=[json.loads(line.removeprefix('INPUT-BINDING ')) for line in raw.splitlines() if line.startswith('INPUT-BINDING ')]
        if launches != [ref['launch']] or bindings != [ref['input_binding']]:
            raise ValueError('S3 raw launch identity mismatch')
        if ref.get('correct')!=32 or ref.get('image')!=IMAGE or ref.get('launch',{}).get('attention_backend')!='ROCM_AITER_UNIFIED_ATTN':
            raise ValueError('S3 reference not correct or changed')
        for field in ('full_ms','one_ms','per_token_ms'):
            if len(ref[field])!=5 or any(not math.isfinite(x) or x<=0 for x in ref[field]):raise ValueError('S3 timing population')
        recomputed=[(a-b)/31 for a,b in zip(ref['full_ms'],ref['one_ms'])]
        if recomputed!=ref['per_token_ms'] or statistics.median(recomputed)!=ref['median_ms']:
            raise ValueError('S3 estimator changed')
        row['s3_ms']=ref['median_ms'];rows.append(row)
    from etx.session.experiment import ns
    savings_ns=[ns(r['baseline_fused']['wall_decimal'])-ns(r['producer_fused']['wall_decimal']) for r in rows]
    med_ns=statistics.median(savings_ns);spread_ns=max(savings_ns)-min(savings_ns)
    savings=[x/1e6 for x in savings_ns];med=med_ns/1e6;spread=spread_ns/1e6
    gains=[(r['producer_graph']['wall_ms']-r['producer_fused']['wall_ms'])/r['producer_graph']['wall_ms'] for r in rows]
    delta=[r['producer_fused']['wall_ms']-r['s3_ms'] for r in rows]
    return dict(status='COMPLETE',diagnostics='COMPLETE' if all((record/f'trace_{a}.json').exists() for a in ARMS) else 'NOT-COMPLETE: optional traces missing',rule=RULE,rows=rows,savings_ms=savings,median_ms=med,spread_ms=spread,
                incremental='KEEP' if min(savings_ns)>0 and med_ns>=30000 and med_ns>2*spread_ns else 'NO-GO',
                fusion=verdict(gains),system_delta_ms=delta,
                reference_scope='S3: strongest legal configuration previously tested; not a new exhaustive search',
                system='LEAD' if max(delta)<0 and statistics.median(delta)<-max(.050,2*(max(delta)-min(delta))) else 'NO-LEAD')


def report():
    try: out=result(g6.record())
    except Exception as exc: out=dict(status='NOT-COMPLETE',reason=str(exc),performance_claim=False)
    save(g6.record()/'report.json',out)
    print('VERDICT '+json.dumps(out),flush=True)


def main():
    global DEADLINE
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('operation',choices=('preflight','build','qualify','measure','report','diagnose','s3-worker'));ap.add_argument('arm',nargs='?',choices=ARMS)
    a=ap.parse_args()
    DEADLINE=time.monotonic()+min(3600,float(os.environ.get('ETX_SESS_REMAINING_SECONDS','3600')))
    try:
        if a.operation=='build': build(a.arm)
        else: globals()[a.operation.replace('-','_')]()
    except Exception as exc:
        print(f'FAIL phase={a.operation} reason={exc}',flush=True)
        print(f'EVIDENCE {g6.record()}',flush=True)
        print('SAFE resume: session.sh status/fetch; inspect failure; no partial timing claim; lease budget is unchanged',flush=True)
        return 2
    return 0

if __name__=='__main__': raise SystemExit(main())
