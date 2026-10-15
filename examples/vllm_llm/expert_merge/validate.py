"""Bounded local acceptance with a shared heavy-stage lock and full receipts.

No native GPU executable is run. Output directories are new and caller-owned.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from .prepare import ROOT, sources


def command(cmd, out, env, cap=600, floor=4*1024**3):
    out=Path(out)
    if out.exists(): raise ValueError('evidence exists: '+str(out))
    start=time.monotonic(); reason=None
    print('LOCAL STAGE',shlex.join(map(str,cmd)),'evidence',out,flush=True)
    with out.open('w') as log:
        p=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
        sel=selectors.DefaultSelector(); sel.register(p.stdout,selectors.EVENT_READ); last=start
        try:
            while sel.get_map():
                for key,_ in sel.select(.5):
                    b=os.read(key.fileobj.fileno(),65536)
                    if not b: sel.unregister(key.fileobj); continue
                    text=b.decode(errors='replace'); log.write(text); log.flush()
                    print(text,end='',flush=True); last=time.monotonic()
                elapsed=time.monotonic()-start
                if elapsed>cap: reason='deadline'
                if shutil.disk_usage(out.parent).free<floor: reason='disk reserve'
                if reason:
                    log.write('STOP '+reason+'\n'); log.flush(); os.killpg(p.pid,signal.SIGKILL); break
                if time.monotonic()-last>30:
                    print('LOCAL PROGRESS pid',p.pid,'elapsed',round(elapsed),'seconds',flush=True); last=time.monotonic()
            rc=p.wait(timeout=5)
        finally:
            if p.poll() is None: os.killpg(p.pid,signal.SIGKILL); p.wait()
            sel.close(); p.stdout.close()
    receipt=dict(command=list(map(str,cmd)),exit=rc,stop=reason,seconds=time.monotonic()-start,
                 free_bytes=shutil.disk_usage(out.parent).free,sources=sources())
    Path(str(out)+'.status.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    if rc or reason: raise RuntimeError(f'local stage failed: rc={rc} reason={reason} log={out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('unit','probe','offline','matrix','full','entrypoints'))
    p.add_argument('out',type=Path)
    p.add_argument('--lock',type=Path,default=Path.home()/'.hermes/cache/scratch/qwen-prep-heavy.lock')
    a=p.parse_args(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy(); env.update(TMPDIR=str(out/'tmp'),PYTHONUNBUFFERED='1')
    (out/'tmp').mkdir()
    with a.lock.open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('BLOCKED: another preparation owns '+str(a.lock))
        f.seek(0); f.truncate(); f.write(f'expert merge pid={os.getpid()} stage={a.stage} out={out}\n'); f.flush()
        py=sys.executable; module='examples.vllm_llm.expert_merge.'
        def run(cmd,name,cap=600): command(cmd,out/(name+'.log'),env,cap)
        if a.stage=='unit':
            run([py,'-m',module+'cpu',out/'cpu'],'arithmetic')
            run([py,'-m',module+'protocol','--tokens','128'],'protocol')
            run([py,'-m','pytest','-q','tests/test_expert_merge.py','--basetemp='+str(out/'pytest'),
                 '-o','tmp_path_retention_policy=failed'],'targeted')
        elif a.stage=='full':
            run([py,'-m','pytest','-q','--basetemp='+str(out/'pytest'),'-o','tmp_path_retention_policy=failed'],
                'full-suite',1200)
        elif a.stage=='probe':
            for head in (False,True):
                for arm in ('control','candidate'):
                    for diag in (False,True):
                        name=f'{"head" if head else "qkv"}-{arm}{"-diag" if diag else ""}'
                        build=out/name
                        run([py,'-m',module+'prepare','generate',build,'--arm',arm,
                             *(['--diagnostic'] if diag else []),*(['--head'] if head else [])],name+'-generate')
                        run([py,'-m',module+'prepare','local',build],name+'-local')
        elif a.stage=='offline':
            # Reuse a complete probe stage in the sibling directory named probe.
            for build in sorted((out.parent/'probe').glob('*/identity.json')):
                run(['bash','examples/vllm_llm/expert_merge/offline.sh',build.parent],build.parent.name,300)
            if len(list((out.parent/'probe').glob('*/identity.json')))!=8:
                raise RuntimeError('offline requires all eight builds in sibling probe directory')
        elif a.stage=='matrix':
            for model,variants in (('qwen3-8b',('base','chain','xcd','xcd_s1','uc','uc_s1')),
                                   ('qwen2.5-1.5b',('base','chain','xcd','xcd_s1','uc','uc_s1')),
                                   ('qwen3-30b-a3b',('base','chain')),('mixtral-8x7b',('base','chain'))):
                for variant in variants:
                    env['LOCAL_MATRIX']=model+':'+variant
                    run(['bash','examples/vllm_llm/local_matrix.sh',out/'builds'],model+'-'+variant,300)
        else:
            run(['bash','-n','examples/vllm_llm/expert_merge/offline.sh'],'shell')
            for name in ('prepare','inputs','capture','runner','protocol','cpu'):
                run([py,'-m',module+name,'--help'],name+'-help')
            run([py,'-m',module+'capture','emit',out/'capture-host'],'capture-emit')
            run([py,'-m','etx.tools.gpu_jobs','examples/sessions/qwen_expert_merge.py','--gpus','0',
                 '--logdir',out/'dry-run','--dry-run'],'session-dry-run')
    print('LOCAL STAGE COMPLETE',a.stage,out,flush=True)


if __name__=='__main__': main()
