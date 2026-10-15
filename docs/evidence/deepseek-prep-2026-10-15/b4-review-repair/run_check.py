import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
base=Path(__file__).parent;root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps')
name=sys.argv[1];cmd=sys.argv[2:]
lock=open(base.parent/'qwen-prep-heavy.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.seek(0);lock.truncate();lock.write(f'deepseek-{name} pid={os.getpid()}\n');lock.flush()
assert shutil.disk_usage(base).free>4.5*1024**3
log=base/(name+'.log');start=time.monotonic();stop=None
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
source={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['examples/dsv2lite/batch_attribution.py','examples/dsv2lite/batch_session.py','tests/test_dsv2_batch_attribution.py']}
env=dict(os.environ,PYTHONPATH=str(base)+os.pathsep+str(root),TMPDIR=str(base/'tmp'),ETX_DATA_ROOT='/Users/ckl/code/etx',ETX_REFERENCE_ROOT='/Users/ckl/code/etx',DEEPSEEK_ARCHIVE_FIXTURE=str(base.parent/'deepseek-closure-evidence/source-fixture'),ETX_DEVLIBS='/Users/ckl/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')
with log.open('x') as out:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  elapsed=time.monotonic()-start
  if elapsed>1800 or shutil.disk_usage(base).free<4*1024**3:
   stop='deadline/reserve';os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  print('PROGRESS',name,int(elapsed),'s; bytes',log.stat().st_size,'free',shutil.disk_usage(base).free,flush=True);time.sleep(5)
r=dict(name=name,command=cmd,cwd=str(root),revision=revision,source_sha256=source,exit=p.returncode,stop=stop,seconds=round(time.monotonic()-start,2),free_bytes=shutil.disk_usage(base).free)
(base/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n');(base/(name+'.exit')).write_text(str(p.returncode)+'\n');print(json.dumps(r),flush=True)
raise SystemExit(p.returncode if not stop else 99)
