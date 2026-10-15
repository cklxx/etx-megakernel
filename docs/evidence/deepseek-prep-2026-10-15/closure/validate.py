import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps');base=Path(__file__).parent
lock=open(base.parent/'qwen-prep-heavy.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.seek(0);lock.truncate();lock.write(f'deepseek-final-tests pid={os.getpid()}\n');lock.flush()
assert shutil.disk_usage(base).free>4.7*1024**3
pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert pin=='1a23336e1bdf1c78c63ec35f3dab136f88acc2b0'
assert subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)==''
(base/'tmp').mkdir(exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(root),TMPDIR=str(base/'tmp'),ETX_DATA_ROOT='/Users/ckl/code/etx',ETX_REFERENCE_ROOT='/Users/ckl/code/etx',ETX_DEVLIBS='/Users/ckl/code/rocm-headers/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode')
commands=[]
def run(name,cmd,limit=1200,extra=None,cwd=root):
 assert shutil.disk_usage(base).free>4.5*1024**3
 log=base/(name+'.log');start=time.monotonic();stop=None;print('STAGE',name,flush=True)
 with log.open('x') as out:
  p=subprocess.Popen(list(map(str,cmd)),cwd=cwd,env=env|dict(extra or {}),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  while p.poll() is None:
   elapsed=time.monotonic()-start
   if elapsed>limit or shutil.disk_usage(base).free<4*1024**3:
    stop='deadline' if elapsed>limit else 'disk reserve'
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    break
   print('PROGRESS',name,round(elapsed), 's; log_bytes',log.stat().st_size,'free_bytes',shutil.disk_usage(base).free,flush=True);time.sleep(5)
 r=dict(name=name,command=list(map(str,cmd)),cwd=str(cwd),exit=p.returncode,stop=stop,seconds=round(time.monotonic()-start,2),log=str(log),free_bytes=shutil.disk_usage(base).free)
 commands.append(r);(base/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');(base/(name+'.exit')).write_text(str(p.returncode)+'\n')
 print('DONE',name,p.returncode,stop,flush=True)
 if p.returncode or stop:raise SystemExit(name+' failed')
py=root/'.venv/bin/python'
selection='every_request or static_queues or routing_handoff or two_requests_keep or more_than_two or shared_kva'
run('remaining-graph-collect',[py,'-m','pytest','--collect-only','-q','tests/test_dsv2_batch.py','-k',selection],60)
assert '14/' in (base/'remaining-graph-collect.log').read_text() or '14 tests collected' in (base/'remaining-graph-collect.log').read_text(),(base/'remaining-graph-collect.log').read_text()
run('remaining-14-graph-tests',[py,'-m','pytest','-q','-ra','tests/test_dsv2_batch.py','-k',selection,'--basetemp='+str(base/'graph-pytest'),'-o','tmp_path_retention_policy=failed'])
# The two existing Qwen transport tests archive all of ROOT. Use a real committed
# source fixture without 2.4 GiB historical DeepSeek measurements, as at Qwen closure.
fixture=base/'source-fixture';fixture.mkdir()
paths=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0');copied={};excluded=[]
for n in paths:
 if not n:continue
 if n.startswith('examples/dsv2lite/results/'):excluded.append(n);continue
 src=root/n;dst=fixture/n;dst.parent.mkdir(parents=True,exist_ok=True)
 if src.is_symlink():dst.symlink_to(os.readlink(src))
 else:shutil.copy2(src,dst);copied[n]=hashlib.sha256(src.read_bytes()).hexdigest()
for label,cmd in [('fixture-init',['git','init','-q']),('fixture-add',['git','add','-f','-A']),('fixture-commit',['git','-c','user.name=Local source acceptance','-c','user.email=fixture@localhost','commit','-qm','Pinned source transport fixture without historical GPU data'])]:run(label,cmd,60,cwd=fixture)
fixture_pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=fixture,text=True).strip()
(base/'fixture.json').write_text(json.dumps(dict(source=pin,fixture_commit=fixture_pin,files=copied,excluded_historical_data=excluded,scope='Only ROOT for the two archive transport tests uses this real committed fixture; all tests and all other inputs use the pinned worktree.'),indent=2)+'\n')
plugin=base/'bounded_archive_plugin.py';plugin.write_text('''import os\nfrom pathlib import Path\nimport pytest\n@pytest.fixture(autouse=True)\ndef bounded_archive(request,monkeypatch):\n    if request.module.__name__.endswith("test_producer_readiness") and request.node.name in ("test_fresh_archive_interpreter_preflight_and_failure_report","test_documented_push_stage_attach_start_with_local_transport"):\n        monkeypatch.setattr(request.module,"ROOT",Path(os.environ["DEEPSEEK_ARCHIVE_FIXTURE"]))\n''')
run('full-suite',[py,'-m','pytest','-q','-ra','-p','bounded_archive_plugin','--basetemp='+str(base/'full-pytest'),'-o','tmp_path_retention_policy=failed'],1800,dict(PYTHONPATH=str(base)+os.pathsep+str(root),DEEPSEEK_ARCHIVE_FIXTURE=str(fixture)))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==pin
assert subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)==''
(base/'acceptance.json').write_text(json.dumps(dict(status='PASS_LOCAL_ONLY',revision=pin,implementation='a0c3e724dbd1a342d96812d232f04ced902d6496',commands=commands,compiler_evidence='Reused unchanged pinned compiler inputs and artifact identity from docs/evidence/deepseek-prep-2026-10-15; no compiler reruns',archive_adaptation='fixture.json',gpu_spend_usd=0,gpu_execution=False),indent=2)+'\n')
print('ALL LOCAL TESTS PASS',flush=True)
