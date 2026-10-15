import fcntl,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
base=Path(__file__).parent;root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps');sys.path.insert(0,str(root))
from examples.vllm_llm import producer_session as ps
lock=open(base.parent/'qwen-prep-heavy.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.seek(0);lock.truncate();lock.write(f'b4-receipt-refresh pid={os.getpid()}\n');lock.flush()
full=json.loads((base/'full-suite.json').read_text());freeze=json.loads((base/'source-freeze.json').read_text())
assert full['exit']==0 and full['stop'] is None and full['revision']==freeze['revision']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==freeze['revision']
assert subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)==''
for n,h in freeze['source_sha256'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
changed=subprocess.check_output(['git','diff','--name-only',freeze['reviewed_base'],freeze['revision']],cwd=root,text=True).splitlines()
assert set(changed)==set(freeze['source_sha256']),changed
closure=json.loads((root/'docs/evidence/deepseek-prep-2026-10-15/source-closure.json').read_text())['device_include_closure']
for n,r in closure.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==r['sha256'],n
raw=json.loads((root/'docs/evidence/deepseek-prep-2026-10-15/raw-evidence-sha256.json').read_text())
for n,r in raw.items():assert hashlib.sha256((base.parent/'deepseek-open-prep-evidence'/n).read_bytes()).hexdigest()==r['sha256'],n
path=ps.ART/'local-ready.json';shutil.copy2(path,base/'qwen-local-ready-before.json');ready=json.loads(path.read_text());current=ps.local_sources()
delta={n:dict(old=h,new=current.get(n)) for n,h in ready['sources'].items() if current.get(n)!=h}
assert set(delta)=={'tests/test_dsv2_batch_attribution.py'} and set(ready['sources'])==set(current),delta
reuse=dict(base=freeze['reviewed_base'],revision=freeze['revision'],changed_sources=changed,device_include_closure_verified=len(closure),original_raw_files_verified=len(raw),qwen_changed_bindings=delta,qwen_unchanged_bindings=len(current)-len(delta),compiler_reruns=0,decision='Reuse existing DeepSeek and Qwen compiler matrices: changes are Python launch selection, traffic parsing and tests only. Device/host C++ sources, plan/emitter/import inputs, compiler flags, build scripts and queues are unchanged from the reviewed base. No new compiler or device result is claimed.')
(base/'compiler-reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
ready['sources']=current
ready['commands'].append(full|dict(log='docs/evidence/deepseek-prep-2026-10-15/b4-review-repair/full-suite.log'))
ready['b4_review_repair']=dict(revision=freeze['revision'],evidence='docs/evidence/deepseek-prep-2026-10-15/b4-review-repair/',compiler_reuse='docs/evidence/deepseek-prep-2026-10-15/b4-review-repair/compiler-reuse.json',archive_acceptance='docs/evidence/deepseek-prep-2026-10-15/b4-review-repair/archive-after-receipt.json',review='Independent scoped R1/R2 recheck pending; no main merge/push or GPU spending authorized')
ready['limitations']=[x.replace('Independent final review and the bounded process-cleanup recheck are still required before publication.', 'Original CC review found no blocking x8/cleanup issue. Its B4 R1/R2 repairs require an independent scoped recheck before publication.') for x in ready['limitations']]
path.write_text(json.dumps(ready,indent=2)+'\n');assert json.loads(path.read_text())['sources']==ps.local_sources()
fixture=Path(json.loads((base/'fixture.json').read_text())['fixture_path']);relative=path.relative_to(root);shutil.copy2(path,fixture/relative)
subprocess.run(['git','add',str(relative)],cwd=fixture,check=True)
subprocess.run(['git','-c','user.name=Local source acceptance','-c','user.email=fixture@localhost','commit','-qm','Bind Qwen readiness to B4 review repair acceptance'],cwd=fixture,check=True)
fpath=base/'fixture.json';f=json.loads(fpath.read_text());f['files'][str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest();f['fixture_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=fixture,text=True).strip();fpath.write_text(json.dumps(f,indent=2)+'\n')
probe=subprocess.run([root/'.venv/bin/python','-c','import json; from examples.vllm_llm import producer_session as p; r=json.loads((p.ART/"local-ready.json").read_text()); assert r["ready"] is True and r["sources"]==p.local_sources(); print("COMMITTED FIXTURE SOURCE BINDINGS MATCH",len(r["sources"]))'],cwd=fixture,env=dict(os.environ,PYTHONPATH=str(fixture)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(base/'qwen-receipt-probe.log').write_text(probe.stdout);(base/'qwen-receipt-probe.exit').write_text(str(probe.returncode)+'\n');assert probe.returncode==0,probe.stdout
print(probe.stdout,flush=True)
