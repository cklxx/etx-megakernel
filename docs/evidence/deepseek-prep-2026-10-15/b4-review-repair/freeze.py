import fcntl,hashlib,json,os,shutil,subprocess
from pathlib import Path
base=Path(__file__).parent;root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps');fixture=base.parent/'deepseek-closure-evidence/source-fixture'
lock=open(base.parent/'qwen-prep-heavy.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.seek(0);lock.truncate();lock.write(f'b4-review-freeze pid={os.getpid()}\n');lock.flush()
assert json.loads((base/'focused-final.json').read_text())['exit']==0
names=['examples/dsv2lite/batch_attribution.py','examples/dsv2lite/batch_session.py','tests/test_dsv2_batch_attribution.py']
changed=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines();assert set(changed)==set(names),changed
commands=[]
for label,cmd in [('whitespace',['git','diff','--check']),('stage',['git','add',*names]),('commit',['git','commit','-m','Fix selected B4 calls and require complete dispatch traffic coverage'])]:
 p=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(base/('freeze-'+label+'.log')).write_text(p.stdout);commands.append(dict(command=cmd,exit=p.returncode));assert p.returncode==0,p.stdout
pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
for n in names:shutil.copy2(root/n,fixture/n)
subprocess.run(['git','add',*names],cwd=fixture,check=True)
subprocess.run(['git','-c','user.name=Local source acceptance','-c','user.email=fixture@localhost','commit','-qm','B4 review repairs for committed-source transport acceptance'],cwd=fixture,check=True)
manifest=json.loads((base.parent/'deepseek-closure-evidence/fixture.json').read_text())
manifest['prior_source']=manifest['source'];manifest['source']=pin
for n in names:manifest['files'][n]=hashlib.sha256((root/n).read_bytes()).hexdigest()
manifest['fixture_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=fixture,text=True).strip();manifest['fixture_path']=str(fixture)
(base/'fixture.json').write_text(json.dumps(manifest,indent=2)+'\n')
result=dict(revision=pin,reviewed_base='3a449a7db98f44eaf42f259011e3bcf996327b9b',implementation_files=names[:2],tests=names[2:],source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},commands=commands,compiler_reruns=0,review='Original CC R1/R2 repair; independent scoped recheck still required',gpu_spend_usd=0)
(base/'source-freeze.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
