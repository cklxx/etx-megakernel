import fcntl,hashlib,json,shutil,subprocess
from pathlib import Path
b=Path(__file__).parent;root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps');main=Path('/Users/ckl/code/.wt/etx-mainverify')
lock=open(b.parent/'qwen-prep-heavy.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(b).free>4*1024**3
f=json.loads((b/'source-freeze.json').read_text());code=f['revision'];assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==code
for name,h in f['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
paths=['docs/DEEPSEEK-B4-PREP-2026-10-15.md','docs/DEEPSEEK-PREP-TWO-GAPS-2026-10-15.md','docs/evidence/deepseek-prep-2026-10-15/README.md','docs/evidence/deepseek-prep-2026-10-15/b4-review-repair','docs/NEXT-GPU-2026-10-15.md','docs/REPORT-STATUS-2026-10-15.md','examples/dsv2lite/results/EXPERIMENT-LOG.md','examples/dsv2lite/results/GPU-QUEUE.md','examples/vllm_llm/results/2026-10-15-producer-prep/local-ready.json']
def allowed(p):return any(p==n or p.startswith(n+'/') for n in paths)
changed=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines()+subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines()
assert changed and all(allowed(p) for p in changed),changed
manifest=root/'docs/evidence/deepseek-prep-2026-10-15/b4-review-repair/sha256.json'
for name,rec in json.loads(manifest.read_text()).items():assert hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest()==rec['sha256'],name
commands=[]
def run(name,cmd,cwd=root):
    r=subprocess.run(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (b/(name+'.log')).write_text(r.stdout);commands.append(dict(command=cmd,exit=r.returncode));assert r.returncode==0,r.stdout
    return r.stdout
probe='import json; from examples.vllm_llm import producer_session as p; r=json.loads((p.ART/"local-ready.json").read_text()); assert r["ready"] is True and r["sources"]==p.local_sources(); print("ACTIVE QWEN SOURCE BINDINGS MATCH",len(r["sources"]))'
run('active-receipt-before-commit',[str(root/'.venv/bin/python'),'-c',probe])
run('metadata-whitespace',['git','diff','--check'])
run('metadata-stage',['git','add',*paths])
run('metadata-commit',['git','commit','-m','Record B4 review repair acceptance and scoped recheck handoff'])
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)==''
assert all(allowed(p) for p in subprocess.check_output(['git','diff','--name-only',code,head],cwd=root,text=True).splitlines())
run('active-receipt-after-commit',[str(root/'.venv/bin/python'),'-c',probe])
for name,h in f['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
state=dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),worktree=str(root),code_revision=code,metadata_commit=head,clean=True,source_sha256=f['source_sha256'],main_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=main,text=True).strip(),main_status=subprocess.check_output(['git','status','--porcelain'],cwd=main,text=True),commands=commands,review='Original CC review received; independent scoped R1/R2 repair recheck PENDING',gpu_spend_usd=0,main_merge=False,push=False,new_experiments=0,compiler_matrix_reruns=0)
(b/'local-closure.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
