import fcntl,hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
b=Path(__file__).parent;root=Path('/Users/ckl/code/.wt/etx-deepseek-two-gaps')
lock=open(b.parent/'qwen-prep-heavy.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert shutil.disk_usage(b).free>4.5*1024**3
f=json.loads((b/'fixture.json').read_text());freeze=json.loads((b/'source-freeze.json').read_text())
commands=[]
with tempfile.TemporaryDirectory(prefix='entrypoint-probe-',dir=b) as td:
    archive=Path(td)
    p=subprocess.Popen(['git','archive',f['fixture_commit']],cwd=f['fixture_path'],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdout,mode='r|') as tar:tar.extractall(archive,filter='data')
    assert p.wait()==0 and not (archive/'.git').exists()
    for name,h in freeze['source_sha256'].items():assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==h
    for name,args in [('archive-b4-help',['-m','examples.dsv2lite.batch_attribution','--help']),('archive-b4-dry-run',['-m','etx.tools.gpu_jobs','examples/sessions/ds_b4_attribution.py','--gpus','0','--dry-run'])]:
        cmd=[str(root/'.venv/bin/python'),*args]
        r=subprocess.run(cmd,cwd=archive,env=dict(os.environ,PYTHONPATH=str(archive)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        (b/(name+'.log')).write_text(r.stdout);(b/(name+'.exit')).write_text(str(r.returncode)+'\n')
        commands.append(dict(name=name,command=cmd,exit=r.returncode,cwd='fresh no-Git source archive of '+f['fixture_commit']))
        assert r.returncode==0,r.stdout
r=subprocess.run(['git','diff','--check'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(b/'final-whitespace.log').write_text(r.stdout);commands.append(dict(command=['git','diff','--check'],exit=r.returncode));assert r.returncode==0
(b/'entrypoint-probe.json').write_text(json.dumps(dict(source_revision=freeze['revision'],fixture_commit=f['fixture_commit'],commands=commands,free_bytes=shutil.disk_usage(b).free,no_gpu_execution=True),indent=2)+'\n')
print(json.dumps(commands))
