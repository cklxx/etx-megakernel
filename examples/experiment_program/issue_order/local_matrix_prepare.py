"""Run the required unchanged16-cell local matrix serially, without devices.

Ordinary invocation only writes its exact command/precondition manifest. Execute
is reserved for the parent's serialized local compiler slot. Outputs are fresh,
all raw logs retained, and storage/deadline failures stop owned processes only.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

ROOT=Path(__file__).resolve().parents[3]
MODELS=('qwen3-8b','qwen2.5-1.5b')
CELLS=tuple((m,v) for m in MODELS for v in ('base','chain','xcd','xcd_s1','uc','uc_s1'))+tuple((m,v) for m in ('qwen3-30b-a3b','mixtral-8x7b') for v in ('base','chain'))
FLOOR=2*1024**3
BUDGET=14*1024**3//10

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_ir_archive(receipt):
    path=Path(receipt['gzip_path'])
    if path.is_symlink() or digest(path)!=receipt['gzip_sha256']:raise ValueError('matrix IR archive hash mismatch')
    h=hashlib.sha256();size=0
    with gzip.open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk);size+=len(chunk)
    if size!=receipt['uncompressed_bytes'] or h.hexdigest()!=receipt['uncompressed_sha256']:raise ValueError('matrix IR archive roundtrip mismatch')
    return True

def archive_new_ir(cell,owned_root,token,*,completed_exit):
    """Losslessly archive only this invocation's newly finished successful IR.

    The caller creates the fresh root and exclusive owner receipt before any
    child runs. Existing build/certificate trees cannot enter this path.
    Native source/assembly and every command log stay raw and unchanged.
    """
    root=Path(owned_root).absolute();cell=Path(cell).absolute()
    if root.is_symlink() or cell.is_symlink() or cell.parent!=root:raise ValueError('archive requires direct new owned cell')
    owner=json.loads((root/'.matrix-owner.json').read_text())
    if owner!={'schema':'etx.local-matrix.owner.v1','root':str(root),'token':token}:raise ValueError('matrix ownership mismatch')
    if type(completed_exit) is not int or completed_exit!=0:raise ValueError('only completed successful cells can archive IR')
    source=cell/'mk.ll';target=cell/'mk.ll.gz';temporary=cell/'mk.ll.gz.writing';receipt_path=cell/'ir-archive.json'
    if source.is_symlink() or not source.is_file() or source.stat().st_nlink!=1:raise ValueError('archive requires new regular unshared IR')
    if any(p.exists() or p.is_symlink() for p in (target,temporary,receipt_path)):raise FileExistsError('refuse occupied IR archive')
    before=source.stat();h=hashlib.sha256();size=0
    with source.open('rb') as src,temporary.open('xb') as output:
        with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0) as compressed:
            for chunk in iter(lambda:src.read(1<<20),b''):
                if shutil.disk_usage(cell).free<FLOOR+(2<<20):raise ValueError('storage floor before IR archive write; source preserved')
                h.update(chunk);size+=len(chunk);compressed.write(chunk)
        output.flush();os.fsync(output.fileno())
    receipt={'original_path':str(source),'uncompressed_sha256':h.hexdigest(),'uncompressed_bytes':size,'gzip_path':str(temporary),'gzip_sha256':digest(temporary),'gzip_bytes':temporary.stat().st_size,'roundtrip_verified':False,'original_command_exit':completed_exit}
    verify_ir_archive(receipt)
    after=source.stat()
    if (before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns) or digest(source)!=receipt['uncompressed_sha256']:raise ValueError('new IR changed during archival; preserve source')
    os.link(temporary,target)  # exclusive publication; never overwrite evidence
    temporary.unlink();receipt['gzip_path']=str(target);verify_ir_archive(receipt)
    receipt['roundtrip_verified']=True
    with receipt_path.open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
    source.unlink()  # only the exact new bytes proven reconstructible above
    return receipt

def allocated_bytes(path):
    # du reports allocated blocks and does not follow links into preserved trees.
    result=subprocess.check_output(['du','-sk',str(path)],text=True)
    return int(result.split()[0])*1024

def storage(out):
    free=shutil.disk_usage(out.parent).free
    total=allocated_bytes(ROOT/'build') if (ROOT/'build').exists() else 0
    return {'free_bytes':free,'new_worktree_build_allocated_bytes':total,'floor_bytes':FLOOR,'total_budget_bytes':BUDGET,'passed':free>=FLOOR and total<BUDGET}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--deadline-seconds',type=int,default=300);p.add_argument('--archive-new-ir',action='store_true',help='losslessly gzip only this invocation\'s successful new mk.ll files, with SHA and roundtrip proof')
    a=p.parse_args();out=a.out.resolve();evidence=a.evidence.resolve()
    if ROOT not in out.parents or out.parent!=ROOT/'build':raise ValueError('matrix output must be a new direct child of this isolated worktree build')
    if out.exists() or evidence.exists():raise ValueError('fresh matrix output and evidence paths required; preserved outputs are never overwritten')
    if not 30<=a.deadline_seconds<=600:raise ValueError('bounded per-cell deadline required')
    out.parent.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True)
    headers=Path.home()/'code/rocm-headers';llvm=Path('/opt/homebrew/opt/llvm/bin');python=Path('/Users/ckl/code/etx/.venv/bin/python');devlibs=headers/'devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode'
    values={'ETX_PYTHON':str(python),'ETX_LLVM':str(llvm),'ETX_ROCM_HEADERS':str(headers),'ETX_DEVLIBS':str(devlibs),'ETX_VLLM':str(Path.home()/'code/vllm'),'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'}
    required=[python,llvm/'clang++',llvm/'llc',llvm/'llvm-link',llvm/'opt',devlibs/'ocml.bc',devlibs/'ockl.bc',headers/'inc/hip/hip_runtime.h']
    missing=[str(x) for x in required if not x.is_file()]
    syntax=subprocess.run(['bash','-n','examples/vllm_llm/local_matrix.sh'],cwd=ROOT,capture_output=True,text=True)
    (evidence/'shell-syntax.log').write_text(syntax.stdout+syntax.stderr)
    manifest={'schema':'etx.local-matrix.serial.v1','status':'PREPARED_NOT_EXECUTED','source_root':str(ROOT),'wrapper_sha256':digest(__file__),'matrix_script_sha256':digest(ROOT/'examples/vllm_llm/local_matrix.sh'),'environment':values,'cells':[{'selection':m+':'+v,'argv':['bash','examples/vllm_llm/local_matrix.sh',str(out)]} for m,v in CELLS],'initial_storage':storage(out),'missing':missing,'syntax_exit':syntax.returncode,'gpu_execution':'NOT_RUN','network':'NOT_USED','results':[]}
    def save():
        temporary=evidence/'manifest.writing';temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(evidence/'manifest.json')
    save()
    if missing or syntax.returncode:raise SystemExit('local prerequisite failed; exact receipt retained')
    if not a.execute:print(json.dumps(manifest,indent=2));return
    out.mkdir();token=uuid.uuid4().hex
    (out/'.matrix-owner.json').write_text(json.dumps({'schema':'etx.local-matrix.owner.v1','root':str(out),'token':token},indent=2)+'\n')
    manifest['archive_new_ir']=a.archive_new_ir;manifest['owner_receipt_sha256']=digest(out/'.matrix-owner.json');manifest['status']='RUNNING';save()
    for model,variant in CELLS:
        selection=model+':'+variant;receipt={'selection':selection,'before':storage(out),'started_unix':time.time(),'command':['bash','examples/vllm_llm/local_matrix.sh',str(out)],'environment':dict(values,LOCAL_MATRIX=selection)}
        if not receipt['before']['passed']:
            receipt.update(status='BLOCKED_STORAGE',exit_code=None);manifest['results'].append(receipt);manifest['status']='BLOCKED_STORAGE';save();raise SystemExit(3)
        env=dict(os.environ,**values,LOCAL_MATRIX=selection)
        print('MATRIX_START',selection,flush=True)
        with (evidence/(model+'_'+variant+'.raw.log')).open('w') as stream:
            child=subprocess.Popen(receipt['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+a.deadline_seconds;reason=None
            while child.poll() is None:
                if time.monotonic()>=deadline:reason='DEADLINE'
                check=storage(out)
                if not check['passed']:reason='STORAGE_FLOOR_OR_BUDGET'
                if reason:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        child.wait()
                    break
                time.sleep(5)
                print('MATRIX_PROGRESS',selection,'elapsed_s',round(time.time()-receipt['started_unix'],1),flush=True)
            receipt.update(exit_code=child.returncode,status=reason or ('PASS' if child.returncode==0 else 'FAILED'),elapsed_seconds=time.time()-receipt['started_unix'],after=storage(out))
        if receipt['status']=='PASS' and a.archive_new_ir:
            try:receipt['ir_archive']=archive_new_ir(out/(model+'_'+variant),out,token,completed_exit=child.returncode)
            except (OSError,ValueError) as exc:
                receipt.update(status='ARCHIVE_FAILED_SOURCE_PRESERVED',archive_error=str(exc));manifest['results'].append(receipt);manifest['status']='BLOCKED_ARCHIVE';save();raise SystemExit(3)
            receipt['after_archive']=storage(out)
        cell=out/(model+'_'+variant)
        receipt['output_files_sha256']={str(p.relative_to(cell)):digest(p) for p in sorted(cell.rglob('*')) if p.is_file()}
        # A failed cell is retained and the remaining independent cells still run.
        manifest['results'].append(receipt);save();print('MATRIX_EXIT',selection,receipt['exit_code'],receipt['status'],flush=True)
        if reason:manifest['status']='BLOCKED_'+reason;save();raise SystemExit(3)
    manifest['status']='PASS' if all(x['status']=='PASS' for x in manifest['results']) else 'COMPLETED_WITH_FAILURES';save()
    raise SystemExit(0 if manifest['status']=='PASS' else 1)

if __name__=='__main__':main()
