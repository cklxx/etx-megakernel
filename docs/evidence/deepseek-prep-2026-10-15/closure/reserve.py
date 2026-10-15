import fcntl,gzip,hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/Users/ckl/.hermes/cache/scratch/deepseek-closure-evidence')
old=base.parent/'deepseek-open-prep-evidence'
lock=open(base.parent/'qwen-prep-heavy.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.seek(0);lock.truncate();lock.write(f'deepseek-reserve pid={os.getpid()}\n');lock.flush()
src=old/'source-a0c3e72.tar';dst=old/'source-a0c3e72.tar.gz'
expected='465824e7d42c90fafd88a20b4050b24751f12d89042ee05bf417668dc3f70e5c'
def digest(f):
 h=hashlib.sha256()
 while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
start=shutil.disk_usage(base).free
with src.open('rb') as f:assert digest(f)==expected
with src.open('rb') as i,gzip.open(dst,'xb',compresslevel=1) as o:shutil.copyfileobj(i,o,8*1024**2)
with gzip.open(dst,'rb') as f:assert digest(f)==expected
src.unlink()
expanded=old/(expected+'-deepseek-prep')
assert expanded.is_dir() and not expanded.is_symlink()
# This directory is only our extracted, immutable git archive plus transient test caches.
shutil.rmtree(expanded)
r=dict(action='lossless compression of own reproducible source tar and removal of duplicate extracted source only',archive=str(dst),decompressed_sha256=expected,compressed_bytes=dst.stat().st_size,removed_duplicate=str(expanded),raw_logs_and_build_artifacts='retained',free_before=start,free_after=shutil.disk_usage(base).free)
(base/'reserve.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
