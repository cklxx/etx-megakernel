"""Lossless archival overlay for this task's NEW completed local-matrix cells.

The original manifest stays byte-identical. An independent overlay binds every
original file, gzip payload and exact decompression hash. Raw logs/configuration
remain raw; no old target build or certificate is admitted by this interface.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
from .local_matrix_prepare import ROOT,CELLS,FLOOR


def require(ok,why):
    if not ok:raise ValueError(why)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(out,value):
    temporary=out/'manifest.writing'
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(out/'manifest.json')


def regular(path):
    require(not path.is_symlink() and path.is_file() and path.stat().st_nlink==1,'requires regular unshared nonsymlink file '+str(path))


def archive_payload(path,expected):
    target=path.with_name(path.name+'.program-archive.gz');temporary=target.with_name(target.name+'.writing')
    require(not target.exists() and not target.is_symlink() and not temporary.exists() and not temporary.is_symlink(),'occupied archival output')
    regular(path);before=path.stat();h=hashlib.sha256();size=0
    with path.open('rb') as source,temporary.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            for chunk in iter(lambda:source.read(1<<20),b''):
                require(shutil.disk_usage(path.parent).free>=FLOOR+(2<<20),'storage floor plus chunk reserve; source preserved')
                h.update(chunk);size+=len(chunk);compressed.write(chunk)
        raw.flush();os.fsync(raw.fileno())
    require(h.hexdigest()==expected['sha256'] and size==expected['bytes'],'source changed during archival')
    row={'original_path':str(path),'uncompressed_sha256':h.hexdigest(),'uncompressed_bytes':size,'gzip_path':str(temporary),'gzip_sha256':digest(temporary),'gzip_bytes':temporary.stat().st_size}
    verify_payload(row)
    after=path.stat();require((before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns) and digest(path)==expected['sha256'],'source changed before archival publication')
    os.link(temporary,target);temporary.unlink();row['gzip_path']=str(target);verify_payload(row)
    return row


def verify_payload(row):
    path=Path(row['gzip_path']);regular(path);require(digest(path)==row['gzip_sha256'],'archive hash mismatch')
    h=hashlib.sha256();size=0
    with gzip.open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk);size+=len(chunk)
    require(h.hexdigest()==row['uncompressed_sha256'] and size==row['uncompressed_bytes'],'archive byte roundtrip mismatch')
    return True


def verify_overlay(path):
    value=json.loads(Path(path).read_text());require(value['status']=='COMPLETE','incomplete archival overlay')
    require(digest(value['original_manifest'])==value['original_manifest_sha256'],'original matrix manifest changed')
    root=Path(value['root']);require(digest(root/'.matrix-owner.json')==value['owner_receipt_sha256'],'owner receipt changed')
    for name,expected in value['original_files'].items():
        if name in value['archives']:
            row=value['archives'][name];verify_payload(row)
            require(row['uncompressed_sha256']==expected['sha256'] and row['uncompressed_bytes']==expected['bytes'],'overlay changed original bytes')
        else:
            p=root/name;regular(p);require(digest(p)==expected['sha256'] and p.stat().st_size==expected['bytes'],'unarchived original file changed')
    return True


def archive_matrix(root,manifest,out,*,expected_manifest_sha256,min_bytes=128<<10):
    root=Path(root).absolute();manifest=Path(manifest).absolute();out=Path(out).absolute()
    require(root==root.resolve() and root.parent==ROOT/'build','only the isolated task new matrix root is admitted')
    regular(manifest);require(digest(manifest)==expected_manifest_sha256,'original manifest SHA mismatch')
    value=json.loads(manifest.read_text());owner=root/'.matrix-owner.json';regular(owner)
    require(value['source_root']==str(ROOT) and digest(owner)==value['owner_receipt_sha256'],'matrix source/owner binding')
    owner_value=json.loads(owner.read_text());require(owner_value.get('schema')=='etx.local-matrix.owner.v1' and owner_value.get('root')==str(root) and bool(owner_value.get('token')),'new matrix owner receipt')
    require(not out.exists(),'refuse existing archival overlay')
    rows=value['results'];require(bool(rows),'no completed native cells')
    selections=[r['selection'] for r in rows];allowed={m+':'+v for m,v in CELLS}
    require(len(selections)==len(set(selections)) and set(selections)<=allowed,'duplicate or unknown native cells')
    for row in rows:
        require(row['exit_code']==0 and row['status'] in ('PASS','ARCHIVE_FAILED_SOURCE_PRESERVED'),'native cell did not finish successfully')
        require(row['command']==['bash','examples/vllm_llm/local_matrix.sh',str(root)],'native cell command/root mismatch')
        cell=root/row['selection'].replace(':','_');configuration=json.loads((cell/'configuration.json').read_text())
        require(configuration['model']+':'+configuration['variant']==row['selection'],'native cell configuration changed')
        for name,h in row.get('output_files_sha256',{}).items():
            require('..' not in Path(name).parts and not Path(name).is_absolute(),'invalid original file seal')
            p=cell/name;regular(p);require(digest(p)==h,'original native cell file changed')
    files={}
    for base,dirs,names in os.walk(root,followlinks=False):
        require(all(not (Path(base)/name).is_symlink() for name in dirs),'symlinked matrix directory')
        for name in names:
            p=Path(base)/name;regular(p);files[str(p.relative_to(root))]={'sha256':digest(p),'bytes':p.stat().st_size}
    candidates=[]
    for name,expected in files.items():
        p=root/name
        if p.parent.name not in {r.replace(':','_') for r in selections}:continue
        if expected['bytes']<min_bytes or p.name=='configuration.json' or p.suffix in ('.log','.gz','.writing','.rc','.exit') or p.name=='ir-archive.json':continue
        require(p.suffix in ('.ll','.bc','.s','.hip','.json','.h','.txt','.inc'),'unregistered large matrix intermediate '+name)
        require(not p.with_name(p.name+'.program-archive.gz').exists(),'occupied archival output')
        candidates.append(name)
    out.mkdir(parents=True)
    overlay={'schema':'etx.local-matrix.archive-overlay.v1','status':'ARCHIVING','root':str(root),'original_manifest':str(manifest),'original_manifest_sha256':expected_manifest_sha256,'owner_receipt_sha256':value['owner_receipt_sha256'],'owner_token_sha256':hashlib.sha256(owner_value['token'].encode()).hexdigest(),'selections':selections,'original_files':files,'archives':{},'raw_logs_and_config_preserved':True,'archiver_sha256':digest(__file__),'free_bytes_before':shutil.disk_usage(root).free}
    save(out,overlay)
    for name in sorted(candidates,key=lambda n:files[n]['bytes'],reverse=True):
        require(digest(manifest)==expected_manifest_sha256,'original manifest changed during archival')
        row=archive_payload(root/name,files[name]);row['state']='VERIFIED_BEFORE_UNLINK';overlay['archives'][name]=row;save(out,overlay)
        (root/name).unlink();row['state']='ARCHIVED';save(out,overlay)
        print('ARCHIVED',name,'raw_bytes',row['uncompressed_bytes'],'gzip_bytes',row['gzip_bytes'],'free_bytes',shutil.disk_usage(root).free,flush=True)
    overlay.update(status='COMPLETE',free_bytes_after=shutil.disk_usage(root).free);save(out,overlay);verify_overlay(out/'manifest.json')
    return overlay


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    v=archive_matrix(a.root,a.manifest,a.out,expected_manifest_sha256=a.manifest_sha256);print(json.dumps({'status':v['status'],'files_archived':len(v['archives']),'selections':v['selections'],'free_bytes_after':v['free_bytes_after']},indent=2))

if __name__=='__main__':main()
