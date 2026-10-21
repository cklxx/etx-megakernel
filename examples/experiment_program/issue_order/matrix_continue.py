"""Finish unbuilt cells after a sealed, losslessly archived local-matrix stop.

No passed cell is compiled again. Each remaining cell owns a fresh root, retains
its immutable native receipt, and is archived before another cell starts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid
from .local_matrix_prepare import ROOT,CELLS,FLOOR,BUDGET,allocated_bytes
from .matrix_archive import archive_matrix,verify_overlay,digest,require


def remaining_cells(previous):
    rows=previous.get('results',[]);names=[r.get('selection') for r in rows]
    all_names=[m+':'+v for m,v in CELLS]
    require(bool(rows) and len(names)==len(set(names)) and names==all_names[:len(names)],'ambiguous completed native prefix')
    require(all(r.get('exit_code')==0 and r.get('status') in ('PASS','ARCHIVE_FAILED_SOURCE_PRESERVED') for r in rows),'native completion required, not pipeline status alone')
    return CELLS[len(names):]


def write(path,value):
    temporary=path.with_name(path.name+'.writing');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original-manifest',type=Path,required=True);p.add_argument('--original-sha256',required=True);p.add_argument('--archive-overlay',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--combined-out',type=Path,required=True);a=p.parse_args()
    require(digest(a.original_manifest)==a.original_sha256,'changed original matrix receipt');verify_overlay(a.archive_overlay)
    previous=json.loads(a.original_manifest.read_text());overlay=json.loads(a.archive_overlay.read_text())
    require(overlay['original_manifest_sha256']==a.original_sha256,'archive/original binding')
    todo=remaining_cells(previous);require(bool(todo),'no unbuilt native cells')
    out=a.out.absolute();combined=a.combined_out.absolute();require(not out.exists() and not combined.exists(),'fresh continuation and combined receipt paths required')
    roots=[ROOT/'build'/('local-matrix-cont-'+str(i+1).zfill(2)+'-'+m+'_'+v) for i,(m,v) in enumerate(todo)]
    require(all(not x.exists() for x in roots),'refuse occupied continuation root')
    out.mkdir(parents=True)
    state={'schema':'etx.local-matrix.continuation.v1','status':'RUNNING','original_manifest':str(a.original_manifest),'original_sha256':a.original_sha256,'original_archive_overlay':str(a.archive_overlay),'original_archive_overlay_sha256':digest(a.archive_overlay),'source_hashes':{str(Path(__file__)):digest(__file__),str(Path(__file__).with_name('matrix_archive.py')):digest(Path(__file__).with_name('matrix_archive.py')),str(ROOT/'examples/vllm_llm/local_matrix.sh'):digest(ROOT/'examples/vllm_llm/local_matrix.sh')},'already_native_pass':len(previous['results']),'planned_selections':[m+':'+v for m,v in todo],'results':[],'gpu_execution':'NOT_RUN','network':'NOT_USED','reserve_bytes_before_cell':40<<20,'early_stop_headroom_bytes':8<<20,'monitor_seconds':.25}
    write(out/'manifest.json',state)
    for index,((model,variant),root) in enumerate(zip(todo,roots)):
        free=shutil.disk_usage(ROOT).free;total=allocated_bytes(ROOT/'build')
        if free<FLOOR+(40<<20) or total>=BUDGET:
            state.update(status='BLOCKED_STORAGE_BEFORE_CELL',blocked_selection=model+':'+variant,free_bytes=free,allocated_bytes=total);write(out/'manifest.json',state);raise SystemExit(3)
        require(digest(a.original_manifest)==a.original_sha256,'original receipt changed')
        root.mkdir();token=uuid.uuid4().hex;owner=root/'.matrix-owner.json';owner.write_text(json.dumps({'schema':'etx.local-matrix.owner.v1','root':str(root),'token':token},indent=2)+'\n')
        cell_out=out/(str(index+1).zfill(2)+'-'+model+'_'+variant);cell_out.mkdir();command=['bash','examples/vllm_llm/local_matrix.sh',str(root)];selection=model+':'+variant
        env=dict(previous['environment'],LOCAL_MATRIX=selection)
        row={'selection':selection,'command':command,'environment':env,'root':str(root),'started_unix':time.time(),'free_bytes_before':free,'reserve_bytes':40<<20};reason=None
        print('CONTINUATION_START',selection,'free_bytes',free,flush=True)
        with (cell_out/'native.raw.log').open('w') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**env),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);last_progress=0
            while child.poll() is None:
                elapsed=time.time()-row['started_unix'];available=shutil.disk_usage(ROOT).free
                if elapsed>600:reason='DEADLINE_600S'
                if available<FLOOR+(8<<20):reason='EARLY_STORAGE_HEADROOM_STOP'
                if reason:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        child.wait()
                    break
                if elapsed-last_progress>=5:print('CONTINUATION_PROGRESS',selection,'elapsed_s',round(elapsed,1),'free_bytes',available,flush=True);last_progress=elapsed
                time.sleep(.25)
        row.update(exit_code=child.returncode,status=reason or ('PASS' if child.returncode==0 else 'FAILED'),elapsed_seconds=time.time()-row['started_unix'],free_bytes_after_native=shutil.disk_usage(ROOT).free,raw_log=str(cell_out/'native.raw.log'),raw_log_sha256=digest(cell_out/'native.raw.log'))
        cell=root/(model+'_'+variant);row['output_files_sha256']={str(x.relative_to(cell)):digest(x) for x in sorted(cell.rglob('*')) if x.is_file()}
        native={'schema':'etx.local-matrix.single-native.v1','source_root':str(ROOT),'environment':previous['environment'],'owner_receipt_sha256':digest(owner),'results':[row]};raw_manifest=cell_out/'native-manifest.json';raw_manifest.write_text(json.dumps(native,indent=2)+'\n')
        row['native_manifest']=str(raw_manifest);row['native_manifest_sha256']=digest(raw_manifest)
        if row['status']!='PASS':state['results'].append(row);state['status']='BLOCKED_NATIVE';write(out/'manifest.json',state);raise SystemExit(1)
        archived=archive_matrix(root,raw_manifest,cell_out/'archive',expected_manifest_sha256=digest(raw_manifest));row.update(archive_overlay=str(cell_out/'archive/manifest.json'),archive_overlay_sha256=digest(cell_out/'archive/manifest.json'),free_bytes_after_archive=archived['free_bytes_after'])
        state['results'].append(row);write(out/'manifest.json',state);print('CONTINUATION_PASS',selection,'free_bytes',archived['free_bytes_after'],flush=True)
    verify_overlay(a.archive_overlay)
    for row in state['results']:verify_overlay(row['archive_overlay'])
    state['status']='PASS';write(out/'manifest.json',state)
    old=[dict(row,native_compile_status='PASS',original_attempt_status=row['status'],archive_overlay=str(a.archive_overlay),archive_overlay_sha256=digest(a.archive_overlay)) for row in previous['results']]
    all_rows=old+state['results'];require([r['selection'] for r in all_rows]==[m+':'+v for m,v in CELLS] and len(all_rows)==16 and all(r['exit_code']==0 for r in all_rows),'exact16 native once-only population')
    combined.mkdir(parents=True)
    final={'schema':'etx.local-matrix.combined.v1','status':'PASS','native_cells':16,'each_cell_native_attempts':1,'original_attempt_manifest':str(a.original_manifest),'original_attempt_sha256':a.original_sha256,'original_attempt_pipeline_status':previous['status'],'original_pipeline_failure_preserved':True,'original_archive_overlay':str(a.archive_overlay),'original_archive_overlay_sha256':digest(a.archive_overlay),'continuation_manifest':str(out/'manifest.json'),'continuation_manifest_sha256':digest(out/'manifest.json'),'results':all_rows,'gpu_execution':'NOT_RUN','network':'NOT_USED','raw_full_logs_preserved':True,'source_hashes':state['source_hashes'],'final_free_bytes':shutil.disk_usage(ROOT).free}
    write(combined/'manifest.json',final);print('ALL16_NATIVE_PASS',str(combined/'manifest.json'),flush=True)

if __name__=='__main__':main()
