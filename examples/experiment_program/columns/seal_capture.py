"""Seal a completed original-model capture; no capture or GPU execution here.

The recipe names the complete original stdout, native row exports, original
checkpoint/input/prefix/golden files, capture-bound source and actual builds.
All raw task populations and per-request correctness lines are checked first.
"""
import argparse,json
from pathlib import Path
from examples.experiment_program.protocol import resolve_path,relocation_seal
from .protocol import require,sha256
from .capture_pack import decode_rows,verify_seal

def seal_capture(recipe,out):
    c=json.loads(resolve_path(recipe).read_text());require(c.get('record') in ('G02','G03','G05','G13'),'native capture target')
    for key in ('raw_files','input_files','source_files','build_files'):
        files=c.get(key);require(isinstance(files,list) and files,'recipe must list complete original '+key)
        c[key]={str(p):sha256(p) for p in files}
    c['input_files'].update(relocation_seal());c['schema']='etx.columns.native-producer.v1'
    c['log_sha256']=sha256(c['log']);c['full_greedy_tokens']=32;c['per_request_greedy']=[32]*c['batch']
    rows=[r for name in c['raw_files'] for r in decode_rows(resolve_path(name).read_bytes(),family=c['family'])]
    expected=c['batch']*296 if c['family']=='fleet' else 896 if c['record']=='G13' else (256 if all(r['h'][32]==2 for r in rows) else 512)
    require(len(rows)==expected,'native full task population differs from source-bound target')
    c['native_task_count']=len(rows);require(c['source_path'] in c['source_files'],'original native source identity must be sealed')
    require(c['weights'] in c['input_files'],'original native checkpoint must be sealed')
    verify_seal(c)
    with Path(out).open('x') as f:f.write(json.dumps(c,indent=2)+'\n')
    return dict(status='NATIVE_PRODUCER_SEALED',record=c['record'],batch=c['batch'],native_task_count=len(rows),seal_sha256=sha256(out))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(seal_capture(a.recipe,a.out),indent=2))
if __name__=='__main__':main()
