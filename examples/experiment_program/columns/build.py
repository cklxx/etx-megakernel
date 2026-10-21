"""Offline module preparation; no GPU execution, pull, or source downloads."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, subprocess
from pathlib import Path
from .protocol import require, sha256

SOURCES = {
 'b5eaa67': 'd1688690864d9d1f98e25abb0077913cc895ee429a2d915957e35ff747d47b45',
 'ff9ef92': 'cfb1d1f155b49e3cf6e57a9f2b426106f14bd0724d0b4c3722f55b279088de07',
}

def extract_stream(source, revision):
    require(sha256(source) == SOURCES[revision], 'historical arithmetic source hash mismatch')
    tree=ast.parse(Path(source).read_text())
    stream = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STREAM' for t in n.targets))
    marker = '// NB activation vectors'
    require(stream.count(marker)==1, 'historical source extraction marker changed')
    return stream[:stream.index(marker)]

def prepare(out, evidence):
    from examples.dsv2lite.pinned import fleet_export
    fleet = fleet_export()
    cooked=Path(os.environ.get('ETX_COLUMN_FLEET_SRC',str(Path.home()/'code/etx-experiment-columns-build/cooked-fleet')))
    from examples.dsv2lite.pinned import source_sha
    require(cooked.is_dir() and source_sha(cooked)=='b75aefc98ce71d0f60c99e7e4e262d37a86faa25a7ade490ea330cebd81e08fb','accepted K1/M1/l1 Fleet source export missing/mismatch')
    os.environ['ETX_COLUMN_FLEET_SRC']=str(cooked.resolve())
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[3]
    sealed={}
    for revision in SOURCES:
        source=Path(evidence)/'source'/revision/'etx/ops/gemv.py'
        text=extract_stream(source,revision)
        (out/f'historical_{revision}.h').write_text(text)
        sealed[revision]={'path':str(source),'sha256':sha256(source),'extracted_sha256':hashlib.sha256(text.encode()).hexdigest()}
    sealed['fleet']={'path':str(fleet),'accepted_cooked_source':str(cooked),'accepted_cooked_sha256':source_sha(cooked),'gemv_sha256':sha256(cooked/'kernels/gemv.h')}
    (out/'sources.json').write_text(json.dumps(sealed,indent=2)+'\n')
    return root,fleet,out

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--evidence',type=Path,default=Path.home()/'.hermes/cache/scratch/etx-shared-weight-causality')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv)
    root,fleet,out=prepare(a.out,a.evidence)
    if a.prepare_only: print(json.dumps({'out':str(out),'status':'SOURCES_VERIFIED'})); return
    script=Path(__file__).with_name('build.sh')
    subprocess.run(['bash',str(script),str(root),str(fleet),str(out)],check=True)
if __name__=='__main__': main()
