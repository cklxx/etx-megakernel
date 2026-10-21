"""G04 complete native capture at original expert selector consumer boundaries.

Use an existing accepted model/prefix/golden set. No HF rerouting or downloads.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from . import decode_records
from ..protocol import require,verify_files,sha,resolve_path,relocation_seal

ROLES={'source','plan','executable','compiler','math_flags','resources','weights','inputs','prefixes','goldens','golden_producers'}


def config(path):
    c=json.loads(resolve_path(path).read_text())
    require(c.get('batch') in (2,4),'capture B2/B4')
    require(set(c.get('roles',{}))==ROLES,'complete capture identity roles required')
    for files in c['roles'].values():verify_files(files)
    require(isinstance(c.get('command'),list) and all(isinstance(x,str) for x in c['command']) and c['command'],'native capture argv')
    require(resolve_path(c['command'][0]) in [resolve_path(p) for p in c['roles']['executable']],'command executable not bound')
    c['command']=[str(resolve_path(s)) if Path(s).is_absolute() else s for s in c['command']]
    all_seals={p:h for files in c['roles'].values() for p,h in files.items()}
    require(all(all_seals.get(p)==h for p,h in relocation_seal().items()),'explicit staging map must be in capture source seal')
    flags='\n'.join(resolve_path(p).read_text() for p in c['roles']['math_flags'])
    require('ETX_PROGRAM_ROUTE_CAPTURE=1' in flags,'capture flag must belong to actual build')
    require(0<c.get('deadline_seconds',0)<=900,'bounded capture deadline')
    return c


def gates(text,batch):
    require(f'G04 CAPTURE COMPLETE requests={batch} steps=32 routed_layers=26 replicas=8 slots=6 dense_layer=0' in text,'incomplete native capture')
    require(not re.search(r'MISMATCH|\[FAIL\]|REFUSED|no reference token',text),'capture correctness failure')
    rows=re.findall(r'^BATCHREQ r=(\d+) tokens=(\d+/\d+) layers=(\d+) warmup=(\d+/\d+)',text,re.M)
    require(len(rows)==batch and sorted(int(x[0]) for x in rows)==list(range(batch)),'missing/duplicate requests')
    require(all(x[1]=='32/32' and x[2]=='27' and x[3]=='32/32' for x in rows),'per-request warmup/greedy/hidden gate')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','dry-run','capture','analyze'])
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    c=config(a.config)
    if a.action in ('preflight','dry-run'):
        print(json.dumps(dict(record='G04',command=c['command'],environment={'ETX_PROGRAM_ROUTE_DIR':str(a.out.resolve())},
                             status='LOCAL_PREFLIGHT_PASS',enabled=False,capture_only=True,performance_claim=False)));return 0
    if a.action=='capture':
        require(a.execute,'native execution disabled by default')
        a.out.mkdir(parents=True,exist_ok=False)
        with (a.out/'capture.log').open('w') as f:
            subprocess.run(c['command'],env=dict(os.environ,ETX_PROGRAM_ROUTE_DIR=str(a.out.resolve())),stdout=f,stderr=subprocess.STDOUT,timeout=c['deadline_seconds'],check=True)
    # Recheck original identities before consuming produced bytes. A capture
    # timeout/failure leaves raw data available but no complete observation seal.
    c=config(a.config);gates((a.out/'capture.log').read_text(),c['batch'])
    files=[a.out/f'routes_r{r}.bin' for r in range(c['batch'])]
    report=decode_records([p.read_bytes() for p in files],batch=c['batch'])
    report.update(record='G04',complete=True,roles=c['roles'],raw_files={str(p):sha(p) for p in files},
                  log_sha256=sha(a.out/'capture.log'),config_sha256=sha(a.config))
    (a.out/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='slot_ids'},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
