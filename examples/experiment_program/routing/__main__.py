"""G04 complete native capture at original expert selector consumer boundaries.

Use an existing accepted model/prefix/golden set. No HF rerouting or downloads.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from . import decode_records
from ..protocol import require,sha
from etx.experiments.sealed_routing import ROLES,config,gates


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
