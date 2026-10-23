"""Local staged experiment session; no provisioning, SSH, cloud or publication.

All device entry points default off. Future execution is separately authorized.
"""
import argparse
from pathlib import Path

from etx.external import SealedSessionRequest, run_sealed_session


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['list','preflight','dry-run','run','export'])
    p.add_argument('--config',type=Path)
    p.add_argument('--out',type=Path)
    records=[row['id'] for row in run_sealed_session(SealedSessionRequest('list'))['records']]
    p.add_argument('--record',action='append',choices=records)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    run_sealed_session(SealedSessionRequest(
        action=a.action,config=str(a.config) if a.config is not None else None,
        output=str(a.out) if a.out is not None else None,records=tuple(a.record or ()),
        resume=a.resume,execute=a.execute),render=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
