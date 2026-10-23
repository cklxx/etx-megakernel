"""Write a concrete thirteen-record local session from staged-path inventory.

This copies no model data and executes no build/device/remote command. Missing
native captures/configuration remain explicit sealed missing prerequisites.
"""
import argparse
import json
from pathlib import Path

from etx.external import SealedPreparationRequest, prepare_sealed_session


def prepare(inventory,destination,output):
    return prepare_sealed_session(SealedPreparationRequest(
        inventory=inventory,destination=str(destination),output=str(output))).to_dict()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--session-output',type=Path,required=True)
    a=p.parse_args(argv)
    r=prepare(json.loads(a.inventory.read_text()),a.out,a.session_output)
    print(json.dumps(dict(configuration=str(a.out.resolve()),records=list(r['records']),device_execution=False,missing_paths=[path for s in r['records'].values() for path,h in s['input_files'].items() if h=='0'*64]),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
