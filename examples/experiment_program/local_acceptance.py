"""Local operational acceptance of prepared paths, never overall GPU readiness.

Synthetic CPU subprocess tests exercise orchestration failures and recovery.
The real retained October14 record is replayed only as historical parser input.
"""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys

from .protocol import require,sha
from .session import MODULES,SPECS
ROOT=Path(__file__).resolve().parents[2]
HISTORICAL=ROOT/'examples/dsv2lite/results/2026-10-14-batch-20261014T183414Z/20261014T183414Z-68d6f6/qwen/pairs_kept_l1.json'


def historical_replay(path=HISTORICAL):
    from .validation import model_rule
    path=Path(path);r=json.loads(path.read_text());rounds=r['rounds'];require(len(rounds)==5,'retained historical five-pair population')
    controls=[];candidates=[]
    for pair in rounds:
        require(set(pair)=={'kept','kept_l1'} and all(v['tokens']==32 for v in pair.values()),'retained historical arm/token identity')
        controls.append(int(Decimal(pair['kept']['wall_ms'])*1000000));candidates.append(int(Decimal(pair['kept_l1']['wall_ms'])*1000000))
    actual=model_rule(controls,candidates)
    require(actual['gains_ns']==[int(Decimal(str(v))*1000) for v in r['wall_ms']['kept']['gains_us']],'historical raw replay differs from retained wall account')
    require(actual['keep']==r['verdict']['passed'],'historical parser regression')
    return dict(historical_only=True,fresh_performance_claim=False,scope='October14 retained record parser replay; no new GPU run, certificate or cross-session comparison',source=str(path.resolve()),source_sha256=sha(path),tokens_per_invocation=32,retained_verdict=r['verdict']['outcome'],recomputed=actual)


def mapping():
    require(set(MODULES)==set(SPECS)=={f'G{i:02}' for i in range(1,14)},'exact13 record coverage')
    return [dict(record=k,module='examples.experiment_program.'+MODULES[k],kind=SPECS[k]['kind'],dependencies=SPECS[k]['dependencies'],gpu_execution='NOT_RUN',scope='entry mapping only; native admission/correctness/timing remains separately gated') for k in sorted(SPECS)]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--historical-record',type=Path,default=HISTORICAL);a=p.parse_args(argv)
    a.out.mkdir(parents=True,exist_ok=False);tmp=a.out/'tmp';tmp.mkdir()
    env=dict(os.environ,TMPDIR=str(tmp.resolve()),PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    result=dict(schema='etx.local-operational-acceptance.v1',gpu_executed=False,overall_gpu_ready='NOT_ASSESSED',independent_review='NOT_PERFORMED',record_mapping=mapping())
    try:
        with (a.out/'list13.log').open('w') as f:
            c=subprocess.run([sys.executable,'-m','examples.experiment_program','list'],cwd=ROOT,env=env,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=15)
        require(c.returncode==0,'real list13 CLI failed')
        values=json.loads((a.out/'list13.log').read_text());require({x['id'] for x in values}==set(SPECS) and len(values)==13 and all(x['enabled'] is False for x in values),'real list13 CLI coverage')
        result['historical_replay']=historical_replay(a.historical_record)
        command=[sys.executable,'-m','pytest','-q','tests/test_experiment_session_integration.py','--basetemp',str((a.out/'test-scratch').resolve())]
        result['cpu_test_command']=command;result['test_scope']='real safe CLI/list/dry-run/preflight + explicitly synthetic CPU orchestration children; no native timing simulation'
        with (a.out/'tests.log').open('w') as f:c=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
        result['cpu_test_exit']=c.returncode;result['status']='LOCAL_OPERATIONAL_TESTS_PASS' if c.returncode==0 else 'LOCAL_OPERATIONAL_TESTS_FAIL'
    except Exception as e:
        result['status']='LOCAL_OPERATIONAL_NOT_COMPLETE';result['error']=str(e)
    result['evidence']={str(v.resolve()):sha(v) for v in a.out.iterdir() if v.is_file()}
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['status']=='LOCAL_OPERATIONAL_TESTS_PASS' else 1
if __name__=='__main__':raise SystemExit(main())
