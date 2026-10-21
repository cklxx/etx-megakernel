"""Bounded G02/G03/G05/G13 preparation and future opt-in native replay."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from examples.experiment_program.protocol import resolve_path
from .protocol import ProtocolError,require
from .runner import admit_native,write_launch,run_native,identity_resource_admitted

RECORDS={
 'G02': {'prerequisites': [], 'mechanism':'one native weight load, two/four ordered accumulators', 'roles':['qkv','gate_up','down']},
 'G03': {'prerequisites':['G02'], 'mechanism':'historical NB1/NB2 layout x simultaneous/retained-arrival readiness', 'roles':['qkv']},
 'G05': {'prerequisites':['G04'], 'mechanism':'stable expert ID grouping, gather, shared columns, original-slot scatter; singleton original tile', 'roles':['gate_up','down']},
 'G13': {'prerequisites':['G02'], 'mechanism':'identical expert up/down bodies; combined/independent request event edges', 'roles':['expert_up_down']},
}
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--record',choices=RECORDS,required=True)
 p.add_argument('--pending-winner',action='append',default=[],help='dry-run conditional prerequisite only; never admitted')
 p.add_argument('--program',type=Path,help='complete registered serial cases; no arbitrary commands')
 p.add_argument('--capture',type=Path,action='append',default=[],help='sealed native capture JSON, Fleet positions1024/1040; historical positions0/1040')
 p.add_argument('--build',type=Path,default=Path.home()/'code/etx-experiment-columns-build')
 p.add_argument('--out',type=Path)
 p.add_argument('--dependency',type=Path,action='append',default=[])
 p.add_argument('--regime',choices=['both','simultaneous','archived'],default='both')
 p.add_argument('--g13-order',choices=['request','both-up-first'],default='request')
 p.add_argument('--warmup',type=int,default=3);p.add_argument('--seed',type=int,default=20261016)
 p.add_argument('--timeout',type=int,default=120)
 modes=p.add_mutually_exclusive_group();modes.add_argument('--execute',action='store_true');modes.add_argument('--dry-run',action='store_true')
 a=p.parse_args(argv)
 try:
  require(not a.pending_winner or not a.execute,'pending winners are dry-run only')
  require(a.warmup>=1 and 1<=a.timeout<=300,'invalid bounded run population/deadline')
  if a.program:
   require(not a.capture and not a.dependency,'program owns its sealed capture/dependency paths')
   from .program import run_program
   result=run_program(a.program,a.record,a.out,execute=a.execute,case_main=main)
   print(json.dumps(result,indent=2));return 0
  spec=RECORDS[a.record]
  report={'record':a.record,'enabled':False,'scope':spec,'device_outcome':'UNMEASURED',
          'native_input_status':'MISSING','claim_scope':'isolated subgraph only; no full-model/fusion/system certificate',
          'source_identity_note':'new native captures cannot recover historical run identity for G03/G13'}
  if not a.capture:
   report['status']='BLOCKED_NATIVE_CAPTURE';report['reason']='Sealed producer-bound post-prologue native operands, original outputs and actual placement not present in retained model goldens.'
   print(json.dumps(report,indent=2));return 2 if a.execute else 0
  captures=[json.loads(resolve_path(path).read_text()) for path in a.capture]
  for c in captures:
   require(c['record']==a.record and c['role'] in spec['roles'],'capture record/role mismatch')
   admit_native(c,a.build)
  require({c['decode_position'] for c in captures}==({1024,1040} if captures[0]['family']=='fleet' else {0,1040}),'both producer-bound decode positions required')
  require(len(captures)==2,'duplicate native capture/decode position')
  require(all(all(c.get(key)==captures[0].get(key) for key in ('batch','role','N','K','ld','source_revision','source_sha256','native_xcd')) for c in captures),'native role/batch/layout/source differs across positions')
  require(all(any(t>0 for t in c['arrivals_ticks']) for c in captures),'one captured nonzero readiness stagger required')
  from .runner import verify_dependency
  receipts=[verify_dependency(path,needed=spec['prerequisites']) for path in a.dependency]
  for dep in spec['prerequisites']:
   require(any(x.get('record')==dep for x in receipts) or bool(a.pending_winner),'dependency not qualified: '+dep)
  if a.record=='G02' and captures[0]['role']!='qkv':
   require(any(x.get('record')=='G02' and x.get('role')=='qkv' and x.get('batch')==captures[0]['batch'] and x.get('admitted') for x in receipts) or a.pending_winner==['qkv-B'+str(captures[0]['batch'])],'shared-expert role requires its exact B QKV pass in both regimes')
  if a.pending_winner:report['pending_winners']=a.pending_winner
  report.update(status='PREPARED_CONDITIONAL' if a.pending_winner else 'PREPARED_AWAITING_NATIVE',native_input_status='VERIFIED',enabled=bool(a.execute))
  if not a.execute: print(json.dumps(report,indent=2));return 0
  require(a.out is not None,'--out required for full raw logs')
  require(not a.out.exists(),'refusing occupied output; no implicit overwrite/resume')
  a.out.mkdir(parents=True)
  regimes=['simultaneous','archived'] if a.regime=='both' else [a.regime]
  decisions=[]
  for c in captures:
   for regime in regimes:
    out=a.out/f"p{c['decode_position']}-{regime}"
    command=write_launch(c,a.build,out,regime,warmup=a.warmup,seed=a.seed,g13_order=a.g13_order)
    decisions.append({'position':c['decode_position'],'regime':regime,**run_native(command,out,timeout=a.timeout)})
  report['relocation_seal']=__import__('examples.experiment_program.protocol',fromlist=['relocation_seal']).relocation_seal()
  report['build']=str(a.build.resolve())
  report['decisions']=decisions
  report.update(native_xcd=captures[0].get('native_xcd'),batch=captures[0]['batch'],role=captures[0]['role'],native_exact=True,resource_admitted=identity_resource_admitted(a.build,captures),native_pick_coverage=a.record=='G05',singleton_fallback_passed=a.record=='G05' and all(c.get('real_mix') is True and any(g['singleton'] for g in __import__('examples.experiment_program.columns.protocol',fromlist=['stable_groups']).stable_groups(c['picks'])) for c in captures) and all(d['verdict']=='MICRO_PASS' for d in decisions),build_identity_sha256=__import__('examples.experiment_program.columns.protocol',fromlist=['sha256']).sha256(a.build/'identity.json'))
  report['status']='MICRO_PASS_BOTH_REGIMES' if len(regimes)==2 and all(d['verdict']=='MICRO_PASS' for d in decisions) else 'NO_MODEL_ADMISSION'
  report['admitted']=a.record in ('G02','G05') and report['status']=='MICRO_PASS_BOTH_REGIMES' and report['resource_admitted']
  report['model_admitted']=False
  if a.record in ('G03','G13'):
   from .runner import diagnose_readiness
   report['diagnostic']=diagnose_readiness(a.out,captures)
   report['independent_order']=a.g13_order if a.record=='G13' else None
  (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0
 except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError) as exc:
  print(json.dumps({'record':a.record,'status':'REFUSED','reason':str(exc),'device_outcome':'UNMEASURED'}),file=sys.stderr);return 2
if __name__=='__main__': raise SystemExit(main())
