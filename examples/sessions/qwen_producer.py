"""One prepared MI300X lane. No GPU operation on import or dry run.

Requires pre-staged inputs, local-ready.json, compiler and cached vLLM image.
The persistent lease prevents a second runner and retains the first budget.
"""
import os
import math

MAX_SESSION_SECONDS=3600
cap=float(os.environ.get('ETX_SESS_MAX_SECONDS',MAX_SESSION_SECONDS))
if not math.isfinite(cap) or not 0 < cap <= MAX_SESSION_SECONDS:
    raise ValueError('producer session cap must be positive and <=3600s')
SESSION_LEASE='~/.cache/etx-qwen-producer-session.lease'
LOCAL_ONLY=() # Retain build, capture and raw evidence for this bounded session.
OP='.venv/bin/python -u -m examples.vllm_llm.producer_session'
CONFIG=os.environ.get('ETX_QWEN_G6_CONFIG',os.path.expanduser('~/qwen-g6.json'))
JOBS=[]

def job(name, operation, lane, reads=(), writes=(), timeout=240):
    JOBS.append(dict(name=name,cmd=f'{OP} {operation}',lane=lane,
                     gpu=0 if lane=='gpu' else None, reads=list(reads),writes=list(writes),
                     timeout=timeout,quiet_s=180,env={'ETX_QWEN_G6_CONFIG':CONFIG},
                     summary=[r'^(PHASE|FAIL|SAFE|EVIDENCE|VERDICT|CERTIFIED|PLAN) ']))

job('preflight','preflight','cpu',writes=('inputs.json',),timeout=240)
for arm in ('baseline','producer'):
    job(f'build_{arm}',f'build {arm}','cpu',reads=('preflight',),writes=(f'b_{arm}/run',),timeout=900)
job('qualify','qualify','gpu',reads=('build_baseline','build_producer'),writes=('correctness.json',),timeout=1500)
job('measure','measure','gpu',reads=('correctness.json',),writes=('measurement.complete.json',),timeout=2700)
job('diagnose','diagnose','gpu',reads=('measurement.complete.json',),writes=('trace_baseline.json','trace_producer.json'),timeout=420)
job('report','report','final',reads=('?qualify','?measure','?diagnose'),writes=('report.json',),timeout=60)
