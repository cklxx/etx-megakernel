"""Actual GLM PP2 model, existing exclusive two-GPU host, no provisioning.

Requires ETX_GLM_STAGE and ETX_GLM_REFERENCE prepared offline. One CPU-lane
process drives both visible devices, so the runner does not mask it to one GPU.
"""
import os
import shlex
MAX_SESSION_SECONDS=18000
stage=shlex.quote(os.environ.get('ETX_GLM_STAGE','/staged/glm53'))
reference=shlex.quote(os.environ.get('ETX_GLM_REFERENCE','/staged/glm53-reference.json'))
batch=int(os.environ.get('ETX_GLM_BATCH','1'))
if batch not in (1,2,4,8):raise ValueError('invalid GLM batch')
base=f'.venv/bin/python -u -m examples.glm53.run --stage {stage} --build "$ETX_SESS_LOGDIR/model" --batch {batch}'
JOBS=[dict(name='model_build',lane='cpu',cmd=base+' build --out "$ETX_SESS_LOGDIR/build-log"',reads=[],writes=['model/run'],timeout=2100,quiet_s=180),
      dict(name='model_qualify',lane='cpu',cmd=base+f' qualify --reference {reference} --out "$ETX_SESS_LOGDIR/qualification"',
           reads=['model/run'],writes=['qualification/qualification.json'],timeout=15000,quiet_s=180,env={'HIP_VISIBLE_DEVICES':'0,1'})]
