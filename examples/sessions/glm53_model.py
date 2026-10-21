"""Actual GLM PP2 model, existing exclusive two-GPU host, no provisioning.

Requires ETX_GLM_STAGE and ETX_GLM_REFERENCE prepared offline; ETX_GLM_BATCH must
match the reference batch. tools/two_gpu_preflight.py checks both without a GPU.
ETX_GLM_GATE=module runs the same jobs on the synthetic module-gate checkpoint
(examples/glm53/module_gate.py): its pinned identity, plus the median-logit rule.
One CPU-lane process drives both visible devices, so the runner does not mask it
to one GPU. The final job re-checks the receipt and fails the session if it is
incomplete.
"""
import os
import shlex
gate=os.environ.get('ETX_GLM_GATE','production')
if gate not in ('production','module'):raise ValueError('invalid GLM gate')
# Caps: the release is sized for hours; the 4-layer module gate builds and runs in minutes
# (estimate, docs/glm/QWEN-BATCH-GLM53-RUNBOOK.md#module-gate), so it gets caps with a margin, not hours.
CAPS=dict(production=dict(session=18000,build=2100,qualify=15000),module=dict(session=2700,build=1200,qualify=1800))[gate]
MAX_SESSION_SECONDS=CAPS['session']
stage=shlex.quote(os.environ.get('ETX_GLM_STAGE','/staged/glm53'))
reference=shlex.quote(os.environ.get('ETX_GLM_REFERENCE','/staged/glm53-reference.json'))
batch=int(os.environ.get('ETX_GLM_BATCH','1'))
if batch not in (1,2,4,8):raise ValueError('invalid GLM batch')
base=f'.venv/bin/python -u -m examples.glm53.run --stage {stage} --build "$ETX_SESS_LOGDIR/model" --batch {batch} --gate {gate}'
# A staged ~/etx is a fresh archive without .venv: create it first (as examples/sessions/ds_interleave.py).
JOBS=[dict(name='apt',lane='cpu',timeout=600,
           cmd="python3 -c 'import ensurepip' 2>/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }"),
      dict(name='etx_env',lane='cpu',reads='apt',timeout=600,
           cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
               ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; "
               ".venv/bin/python -c 'import numpy' 2>/dev/null || .venv/bin/pip install -q numpy; "
               ".venv/bin/python -c 'import etx, numpy; print(\"etx ok\")'"),
      dict(name='model_build',lane='cpu',cmd=base+' build --out "$ETX_SESS_LOGDIR/build-log"',reads=['etx_env'],writes=['model/run'],timeout=CAPS['build'],quiet_s=180),
      dict(name='model_qualify',lane='cpu',cmd=base+f' qualify --reference {reference} --out "$ETX_SESS_LOGDIR/qualification"',
           reads=['model/run'],writes=['qualification/qualification.json'],timeout=CAPS['qualify'],quiet_s=180,env={'HIP_VISIBLE_DEVICES':'0,1'}),
      # "?" keeps this job running after a failed qualification, so the record is
      # refused rather than silently lacking a verdict. The runner's python3 (stdlib
      # only) keeps it working when the .venv setup failed.
      dict(name='verify',lane='final',cmd=f'python3 tools/two_gpu_verify.py "$ETX_SESS_LOGDIR" --glm-only --gate {gate} '
           '--out "$ETX_SESS_LOGDIR/glm-verification.json"',
           reads=['?qualification/qualification.json'],writes=['glm-verification.json'],timeout=300,quiet_s=60)]
