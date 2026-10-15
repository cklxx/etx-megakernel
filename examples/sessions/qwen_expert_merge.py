"""Prepared one-GPU lane. No provisioning. Run only on an already authorized GPU.

Set EEM_INPUT to a validated pack. The default is a labelled synthetic probe.
Use a new ETX_SESS_LOGDIR for each budgeted session. gpu_jobs keeps raw logs.
"""
import os

MAX_SESSION_SECONDS=1800
ROOT=os.getcwd()
INPUT=os.environ.get("EEM_INPUT", "$ETX_SESS_LOGDIR/synthetic")
P="python3 -m examples.vllm_llm.expert_merge."
R='"$ETX_SESS_LOGDIR/probe"'
JOBS=[]
if "EEM_INPUT" not in os.environ:
    JOBS.append(dict(name="synthetic",lane="cpu",cwd=ROOT,timeout=120,quiet_s=45,
                     cmd=P+'inputs synthetic "$ETX_SESS_LOGDIR/synthetic"'))
JOBS.append(dict(name="build",lane="cpu",cwd=ROOT,reads=["synthetic"] if "EEM_INPUT" not in os.environ else [],
                 timeout=1200,quiet_s=45,cmd=P+f'runner build {R} --input "{INPUT}"'))
JOBS.append(dict(name="correctness",gpu=0,cwd=ROOT,reads=["build"],timeout=360,quiet_s=45,
                 cmd=P+f"runner check {R}"))
JOBS.append(dict(name="pairs",gpu=0,cwd=ROOT,reads=["correctness"],timeout=240,quiet_s=45,
                 cmd=P+f"runner time {R}"))
