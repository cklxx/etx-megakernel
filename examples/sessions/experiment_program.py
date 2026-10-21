"""One pinned GPU lane for the prepared thirteen-record program.

This job file only consumes already staged, sealed inputs. It never provisions,
installs, downloads or stages remotely. Loading/dry-running it executes no GPU.
Set ETX_EXPERIMENT_CONFIG and ETX_EXPERIMENT_PYTHON to staged local paths before
an explicitly authorized gpu_jobs invocation. Conditional misses stay closed.
"""
import os
from pathlib import Path
import shlex

MAX_SESSION_SECONDS=14400
configuration=Path(os.environ.get('ETX_EXPERIMENT_CONFIG','EXPERIMENT_SESSION.json')).expanduser().absolute()
python=os.environ.get('ETX_EXPERIMENT_PYTHON','.venv/bin/python')
base=shlex.quote(python)+' -u -m examples.experiment_program '
args=' --config '+shlex.quote(str(configuration))
JOBS=[dict(name='experiment_preflight',lane='cpu',timeout=900,quiet_s=30,
           cwd=str(Path(__file__).resolve().parents[2]),
           cmd=base+'preflight'+args+' --out "$ETX_SESS_LOGDIR/preflight"',
           writes=['$ETX_SESS_LOGDIR/preflight/session.json']),
      dict(name='experiment_program',gpu=0,reads=['experiment_preflight'],timeout=13500,quiet_s=30,
           cwd=str(Path(__file__).resolve().parents[2]),
           cmd=base+'run --execute'+args+' --out "$ETX_SESS_LOGDIR/program"',
           writes=['$ETX_SESS_LOGDIR/program/session.json'])]
