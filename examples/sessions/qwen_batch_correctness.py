"""Pre-staged single-MI300X Qwen B2/B4/B8 correctness only; no provisioning.

ETX_QWEN_BATCH_MODEL=/absolute/packed/model tools/session.sh start <ip> examples/sessions/qwen_batch_correctness.py
All golden/reference preparation must finish first. CPU builds may overlap;
GPU qualifications are serialized by lane and dependencies. No tuning/timing verdict.
"""
import os
import shlex

MAX_SESSION_SECONDS = 7200
MODEL = os.environ.get('ETX_QWEN_BATCH_MODEL', os.path.expanduser('~/llm/qwen3-30b-a3b'))
OP = '.venv/bin/python -u -m examples.vllm_llm.batch_correctness'
JOBS = []
for b in (2, 4, 8):
    build = f'$ETX_SESS_LOGDIR/build_b{b}'
    common = f'--batch {b} --model {shlex.quote(MODEL)} --build "{build}"'
    JOBS.append(dict(name=f'build_b{b}', lane='cpu',
                     cmd=f'{OP} build {common} --out "$ETX_SESS_LOGDIR/log_b{b}"',
                     reads=[], writes=[f'build_b{b}/run'], timeout=2100, quiet_s=180))
    JOBS.append(dict(name=f'qualify_b{b}', lane='gpu', gpu=0,
                     cmd=f'{OP} qualify {common} --out "$ETX_SESS_LOGDIR/qualify_b{b}"',
                     reads=[f'build_b{b}/run'] + ([f'qualify_b{b//2}/correctness.json'] if b > 2 else []),
                     writes=[f'qualify_b{b}/correctness.json'], timeout=4800, quiet_s=180))
