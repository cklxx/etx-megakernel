"""Pre-staged single-MI300X Qwen B2/B4/B8 correctness only; no provisioning.

ETX_QWEN_BATCH_MODEL=/absolute/packed/model tools/session.sh start <ip> examples/sessions/qwen_batch_correctness.py 0
All golden/reference preparation must finish first; tools/two_gpu_preflight.py checks it without a GPU.
CPU builds may overlap; GPU qualifications are serialized by lane and dependencies.
The final job re-checks every receipt and fails the session if one is incomplete.
No tuning/timing verdict.
"""
import os
import shlex

MAX_SESSION_SECONDS = 7200
MODEL = os.environ.get('ETX_QWEN_BATCH_MODEL', os.path.expanduser('~/llm/qwen3-30b-a3b'))
OP = '.venv/bin/python -u -m examples.vllm_llm.batch_correctness'
# A staged ~/etx is a fresh archive without .venv: create it first (as examples/sessions/ds_interleave.py).
JOBS = [dict(name='apt', lane='cpu', timeout=600,
             cmd="python3 -c 'import ensurepip' 2>/dev/null || "
                 "{ sudo apt-get update -qq; sudo apt-get install -y -qq python3.12-venv; }"),
        dict(name='etx_env', lane='cpu', reads='apt', timeout=600,
             cmd=".venv/bin/python -c 1 2>/dev/null || { rm -rf .venv; python3 -m venv --system-site-packages .venv; }; "
                 ".venv/bin/python -c 'import yaml' 2>/dev/null || .venv/bin/pip install -q pyyaml; "
                 ".venv/bin/python -c 'import etx; print(\"etx ok\")'")]
for b in (2, 4, 8):
    build = f'$ETX_SESS_LOGDIR/build_b{b}'
    common = f'--batch {b} --model {shlex.quote(MODEL)} --build "{build}"'
    JOBS.append(dict(name=f'build_b{b}', lane='cpu',
                     cmd=f'{OP} build {common} --out "$ETX_SESS_LOGDIR/log_b{b}"',
                     reads=['etx_env'], writes=[f'build_b{b}/run'], timeout=2100, quiet_s=180))
    JOBS.append(dict(name=f'qualify_b{b}', lane='gpu', gpu=0,
                     cmd=f'{OP} qualify {common} --out "$ETX_SESS_LOGDIR/qualify_b{b}"',
                     reads=[f'build_b{b}/run'] + ([f'qualify_b{b//2}/correctness.json'] if b > 2 else []),
                     writes=[f'qualify_b{b}/correctness.json'], timeout=4800, quiet_s=180))
# Reads the receipts with "?" so it runs even when a qualification failed or was skipped, and
# its non-zero exit is what makes an incomplete record fail the session. The runner's python3
# (stdlib only) keeps it working when the .venv setup failed.
JOBS.append(dict(name='verify', lane='final',
                 cmd='python3 tools/two_gpu_verify.py "$ETX_SESS_LOGDIR" --qwen-only '
                     '--out "$ETX_SESS_LOGDIR/qwen-verification.json"',
                 reads=[f'?qualify_b{b}/correctness.json' for b in (2, 4, 8)],
                 writes=['qwen-verification.json'], timeout=300, quiet_s=60))
