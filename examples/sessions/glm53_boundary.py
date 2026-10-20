"""Both GPUs in one bounded process; CPU lane deliberately preserves two visible GPUs.

The general runner's GPU lane masks to one GPU, so this exclusive two-GPU
experiment uses CPU lane and explicitly sets HIP_VISIBLE_DEVICES=0,1.
Run alone on an authorized, pre-staged host; no GLM model runs here.
"""
MAX_SESSION_SECONDS = 300
JOBS = [dict(name='build_transport', lane='cpu', reads=[], writes=['boundary/run'],
             timeout=180, quiet_s=60,
             cmd='bash examples/glm53/build_boundary.sh "$ETX_SESS_LOGDIR/boundary"'),
        dict(name='transport', lane='cpu', reads=['boundary/run'], writes=[],
             timeout=90, quiet_s=15,
             env={'HIP_VISIBLE_DEVICES': '0,1', 'ETX_SYNC': 'poll'},
             cmd='stdbuf -oL "$ETX_SESS_LOGDIR/boundary/run" 8')]
