"""Standalone, opt-in two-GPU seam qualification, no model inference.

Run exclusively on an already authorized/pre-staged host. CPU lane is deliberate:
GPU lanes mask to one GPU, but each run owns both persistent rank kernels.
Full native logs live in seam-logs; runner stdout carries bounded progress and
an offline-parsed summary. No downloads, provisioning, model weights or tokens.
"""
MAX_SESSION_SECONDS = 2400
JOBS = [
    dict(name='build_seam', lane='cpu', reads=[],
         writes=[f'seam-build/b{b}/run' for b in (1, 2, 4, 8)],
         timeout=360, quiet_s=60,
         cmd='for b in 1 2 4 8; do ETX_GLM_BATCH=$b bash examples/glm53/build_seam.sh "$ETX_SESS_LOGDIR/seam-build/b$b" || exit $?; done'),
    dict(name='seam', lane='cpu', reads=[f'seam-build/b{b}/run' for b in (1, 2, 4, 8)],
         writes=[], timeout=2000, quiet_s=15,
         env={'HIP_VISIBLE_DEVICES': '0,1', 'ETX_SYNC': 'poll', 'ETX_DEV_RESET': '0'},
         cmd='stdbuf -oL .venv/bin/python -m examples.glm53.seam run '
             '--build-root "$ETX_SESS_LOGDIR/seam-build" --logdir "$ETX_SESS_LOGDIR/seam-logs" '
             '--iterations 1000 --rounds 5 --warmup 20 --timeout 240'),
]
