#!/bin/bash
set -euo pipefail
cd /root/etx
export PATH=/root/etx-python/bin:/opt/rocm/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export ETX_DSV2_FLEET=/root/fleet-3fb67a6
export ETX_DSV2_DATA=/root/sess/x8-live-20261015/20261015T154549Z-7cdcb2/references
# Fixed global stop minus five minutes for export. A retry must not reset this bound.
export ETX_SESS_MAX_SECONDS=$(python3 -c 'import time; print(max(0,min(7200,int(1792084930-time.time()-300))))')
test "$ETX_SESS_MAX_SECONDS" -gt 0
test -s /root/setup-live/c6-container-run.exit
printf "C6 preceding terminal exit: "; cat /root/setup-live/c6-container-run.exit
if docker ps --format '{{.Names}}' | grep -q '^etx-c6-run-723$'; then echo 'REFUSE: C6 container still present'; exit 4; fi
printf 'B4 original-global-bound cap %s; UTC ' "$ETX_SESS_MAX_SECONDS"; date -u
cat SOURCE_IDENTITY
hipcc --version
python3 -u -m etx.tools.gpu_jobs examples/sessions/ds_b4_attribution.py --gpus 0 --cpu-slots 2 --logdir /root/sess/b4-live-20261015
