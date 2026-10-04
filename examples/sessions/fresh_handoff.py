"""Dedicated MI300X mechanism probe; no provisioning or model validation.

Use only after approving the session budget and collecting machine configuration.
The 2-hour run cap may yield an incomplete matrix; incomplete is never a pass.
"""
JOBS = [
    dict(name='fresh_build', lane='cpu', timeout=300,
         cmd='mkdir -p build && hipcc -O3 -std=c++17 --offload-arch=gfx942 '
             'bench/sync/sync_bench.hip -o build/fresh-sync-bench'),
    dict(name='fresh_matrix', gpu=0, after='fresh_build', timeout=7500,
         cmd='.venv/bin/python -u -m etx.tools.fresh_handoff run '
             '--binary build/fresh-sync-bench --out build/fresh-results '
             '--budget-seconds 7200 --case-seconds 420'),
]
