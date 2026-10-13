"""Dedicated MI300X mechanism probe; no provisioning or model validation.

Use only after approving the session budget and collecting machine configuration.
One million handoffs per variant makes this an 8-hour GPU job (28 TB of payload
traffic, 61 GiB peak allocation); a run cap below that yields an incomplete matrix,
and incomplete is never a pass.
"""
JOBS = [
    dict(name='fresh_build', lane='cpu', timeout=300,
         cmd='mkdir -p build && hipcc -O3 -std=c++17 --offload-arch=gfx942 '
             'bench/sync/sync_bench.hip -o build/fresh-sync-bench'),
    dict(name='fresh_matrix', gpu=0, reads='fresh_build', timeout=32000,
         cmd='.venv/bin/python -u -m etx.tools.fresh_handoff run '
             '--binary build/fresh-sync-bench --out build/fresh-results '
             '--budget-seconds 28800 --case-seconds 1800'),
]
