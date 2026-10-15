"""PREPARED, NOT RUN: existing B4 slowdown follow-up, no new kernels and no x8 timing.

One MI300X, GPU 0; planned 60-90 minutes, hard runner cap 7200 seconds including prep.
No provider operation, tariff assumption or spend approval is part of this job. DigitalOcean
availability, actual price and native/runtime gates must be checked before a separately authorized run.
See docs/DEEPSEEK-B4-PREP-2026-10-15.md for acceptance and the exact pre-registered branches.
"""
import copy
import importlib.util
from pathlib import Path

from examples.dsv2lite.arms import BATCH_ARMS

MAX_SESSION_SECONDS = 7200
_spec = importlib.util.spec_from_file_location('ds_b4_base', Path(__file__).with_name('ds_batch.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_by = {j['name']: j for j in _base.JOBS}
JOBS = [copy.deepcopy(_by[n]) for n in ('apt','etx_env','prep_env','source','prep_dl','prep_pack','docker_pull')]
# Reuse the pinned pack recipe, without generating an old unbound r0 tuple first.
pack = next(j for j in JOBS if j['name'] == 'prep_pack')
pack['reads'] = [_base.MODEL+'/config.json', 'source.json']
pack['cmd'] = 'mkdir -p "$ETX_DSV2_DATA" && ' + pack['cmd']
ENV = dict(_base._full.ENV)
OP = f'{_base.PREP_PY} -u -m examples.dsv2lite.batch_attribution'


def job(name, operation, *, lane='gpu', reads=(), writes=(), timeout=600, cmd=None):
    JOBS.append(dict(name=name, cmd=cmd or f'{OP} {operation}', lane=lane, gpu=0 if lane == 'gpu' else None,
                     env=ENV, quiet_s=90, reads=list(reads), writes=list(writes), timeout=timeout,
                     summary=[r'^(REFERENCES|BGATE|VSEL|BROUNDS|BREPORT|B4-RESULT|PROGRESS) ']))


job('references', '', reads=('source.json', 'prep_env', 'prep_pack', _base.MODEL+'/config.json'),
    writes=('references_ready.json',), timeout=1500,
    cmd=f'{_base.PREP_PY} -u -m examples.dsv2lite.reference_prepare --model {_base.MODEL} --requests 4 --regenerate-missing')
for b in (1,2,4):
    arm = BATCH_ARMS[b]
    build = copy.deepcopy(_by['build_'+arm])
    JOBS.append(build)
    job('bgate_'+arm, 'gate '+arm, reads=('references_ready.json', f'b_{arm}/run'), writes=(f'bgate_{arm}.json',))
for b in (1,2,4):
    arm = BATCH_ARMS[b]
    job('graph_gate_'+arm, '', lane='cpu', reads=(f'b_{arm}/run',), writes=(f'graph_gate_{arm}.json',), timeout=60,
        cmd='.venv/bin/python -u -m examples.dsv2lite.batch_attribution graph-eligibility '
            f'"$ETX_SESS_LOGDIR/b_{arm}/plan.json" --output "$ETX_SESS_LOGDIR/graph_gate_{arm}.json"')
GATES = tuple(f'{kind}_{BATCH_ARMS[b]}.json' for b in (1,2,4) for kind in ('bgate','graph_gate'))
job('vsel', 'vsel', reads=('references_ready.json','docker_pull', *GATES), writes=('vsel.json',), timeout=1900)
job('brounds', 'rounds', reads=('vsel.json', *GATES), writes=('brounds.json',), timeout=3600)
# Captures are separate fresh invocations AFTER all unprofiled five-pair rounds.
job('captures', 'captures', reads=('vsel.json', 'brounds.json', *GATES), writes=('attribution.json',), timeout=3300)
job('routing', 'routing', reads=('brounds.json','references_ready.json'), writes=('routing.json', 'routing_b1.json', 'routing_b2.json', 'routing_b4.json'), timeout=1000)
job('breport', 'report', lane='final', reads=('brounds.json','vsel.json','?routing.json'), writes=('batch_report.json',), timeout=120)
# Always preserve a structured result, including skipped/failed capture and dependency branches.
job('b4_result', 'final', lane='final', reads=('?breport','?captures','?routing'), writes=('b4_result.json',), timeout=120)
STUDIES = {'DeepSeek B4 attribution': ['b4_result']}
