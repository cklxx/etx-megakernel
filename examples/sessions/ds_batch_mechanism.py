"""PREPARED, NOT RUN: why DeepSeek B4/B8 do not scale past B2 (docs/deepseek/DEEPSEEK-BATCH-MECHANISM-2026-10-18.md).

One MI300X, GPU 0. No vLLM, no new kernel, no pairs: this session tests the mechanism, it does not re-time ETX against
vLLM (the 2026-10-15 rounds already did). Order on the GPU lane: references, the x2/x4 gates, the isolation
microbenchmark (the deciding unknown), the slot traces, the x8 gate and its trace if it passes, counters last
(rocprofv3 crashed once on the 2026-10-15 box; a crash leaves that group INCOMPLETE, nothing is salvaged).
Budget [E]: 60-75 min with fresh prep, hard cap 7200 s; about $2.0-2.5 at $1.99/h. No provider action here.

Run (one command after stage, on a DigitalOcean box where login is root):
  examples/dsv2lite/reproduce.sh stage <ip> && examples/dsv2lite/reproduce.sh box-setup <ip> &&
  ETX_SESS_USER=root ETX_SESS_MAX_SECONDS=7200 tools/session.sh start <ip> examples/sessions/ds_batch_mechanism.py 0
Then: ETX_SESS_USER=root tools/session.sh fetch <ip> ~/etx-captures; the verdicts are in mechanism_report.json.
"""
import copy
import importlib.util
from pathlib import Path

MAX_SESSION_SECONDS = 7200
_spec = importlib.util.spec_from_file_location('ds_mech_b4', Path(__file__).with_name('ds_b4_attribution.py'))
_b4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b4)
_base = _b4._base
_by = {j['name']: j for j in _base.JOBS}
_b4_by = {j['name']: j for j in _b4.JOBS}
# The B4 queue's prep (its prep_pack already reads no r0 tuple first); no docker image: vLLM is not invoked here.
JOBS = [copy.deepcopy(_b4_by[n]) for n in ('apt', 'etx_env', 'prep_env', 'source', 'prep_dl', 'prep_pack')]
ENV = dict(_base._full.ENV)
OP = '.venv/bin/python -u -m examples.dsv2lite.batch_mechanism'
SUMMARY = [r'^(REFERENCES|BGATE|MTRACE|MISO|MCOUNT|MREPORT|BUILD|PROGRESS) ']


def job(name, cmd, *, lane='gpu', reads=(), writes=(), timeout=600):
    JOBS.append(dict(name=name, cmd=cmd, lane=lane, gpu=0 if lane == 'gpu' else None, env=ENV, quiet_s=90,
                     reads=list(reads), writes=list(writes), timeout=timeout, summary=SUMMARY))


# eight producer-bound request tuples: B8 needs r0..r7 (B2/B4 use their prefixes)
job('references', f'{_base.PREP_PY} -u -m examples.dsv2lite.reference_prepare --model {_base.MODEL} --requests 8 '
    '--regenerate-missing', reads=('source.json', 'prep_env', 'prep_pack', _base.MODEL + '/config.json'),
    writes=('references_ready.json',), timeout=2400)
# builds (CPU lane, in parallel with the references)
for arm in ('x2', 'x4', 'x8'):
    JOBS.append(copy.deepcopy(_by['build_' + arm]))
iso = copy.deepcopy(_by['build_x4'])
iso.update(name='build_x4iso', cmd=iso['cmd'].replace('build x4', 'build x4iso'), writes=['b_x4iso/run'])
JOBS.append(iso)

# GPU, in order: correctness, the deciding microbenchmark, traces, B8, counters
for arm in ('x2', 'x4'):
    job('bgate_' + arm, f'{OP} gate {arm}', reads=('references_ready.json', f'b_{arm}/run'), writes=(f'bgate_{arm}.json',))
job('iso_x4iso', f'{OP} iso x4iso', reads=('references_ready.json', 'b_x4iso/run'), writes=('iso_x4iso.json',), timeout=600)
for arm in ('x2', 'x4'):
    job('trace_' + arm, f'{OP} trace {arm}', reads=(f'bgate_{arm}.json',), writes=(f'trace_{arm}.json',), timeout=900)
job('bgate_x8', f'{OP} gate x8', reads=('references_ready.json', 'b_x8/run'), writes=('bgate_x8.json',))
# a failed or skipped x8 gate still yields trace_x8.json / counters_x8.json saying NOT-RUN and why
job('trace_x8', f'{OP} trace x8', reads=('references_ready.json', '?bgate_x8.json'), writes=('trace_x8.json',), timeout=900)
for arm, gate in (('x2', 'bgate_x2.json'), ('x4', 'bgate_x4.json'), ('x8', '?bgate_x8.json')):
    job('counters_' + arm, f'{OP} counters {arm}', reads=('references_ready.json', gate, '?trace_x8.json'),
        writes=(f'counters_{arm}.json',), timeout=1500)
job('report', f'{OP} report', lane='final', reads=('?trace_x2.json', '?trace_x4.json', '?trace_x8.json', '?iso_x4iso.json',
    '?counters_x2.json', '?counters_x4.json', '?counters_x8.json'), writes=('mechanism_report.json',), timeout=120)
STUDIES = {'DeepSeek batch mechanism': ['report']}
