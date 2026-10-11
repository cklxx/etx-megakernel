"""CPU identity gate and the registered same-session device-reset comparison."""
import json
from pathlib import Path

PATH_OVERRIDES = {'ETX_VL_IMPORTS', 'ETX_VL_ADAPTERS', 'ETX_VL_PLAN',
                  'ETX_VL_CHAINS', 'ETX_VL_DSL_TILES'}


def check_pair(base, candidate, base_plan, candidate_plan):
    from examples.sessions.qwen300_hour import schedule_signature
    clean = lambda x: {k: v for k, v in x.items() if k not in PATH_OVERRIDES}
    expected = dict(clean(base['overrides']), ETX_DEV_RESET='1')
    if base['overrides'].get('ETX_DEV_RESET') != '0':
        raise ValueError('reset baseline must have ETX_DEV_RESET=0')
    if clean(candidate['overrides']) != expected or candidate['hipcc_flags'] != base['hipcc_flags']:
        raise ValueError('reset pair differs in flags other than ETX_DEV_RESET')
    keys = {'imports', 'import_metadata', 'device_source/dsl_tiles.hip', 'device_source/imp_tiles.hip'}
    keys.update(k for k in base['files'] if k.startswith('original/'))
    if base.get('compiler') != candidate.get('compiler'):
        raise ValueError('reset pair changed compiler')
    for key in keys:
        if key not in base['files'] or base['files'][key] != candidate['files'].get(key):
            raise ValueError('reset pair changed body/import identity: '+key)
    if schedule_signature(base_plan) != schedule_signature(candidate_plan):
        raise ValueError('reset pair changed schedule')


def run(s):
    from examples.sessions import qwen300_hour as h
    from examples.vllm_llm import import_certificate as cert
    arm = dict(id='DEV_RESET', build='reset', enabled=True,
               flags={'ETX_DEV_RESET': '1'}, overlap='A1 reset path; wall saving only')
    h.eligible_arm(s, arm)
    b, c = (Path(s.c['builds'][name]) for name in ('kept', 'reset'))
    check_pair(cert.verified_build(b, Path(s.c['model'])), cert.verified_build(c, Path(s.c['model'])),
               json.loads((b/'plan.json').read_text()), json.loads((c/'plan.json').read_text()))
    result = h.candidate_pairs(s, arm, 'RESET', publish=False)
    # Compare both directions on the identical populations. Do not reuse a
    # profile-build event delta as a candidate wall saving.
    result['device_deltas_ms'] = [r['reset']['device_ms']-r['kept']['device_ms'] for r in result['rounds']]
    s.row('RESET.keep', result, result['status'], h.REGISTERED['RESET.keep'])
    return result
