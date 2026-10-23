"""Fail closed on unqualified sparse timestamps. No GPU data is generated here.

Resource mismatches are a blocker, not something to subtract away. The caller
must supply same-session correctness evidence, compiler allocation, actual last
completing TASK AND WAVE, and body A/B perturbation. Sparse sampling alone proves
neither last-wave identity nor exclusive removable work.
"""
from .protocol import require


def attribution(sample, control, empty):
    allocation = ('vgpr', 'agpr', 'vgpr_spills', 'sgpr_spills', 'scratch_bytes_lane', 'lds_bytes', 'static_blocks_per_cu')
    resources = lambda x: tuple(x['resources'][k] for k in allocation)
    require(sample['end'] >= sample['begin'] > 0 and empty['end'] >= empty['begin'] > 0 and
            sample['task'] == sample['last_task'] and sample['wave'] == sample['last_wave'] and
            abs(sample['body_us'] - control['body_us']) <= 0.5 and
            all(x['tokens_matched'] == 32 and x['all_imports_byte_equal'] for x in (sample, control, empty)) and
            len({x['session'] for x in (sample, control, empty)}) == 1 and
            len({x['compiler'] for x in (sample, control, empty)}) == 1 and
            len({x['math_identity'] for x in (sample, control, empty)}) == 1 and
            resources(sample) == resources(control) == resources(empty), 'attribution_gate')
    return {'issue_interval_us': (sample['end'] - sample['begin']) * 0.01,
            'empty_pair_us': (empty['end'] - empty['begin']) * 0.01,
            'perturbation_us': sample['body_us'] - control['body_us'],
            'exclusive_removable_us_layer': None, 'pursuit_threshold_us_layer': 2.083}
