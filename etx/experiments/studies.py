"""Legacy phase-0 studies using the shared simulation/trace validation boundary."""
from __future__ import annotations

import math
import os
import random

from etx.passes import PassOptions, compile_graph
from .protocol import MODES, contract, evaluate


DEFAULTS = {
    'variance': ((0.0, 0.1, 0.3, 0.6, 1.0), 256),
    'routing-skew': ((0.0, 0.5, 1.0, 2.0, 4.0), 32),
}


def routing(mod, size, skew):
    rng = random.Random(3)
    weights = [pow(2.0, -skew * e) for e in range(mod.NE)]
    topk = []
    for _ in range(size):
        picks = []
        while len(picks) < mod.K:
            expert = rng.choices(range(mod.NE), weights=weights)[0]
            if expert not in picks:
                picks.append(expert)
        topk.append(sorted(picks))
    counts = [0] * mod.NE
    for row in topk:
        for expert in row:
            counts[expert] += 1
    tiles = [-(-count // mod.TPT) for count in counts]
    tok_indptr, indptr = [0], [0]
    for expert in range(mod.NE):
        tok_indptr.append(tok_indptr[-1] + counts[expert])
        indptr.append(indptr[-1] + tiles[expert])
    fill = [0] * mod.NE
    tok_slot = []
    for row in topk:
        slots = []
        for expert in row:
            slots.append(tok_indptr[expert] + fill[expert])
            fill[expert] += 1
        tok_slot.append(slots)
    runtime = {'topk': topk, 'expert_counts': counts, 'expert_tiles': tiles,
               'tok_indptr': tok_indptr, 'indptr': indptr, 'tok_slot': tok_slot,
               'tile_expert': [[e] for e in range(mod.NE) for _ in range(tiles[e])]}
    return {'B': size, 'seed': 0, 'n_gg': indptr[-1]}, runtime


def run(name, *, values=None, size=None):
    if name not in DEFAULTS:
        raise ValueError('study must be variance or routing-skew')
    defaults, default_size = DEFAULTS[name]
    values = defaults if values is None else tuple(values)
    size = default_size if size is None else size
    if type(size) is not int or not 1 <= size <= 256:
        raise ValueError('study size must be an integer in [1, 256]')
    if not values or any(type(value) not in (int, float) or not math.isfinite(value)
                         or not 0 <= value <= (2 if name == 'variance' else 4)
                         for value in values):
        raise ValueError('study values must be nonempty finite numbers in the supported range')
    from etx.operators import splitk_sum
    from etx.operators.moe_layer import provider
    moe_layer = provider(small=os.environ.get('ETX_MOE_SMALL') == '1')
    rows = []
    for value in values:
        if name == 'variance':
            bindings, runtime = splitk_sum.bindings(n=size), {}
        else:
            bindings, runtime = routing(moe_layer, size, value)
        times = {}
        for mode in MODES:
            graph = (splitk_sum if name == 'variance' else moe_layer).build()
            if name == 'variance':
                for grid in graph.grids:
                    grid.duration_cv = value
            # Preserve historical study defaults, distinct from the frozen paired benchmark.
            plan = compile_graph(graph, 'gfx942', bindings, runtime, PassOptions(force_mode=mode))
            result, errors = evaluate(plan, contract(plan), seed=1)
            if errors:
                raise ValueError(f'{name}:{value}:{mode}: ' + '; '.join(errors))
            times[mode] = result['makespan_us']
        row = {'value': value, 'makespan_us': times, 'best': min(times, key=times.get)}
        if name == 'routing-skew':
            counts = runtime['expert_counts']
            row['max_mean'] = max(counts) / (sum(counts) / len(counts))
        rows.append(row)
    return {'schema_version': 1, 'kind': 'protocol_study', 'backend': 'simulation',
            'timing_scope': 'simulated_protocol', 'gpu_measurement': False,
            'tensor_correctness': 'not_executed', 'name': name, 'size': size,
            'seed': 1, 'rows': rows}
