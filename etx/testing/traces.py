"""A simulated DeepSeek-V2-Lite schedule written as the files the timeline/account commands read; not GPU data."""
from array import array
import json
from pathlib import Path


def simulated_trace_files(directory, layers=3, seed=0):
    """Write plan.json, t.bin (raw trace rows) and b.json (per-task byte spec); return (paths, domains)."""
    from etx.passes import compile_graph
    from etx.sim import simulate
    from etx.tools import timeline
    from examples.dsv2lite import bytes as dsb
    from examples.dsv2lite import graph as dsv
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plan = compile_graph(dsv.build(layers=layers), 'gfx942', {}, {})
    result = simulate(plan, seed=seed)
    if result.deadlock:
        raise ValueError('simulated schedule deadlocked')
    plan_json = timeline.plan_json_from_obj(plan)
    rows = timeline.trace_from_sim(plan, result)
    paths = dict(plan=directory / 'plan.json', raw=directory / 't.bin', bytes=directory / 'b.json')
    paths['plan'].write_text(json.dumps(plan_json))
    paths['raw'].write_bytes(array('q', [v for row in rows for v in row]).tobytes())
    paths['bytes'].write_text(json.dumps(dsb.spec()))
    return paths, plan_json['domains']
