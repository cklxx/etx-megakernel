"""Lossless diagnostic reader and per-layer/phase attribution (CPU only).

python -m etx.tools.inbody PLAN RAW --bytes BYTES_JSON --output REPORT_JSON
Legacy raw files remain readable by timeline; diagnostic files are versioned.
No issue timestamp can certify exclusive memory latency or removable work.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import struct

from . import timeline as tl

MAGIC = 0x3142444F424E4954


def read_raw(path):
    data = Path(path).read_bytes()
    if len(data) < 48 or len(data) % 8:
        raise ValueError('truncated inbody header/word')
    magic, version, nt, ne, nw, tick_ns = struct.unpack_from('<6Q', data)
    if magic != MAGIC or version != 1 or not nt or not ne or not 0 < nw <= 16 or tick_ns != 10:
        raise ValueError('invalid inbody header')
    count = 6 + nt * 5 + ne * 2 + nt * nw * 2
    if len(data) != count * 8:
        raise ValueError('truncated/extra inbody records')
    words = struct.unpack('<' + 'Q' * count, data)
    pos = 6
    def take(n, width):
        nonlocal pos
        rows = [list(words[i:i+width]) for i in range(pos, pos + n * width, width)]
        pos += n * width
        return rows
    return dict(version=version, tick_ns=tick_ns, legacy=take(nt, 4),
                poll=take(nt, 1), events=take(ne, 2), waves=take(nt*nw, 2), waves_per_task=nw)


def event_words(plan):
    events = {e['name']: e for e in plan['events']}
    result = []
    for name, coord, _ in plan['ev_inst']:
        ev = events[name]
        linear = 0
        for index, extent in zip(coord, ev['shape']):
            if not 0 <= index < extent:
                raise ValueError('invalid event coordinate')
            linear = linear * extent + index
        result.append(ev['offset'] + linear)
    return result


def analyze(plan, raw, byte_rules=None, hop_us=None, bandwidth=4.3):
    if bandwidth <= 0 or len(raw['legacy']) != len(plan['descs']):
        raise ValueError('incomplete plan/trace or invalid bandwidth')
    names = {v['id']: k for k, v in plan['types'].items()}
    words = event_words(plan)
    legacy = []
    tasks = []
    scale = raw['tick_ns'] / 1000
    nb = tl.bytes_spec(byte_rules)
    for i, (d, r) in enumerate(zip(plan['descs'], raw['legacy'])):
        taken, body, done, worker = r
        if not 0 < taken <= body <= done:
            raise ValueError(f'incomplete legacy task {i}')
        coords = (d['coord'] + [0]*4)[:4]
        legacy.append((d['type'], *coords, *r))
        dependencies = plan['task_in'][i]
        brackets = [raw['events'][words[e]] for e in dependencies]
        if any(not 0 < a <= b for a, b in brackets):
            raise ValueError(f'missing final-arrival bracket for task {i}')
        # Root tasks have no input poll: use taken as an explicitly synthetic
        # runnable origin. Never claim it is a producer timestamp.
        lo = max((a for a, b in brackets), default=taken)
        hi = max((b for a, b in brackets), default=taken)
        poll = raw['poll'][i][0] if dependencies else taken
        if not taken <= poll <= body or lo > poll:
            raise ValueError(f'invalid success order/clock for task {i}')
        nw = raw['waves_per_task']
        waves = raw['waves'][i*nw:(i+1)*nw]
        if any(not body <= a <= b <= done for a, b in waves):
            raise ValueError(f'missing/invalid task {i} wave completion')
        compute = max(a for a, b in waves)
        drain = max(b for a, b in waves)
        name = names[d['type']]
        match = re.search(r'_(\d+)(?:_x)?$', name)
        task = dict(task=i, phase=re.sub(r'_\d+(?:_x)?$', '', name),
                    layer=int(match[1]) if match else None, type=name,
                    record=[*r, hi, poll], arrive_bounds_ticks=[lo, hi],
                    arrive_is_synthetic=not dependencies,
                    poll_minus_arrive_bounds_us=[max(0, poll-hi)*scale, (poll-lo)*scale],
                    acquire_setup_us=(body-poll)*scale,
                    tile_return_us=(compute-body)*scale,
                    all_wave_drain_tail_us=(drain-compute)*scale,
                    barrier_tail_us=(done-drain)*scale,
                    last_return_waves=[w for w, (a, b) in enumerate(waves) if a == compute],
                    last_drain_waves=[w for w, (a, b) in enumerate(waves) if b == drain],
                    bytes=nb(name), body_us=(done-body)*scale)
        tasks.append(task)
    trace = tl.build_trace(plan, legacy, tick_ns=raw['tick_ns'])
    path = tl.critical_path(trace)
    phases = defaultdict(lambda: dict(tasks=0, body_us=0., gap_us=0., byte_floor_estimate_us=0.,
        hop_estimate_us=0., machinery_residual_estimate_us=0., unpriced_body_us=0.,
        wait_bounds_us=[0., 0.], tile_return_us=0., drain_tail_us=0., barrier_tail_us=0.))
    for step in path:
        t = tasks[step['task'].id]
        t['critical'] = True
        t['gap_us'] = step['gap']
        body = t['body_us']
        floor = t['bytes'] / (bandwidth*1e6)
        hop = (hop_us or {}).get(t['type'])
        # Logical-byte floors are estimates; if they exceed measured body,
        # retain the discrepancy rather than silently clamp it to a proof.
        valid = hop is not None and 0 <= floor + hop <= body
        t['classes_us'] = dict(bytes_estimate=floor, gap=step['gap'],
            hop_estimate=hop if valid else None,
            machinery_residual_estimate=body-floor-hop if valid else None)
        t['exclusive_removable_us'] = None
        group = phases[(t['layer'], t['phase'])]
        group['tasks'] += 1
        for key, val in [('body_us', body), ('gap_us', step['gap']), ('byte_floor_estimate_us', floor),
                         ('hop_estimate_us', hop if valid else 0.),
                         ('machinery_residual_estimate_us', body-floor-hop if valid else 0.),
                         ('unpriced_body_us', 0. if valid else body),
                         ('tile_return_us', t['tile_return_us']), ('drain_tail_us', t['all_wave_drain_tail_us']),
                         ('barrier_tail_us', t['barrier_tail_us'])]:
            group[key] += val
        group['wait_bounds_us'] = [a+b for a, b in zip(group['wait_bounds_us'], t['poll_minus_arrive_bounds_us'])]
    return dict(raw=raw, tasks=tasks, critical_task_ids=[s['task'].id for s in path],
                phases=[dict(layer=l, phase=p, **v) for (l, p), v in phases.items()],
                span_us=trace.span, critical_us=sum(s['body']+s['gap'] for s in path),
                acceptance=dict(verdict='NOT-ESTABLISHED', exclusive_attributed_fraction=0.,
                    largest_removable_item=None,
                    reason='issue intervals and byte/hop estimates do not prove exclusive removable atomic work'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('raw', type=Path)
    p.add_argument('--bytes', type=Path)
    p.add_argument('--hops', type=Path, help='independent per-type estimates, never exclusive credit')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(json.loads(a.plan.read_text()), read_raw(a.raw),
        json.loads(a.bytes.read_text()) if a.bytes else None,
        json.loads(a.hops.read_text()) if a.hops else None)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['acceptance']))


if __name__ == '__main__':
    main()
