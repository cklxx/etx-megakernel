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
    if magic != MAGIC or version not in (1, 2, 3, 4) or not nt or not ne or not 0 < nw <= 16 or tick_ns != 10:
        raise ValueError('invalid inbody header')
    count = 6 + nt * 5 + ne * 3 + nt * nw * 2
    if len(data) < count*8 or (version==1 and len(data)!=count*8):
        raise ValueError('truncated/extra inbody records')
    words = struct.unpack('<' + 'Q' * (len(data)//8), data)
    pos = 6
    def take(n, width):
        nonlocal pos
        rows = [list(words[i:i+width]) for i in range(pos, pos + n * width, width)]
        pos += n * width
        return rows
    result=dict(version=version, tick_ns=tick_ns, legacy=take(nt, 4),
                poll=take(nt, 1), events=take(ne, 3), waves=take(nt*nw, 2), waves_per_task=nw)
    if version in (2, 4):
        if len(words)<pos+3: raise ValueError('truncated ledger header')
        ni,nc,nd=words[pos:pos+3];pos+=3
        expected=count+3+(nt+1)+ni*3+nc*(nd+1)
        if nc!=ne or not nd or (len(words)!=expected if version==2 else len(words)<expected): raise ValueError('truncated/extra ledger records')
        offsets=[r[0] for r in take(nt+1,1)]; event_words=[r[0] for r in take(ni,1)]
        stamps=take(ni,2)
        if offsets[0]!=0 or offsets[-1]!=ni or offsets!=sorted(offsets) or any(w>=ne for w in event_words):
            raise ValueError('invalid producer ledger CSR')
        signed=lambda v:v-(1<<64) if v>=(1<<63) else v
        remaining=[signed(r[0]) for r in take(nc,1)]
        sub=[[signed(v) for v in r] for r in take(nd,nc)]
        result['producer_ledger']=dict(offsets=offsets,words=event_words,stamps=stamps,remaining=remaining,sub=sub)
    if version in (3, 4):
        if len(words) < pos+2: raise ValueError('missing load-phase header')
        capacity, size = words[pos:pos+2]; pos += 2
        if not 0 < capacity <= 64 or len(words) != pos+nt+1+size:
            raise ValueError('truncated/extra load-phase records')
        offsets = [r[0] for r in take(nt+1, 1)]
        width = nw*(1+capacity*3)
        if offsets[0] or offsets[-1] != size or any(b-a not in (0, width) for a,b in zip(offsets,offsets[1:])):
            raise ValueError('invalid load-phase CSR')
        result['loads'] = dict(capacity=capacity, offsets=offsets, data=list(words[pos:]))
    return result


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
    types = {v['id']: v for v in plan['types'].values()}
    words = event_words(plan)
    legacy = []
    tasks = []
    scale = raw['tick_ns'] / 1000
    nb = tl.bytes_spec(byte_rules)
    known_bytes = [re.compile(p) for p in (byte_rules or {})]
    for i, (d, r) in enumerate(zip(plan['descs'], raw['legacy'])):
        taken, body, done, worker = r
        if not 0 < taken <= body <= done:
            raise ValueError(f'incomplete legacy task {i}')
        coords = (d['coord'] + [0]*4)[:4]
        legacy.append((d['type'], *coords, *r))
        dependencies = plan['task_in'][i]
        brackets = [raw['events'][words[e]] for e in dependencies]
        if any(not 0 < a <= b for a, b, owner in brackets):
            raise ValueError(f'missing final-arrival bracket for task {i}')
        if any(owner not in plan['ev_inst'][e][2] for e, (_, _, owner) in zip(dependencies, brackets)):
            raise ValueError(f'final arrival has invalid producer for task {i}')
        # Root tasks have no input poll: use taken as an explicitly synthetic
        # runnable origin. Never claim it is a producer timestamp.
        lo = max((a for a, b, _ in brackets), default=taken)
        hi = max((b for a, b, _ in brackets), default=taken)
        poll = raw['poll'][i][0] if dependencies else taken
        # Zero is the reset value; one is also a missing/stale stamp when the
        # task was taken later. Do not turn either into a synthetic success.
        if dependencies and poll <= 1 and poll < taken:
            raise ValueError(f'poll buffer not written with a valid timestamp for task {i}: poll={poll}')
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
        ty=types[d['type']]
        coord=d['coord']
        tile_block=None
        if 'dsl_moe_' in ty.get('symbol',''):
            tile_block=coord[0]*ty['grid'][1]+coord[1] if ty['symbol'].startswith('etx_tile_x_') else coord[0]
        task = dict(task=i, phase=re.sub(r'_\d+(?:_x)?$', '', name),
                    layer=int(match[1]) if match else None, type=name,
                    coord=coord, tile_block=tile_block, body_symbol=ty.get('symbol'),
                    record=[*r, hi, poll], arrive_bounds_ticks=[lo, hi],
                    arrive_is_synthetic=not dependencies,
                    arrive_producer_tasks=sorted({owner for _, b, owner in brackets if b == hi}),
                    poll_minus_arrive_bounds_us=[max(0, poll-hi)*scale, (poll-lo)*scale],
                    acquire_setup_us=(body-poll)*scale,
                    tile_return_us=(compute-body)*scale,
                    all_wave_drain_tail_us=(drain-compute)*scale,
                    barrier_tail_us=(done-drain)*scale,
                    last_return_waves=[w for w, (a, b) in enumerate(waves) if a == compute],
                    last_drain_waves=[w for w, (a, b) in enumerate(waves) if b == drain],
                    bytes=nb(name) if any(p.search(name) for p in known_bytes) else None,
                    body_us=(done-body)*scale)
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
        floor = t['bytes'] / (bandwidth*1e6) if t['bytes'] is not None else None
        hop = (hop_us or {}).get(t['type'])
        # Logical-byte floors are estimates; if they exceed measured body,
        # retain the discrepancy rather than silently clamp it to a proof.
        valid = floor is not None and hop is not None and 0 <= floor + hop <= body
        t['classes_us'] = dict(bytes_estimate=floor, gap=step['gap'],
            hop_estimate=hop if valid else None,
            machinery_residual_estimate=body-floor-hop if valid else None)
        t['exclusive_removable_us'] = None
        group = phases[(t['layer'], t['phase'])]
        group['tasks'] += 1
        for key, val in [('body_us', body), ('gap_us', step['gap']), ('byte_floor_estimate_us', floor or 0.),
                         ('hop_estimate_us', hop if valid else 0.),
                         ('machinery_residual_estimate_us', body-floor-hop if valid else 0.),
                         ('unpriced_body_us', 0. if valid else body),
                         ('tile_return_us', t['tile_return_us']), ('drain_tail_us', t['all_wave_drain_tail_us']),
                         ('barrier_tail_us', t['barrier_tail_us'])]:
            group[key] += val
        group['wait_bounds_us'] = [a+b for a, b in zip(group['wait_bounds_us'], t['poll_minus_arrive_bounds_us'])]
    all_bytes=defaultdict(float)
    unknown_bytes=set()
    for t in tasks:
        if t['bytes'] is not None: all_bytes[(t['layer'],t['phase'])]+=t['bytes']
        else: unknown_bytes.add((t['layer'],t['phase']))
    return dict(raw=raw, tasks=tasks, critical_task_ids=[s['task'].id for s in path],
                phases=[dict(layer=l, phase=p, all_grid_byte_floor_estimate_us=None if (l,p) in unknown_bytes else all_bytes[(l,p)]/(bandwidth*1e6), **v)
                        for (l, p), v in phases.items()],
                global_known_byte_floor_estimate_us=sum(all_bytes.values())/(bandwidth*1e6),
                byte_floor_limit='sum of per-critical-task full-bandwidth floors is NOT the full-grid/global byte floor',
                span_us=trace.span, critical_us=sum(s['body']+s['gap'] for s in path),
                acceptance=dict(verdict='NOT-ESTABLISHED', exclusive_attributed_fraction=0.,
                    largest_removable_item=None,
                    reason='issue intervals and byte/hop estimates do not prove exclusive removable atomic work'))



# Logical streamed weight bytes per full task at the registered exact shape.
# MASTER-TABLE: QK6, OP3 and M.J. Touch reads, activations and cache hits are
# not HBM traffic; this is a logical stream rate, not a hardware counter.
LOAD_ROLES = {'moe_up': (192, 262144), 'moe_down': (256, 98304),
              'qkv': (256, 81920), 'o_proj': (256, 65536)}


def union_ticks(intervals):
    end = None; total = 0
    for a, b in sorted(intervals):
        total += b-a if end is None or a >= end else max(0, b-end)
        end = b if end is None else max(end, b)
    return total


def load_phases(plan, raw, layers=range(48)):
    """Complete load groups across the grid; no whole-body denominator.

    Union within each layer/role to count concurrent waves once. Sum layer
    durations, never sum parallel task durations or divide all-grid bytes by
    one critical task's duration. Missing records/bytes fail closed.
    """
    if 'loads' not in raw:
        raise ValueError('missing load-phase stamps: rebuild stamps with load-group begin/ready/bytes (inbody v3/v4)')
    load = raw['loads']; nw = raw['waves_per_task']; cap = load['capacity']; width = 1+cap*3
    names = {v['id']: name for name, v in plan['types'].items()}
    groups = defaultdict(list); populations = defaultdict(list)
    for i, d in enumerate(plan['descs']):
        name = names[d['type']]
        match = re.fullmatch(r'(moe_up|moe_down|qkv|o_proj)_(\d+)(?:_x)?', name)
        if not match: continue
        role, layer = match[1], int(match[2]); key = (layer, role)
        begin, end = load['offsets'][i:i+2]
        if end-begin != nw*width:
            raise ValueError(f'missing load-phase stamps for {name} task {i}')
        body, done = raw['legacy'][i][1:3]
        byte_count = 0
        for wave in range(nw):
            row = load['data'][begin+wave*width:begin+(wave+1)*width]
            count = row[0]
            if not 0 < count <= cap:
                raise ValueError(f'missing/overflow load-phase groups for {name} task {i} wave {wave}: {count}')
            last = body
            for j in range(count):
                a, b, n = row[1+j*3:4+j*3]
                wave_return = raw['waves'][i*nw+wave][0]
                if not 0 < body <= last <= a < b <= wave_return <= done or not n:
                    raise ValueError(f'missing/invalid load-phase stamp for {name} task {i} wave {wave} group {j}')
                groups[key].append((a,b)); byte_count += n; last = b
        if byte_count != LOAD_ROLES[role][1]:
            raise ValueError(f'load-phase byte coverage for {name} task {i}: {byte_count} != {LOAD_ROLES[role][1]}')
        populations[key].append(byte_count)
    rows = []
    for layer in layers:
        for role, (nt, _) in LOAD_ROLES.items():
            key = (layer, role)
            if len(populations[key]) != nt:
                raise ValueError(f'incomplete load-phase {role} layer {layer}: {len(populations[key])}/{nt} tasks')
            us = union_ticks(groups[key])*raw['tick_ns']/1000
            size = sum(populations[key]); rate = size/(us*1e6)
            rows.append(dict(layer=layer, role=role, tasks=nt, groups=len(groups[key]),
                             bytes=size, load_union_us=us, logical_TB_s=rate))
    totals = []
    for role in LOAD_ROLES:
        rs = [r for r in rows if r['role']==role]
        us = sum(r['load_union_us'] for r in rs); size = sum(r['bytes'] for r in rs)
        rate = size/(us*1e6)
        totals.append(dict(role=role, layers=len(rs), bytes=size, load_union_us=us,
            logical_TB_s=rate, decision='non-byte body work' if rate >= 4.0 else
            'memory path' if rate <= 3.3 else 'UNRESOLVED between 3.3 and 4.0',
            actionable=False))
    return dict(layers=rows, totals=totals,
        basis='logical streamed weights / union of load issue-to-ready groups per layer; ON diagnostic, not exclusive HBM latency',
        validity='UNVERIFIED perturbation: decisions actionable only after unchanged OFF/ON <=0.5 us sampled-body gate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('raw', type=Path)
    p.add_argument('--bytes', type=Path)
    p.add_argument('--hops', type=Path, help='independent per-type estimates, never exclusive credit')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--legacy-output', type=Path, help='export unchanged nine-i64 rows for existing analyses')
    a = p.parse_args()
    result = analyze(json.loads(a.plan.read_text()), read_raw(a.raw),
        json.loads(a.bytes.read_text()) if a.bytes else None,
        json.loads(a.hops.read_text()) if a.hops else None)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    if a.legacy_output:
        rows=[]
        for d,r in zip(json.loads(a.plan.read_text())['descs'],result['raw']['legacy']):
            rows.append(struct.pack('<9q',d['type'],*(d['coord']+[0]*4)[:4],*r))
        a.legacy_output.write_bytes(b''.join(rows))
    print(json.dumps(result['acceptance']))


if __name__ == '__main__':
    main()
