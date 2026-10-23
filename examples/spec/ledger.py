"""Weights-free verify weight-byte ledger.

Counts logical payload and FP8 block-scale bytes, excluding blob padding,
descriptor loads, activations and cache traffic. These are NOT measured HBM
transactions. A shared IR buffer does not imply a shared weight load.

``ledger`` compares sequential position reads with the distinct-weight lower
bound. ``ReadLedger`` audits observed element reads (e.g. an injected CPU tile
reader); only that audit can establish the read-once condition. Repeated scale
lookups are deliberately separate from payload read-once qualification.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

ITEM_BYTES = {'torch.bfloat16': 2, 'torch.float32': 4,
              'torch.float8_e4m3fn': 1}
EXPERT = re.compile(r'mlp\.experts\.(\d+)\.')


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def tensor_bytes(record):
    """Return (payload, scale) bytes, including flattened convolution columns."""
    shape = record['shape']
    if not shape or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('invalid tensor shape')
    try:
        item = ITEM_BYTES[record['dtype']]
    except KeyError as exc:
        raise ValueError(f'unsupported dtype: {record["dtype"]}') from exc
    rows, cols = shape[0], math.prod(shape[1:])
    scales = (4 * math.ceil(rows / 128) * math.ceil(cols / 128)
              if record['dtype'] == 'torch.float8_e4m3fn' else 0)
    return math.prod(shape) * item, scales


def _routes(spec, k, experts):
    """Explicit IDs per position, per sparse layer; never assume expert overlap."""
    t = spec['config']
    sparse = {f'l{i}' for i, kind in enumerate(t['mlp_layer_types']) if kind == 'sparse'}
    if set(experts) != sparse:
        raise ValueError('expert IDs required for exactly the sparse layers')
    out = {}
    for layer, positions in experts.items():
        if len(positions) != k + 1:
            raise ValueError(f'{layer}: expected k+1 routing rows')
        rows = []
        for row in positions:
            row = tuple(row)
            if (len(row) != t['num_experts_per_tok'] or len(set(row)) != len(row)
                    or any(type(e) is not int or not 0 <= e < t['n_routed_experts'] for e in row)):
                raise ValueError(f'{layer}: invalid top-k expert IDs')
            rows.append(set(row))
        out[layer] = rows
    return out


def ledger(spec, k, *, experts):
    """B1 logical bytes for last accepted token + k drafts, with explicit routing.

    ``distinct`` is an ideal reuse bound, not a statement about verify_phases.h.
    Sequential bytes also count each active tensor only once per position;
    tile-level repeated norms/scales are outside this logical-byte basis.
    The present WP-S2 tile loops read matrices per position even when experts
    are grouped. No read-once verdict is inferred from this analytical report.
    Embedding counts one row per position; its distinct bound conservatively
    keeps those rows separate because token IDs are not supplied.
    """
    _integer(k, 'k')
    _validate_schema(spec)
    routes = _routes(spec, k, experts)
    rows = []
    for group, blob in spec['weights'].items():
        for key, record in blob['tensors'].items():
            payload, scales = tensor_bytes(record)
            selected = k + 1
            match = EXPERT.match(key)
            if match:
                if group not in routes:
                    raise ValueError(f'{group}: expert descriptor outside sparse layer')
                selected = sum(int(match[1]) in row for row in routes[group])
            distinct = int(selected > 0)
            if group == 'embedding':
                payload //= record['shape'][0]
                if scales:
                    raise ValueError('FP8 embedding row accounting is unsupported')
                distinct = selected
            rows.append(dict(group=group, tensor=key, payload_bytes=payload,
                             scale_bytes=scales, sequential_reads=selected,
                             distinct_reads=distinct))
    total = lambda field: sum((r['payload_bytes'] + r['scale_bytes']) * r[field] for r in rows)
    return dict(k=k, positions=k+1, batch=1, basis='logical weight bytes; not HBM traffic',
                rows=rows, sequential_bytes=total('sequential_reads'),
                distinct_bytes=total('distinct_reads'), read_once_established=False)


def _validate_schema(spec):
    from examples.glm53.stage import shapes
    t = spec['config']
    groups = {'embedding', 'norm', 'head'} | {f'l{i}' for i in range(t['num_hidden_layers'])}
    if set(spec['weights']) != groups:
        raise ValueError('incomplete weight groups')
    for group, blob in spec['weights'].items():
        if group.startswith('l') and group[1:].isdigit():
            expected = shapes(t, int(group[1:]))
        else:
            key = {'embedding': 'model.embed_tokens.weight', 'norm': 'model.norm.weight',
                   'head': 'lm_head.weight'}[group]
            shape = ((t['hidden_size'],) if group == 'norm'
                     else (t['vocab_size'], t['hidden_size']))
            expected = {group if group in blob['tensors'] else key: shape}
        if set(blob['tensors']) != set(expected):
            raise ValueError(f'{group}: incomplete tensor schema')
        for key, shape in expected.items():
            if tuple(blob['tensors'][key]['shape']) != tuple(shape):
                raise ValueError(f'{group}/{key}: descriptor shape mismatch')


def graph_ledger(graph, *, batch=1):
    """IR tensor-reference byte ledger, broken down by launch and tensor role.

    Every declared read/write counts its entire tensor once per grid, NOT per
    workgroup. This exposes replicated position buffers and shared state but
    cannot describe subregions, repeated loads or actual DRAM traffic. Weight
    blobs include padding here; use ``ledger`` for active payload bytes.
    """
    _integer(batch, 'batch', 1)
    rows, totals = [], {}
    for grid in graph.grids:
        for access, names in [('read', grid.reads), ('write', grid.writes)]:
            for name in names:
                tensor = graph.tensors[name]
                n = tensor.nbytes({'B': batch})
                row = dict(grid=grid.name, tensor=name, role=tensor.role,
                           access=access, bytes=n, device=grid.device)
                rows.append(row)
                key = (access, tensor.role)
                totals[key] = totals.get(key, 0) + n
    storage = {}
    for tensor in graph.tensors.values():
        storage[tensor.role] = storage.get(tensor.role, 0) + tensor.nbytes({'B': batch})
    return dict(basis='whole-tensor IR references; not HBM traffic',
                rows=rows, reference_bytes=totals, storage_bytes=storage)


@dataclass
class ReadLedger:
    """Audit actual payload element reads against an explicit expected union.

    Keys are (weight group, tensor key, row, column). The expected union is
    supplied independently of observations: missing reads must fail as well
    as duplicates and unexpected experts. Scales are not payload elements.
    """
    expected: set

    def __post_init__(self):
        self.expected = frozenset(self.expected)
        self.counts = Counter()

    def read(self, group, tensor, row, column):
        self.counts[group, tensor, row, column] += 1

    def report(self):
        missing = self.expected - self.counts.keys()
        unexpected = self.counts.keys() - self.expected
        repeated = {key: n for key, n in self.counts.items() if n != 1}
        return dict(read_once=bool(self.expected) and not (missing or unexpected or repeated),
                    missing=missing, unexpected=unexpected, repeated=repeated,
                    element_reads=sum(self.counts.values()))


class ObservedBytes:
    """Sink for instrumented logical weight reads, including repeated scales.

    Call ``read`` at the reader boundary, never from graph declarations or
    descriptor allocation. Counts describe bytes requested by the reader, not
    physical HBM transactions. Storage is O(tensors), also for production.
    """

    def __init__(self, spec):
        _validate_schema(spec)
        self.spec = spec
        self.payload = Counter()
        self.scales = Counter()

    def read(self, group, tensor, *, elements=1, scales=0):
        """Record payload elements and FP32 scale loads actually performed."""
        _integer(elements, 'elements')
        _integer(scales, 'scales')
        record = self.spec['weights'][group]['tensors'][tensor]
        if scales and record['dtype'] != 'torch.float8_e4m3fn':
            raise ValueError('scale reads require an FP8 descriptor')
        key = group, tensor
        self.payload[key] += elements * ITEM_BYTES[record['dtype']]
        self.scales[key] += scales * 4

    def snapshot(self):
        return {key: self.payload[key] + self.scales[key]
                for key in self.payload.keys() | self.scales.keys()}


def fc2a(spec, observations, *, experts):
    """Qualify instrumented per-step byte snapshots for k=0,1,3,5.

    ``observations[k]`` maps (group, tensor) to observed bytes. Missing or
    extra tensors fail closed; analytical ledgers are not observations.
    Routed expectations include FP8 scales once per block. Fixed tensors
    are compared individually to their observed decode baseline (0.1%).
    """
    _validate_schema(spec)
    ks = (0, 1, 3, 5)
    reasons = []
    expected_keys = {(g, t) for g, b in spec['weights'].items() for t in b['tensors']}
    for k in ks:
        if k not in observations:
            reasons.append(f'k={k}: missing observed instrumented reads')
            continue
        observed = observations[k]
        routes = _routes(spec, k, experts[k])
        if set(observed) != expected_keys:
            reasons.append(f'k={k}: incomplete observed tensor coverage '
                           f'(missing={len(expected_keys-set(observed))}, '
                           f'unexpected={len(set(observed)-expected_keys)})')
        for key, n in observed.items():
            _integer(n, 'observed bytes')
            if key not in expected_keys:
                continue
            group, tensor = key
            match = EXPERT.match(tensor)
            if match:
                union = set().union(*routes[group])
                target = (sum(tensor_bytes(spec['weights'][group]['tensors'][tensor]))
                          if int(match[1]) in union else 0)
                if n != target:
                    reasons.append(f'k={k} {group}/{tensor}: routed observed={n} '
                                   f'expected={target} excess={n-target} bytes')
            else:
                target = observations.get(0, {}).get(key)
                if target is None or target <= 0:
                    reasons.append(f'k={k} {group}/{tensor}: missing positive observed baseline')
                elif abs(n-target)*1000 > target:
                    reasons.append(f'k={k} {group}/{tensor}: fixed observed={n} '
                                   f'baseline={target} excess={n-target} bytes')
    return dict(passed=not reasons, basis='observed logical reader bytes; not HBM traffic',
                reasons=reasons)


def gate_spec(name):
    """Build gate descriptors without materializing any checkpoint weights."""
    from examples.glm53 import model_build, module_gate, stage
    from examples.glm53.prepare import partition
    if name == 'production':
        return model_build.production_compile_spec()
    if name != 'module8':
        raise ValueError('unknown gate')
    t = module_gate.module_config('module8')['text_config']
    slots = stage.slot_names(t)
    weights = {}
    for i, kind in enumerate(t['layer_types']):
        weights[f'l{i}'] = model_build.compile_descriptors(
            f'l{i}', stage.shapes(t, i), slots,
            lambda key, shape, kind=kind: module_gate.storage(kind, key, shape))
    for name, shape in [('embedding', (t['vocab_size'], t['hidden_size'])),
                        ('norm', (t['hidden_size'],)),
                        ('head', (t['vocab_size'], t['hidden_size']))]:
        weights[name] = model_build.compile_descriptors(
            name, {name: shape}, [name], lambda key, shape: 'BF16')
    return dict(config=t, dimensions=stage.dimensions(t, 160),
                ranks=partition(t), weights=weights)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fc2a', action='store_true', required=True)
    parser.parse_args(argv)
    # The wired graph currently has no full-graph instrumented read provider.
    # In particular, IR reads cannot be substituted for observed reader events.
    # Report both descriptor gates, and refuse to certify from ideal arithmetic.
    reasons = []
    for name in ('module8', 'production'):
        spec = gate_spec(name)
        report = fc2a(spec, {}, experts={})
        reasons.extend(f'{name} {reason}' for reason in report['reasons'])
    print('FC-2a FAIL ' + '; '.join(reasons))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
