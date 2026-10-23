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
