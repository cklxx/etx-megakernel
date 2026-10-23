"""Offline descriptor arithmetic and fail-closed payload-read auditing."""
import math

import pytest

from examples.glm53 import model_build, module_gate, stage
from examples.glm53.prepare import partition
from examples.spec.ledger import ReadLedger, graph_ledger, ledger, tensor_bytes


@pytest.fixture(scope='module', params=['module8', 'production'])
def spec(request):
    if request.param == 'production':
        return model_build.production_compile_spec()
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


def routes(spec, k, pattern='same'):
    t = spec['config']
    topk, n = t['num_experts_per_tok'], t['n_routed_experts']
    return {f'l{i}': [[(j + (p*topk if pattern == 'rotate' else 0)) % n
                       for j in range(topk)] for p in range(k+1)]
            for i, kind in enumerate(t['mlp_layer_types']) if kind == 'sparse'}


@pytest.mark.parametrize('k', [0, 1, 3, 5])
@pytest.mark.parametrize('pattern', ['same', 'rotate'])
def test_verify_byte_ledger(spec, k, pattern):
    ids = routes(spec, k, pattern)
    report = ledger(spec, k, experts=ids)
    # Independent sum from raw shapes, dtypes and explicitly selected IDs.
    sequential = distinct = 0
    for group, blob in spec['weights'].items():
        for key, rec in blob['tensors'].items():
            shape, dtype = rec['shape'], rec['dtype']
            n = math.prod(shape) * {'torch.bfloat16': 2, 'torch.float32': 4,
                                    'torch.float8_e4m3fn': 1}[dtype]
            if dtype == 'torch.float8_e4m3fn':
                n += 4*((shape[0]+127)//128)*((math.prod(shape[1:])+127)//128)
            hits = k+1
            if key.startswith('mlp.experts.'):
                expert = int(key.split('.')[2])
                hits = sum(expert in row for row in ids[group])
            unique = int(hits > 0)
            if group == 'embedding':
                n = shape[1]*2
                unique = hits
            sequential += n*hits
            distinct += n*unique
    assert report['sequential_bytes'] == sequential
    assert report['distinct_bytes'] == distinct
    assert 0 < distinct <= sequential
    assert not report['read_once_established']
    if k == 0:
        assert distinct == sequential
    else:
        assert distinct < sequential
    # Each selected expert's three matrices are counted once in the union,
    # including shared experts, independently of selection multiplicity.
    for layer, positions in ids.items():
        union = set().union(*map(set, positions))
        selected = [r for r in report['rows'] if r['group'] == layer
                    and r['tensor'].startswith('mlp.experts.') and r['distinct_reads']]
        assert len(selected) == 3*len(union)


@pytest.mark.parametrize('k', [0, 1, 3, 5])
def test_actual_graph_does_not_prove_read_reuse(spec, k):
    from examples.glm53 import model, verify_graph
    graph = model.build_from(spec) if k == 0 else verify_graph.build_from(spec, k)
    # Full head matrix is consumed by a separate head call per position.
    heads = [g for g in graph.grids if g.body.symbol == 'glm53_lm']
    assert len(heads) == k+1
    assert all('head_weights' in g.reads for g in heads)
    report = graph_ledger(graph)
    weight_storage = sum(t.nbytes({'B': 1}) for t in graph.tensors.values() if t.role == 'weight')
    assert report['storage_bytes']['weight'] == weight_storage
    head_reads = [r for r in report['rows'] if r['tensor'] == 'head_weights']
    assert len(head_reads) == k+1
    assert sum(r['bytes'] for r in head_reads) == (k+1)*spec['weights']['head']['blob_bytes']
    # No copy of recurrent state or weight storage per draft position.
    baseline = graph_ledger(model.build_from(spec))
    assert report['storage_bytes']['weight'] == baseline['storage_bytes']['weight']
    assert report['storage_bytes']['runtime'] == baseline['storage_bytes']['runtime']


@pytest.mark.parametrize('k', [0, 1, 3, 5])
def test_observed_union_read_once_and_failures(k):
    # Reader events deliberately distinct from descriptor allocation counts.
    ids = [{p % 3, (p+1) % 3} for p in range(k+1)]
    expected = {('l1', f'mlp.experts.{e}.{matrix}', r, c)
                for e in set.union(*ids) for matrix in ('gate', 'up', 'down')
                for r in range(3) for c in range(5)}
    audit = ReadLedger(expected)
    assert not audit.report()['read_once']
    for key in expected:
        audit.read(*key)
    assert audit.report()['read_once']
    assert audit.report()['element_reads'] == len(expected)
    audit.read(*next(iter(expected)))
    assert not audit.report()['read_once']
    audit = ReadLedger(expected)
    for key in expected:
        audit.read(*key)
    audit.read('l1', 'unselected', 0, 0)
    assert audit.report()['unexpected']
    assert not audit.report()['read_once']


def test_fp8_scales_and_padding():
    rec = dict(shape=[129, 257], dtype='torch.float8_e4m3fn', offset=128)
    assert tensor_bytes(rec) == (129*257, 2*3*4)
    assert tensor_bytes(dict(shape=[3, 2, 129], dtype='torch.float8_e4m3fn')) == (774, 12)


@pytest.mark.parametrize('k', [-1, True, 1.5])
def test_invalid_k(spec, k):
    with pytest.raises(ValueError, match='k must'):
        ledger(spec, k, experts={})


def test_missing_or_invalid_routes(spec):
    with pytest.raises(ValueError, match='exactly the sparse'):
        ledger(spec, 1, experts={})
    ids = routes(spec, 1)
    first = next(iter(ids))
    ids[first][0] = [0]*spec['config']['num_experts_per_tok']
    with pytest.raises(ValueError, match='invalid top-k'):
        ledger(spec, 1, experts=ids)


def test_missing_descriptors_refuse(spec):
    from copy import deepcopy
    bad = deepcopy(spec)
    bad['weights']['l0']['tensors'].pop('self_attn.q_proj.weight')
    with pytest.raises(ValueError, match='incomplete tensor schema'):
        ledger(bad, 0, experts=routes(spec, 0))


def test_empty_audit_does_not_establish_read_once():
    assert not ReadLedger(set()).report()['read_once']
