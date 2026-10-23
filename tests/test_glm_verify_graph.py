"""Weights-free module8 verification: actual CPU tiles, no GPU or downloads."""
import numpy as np
import pytest

from etx.passes import compile_graph, verify_plan
from examples.glm53 import graph_cpu, model_build, module_gate, stage
from examples.glm53.options import Options
from examples.glm53.verify_graph import VerifyGraphCPU, build_from


@pytest.fixture(scope='module')
def module8(tmp_path_factory):
    root = tmp_path_factory.mktemp('verify-module8')
    module_gate.checkpoint(root / 'checkpoint', variant='module8')
    stage.stage(root / 'checkpoint', root / 'stage', maxlen=160, gate='module8')
    return root


@pytest.fixture(scope='module')
def sequential(module8):
    return graph_cpu.GraphCPU(module8 / 'stage', Options(), 2, 'module8',
                              module8 / 'sequential', poison=True)


def _reset_state(cpu):
    for name, tensor in cpu.graph.tensors.items():
        if tensor.role == 'runtime':
            cpu.buffers[name][:] = 0


@pytest.mark.parametrize('k', [1, 3, 5])
def test_module8_logits_and_caches_equal_sequential(module8, sequential, k):
    verify = VerifyGraphCPU(module8 / 'stage', k, 2, 'module8', module8 / f'k{k}', poison=True)
    _reset_state(sequential)
    rng = np.random.default_rng(20261023)
    # Exercise DSA's pruned-pool branch (index_topk=64), pool boundaries,
    # KDA convolution history, and a second block using the first one's state.
    # Seed both with exactly the same valid prefix; no model/attention mocks.
    for pos in range(65):
        sequential.step(pos, rng.integers(0, sequential.d['VOCAB'], size=2))
    for name, tensor in sequential.graph.tensors.items():
        if tensor.role == 'runtime':
            verify.buffers[name][:] = sequential.buffers[name]
    for pos in (65, 65 + k + 1):
        tokens = rng.integers(0, sequential.d['VOCAB'], size=(k + 1, 2))
        got_ids, got_logits = verify.step(pos, tokens)
        expected = [sequential.step(pos + p, row) for p, row in enumerate(tokens)]
        np.testing.assert_array_equal(got_ids, np.stack([x[0] for x in expected]))
        np.testing.assert_array_equal(got_logits, np.stack([x[1] for x in expected]))
        assert np.isfinite(got_logits).all()
        for name, tensor in sequential.graph.tensors.items():
            if tensor.role == 'runtime':
                np.testing.assert_array_equal(verify.buffers[name], sequential.buffers[name], err_msg=name)
    # Start from empty history too; all k+1 outputs (including position zero)
    # are returned, rather than just the last draft's prediction.
    _reset_state(verify)
    _reset_state(sequential)
    tokens = rng.integers(0, sequential.d['VOCAB'], size=(k + 1, 2))
    ids, logits = verify.step(0, tokens)
    expected = [sequential.step(p, row) for p, row in enumerate(tokens)]
    np.testing.assert_array_equal(ids, np.stack([x[0] for x in expected]))
    np.testing.assert_array_equal(logits, np.stack([x[1] for x in expected]))
    before = {n: b.copy() for n, b in verify.buffers.items()
              if verify.graph.tensors[n].role == 'runtime'}
    for pos, bad in [(0, tokens[:-1]), (0, tokens.astype(float)),
                     (0, np.full_like(tokens, -1)), (160-k, tokens), (-1, tokens)]:
        with pytest.raises(ValueError):
            verify.step(pos, bad)
    for n, b in before.items():
        np.testing.assert_array_equal(verify.buffers[n], b)


@pytest.mark.parametrize('k', [1, 3, 5])
def test_verify_plan_has_causal_scan_and_expert_groups(k):
    spec = model_build.production_compile_spec()
    spec['config'] = dict(spec['config'], num_hidden_layers=2,
        layer_types=['linear_attention', 'deepseek_sparse_attention'],
        mlp_layer_types=['dense', 'sparse'])
    spec['weights'] = {n: w for n, w in spec['weights'].items()
                       if n in ('embedding', 'norm', 'head', 'l0', 'l1')}
    spec['ranks'] = [dict(rank=0, layers=[0]), dict(rank=1, layers=[1])]
    g = build_from(spec, k, batch=2)
    plan = compile_graph(g, 'gfx942', {'B': 2}, {})
    assert not verify_plan(plan)
    scan = g.grid('l0_kda_scan')
    assert scan.consts[2] == k + 1
    assert all(f'p{p}_l0_att' in scan.writes for p in range(k + 1))
    assert g.tensors['l0_state'].shape[0] == 'B'
    assert not any(n.startswith('p0_l0_state') for n in g.tensors)
    dsa = g.grid('l1_dsa_scan')
    assert dsa.consts[2] == k + 1 and dsa.grid == ('B', 1)
    assert 'l1_state' in dsa.writes
    for p in range(k + 1):
        assert f'p{p}_step' in dsa.reads
        assert f'p{p}_l1_att' in dsa.writes
    groups = [x for x in g.grids if x.body.symbol == 'glm53_verify_expert_up']
    assert len(groups) == 1
    assert groups[0].grid[1] == groups[0].consts[1] * (spec['dimensions']['EXPERTS'] + 1)
    for gr in groups:
        for p in range(k + 1):
            assert f'p{p}_l1_sel' in gr.reads
            assert f'p{p}_l1_act' in gr.writes
    assert all(f'p{p}_head_scratch' in g.tensors for p in range(k + 1))


@pytest.mark.parametrize('k', [0, -1, True, 1.5])
def test_invalid_draft_counts(k):
    with pytest.raises(ValueError, match='positive integer'):
        build_from({}, k)


@pytest.mark.parametrize('options', [Options(phased=0), Options(parallel=1)])
def test_unsupported_graphs_fail_explicitly(options):
    with pytest.raises(ValueError, match='verify requires'):
        build_from({}, 1, options=options)


@pytest.mark.parametrize('k', [1, 3, 5])
def test_read_once_moe_boundaries(module8, k):
    spec = stage.validate(module8 / 'stage', allow_fixture=True, gate='module8')
    g = build_from(spec, k, batch=2)
    for i, kind in enumerate(spec['config']['mlp_layer_types']):
        if kind != 'sparse':
            continue
        layer = f'l{i}'
        up = g.grid(layer + '_expert_up')
        down = g.grid(layer + '_expert_down')
        assert down.body.symbol == 'glm53_verify_expert_down'
        assert 'e_' + up.name in down.in_edges
        assert down.consts[2] == k + 1
        for p in range(k + 1):
            partial = f'p{p}_{layer}_partial'
            assert partial in down.writes
            assert g.tensors[partial].shape == ('B', (spec['dimensions']['TOPK'] + 1) * spec['dimensions']['H'])
            mix = g.grid(f'p{p}_{layer}_mix')
            assert mix.body.symbol == 'glm53_verify_expert_mix'
            assert partial in mix.reads
            assert 'e_' + down.name in mix.in_edges
            assert not any(x.name == f'p{p}_{layer}_down' for x in g.grids)


def test_read_once_position_limit():
    with pytest.raises(ValueError, match='at most 6 positions'):
        build_from({}, 6)
