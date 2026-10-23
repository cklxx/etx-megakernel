"""CPU rollback qualification with real KDA, DSA, and model logit tiles."""
import numpy as np
import pytest

from etx.passes import compile_graph, verify_plan
from examples.glm53 import graph_cpu, module_gate, stage
from examples.glm53.options import Options
from examples.glm53.verify_state import VerifyStateCPU


@pytest.fixture(scope='module')
def runners(tmp_path_factory):
    root = tmp_path_factory.mktemp('rollback')
    module_gate.checkpoint(root / 'checkpoint', variant='module8')
    stage.stage(root / 'checkpoint', root / 'stage', maxlen=96, gate='module8')
    seq = graph_cpu.GraphCPU(root / 'stage', Options(), 2, 'module8', root / 'seq', poison=True)
    verify = VerifyStateCPU(root / 'stage', 3, 2, 'module8', root / 'verify', poison=True)
    rng = np.random.default_rng(123)
    for p in range(65):
        seq.step(p, rng.integers(0, seq.d['VOCAB'], size=2))
    seeded = {n: seq.buffers[n].copy() for n in verify.states}
    return seq, verify, seeded


def reset(cpu):
    for name, tensor in cpu.graph.tensors.items():
        if tensor.role == 'runtime':
            cpu.buffers[name].fill(0)


def equal_state(seq, verify):
    for name in verify.states:
        np.testing.assert_array_equal(verify.buffers[name], seq.buffers[name], err_msg=name)


@pytest.mark.parametrize('prefix', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('start', [0, 65])
def test_every_rejection_matches_never_drafted(runners, prefix, start):
    seq, verify, seeded = runners
    reset(seq)
    reset(verify)
    rng = np.random.default_rng(123)
    if start:
        for name, data in seeded.items():
            seq.buffers[name][:] = data
    for name in verify.states:
        verify.buffers[name][:] = seq.buffers[name]
    tokens = rng.integers(0, seq.d['VOCAB'], size=(4, 2))
    ids, logits = verify.step(start, tokens, accepted=prefix)
    for p in range(prefix):
        expected_ids, expected_logits = seq.step(start + p, tokens[p])
        np.testing.assert_array_equal(ids[p], expected_ids)
        np.testing.assert_array_equal(logits[p], expected_logits)
    equal_state(seq, verify)
    # Replacement and continuation must observe only the accepted prefix.
    replacement = rng.integers(0, seq.d['VOCAB'], size=(4, 2))
    ids, logits = verify.step(start + prefix, replacement, accepted=4)
    for p in range(4):
        expected_ids, expected_logits = seq.step(start + prefix + p, replacement[p])
        np.testing.assert_array_equal(ids[p], expected_ids)
        np.testing.assert_array_equal(logits[p], expected_logits)
    equal_state(seq, verify)


def test_independent_request_prefixes_and_validation(runners):
    seq, verify, seeded = runners
    reset(seq)
    reset(verify)
    tokens = np.arange(8).reshape(4, 2)
    before = {n: verify.buffers[n].copy() for n in verify.states}
    for bad in (-1, 5, True, 1.5, [1], [0, 5]):
        with pytest.raises(ValueError):
            verify.step(0, tokens, accepted=bad)
    for n, b in before.items():
        np.testing.assert_array_equal(verify.buffers[n], b)
    verify.step(0, tokens)
    with pytest.raises(RuntimeError):
        verify.step(0, tokens)
    with pytest.raises(ValueError):
        verify.commit([1, -1])
    verify.commit([1, 3])
    for r, count in enumerate([1, 3]):
        reset(seq)
        for p in range(count):
            seq.step(p, tokens[p])
        for name in verify.states:
            got = verify.buffers[name].reshape(2, -1)[r]
            expected = seq.buffers[name].reshape(2, -1)[r]
            np.testing.assert_array_equal(got, expected)
    with pytest.raises(RuntimeError):
        verify.commit(0)


def test_checkpoint_graph_plan(runners):
    _, verify, _ = runners
    assert not verify_plan(compile_graph(verify.graph, 'gfx942', {'B': 2}, {}))
    for name in verify.states:
        assert verify.graph.tensors[name + '_checkpoints'].shape[:2] == ('B', 5)
