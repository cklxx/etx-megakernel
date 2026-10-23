"""CPU draft equivalence; no GPU performance or device execution claim."""
import numpy as np
import pytest
import torch

from examples.glm53 import module_gate
from examples.glm53.mtp_graph import MTPGraphCPU, MTPUnavailable, prerequisites, require_prerequisites
from examples.spec.mtp_reference import MTPReference, dequantize


@pytest.fixture(scope='module')
def fixture():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = module_gate.module_config('module8')
    tensors = module_gate.tensors(config)
    yield config, tensors, MTPGraphCPU(config, tensors), MTPReference(config, dequantize(tensors))
    torch.set_num_threads(previous_threads)


def inputs(config, n=32):
    c = config['text_config']
    rng = np.random.default_rng(20261023)
    return rng.integers(c['vocab_size'], size=n), rng.standard_normal((n, c['hidden_size'])).astype(np.float32)


def test_preflight(fixture):
    config, tensors, _, _ = fixture
    report = prerequisites(config, tensors)
    assert report['qualified'] is False
    assert not report['blockers']
    assert report['required_comparison'] == dict(positions=32, k=5, reference='vLLM v0.30.0 NumPy MTP')
    prefix = f"model.language_model.layers.{config['text_config']['num_hidden_layers']}."
    main_only = {n: t for n, t in tensors.items() if not n.startswith(prefix)}
    with pytest.raises(MTPUnavailable, match='missing MTP tensors'):
        require_prerequisites(config, main_only)
    with pytest.raises(MTPUnavailable):
        MTPGraphCPU(config, main_only)
    incomplete = tensors.copy()
    del incomplete[prefix + 'shared_head.head.weight']
    with pytest.raises(MTPUnavailable, match='shared_head.head'):
        MTPGraphCPU(config, incomplete)


def test_module8_mtp_32_positions_k5_equal_reference(fixture, monkeypatch):
    config, _, draft, reference = fixture
    ids, hidden = inputs(config)
    calls = []
    select = draft.selected
    def counted(*args):
        calls.append(len(args[0]))
        return select(*args)
    monkeypatch.setattr(draft, 'selected', counted)
    actual = draft.chain(ids, hidden, 5)
    expected = reference.chain(ids, hidden, 5)
    assert actual[0].shape == (5, 32)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(actual[2], expected[2], atol=3e-5, rtol=3e-5)
    assert np.isfinite(actual[1]).all()
    # One validation forward plus one index selection per seed; later draft
    # iterations reuse the index. All 36 positions fit the dense topk prefix,
    # so the oracle's recomputed selection is exactly the shared selection.
    assert len(calls) == 33
    np.testing.assert_array_equal(actual[0], actual[1].argmax(-1))


def test_causal_prefix_and_position_zero(fixture):
    config, _, draft, _ = fixture
    ids, hidden = inputs(config, 7)
    full, _ = draft.forward(ids, hidden)
    short, _ = draft.forward(ids[:4], hidden[:4])
    torch.testing.assert_close(short, full[:4], atol=3e-5, rtol=3e-5)
    a, _ = draft.forward([1], hidden[:1])
    b, _ = draft.forward([2], hidden[:1])
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError, match='contiguous'):
        draft.forward(ids, hidden, np.arange(1, 8))
    with pytest.raises(ValueError, match='positive'):
        draft.chain(ids, hidden, 0)
    with pytest.raises(ValueError, match='nonempty'):
        draft.chain([], hidden[:0], 1)


def test_sparse_selection_matches_independent_reference(fixture):
    config, _, draft, reference = fixture
    rng = np.random.default_rng(8)
    c = config['text_config']
    x = rng.standard_normal((71, c['hidden_size'])).astype(np.float32)
    q = rng.standard_normal((71, c['q_lora_rank'])).astype(np.float32)
    actual = draft.selected(torch.from_numpy(x), torch.from_numpy(q))
    for t, (a, b) in enumerate(zip(actual, reference.selected(x, q))):
        np.testing.assert_array_equal(a.numpy(), b)
        assert torch.all(a <= t)


def test_chain_recycles_normalized_state_and_appends_greedy_token(fixture):
    config, _, draft, _ = fixture
    ids, previous = inputs(config, 3)
    tokens, logits, recycled = draft.chain(ids, previous, 2)
    first, hidden = draft.forward(ids, previous)
    second, last_hidden = draft.forward(
        np.append(ids, first[-1].argmax().item()),
        torch.cat((torch.from_numpy(previous), hidden[-1:])).numpy())
    np.testing.assert_allclose(logits[1, -1], second[-1].numpy(), atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(recycled[-1], last_hidden[-1].numpy(), atol=3e-5, rtol=3e-5)
    assert tokens[1, -1] == second[-1].argmax().item()
