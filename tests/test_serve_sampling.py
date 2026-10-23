"""CPU correctness and reproducibility checks for serving token sampling."""
import numpy as np
import pytest

from etx.serve.sampling import sample


def reference(logits, temperature, top_p, rng):
    """Explicit nucleus construction, independently of the sampler's mask."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    weights = np.exp(scaled - np.max(scaled))
    probs = weights / weights.sum()
    if top_p < 1:
        ranked = sorted(range(len(probs)), key=lambda token: (-probs[token], token))
        kept, mass = [], 0.0
        for token in ranked:
            kept.append(token)
            mass += probs[token]
            if mass >= top_p:
                break
        probs = np.array([p if i in kept else 0.0 for i, p in enumerate(probs)])
        probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_greedy_matches_argmax(dtype):
    rng = np.random.default_rng(52)
    for _ in range(20):
        logits = rng.normal(size=1024).astype(dtype)
        assert sample(logits) == int(np.argmax(logits))
    assert sample([2, 2, 1], top_p=0.01) == 0
    assert sample([-np.inf, -3, -2]) == 2


@pytest.mark.parametrize('temperature', [0.1, 0.7, 1.0, 2.5])
@pytest.mark.parametrize('top_p', [0.01, 0.5, 0.9, 1.0])
def test_matches_numpy_reference(temperature, top_p):
    logits = np.random.default_rng(73).normal(size=37)
    actual_rng = np.random.default_rng(91)
    expected_rng = np.random.default_rng(91)
    actual = [sample(logits, temperature, top_p, rng=actual_rng) for _ in range(100)]
    expected = [reference(logits, temperature, top_p, expected_rng) for _ in range(100)]
    assert actual == expected


def test_seed_and_input_preservation():
    logits = np.array([0., 1., 2., -np.inf])
    original = logits.copy()
    assert sample(logits, 0.8, 0.9, seed=41) == reference(
        logits, 0.8, 0.9, np.random.default_rng(41))
    np.testing.assert_array_equal(logits, original)


def test_greedy_preserves_random_state():
    rng = np.random.default_rng(14)
    sample([1, 2], rng=rng)
    assert rng.random() == np.random.default_rng(14).random()


def test_nucleus_boundary_and_ties():
    # Exactly 0.5 mass retains two tokens; a tie resolves by ascending ID.
    rng = np.random.default_rng(1)
    draws = {sample([0, 0, 0, 0], 1, 0.5, rng=rng) for _ in range(100)}
    assert draws == {0, 1}
    assert sample([0, 0, 0], 1, 0.01, seed=4) == 0


@pytest.mark.parametrize('logits', [[7], [-np.inf, 7, -np.inf], [1e308, -1e308]])
@pytest.mark.parametrize('temperature', [1.0, 1e-300])
def test_degenerate_and_extreme_distributions(logits, temperature):
    assert sample(logits, temperature, seed=3) == int(np.argmax(logits))


@pytest.mark.parametrize('logits', [[], [[1, 2]], 1, [np.nan, 0], [np.inf, 0], [-np.inf]])
def test_invalid_logits(logits):
    with pytest.raises(ValueError):
        sample(logits)


@pytest.mark.parametrize('temperature', [-1, np.nan, np.inf])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match='temperature'):
        sample([1, 2], temperature)


@pytest.mark.parametrize('top_p', [0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_top_p(top_p):
    with pytest.raises(ValueError, match='top_p'):
        sample([1, 2], top_p=top_p)


def test_invalid_random_sources():
    with pytest.raises(ValueError, match='seed or rng'):
        sample([1], seed=1, rng=np.random.default_rng(1))
    with pytest.raises(TypeError, match='Generator'):
        sample([1], rng=object())
