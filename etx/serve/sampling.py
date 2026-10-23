"""CPU token sampling from one vocabulary-sized vector of logits.

NumPy is required when using this module. No GPU framework is imported.
"""
from __future__ import annotations

import numpy as np


def sample(
    logits,
    temperature: float = 0.0,
    top_p: float = 1.0,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Return a token ID, without modifying ``logits``.

    Zero temperature is greedy (first token wins an argmax tie) and does not
    consume random state. Positive temperature applies softmax, then keeps the
    smallest descending-probability prefix with cumulative mass >= ``top_p``.
    Ties in that prefix are ordered by token ID. Sampling uses token-ID order.
    Pass a persistent ``rng`` for a reproducible sequence, or ``seed`` for one
    reproducible draw; supplying both is an error.

    Negative infinity masks a token. NaN, positive infinity, empty/non-vector
    inputs and vectors with no unmasked token are rejected.
    """
    temperature = float(temperature)
    top_p = float(top_p)
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if not np.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')
    if seed is not None and rng is not None:
        raise ValueError('provide seed or rng, not both')
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be a numpy.random.Generator')

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('logits must be a nonempty one-dimensional vector')
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('logits must not contain NaN or positive infinity')
    if not np.isfinite(values).any():
        raise ValueError('logits must contain at least one finite value')
    if temperature == 0:
        return int(np.argmax(values))

    # Subtract before scaling to avoid overflow for large positive logits.
    # Extreme differences or tiny temperatures may safely saturate to -inf.
    with np.errstate(over='ignore', under='ignore'):
        weights = np.exp((values - values.max()) / temperature)
    probabilities = weights / weights.sum()
    if top_p < 1:
        order = np.argsort(-probabilities, kind='stable')
        count = min(
            int(np.searchsorted(np.cumsum(probabilities[order]), top_p)) + 1,
            values.size,
        )
        probabilities[order[count:]] = 0
        probabilities /= probabilities.sum()
    generator = rng if rng is not None else np.random.default_rng(seed)
    return int(generator.choice(values.size, p=probabilities))
