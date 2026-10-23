"""CPU-only draft-token proposals for speculative decoding.

Proposals are candidates, not committed output: the caller must verify them with
its target model before delivering tokens or updating the KV cache. This module
has no model, tensor, tokenizer, or GPU dependency.
"""
from __future__ import annotations

from collections.abc import Sequence
from operator import index
from typing import Protocol, runtime_checkable


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@runtime_checkable
class Proposer(Protocol):
    """A stateless source of at most ``max_tokens`` draft token IDs.

    ``token_ids`` is the complete committed history (prompt plus output).
    Implementations must leave it unchanged and return a fresh list. An empty
    list means no proposal; callers can fall back to ordinary target decoding.
    No acceptance probabilities or correctness guarantees are implied.
    """

    def propose(self, token_ids: Sequence[int], max_tokens: int) -> list[int]:
        ...


class NGramProposer:
    """Copy a continuation of a previous occurrence of the history's suffix.

    Search suffix lengths from ``max_n`` down to ``min_n`` (inclusive), then
    choose the most recent earlier match with at least one following token.
    Return up to ``max_tokens`` existing tokens following that match. Matching
    and continuation may overlap the current suffix, but copying stops at the
    end of the supplied history: drafts are never recursively fed back in.

    There is no cross-request state. Search costs O(max_n * history length *
    max_n) in the worst case and uses O(history length) temporary storage.
    """

    def __init__(self, min_n: int = 1, max_n: int = 4):
        self._min_n = _integer(min_n, "min_n", 1)
        self._max_n = _integer(max_n, "max_n", 1)
        if self._min_n > self._max_n:
            raise ValueError("min_n must be <= max_n")

    @property
    def min_n(self) -> int:
        return self._min_n

    @property
    def max_n(self) -> int:
        return self._max_n

    def propose(self, token_ids: Sequence[int], max_tokens: int) -> list[int]:
        """Return a bounded continuation, or [] if no earlier suffix matches.

        Token IDs and the limit must be nonnegative integers (booleans are
        rejected). Empty histories and zero limits are valid.
        """
        max_tokens = _integer(max_tokens, "max_tokens", 0)
        tokens = [_integer(token, "token ID", 0) for token in token_ids]
        if max_tokens == 0:
            return []
        size = len(tokens)
        for n in range(min(self.max_n, size - 1), self.min_n - 1, -1):
            suffix = tokens[size - n:]
            for start in range(size - n - 1, -1, -1):
                if tokens[start:start + n] == suffix:
                    return tokens[start + n:start + n + max_tokens]
        return []
