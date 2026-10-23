"""CPU contract and reference checks for history-based draft proposals."""
import random

import pytest

from etx.serve.proposer import NGramProposer, Proposer


def test_protocol_supports_other_draft_sources():
    class FixedProposer:
        def propose(self, token_ids, max_tokens):
            return [7][:max_tokens]

    assert isinstance(FixedProposer(), Proposer)
    assert isinstance(NGramProposer(), Proposer)


def test_copies_continuation_and_caps_length():
    proposer = NGramProposer()
    history = [1, 2, 3, 4, 1, 2]
    assert proposer.propose(history, 10) == [3, 4, 1, 2]
    assert proposer.propose(history, 2) == [3, 4]
    assert history == [1, 2, 3, 4, 1, 2]


def test_longest_match_wins_over_more_recent_short_match():
    assert NGramProposer().propose([1, 2, 8, 2, 9, 1, 2], 1) == [8]


def test_most_recent_match_breaks_ties():
    assert NGramProposer(max_n=2).propose([1, 2, 8, 1, 2, 9, 1, 2], 1) == [9]


def test_ngram_bounds_control_matching():
    history = [1, 2, 8, 2, 9, 1, 2]
    assert NGramProposer(max_n=1).propose(history, 1) == [9]
    assert NGramProposer(min_n=3).propose(history, 1) == []
    assert NGramProposer(min_n=2, max_n=2).propose(history, 1) == [8]


@pytest.mark.parametrize('history', [[], [1], [1, 2, 3], [0]])
def test_no_match(history):
    assert NGramProposer().propose(history, 4) == []


def test_overlap_is_bounded_by_supplied_history():
    assert NGramProposer().propose([5, 5, 5], 10) == [5]
    assert NGramProposer(max_n=2).propose([1, 2, 1, 2, 1], 10) == [2, 1]


def test_zero_limit_and_zero_token_id():
    assert NGramProposer().propose([0, 7, 0], 0) == []
    assert NGramProposer().propose([0, 7, 0], 1) == [7]


def test_results_are_fresh_and_requests_are_isolated():
    proposer = NGramProposer()
    history = (1, 9, 1)
    result = proposer.propose(history, 2)
    result.append(99)
    assert proposer.propose([1], 2) == []
    assert proposer.propose(history, 2) == [9, 1]


@pytest.mark.parametrize('kwargs', [
    {'min_n': 0}, {'max_n': 0}, {'min_n': -1},
    {'min_n': 3, 'max_n': 2},
])
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        NGramProposer(**kwargs)


@pytest.mark.parametrize('value', [True, 1.5, '2', None])
def test_noninteger_parameters(value):
    with pytest.raises(TypeError):
        NGramProposer(min_n=value)
    with pytest.raises(TypeError):
        NGramProposer(max_n=value)
    with pytest.raises(TypeError):
        NGramProposer().propose([1, 1], value)
    with pytest.raises(TypeError):
        NGramProposer().propose([value], 1)


def test_negative_limit_and_token():
    with pytest.raises(ValueError):
        NGramProposer().propose([], -1)
    with pytest.raises(ValueError):
        NGramProposer().propose([1, -1], 2)


def test_matches_independent_reference_on_random_histories():
    rng = random.Random(42)
    for _ in range(500):
        history = [rng.randrange(4) for _ in range(rng.randrange(30))]
        min_n = rng.randint(1, 4)
        max_n = rng.randint(min_n, 7)
        limit = rng.randrange(8)
        matches = [
            (n, start)
            for n in range(min_n, max_n + 1)
            for start in range(max(0, len(history) - n))
            if history[start:start + n] == history[-n:]
        ]
        expected = []
        if matches:
            n, start = max(matches)
            expected = history[start + n:start + n + limit]
        assert NGramProposer(min_n, max_n).propose(history, limit) == expected
