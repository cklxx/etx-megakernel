import pytest

from etx.models.dsv2lite.multibatch_trace_summary import stats, summarize


def test_median_is_not_midrange():
    assert stats([1., 2., 100.]) == dict(median=2., minimum=1., maximum=100.)


@pytest.mark.parametrize('results', [[], [{'round': 1, 'batch': 1}] * 9])
def test_no_summary_from_incomplete_or_duplicated_tuples(results):
    with pytest.raises(ValueError, match='nine complete'):
        summarize(dict(status='COMPLETE', results=results), {})
